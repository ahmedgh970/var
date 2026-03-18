import argparse

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, transforms

from var.models.tokenizer.vqvae import VQVAE


def normalize_01_to_pm1(x: torch.Tensor) -> torch.Tensor:
    return x.mul(2.0).sub_(1.0)


def denormalize_pm1_to_01(x: torch.Tensor) -> torch.Tensor:
    return x.add(1.0).mul(0.5).clamp(0.0, 1.0)


def load_image_tensor(image_path: str, image_size: int) -> torch.Tensor:
    tf = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=InterpolationMode.LANCZOS),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            normalize_01_to_pm1,
        ]
    )
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--z-channels", type=int, default=32)
    parser.add_argument("--ch", type=int, default=128)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--using-znorm", action="store_true")
    parser.add_argument("--quantizer-type", type=str, default="multi", choices=["single", "multi"])
    parser.add_argument("--patch-nums", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", type=str, default="experiments/recon2.png")
    args = parser.parse_args()

    device = torch.device(args.device)

    model = VQVAE(
        vocab_size=args.vocab_size,
        z_channels=args.z_channels,
        ch=args.ch,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        dropout=0.0,
        beta=args.beta,
        using_znorm=args.using_znorm,
        patch_nums=tuple(args.patch_nums),
        quantizer_type=args.quantizer_type,
    ).to(device)

    model.eval()

    x = load_image_tensor(args.image, args.image_size).to(device)

    with torch.no_grad():
        z = model.encode_latent(x)
        h, ms_idx, vq_loss = model(x)

    print(f"input shape: {tuple(x.shape)}")
    print(f"latent shape: {tuple(z.shape)}")
    print(f"reconstructed shape: {tuple(h.shape)}")
    print(f"quantizer type: {args.quantizer_type}")
    print(f"num scales: {len(ms_idx)}")
    print(f"indices shapes: {[tuple(idx.shape) for idx in ms_idx]}")
    print(f"indices dtype: {ms_idx[0].dtype}")
    print(f"vq loss: {float(vq_loss):.6f}")

    x_vis = denormalize_pm1_to_01(x[0].detach().cpu())
    h_vis = denormalize_pm1_to_01(h[0].detach().cpu())
    side_by_side = torch.cat([x_vis, h_vis], dim=2)  # C x H x (2W)
    out = (side_by_side.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    Image.fromarray(out).save(args.save_path)
    print(f"saved visualization: {args.save_path}")


if __name__ == "__main__":
    main()
