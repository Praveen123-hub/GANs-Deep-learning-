import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from load_weights import load_model

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(device=device)

@torch.no_grad()
def upscale_image(
    img_path,
    save_path="output.png",
    tile_size=256,
    tile_pad=10,
    scale=4
):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    print("Input size:", img.size)

    img = to_tensor(img).unsqueeze(0).to(device)

    output = torch.zeros(
        (1, 3, h * scale, w * scale),
        device=device
    )

    tiles_x = (w + tile_size - 1) // tile_size
    tiles_y = (h + tile_size - 1) // tile_size

    for y in range(tiles_y):
        for x in range(tiles_x):
            x0 = x * tile_size
            y0 = y * tile_size
            x1 = min(x0 + tile_size, w)
            y1 = min(y0 + tile_size, h)

            px0 = max(x0 - tile_pad, 0)
            py0 = max(y0 - tile_pad, 0)
            px1 = min(x1 + tile_pad, w)
            py1 = min(y1 + tile_pad, h)

            tile = img[:, :, py0:py1, px0:px1]
            sr_tile = model(tile).clamp(0, 1)

            ox0 = (x0 - px0) * scale
            oy0 = (y0 - py0) * scale
            ox1 = ox0 + (x1 - x0) * scale
            oy1 = oy0 + (y1 - y0) * scale

            output[
                :, :,
                y0 * scale:y1 * scale,
                x0 * scale:x1 * scale
            ] = sr_tile[:, :, oy0:oy1, ox0:ox1]

    out_img = to_pil_image(output.squeeze().cpu())
    out_img.save(save_path)
    print("Saved:", save_path)


if __name__ == "__main__":
    input_img = r"path to input image"
    output_img = r"path to save the image"

    upscale_image(
        input_img,
        output_img,
        tile_size=256,   
        tile_pad=10
    )
