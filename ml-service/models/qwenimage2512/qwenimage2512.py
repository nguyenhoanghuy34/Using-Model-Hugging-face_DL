# qwenimage2512.py
from diffusers import DiffusionPipeline
import torch
import os

def main():
    # Chạy trên CPU
    device = "cpu"
    torch_dtype = torch.float32

    print("Đang load model Qwen-Image-2512 (CPU, có thể mất vài phút)...")
    model_name = "Qwen/Qwen-Image-2512"
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)

    # Prompt sinh ảnh
    prompt = (
        "Một cô gái trẻ châu Á, tóc dài xoăn, mặc váy sáng màu, "
        "đứng trong hội chợ anime, ánh sáng tự nhiên, gương mặt tươi vui."
    )

    # Negative prompt tránh lỗi
    negative_prompt = (
        "mờ, biến dạng, tay/chân sai, AI-look, quá bóng, chữ nhòe, "
        "quá bão hòa, biến dạng cơ thể"
    )

    # Kích thước ảnh nhỏ để test nhanh
    width, height = 512, 512

    # Tạo folder outputs nếu chưa có
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "qwen_image_output.png")

    print("Đang sinh ảnh, vui lòng chờ...")
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=30,  # CPU -> ít bước để nhanh
        true_cfg_scale=4.0,
        generator=torch.Generator(device=device).manual_seed(42)
    ).images[0]

    # Lưu ảnh
    image.save(output_path)
    print(f"Ảnh đã được lưu: {output_path}")

if __name__ == "__main__":
    main()
