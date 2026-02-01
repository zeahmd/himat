import torch
from app.sana_pipeline import SanaPipeline
from torchvision.utils import save_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device=device).manual_seed(42)

sana = SanaPipeline("configs/sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml")
# checkpoint = torch.load("/home/woody/vlgm/vlgm116v/output/debug/checkpoints/latest.pth", map_location=device, weights_only=False)
# checkpoint = torch.load("/home/woody/vlgm/vlgm116v/old_checkpoints/latest.pth", map_location=device, weights_only=False)
# sana.load_state_dict(checkpoint["state_dict"], strict=False)
sana.from_pretrained("/home/woody/vlgm/vlgm116v/output/debug/checkpoints/latest.pth")
sana.to(device)
sana.eval()


prompts = ['a surface with white and black checkered pattern'] * 3
    
while True:
    prompt = input("Enter your prompt: ")  # Get user input for the prompt
    img_name = input("Enter the output image name (without .png extension): ")  # Get user input for the image name
    
    # prompt = [prompt] * 3  # Create a list of identical prompts for batch processing
    prompt = ["albedo: " + prompt, "normal: " + prompt, "roughness, metallic, height: " + prompt]
    image = sana(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=4.5,
        pag_guidance_scale=1.0,
        num_inference_steps=20,
        generator=generator,
    )
    image = (image / 2 + 0.5).clamp(0, 1)
    save_image(image[0], f'output/{img_name}_albedo.png', nrow=1, normalize=True, value_range=(-1, 1))
    save_image(image[1], f'output/{img_name}_normal.png', nrow=1, normalize=True, value_range=(-1, 1))
    save_image(image[2], f'output/{img_name}_roughness_metallic_height.png', nrow=1, normalize=True, value_range=(-1, 1))