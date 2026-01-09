import torch
from app.sana_pipeline import SanaPipeline
from torchvision.utils import save_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device=device).manual_seed(42)

sana = SanaPipeline("configs/sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml")
sana.from_pretrained("/home/hpc/vlgm/vlgm116v/.cache/huggingface/hub/models--Efficient-Large-Model--SANA1.5_1.6B_1024px/snapshots/a7b7e39161522e12d98045d2c1b5da5101d712f8/checkpoints/SANA1.5_1.6B_1024px.pth")
prompt = 'a cyberpunk cat with a neon sign that says "Sana"'

while True:
    prompt = input("Enter your prompt: ")  # Get user input for the prompt
    img_name = input("Enter the output image name (with .png extension): ")  # Get user input for the image name
    
    image = sana(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=4.5,
        pag_guidance_scale=1.0,
        num_inference_steps=20,
        generator=generator,
    )
    save_image(image, f'output/{img_name}', nrow=1, normalize=True, value_range=(-1, 1))
