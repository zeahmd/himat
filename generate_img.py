import torch
from app.sana_pipeline import SanaPipeline
from torchvision.utils import save_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device=device).manual_seed(42)

sana = SanaPipeline("configs/sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml")
# the following from_pretrained loads checkpoints but contains only 1 dict with key 'state_dict'
# these weights are only for the SANA model, not the entire pipeline (such as DC-AE and Gemma-2)
sana.from_pretrained("/home/hpc/vlgm/vlgm116v/.cache/huggingface/hub/models--Efficient-Large-Model--SANA1.5_1.6B_1024px/snapshots/a7b7e39161522e12d98045d2c1b5da5101d712f8/checkpoints/SANA1.5_1.6B_1024px.pth")

# these training checkpoints contain the many dicts including 'state_dict' for SANA model, 'optimizer', 'lr_scheduler', etc.
# we only need 'state_dict' for inference, so we load that into the sana model
# Sanity Check: set(new_checkpoint['state_dict'].keys()) - set(sana.state_dict().keys()) should contain only
# the CrossStitchBlock parameters if those were newly added
# checkpoint = torch.load("/home/woody/vlgm/vlgm116v/output/debug/checkpoints/latest.pth", map_location=device, weights_only=False)
# checkpoint = torch.load("/home/hpc/vlgm/vlgm116v/.cache/huggingface/hub/models--Efficient-Large-Model--SANA1.5_1.6B_1024px/snapshots/a7b7e39161522e12d98045d2c1b5da5101d712f8/checkpoints/SANA1.5_1.6B_1024px.pth", map_location=device, weights_only=False)
# sana.load_state_dict(checkpoint["state_dict"], strict=False)
# sana.from_pretrained("/home/hpc/vlgm/vlgm116v/output/debug/checkpoints/latest.pth")
sana.to(device)
sana.eval()



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
