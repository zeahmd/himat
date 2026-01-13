import torch
from app.sana_pipeline import SanaPipeline
from torchvision.utils import save_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device=device).manual_seed(42)

sana = SanaPipeline("configs/sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml")
checkpoint = torch.load("/home/woody/vlgm/vlgm116v/output/debug/checkpoints/epoch_17_step_5000.pth", map_location=device, weights_only=False)
sana.load_state_dict(checkpoint["state_dict"], strict=False)
sana.to(device)
sana.eval()


prompts = ['a surface with white and black checkered pattern'] * 3
    
image = sana(
    prompt=prompts,
    height=1024,
    width=1024,
    guidance_scale=4.5,
    pag_guidance_scale=1.0,
    num_inference_steps=20,
    generator=generator,
)
save_image(image[0], f'output/tex_map_basecolor.png', nrow=1, normalize=True, value_range=(-1, 1))
