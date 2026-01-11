import torch
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_decode, vae_encode
from diffusion.model.utils import get_weight_dtype
from omegaconf import OmegaConf

from torchvision import transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode, to_pil_image
from diffusion.data.datasets.utils import ASPECT_RATIO_2048, ASPECT_RATIO_2880, ASPECT_RATIO_1024
from diffusion.data.transforms import get_closest_ratio

import io
import time
import json
import tarfile
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tar_file(tar_filename):
    data_dict = {}
    with tarfile.open(tar_filename, 'r') as tar:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is not None:
                if member.name.endswith('.npy'):
                    buffer = io.BytesIO(f.read())
                    data = np.load(buffer)
                    data_dict[member.name] = data
                elif member.name.endswith('.json'):
                    data = json.load(f)
                    data_dict[member.name] = data
    return data_dict


def load_matsynth_and_save_tar(filename, config):
    df = pd.read_parquet(filename)
    df['caption'] = df['metadata'].apply(lambda x: x['gemini2.5_basecolor_tags_desc'])
    df = df[['name', 'basecolor', 'normal', 'roughness', 'metallic', 'height', 'caption']]

    feat_list = []
    caption_list = []
    for idx in range(len(df)):
        basecolor = preprocess_img(Image.open(io.BytesIO(df.loc[idx, 'basecolor']['bytes'])))
        normal = preprocess_img(Image.open(io.BytesIO(df.loc[idx, 'normal']['bytes'])))
        roughness = preprocess_img(Image.open(io.BytesIO(df.loc[idx, 'roughness']['bytes'])))
        metallic = preprocess_img(Image.open(io.BytesIO(df.loc[idx, 'metallic']['bytes'])))
        height = preprocess_img(Image.open(io.BytesIO(df.loc[idx, 'height']['bytes'])))
        rmh = torch.cat([roughness, metallic, height], dim=0)
        caption = [df.loc[idx, 'caption']]

        basecolor = basecolor[None, ...]
        normal = normal[None, ...]
        rmh = rmh[None, ...]

        batch = torch.cat([basecolor, normal, rmh], dim=0).to(device) # Mx3xHxW, M=3; as in HiMat paper
        z = create_img_features(batch, config)
        z = z.cpu().numpy()

        feat_list.append(z)
        caption_list.append(caption)
        print(f"Processed {df.loc[idx, 'name']} with caption: {caption[0]}")
        break

    # save to tar file
    with tarfile.open(filename.replace('.parquet', '.tar'), 'w') as tar:
        for i in range(len(feat_list)):
            name = df.loc[i, 'name']
            feat_np = feat_list[i]
            caption = caption_list[i][0]
            # save numpy array to bytes
            np_bytes = io.BytesIO()
            np.save(np_bytes, feat_np)
            np_bytes.seek(0)

            # create tarinfo for numpy array
            np_info = tarfile.TarInfo(name=f"{name}.npy")
            np_info.size = len(np_bytes.getbuffer())
            tar.addfile(tarinfo=np_info, fileobj=np_bytes)

            # write a dictionary to json file in tar
            json_bytes = io.BytesIO(
                json.dumps({
                    'height': 1024,
                    'width': 1024,
                    'prompt': caption
                }).encode('utf-8')
            )
            json_bytes.seek(0)
            json_info = tarfile.TarInfo(name=f"{name}.json")
            json_info.size = len(json_bytes.getbuffer())
            tar.addfile(tarinfo=json_info, fileobj=json_bytes)

    print(f"Saved features and captions to {filename.replace('.parquet', '.tar')}")

    ###################### TEST PROCESSING SPEED AND EQUALITY ######################
    # now = time.time()
    # basecolor_processedv1 = apply_transformations(basecolor, config)(basecolor)
    # print("Basecolor processed v1 time:", time.time() - now)
    # now = time.time()
    # basecolor_processedv2 = preprocess_img(basecolor)
    # print("Basecolor processed v2 time:", time.time() - now)
    # print("Basecolor processed v1 shape:", basecolor_processedv1.shape)
    # print("Basecolor processed v2 shape:", basecolor_processedv2.shape)
    # print("All equal: ", torch.equal(basecolor_processedv1, basecolor_processedv2))
    # print("Difference between v1 and v2:", torch.abs(basecolor_processedv1 - basecolor_processedv2).max())
    ################################################################################


def create_text_features(text, config):
    with torch.no_grad():
        if not config.text_encoder.chi_prompt:
            max_length_all = config.text_encoder.model_max_length
            prompt = text
        else:
            chi_prompt = "\n".join(config.text_encoder.chi_prompt)
            # prompt = [chi_prompt + i for i in batch[1]]
            prompt = [chi_prompt + text[0]]
            num_sys_prompt_tokens = len(tokenizer.encode(chi_prompt))
            max_length_all = (
                num_sys_prompt_tokens + config.text_encoder.model_max_length - 2
            )  # magic number 2: [bos], [_]
        txt_tokens = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length_all,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        select_index = [0] + list(
            range(-config.text_encoder.model_max_length + 1, 0)
        )  # first bos and end N-1
        y = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:, None][
            :, :, select_index
        ]
        y_mask = txt_tokens.attention_mask[:, None, None][:, :, :, select_index]

    return y, y_mask


def preprocess_img(x):
    ############# OLD VERSION: to_tensor was not handinling height map's uint16 properly #############
    # x = TF.resize(x, (1024, 1024), interpolation=InterpolationMode.BICUBIC)
    # x = TF.to_tensor(x).to(torch.float32)
    # x = TF.normalize(x, mean=[0.5], std=[0.5])
    # return x
    ##################################################################################################
    x = TF.resize(x, (1024, 1024), interpolation=TF.InterpolationMode.BICUBIC)
    x_np = np.array(x)
    if x_np.dtype == np.uint8:
        x_np = x_np.astype(np.float32) / 255.0
    elif x_np.dtype == np.uint16:
        x_np = x_np.astype(np.float32) / 65535.0
    else:
        raise ValueError(f"Unsupported image dtype: {x_np.dtype}")
    if x_np.ndim == 3:
        x_np = np.transpose(x_np, (2, 0, 1))  # HWC to CHW
    elif x_np.ndim == 2:
        x_np = x_np[np.newaxis, :, :]  # HW to 1HW

    x = torch.from_numpy(x_np)
    x = TF.normalize(x, mean=[0.5], std=[0.5])
    return x


def apply_transformations(img, config):
    aspect_ratio = eval(config.model.aspect_ratio_type)
    interpolate_model = InterpolationMode.BICUBIC
    interpolate_model = (
        InterpolationMode.BICUBIC
        if aspect_ratio not in [ASPECT_RATIO_2048, ASPECT_RATIO_2880]
        else InterpolationMode.LANCZOS
    )
    
    # calculate resize and crop sizes
    # height, width = img.shape[2], img.shape[3]
    height, width = img.height, img.width
    closest_size, closest_ratio = get_closest_ratio(height, width, aspect_ratio)
    closest_size = list(map(lambda x: int(x), closest_size))

    if closest_size[0] / height > closest_size[1] / width:
        resize_size = closest_size[0], int(closest_size[0] * width / height)
    else:
        resize_size = int(height * closest_size[1] / width), closest_size[1]

    transforms = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize(resize_size, interpolation=interpolate_model),  # Image.BICUBIC
            T.CenterCrop(closest_size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )

    return transforms


def create_img_features(img, config):
    with torch.no_grad():
        z = vae_encode(config.vae.vae_type, vae, img, config.vae.sample_posterior, device)

    return z


def create_img_text_features(img, text, config):
    img_feat = create_img_features(img, config)
    text_feat, text_mask = create_text_features(text, config)

    return img_feat, text_feat, text_mask
    

def main(cfg):
    ################# Testing tarfile Loading #################
    # tar_filename = glob("/home/woody/vlgm/vlgm116v/matsynth/*.tar")[0]
    # tar_filename = "/home/hpc/vlgm/vlgm116v/MatGen/Sana/data/toy_data/00000000.tar"
    # data_dict = load_tar_file(tar_filename)
    ###########################################################


    config = cfg  # Load or define your configuration here
    global tokenizer, text_encoder
    global vae, vae_dtype

    # build model components
    tokenizer, text_encoder = get_tokenizer_and_text_encoder(
        name=config.text_encoder.text_encoder_name, device=device
    )
    config.vae.weight_dtype = "float32" # set as per default value in AEConfig
    vae_dtype = get_weight_dtype(config.vae.weight_dtype)
    vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, device).to(vae_dtype)
    print(f"Successfully built model components.")

    # create example inputs
    # img = torch.randn(1, 3, 1024, 1024).to(device)
    # img = torch.randn(1, 3, 2048, 2048).to(device)
    # img = to_pil_image(img.squeeze(0).cpu())
    # text = ["a cyberpunk cat with a neon sign that says 'Sana'"]
    # img_feat, text_feat, text_mask = create_img_text_features(img, text, config)
    # print("Image features shape:", img_feat.shape)
    # print("Text features shape:", text_feat.shape)
    # print("Text mask shape:", text_mask.shape)

    # load data here
    parquet_files = glob("/home/woody/vlgm/vlgm116v/matsynth/*.parquet")
    for pf in parquet_files:
        print(f"Processing file: {pf}")
        load_matsynth_and_save_tar(pf, config)


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml")
    main(cfg)