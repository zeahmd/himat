# HiMat: DiT-based Ultra-High Resolution SVBRDF Generation
> This is an unofficial implementation of HiMat ([https://arxiv.org/abs/2508.07011](https://arxiv.org/abs/2508.07011)) generating materials only at 1K resolution. It was implemented without the supplementary material so there can be discrepancies when compared the original work. <br><br>
<font color="red">**Note:**</font> This codebase was built on top of [SANA](https://github.com/NVlabs/Sana).

## HiMat Pipeline
![jpg](asset/himat_pipeline.jpg)

## Environment Setup
First create a conda environment, activate it, install ninja, and verify the installation as below.

```
conda create -n <env_name> python=3.10.0 -y
conda activate <env_name>
pip install ninja
ninja --version
echo $? (should return exit code 0)
```
If exit code is not 0, please refer to flash-attn [Github](https://github.com/Dao-AILab/flash-attention).


Once the ninja is successfully installed, run the following command to install essential dependencies including flash-attn: <br>

```
bash environment_setup.sh
```

There can be problems with flash-attn installation, in that case please refer to the flash-attn [Github](https://github.com/Dao-AILab/flash-attention). Moreover, do not forget to check the issues section especially this [one](https://github.com/Dao-AILab/flash-attention/issues/1038#issuecomment-2439430999).

## MatSynth Preprocessing

### Tags to Prompt Generation
The text prompts for materials were generated using **Gemini-2.5**. Both tags from MatSynth and the rendered material images were used to capture all the essential material properties.

```
In approximately 50-70 words, describe a [{tags}] surface. Include specific colors, texture details, arrangement patterns, and notable features. Write one detailed paragraph that captures all essential visual properties. Make sure the properties like shape, color, and texture details are consistent with the provided image.
```

### Storing DC-AE Features
Generate latents for SVBRDF maps using DC-AE and store them locally in a **.tar** along with text prompts which can be used with SANA's *`SanaWebDatasetMS`* dataloader.

To generate features:

```
python myscripts/create_features.py
```

Please change the datapaths inside the file and feel free to make other necessary changes.

### Data Format
The latents and text prompts are stored in **.tar** files but we still need to generate metadata *`wids-meta.json`* using the following command:

```
python tools/convert_scripts/convert_ImgDataset_to_WebDatasetMS_format.py
```

Once you have created the metadata, please verify it has information about all of the tar files you have created and it should look as follows:

```
{
    "name": "sana-dev",
    "__kind__": "SANA-WebDataset",
    "wids_version": 1,
    "shardlist": [
        {
            "url": "train-00000-of-00431.tar",
            "nsamples": 9,
            "filesize": 3563520
        },
        {
            "url": "train-00001-of-00431.tar",
            "nsamples": 3,
            "filesize": 1198080
        },
        .....
        .....
        .....
    ]
}
```

## Training Details
To run the training, please edit `train.sh` as per requirement and run it as follows:

```
bash train_scripts/train.sh
```

Please remember to edit `Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml` to update the paths to the pretrained models such as Gemma-2, DC-AE, and SANA1.5@1024px.

> <font color="red">**Note:**</font> To train HiMat, you need at least 2 GPUs each having a VRAM of $\ge$ 40GB.

## Inference
To run inference, please first download the himat model from Hugging Face, e.g., as below or some other way:

```
hf download zeahmd/himat_unofficial --repo-type model --local-dir .
```

Please update the model path in the aforementioned `Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml` config file. Apart from himat, you will also have to specify the paths for Gemma-2 and DC-AE. <br>

Once the environment is setup and the models are downloaded, you simply need to run inference as below:

```
python generate_maps.py
```

## Results
| Text Prompt | Rendering | Basecolor | Normal | Roughness | Metallic | Height |
| --- | --- | --- | --- | --- | --- | --- |
| a surface showing white bathroom tiles with an exquisite pattern | ![gif](asset/tiles/rendering.gif) | ![png](asset/tiles/albedo.png) | ![png](asset/tiles/normal.png) | ![png](asset/tiles/roughness.png) | ![png](asset/tiles/metallic.png) | ![png](asset/tiles/height.png) |
| closeup detail of red leather texture background | ![gif](asset/leather/rendering.gif) | ![png](asset/leather/albedo.png) | ![png](asset/leather/normal.png) | ![png](asset/leather/roughness.png) | ![png](asset/leather/metallic.png) | ![png](asset/leather/height.png) |
| a sci-fi golden circuit agains the dark background | ![gif](asset/circuit/rendering.gif) | ![png](asset/circuit/albedo.png) | ![png](asset/circuit/normal.png) | ![png](asset/circuit/roughness.png) | ![png](asset/circuit/metallic.png) | ![png](asset/circuit/height.png) |
| a stone with shiny patches | ![gif](asset/stone/rendering.gif) | ![png](asset/stone/albedo.png) | ![png](asset/stone/normal.png) | ![png](asset/stone/roughness.png) | ![png](asset/stone/metallic.png) | ![png](asset/stone/height.png) |
| a brown wood texture background | ![gif](asset/wood/rendering.gif) | ![png](asset/wood/albedo.png) | ![png](asset/wood/normal.png) | ![png](asset/wood/roughness.png) | ![png](asset/wood/metallic.png) | ![png](asset/wood/height.png) | 

