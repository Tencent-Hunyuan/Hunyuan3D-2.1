
<p align="center">
  <img src="assets/images/teaser.jpg">
</p>


<div align="center">
  <a href=https://3d.hunyuan.tencent.com target="_blank"><img src=https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage height=22px></a>
  <a href=https://huggingface.co/spaces/tencent/Hunyuan3D-2.1  target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Demo-276cb4.svg height=22px></a>
  <a href=https://huggingface.co/tencent/Hunyuan3D-2.1 target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px></a>
  <a href=https://3d-models.hunyuan.tencent.com/ target="_blank"><img src= https://img.shields.io/badge/Page-bb8a2e.svg?logo=github height=22px></a>
  <a href=https://discord.gg/dNBrdrGGMa target="_blank"><img src= https://img.shields.io/badge/Discord-white.svg?logo=discord height=22px></a>
  <a href=https://arxiv.org/abs/2501.12202 target="_blank"><img src=https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px></a>
  <a href=https://x.com/TencentHunyuan target="_blank"><img src=https://img.shields.io/badge/Hunyuan-black.svg?logo=x height=22px></a>
 <a href="#community-resources" target="_blank"><img src=https://img.shields.io/badge/Community-lavender.svg?logo=homeassistantcommunitystore height=22px></a>
</div>

[//]: # (  <a href=# target="_blank"><img src=https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px></a>)

[//]: # (  <a href=# target="_blank"><img src= https://img.shields.io/badge/Colab-8f2628.svg?logo=googlecolab height=22px></a>)

[//]: # (  <a href="#"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/v/mulankit?logo=pypi"  height=22px></a>)
<br>

## 🔥 News

- Jun 13, 2025: 🤗 We release the first production-ready 3D asset generation model, Hunyuan3D-2.1!

> Join our **[Wechat](#)** and **[Discord](https://discord.gg/dNBrdrGGMa)** group to discuss and find help from us.

| Wechat Group                                     | Xiaohongshu                                           | X                                           | Discord                                           |
|--------------------------------------------------|-------------------------------------------------------|---------------------------------------------|---------------------------------------------------|
| <img src="assets/qrcode/wechat.png"  height=140> | <img src="assets/qrcode/xiaohongshu.png"  height=140> | <img src="assets/qrcode/x.png"  height=140> | <img src="assets/qrcode/discord.png"  height=140> |        



## ☯️ **Hunyuan3D 2.1**

### Architecture

Tencent Hunyuan3D-2.1 is a scalable 3D asset creation system that advances state-of-the-art 3D generation through two pivotal innovations: Fully Open-Source Framework and  Physically-Based Rendering (PBR) Texture Synthesis. For the first time, the system releases full model weights and training code, enabling community developers to directly fine-tune and extend the model for diverse downstream applications. This transparency accelerates academic research and industrial deployment. Moreover, replacing the prior RGB-based texture model, the upgraded PBR pipeline leverages  physics-grounded material simulation  to generate textures with photorealistic light interaction (e.g., metallic reflections, subsurface scattering).

<p align="left">
  <img src="assets/images/pipeline.png">
</p>

### Performance

We have evaluated Hunyuan3D 2.1 with other open-source as well as close-source 3d-generation methods.
The numerical results indicate that Hunyuan3D 2.1 surpasses all baselines in the quality of generated textured 3D assets
and the condition following ability.

| Model                   | ULIP-T(⬆)   | ULIP-I(⬆) | Uni3D-T(⬆)      | Uni3D-I(⬆) |
|-------------------------|-----------|-------------|-------------|---------------|
| Michelangelo  | 0.0752     | 0.1152      | 0.2133     | 0.2611         |
| Craftsman | 0.0745     | 0.1296      | 0.2375     | 0.2987         |
| TripoSG | 0.0767     | 0.1225      | 0.2506     | 0.3129       |
| Step1X-3D | 0.0735     | 0.1183      | 0.2554     | 0.3195         |
| Trellis | 0.0769     | 0.1267      | 0.2496     | 0.3116         |
| Direct3D-S2 | 0.0706     | 0.1134      | 0.2346     | 0.2930         |
| Hunyuan3D-Shape-2.1           | **0.0774** | **0.1395**  | **0.2556** | **0.3213** |


| Model                   | CLIP-FiD(⬇)   | CMMD(⬇) | CLIP-I(⬆)      | LPIPS(⬇) |
|-------------------------|-----------|-------------|-------------|---------------|
| SyncMVD-IPA  | 28.39     | 2.397      | 0.8823     | 0.1423         |
| TexGen | 28.24     | 2.448      | 0.8818     | 0.1331         |
| Hunyuan3D-2.0 | 26.44     | 2.318     | 0.8893     | 0.1261         |
| Hunyuan3D-Paint-2.1           | **24.78** | **2.191**  | **0.9207** | **0.1211**     |



## 🎁 Models Zoo

It takes 10 GB VRAM for shape generation, 21GB for texture generation and 29GB for shape and texture generation in total.


| Model                      | Description                 | Date       | Size | Huggingface                                                                               |
|----------------------------|-----------------------------|------------|------|-------------------------------------------------------------------------------------------| 
| Hunyuan3D-Shape-v2-1         | Image to Shape Model        | 2025-01-21 | 3.3B | [Download](https://huggingface.co/tencent/Hunyuan3D-2.1/tree/main/hunyuan3d-dit-v2-1)         |
| Hunyuan3D-Paint-v2-1       | Texture Generation Model    | 2025-01-21 | 2B | [Download](https://huggingface.co/tencent/Hunyuan3D-2.1/tree/main/hunyuan3d-paint-v2-1)       |


## 🤗 Get Started with Hunyuan3D 2.1

Hunyuan3D 2.1 supports Macos, Windows, Linux. You may follow the next steps to use Hunyuan3D 2.1 via:

### Install Requirements
We test our model on an A100 GPU with Python 3.10 and PyTorch 2.5.1+cu124.
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

cd hy3dpaint/custom_rasterizer
pip install -e .
cd ../..
cd hy3dpaint/DifferentiableRenderer
bash compile_mesh_painter.sh
cd ../..
```

### Code Usage

We designed a diffusers-like API to use our shape generation model - Hunyuan3D-Shape and texture synthesis model -
Hunyuan3D-Paint.

```python
import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')
from textureGenPipeline import Hunyuan3DPaintPipeline
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

# let's generate a mesh first
shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')
mesh_untextured = shape_pipeline(image='assets/demo.png')[0]

from utils.torchvision_fix import apply_fix

apply_fix()

paint_pipeline = Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(max_num_view=6, resolution=512))
mesh_textured = paint_pipeline(mesh_path, image_path='assets/demo.png')
```


### Gradio App

You could also host a [Gradio](https://www.gradio.app/) App in your own computer via:


```bash
python3 gradio_app.py \
  --model_path tencent/Hunyuan3D-2.1 \
  --subfolder hunyuan3d-dit-v2-1 \
  --texgen_model_path tencent/Hunyuan3D-2.1 \
  --low_vram_mode
```


## 🔗 BibTeX

If you found this repository helpful, please cite our reports:

```bibtex
@misc{hunyuan3d22025tencent,
    title={Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation},
    author={Tencent Hunyuan3D Team},
    year={2025},
    eprint={2501.12202},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{yang2024hunyuan3d,
    title={Hunyuan3D 1.0: A Unified Framework for Text-to-3D and Image-to-3D Generation},
    author={Tencent Hunyuan3D Team},
    year={2024},
    eprint={2411.02293},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Acknowledgements

We would like to thank the contributors to
the [TripoSG](https://github.com/VAST-AI-Research/TripoSG), [Trellis](https://github.com/microsoft/TRELLIS),  [DINOv2](https://github.com/facebookresearch/dinov2), [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), [HuggingFace](https://huggingface.co), [CraftsMan3D](https://github.com/wyysf-98/CraftsMan3D), 
and [Michelangelo](https://github.com/NeuralCarver/Michelangelo/tree/main) repositories, for their open research and
exploration.

## Star History

<a href="https://star-history.com/#Tencent/Hunyuan3D-2.1&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Tencent/Hunyuan3D-2.1&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Tencent/Hunyuan3D-2.1&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Tencent/Hunyuan3D-2.1&type=Date" />
 </picture>
</a>
