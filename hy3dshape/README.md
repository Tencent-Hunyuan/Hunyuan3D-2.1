# Hunyuan3D-2.1-Shape

## Quick Inference

Given a reference image `image.png`, you can run inference using the following code. The result will be saved as `demo.glb`.

```bash
python3 minimal_demo.py
```

**Memory Recommendation:** For we recommend using a GPU with at least **10GB VRAM**.

## LoRA inference

Install `peft==0.14.0` alongside the repository dependencies. For the official
2.1 denoiser, add this under `model.params` in the training YAML:

```yaml
lora_config:
  rank: 16
  target_modules: [to_q, to_k, to_v]
```

The trainer uses `lora_alpha=rank`; keep the same rank and target modules when
loading a complete checkpoint. Save the trained PEFT denoiser as an adapter directory:

```python
# diffuser is the trained Diffuser; export after gathering distributed weights.
diffuser.model.save_pretrained("shape_adapter")
```

Load that adapter onto the same base model used for training:

```python
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
pipe.load_lora_weights("shape_adapter")
mesh = pipe(image="image.png")[0]
mesh.export("lora.glb")
```

For a complete training checkpoint, pass its original LoRA configuration:

```python
pipe.load_lora_weights(
    "pytorch_model.bin",
    lora_config={"rank": 16, "target_modules": ["to_q", "to_k", "to_v"]},
)
```

This accepts a consolidated DeepSpeed state dict (`_forward_module.model.*`)
or a Lightning checkpoint (`state_dict` containing `model.*`). The denoiser
architecture must match training; `minimal_demo_with_ckpt.py` reconstructs it
from the training YAML for custom architectures. All denoiser weights, including
LoRA weights, must be present in a complete checkpoint. Keys and tensor shapes
are checked before copying any weights. A mismatch raises `ValueError` and
removes the temporary adapter layers, so the base model can be used again.
Adapter directories must contain exactly the expected adapter keys.
Load before enabling CPU offload or compiling the pipeline, and use a fresh
base model to switch adapters. This loader restores denoiser weights; it does
not restore a separately trained image encoder or resume optimizer state.

# Training

Here we demonstrate the complete training workflow of DiT on a small dataset.

## Data Preprocessing

The rendering and watertight mesh generation process is described in detail in [this document](tools/README.md). After preprocessing, the dataset directory structure should look like the following:

```yaml
dataset/preprocessed/{uid}
├── geo_data
│   ├── {uid}_sdf.npz
│   ├── {uid}_surface.npz
│   └── {uid}_watertight.obj
└── render_cond
    ├── 000.png
    ├── ...
    ├── 023.png
    ├── mesh.ply
    └── transforms.json
```

We provide a preprocessed mini_dataset containing 8 cases (all sourced from Objaverse-XL) as `tools/mini_trainset`, which can be used directly for DiT overfitting training experiments.

## Launching Training

We provide example configuration files and launch scripts for reference. By default, the training runs on a single node with 8 GPUs using DeepSpeed. Users can modify the configurations and scripts as needed to suit their environment.

Configuration File
```
configs/hunyuandit-mini-overfitting-flowmatching-dinog518-bf16-lr1e4-512.yaml
```
Launch Script

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export num_gpu_per_node=8

export node_num=1
export node_rank=0
export master_ip=0.0.0.0 # set your master_ip

# export config=configs/hunyuandit-finetuning-flowmatching-dinol518-bf16-lr1e5-4096.yaml
# export output_dir=output_folder/dit/fintuning_lr1e5
export config=configs/hunyuandit-mini-overfitting-flowmatching-dinol518-bf16-lr1e4-4096.yaml
export output_dir=output_folder/dit/overfitting_depth_16_token_4096_lr1e4

bash scripts/train_deepspeed.sh $node_num $node_rank $num_gpu_per_node $master_ip $config $output_dir
```
