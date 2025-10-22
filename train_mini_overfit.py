"""
train_mini_overfit.py
单卡 mini 数据集过拟合专用脚本
用法:
    python train_mini_overfit.py
"""
import os
import warnings
warnings.filterwarnings("ignore")

import torch
from pathlib import Path
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

CONFIG_YAML   = "configs/overfit_mini.yaml"     
OUTPUT_DIR    = "runs/overfit_mini"


def main():
    # 1. 读取配置
    cfg = OmegaConf.load(CONFIG_YAML)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 2. 构造 data & model
    from hy3dshape.utils import instantiate_from_config
    data  = instantiate_from_config(cfg.dataset)
    model = instantiate_from_config(cfg.model)

    # 3. 回调：每 200 步保存一次
    ckpt_cb = ModelCheckpoint(
        dirpath  = os.path.join(OUTPUT_DIR, "ckpt"),
        filename = "step-{step:06d}",
        save_top_k = -1,
        every_n_train_steps = 200
    )

    # 4. Trainer
    trainer = pl.Trainer(
        max_steps       = cfg.training.steps,
        accelerator     = "gpu",
        devices         = 1,
        precision       = "bf16",
        default_root_dir= OUTPUT_DIR,
        callbacks       = [ckpt_cb],
        log_every_n_steps = 50
    )

    # 5. 开始训练小样本模型
    trainer.fit(model, datamodule=data)

if __name__ == "__main__":
    main()

