# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import numpy as np
from PIL import Image


class imageSuperNet:
    def __init__(self, config) -> None:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=4,
            model_path=config.realesrgan_ckpt_path,
            dni_weight=None,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
            gpu_id=None,
        )
        self.upsampler = upsampler
        self.output_size = config.realesrgan_output_size

    def __call__(self, image):
        # Perform the initial 4x upscale (e.g., 512x512 -> 2048x2048)
        output, _ = self.upsampler.enhance(np.array(image))
        
        # Convert to PIL Image for resizing
        output_image = Image.fromarray(output)

        # If the upscaled image is not at the final target resolution, resize it.
        # This handles cases where we need to go from 2048 -> 4096 for 4K textures.
        if output_image.size[0] != self.output_size:
            output_image = output_image.resize((self.output_size, self.output_size), Image.LANCZOS)
            
        return output_image