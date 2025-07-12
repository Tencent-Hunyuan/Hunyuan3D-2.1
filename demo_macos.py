import sys
import os
import torch
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(current_dir, 'hy3dshape'))
sys.path.insert(0, os.path.join(current_dir, 'hy3dpaint'))

INPUT_IMAGE_PATH = os.path.join(current_dir, 'assets', 'demo.png')


def setup_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        if hasattr(torch.backends.mps, "enable_mem_efficient"):
            torch.backends.mps.enable_mem_efficient()
        print("Using Apple MPS acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU (MPS support not detected)")

    return device


def generate_mesh(low_vram=False):

    device = setup_device()

    if low_vram:
        print("Low VRAM mode enabled")
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        torch.set_num_threads(1)

    try:
        print(f"Loading fixed input image: {INPUT_IMAGE_PATH}")
        image = Image.open(INPUT_IMAGE_PATH).convert("RGBA")
        # Attempt to remove background
        from hy3dshape.rembg import BackgroundRemover
        if image.mode == 'RGB':
            remover = BackgroundRemover()
            image = remover(image)

        # Load model
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

        shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            'tencent/Hunyuan3D-2.1',
            torch_dtype=torch.float32,
            device=device
        )

        shape_pipeline.to(device)

        # Generate 3D mesh
        print("Generating 3D mesh...")
        try:
            mesh = shape_pipeline(image=image, batch_size=1)[0]
        except Exception as e:
            print(f"Mesh generation failed: {e}")
            return None

        output_path = os.path.join(current_dir, "demo.obj")
        mesh.export(output_path)
        print(f"Mesh saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"Unexpected error occurred during processing: {e}")
        return None


def main():
    print("Hunyuan3D-2.1 Mesh Generator")
    print(f"Fixed input image: {INPUT_IMAGE_PATH}")

    result = generate_mesh(
        low_vram=False
    )

    if result:
        print("Mesh generation succeeded!")
        print(f"MESH_OUTPUT:{result}")
    else:
        print("Mesh generation failed")
        sys.exit(1)


if __name__ == "__main__":
    # macOS-specific settings
    if sys.platform == "darwin":
        # Disable CUDA
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        # Enable MPS fallback
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    main()