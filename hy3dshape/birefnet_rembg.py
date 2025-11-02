import torch
from torchvision import transforms
from PIL import Image
from transformers import AutoModelForImageSegmentation

class BiRefNetRemover:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        )
        self.model.to(self.device)
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    @torch.no_grad()
    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Removes the background from a PIL image.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to(self.device)
        
        preds = self.model(input_images)[-1].sigmoid().cpu()
        
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        
        mask = pred_pil.resize(image_size)
        
        image.putalpha(mask)
        return image 