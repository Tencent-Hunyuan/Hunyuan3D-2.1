import torch
from torchvision import transforms
from PIL import Image
from modelscope import AutoModelForImageSegmentation

class RMBG:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.transform = None
    
    def _load_model(self):
        if self.model is None:
            
            self.model = AutoModelForImageSegmentation.from_pretrained(
                'AI-ModelScope/RMBG-2.0', 
                trust_remote_code=True
            )
            torch.set_float32_matmul_precision('high')
            self.model.to(self.device)
            self.model.eval()
            
            self.transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __call__(self, image):

        self._load_model()
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        original_size = image.size
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred = self.model(input_tensor)[-1].sigmoid().cpu()
        
        mask = pred[0].squeeze()
        mask_pil = transforms.functional.to_pil_image(mask)
        mask_resized = mask_pil.resize(original_size, Image.LANCZOS)
        
        result = image.convert("RGBA")
        result.putalpha(mask_resized)
        
        return result