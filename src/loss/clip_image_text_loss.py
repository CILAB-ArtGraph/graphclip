import torch
from torch.nn.functional import cross_entropy

class CLIPImageTextLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def get_logits(self, pred: dict[str, torch.Tensor]) -> torch.Tensor:
        # gets image scaled logits
        image_embeddings = pred.get("image")
        text_embeddings = pred.get("text")
        scale = pred.get("scale")
        return scale * image_embeddings @ text_embeddings.T 
    
    def forward(self, pred: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        image_logits = self.get_logits(pred=pred)
        return cross_entropy(input=image_logits, target=target)