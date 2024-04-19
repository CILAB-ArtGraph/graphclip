from torch._tensor import Tensor
from .clip_image_text_loss import CLIPImageTextLoss

class CLIPImageGraphLoss(CLIPImageTextLoss):
    def __init__(self, return_logits: bool = False) -> None:
        super().__init__(return_logits)
    
    def get_logits(self, pred: dict[str, Tensor]) -> Tensor:
        # gets image scaled logits
        image_embeddings = pred.get("image_features")
        graph_embeddings = pred.get("graph_features")
        scale = pred.get("logit_scale")
        return scale * image_embeddings @ graph_embeddings.T