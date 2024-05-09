from torch._tensor import Tensor
from .clip_image_text_loss import CLIPImageTextLoss

class CLIPImageGraphLoss(CLIPImageTextLoss):
    def __init__(self, return_logits: bool = False, target_node: str = None) -> None:
        super().__init__(return_logits)
        self.target_node = target_node
    
    def get_logits(self, pred: dict[str, Tensor]) -> Tensor:
        # gets image scaled logits
        image_embeddings = pred.get("image_features")
        graph_embeddings = pred.get("graph_features")
        if self.target_node:
            graph_embeddings = graph_embeddings[self.target_node]
        scale = pred.get("logit_scale")
        return scale * image_embeddings @ graph_embeddings.T