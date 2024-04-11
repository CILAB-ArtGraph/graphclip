import torch
from torch.nn.functional import cross_entropy


class CLIPImageTextLoss(torch.nn.Module):
    def __init__(self, return_logits: bool = False) -> None:
        super().__init__()
        self.return_logits=return_logits

    def get_logits(self, pred: dict[str, torch.Tensor]) -> torch.Tensor:
        # gets image scaled logits
        image_embeddings = pred.get("image_features")
        text_embeddings = pred.get("text_features")
        scale = pred.get("logit_scale")
        return scale * image_embeddings @ text_embeddings.T

    def forward(
        self, pred: dict[str, torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        image_logits = self.get_logits(pred=pred)
        out =  {"loss": cross_entropy(input=image_logits, target=target)}
        if self.return_logits:
            out["logits"] = image_logits
        return out 
