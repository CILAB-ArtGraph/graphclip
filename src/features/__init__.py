from .extract_images_feats import extract_vis_clip
from .extract_label_feats import extract_label_clip
from src.utils import StrEnum



class TypeExtraction(StrEnum):
    VISUAL = "visual"
    LABEL = "label"


def extract_features(parameters, t):
    if t == TypeExtraction.VISUAL:
        extract_vis_clip(parameters=parameters)
    elif t == TypeExtraction.LABEL:
        extract_label_clip(parameters=parameters)
    else:
        raise ValueError(f"Unknown type {t}. Correct types are [visual, label]")