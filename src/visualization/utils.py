from src.utils import StrEnum

class SessionStateKey(StrEnum):
    IDX2CLASS = "idx2class"
    CLASS2IDX = "class2idx"
    PREDICTION = "prediction"
    LOGITS = "logits"
    IMG_FEATS = "img_feats"
    TXT_FEATS = "class_feats"
    CASE_IMG = "case_img"
    IMG = "image"
    CAPTION = "caption"
    USE_COL_WIDTH = "use_column_width"
    ST_IMG = "st_img"
    IMG_PTH = "img_pth"
    IMG_EXP = "img_exp"
    TXT_BOX = "txt_box"
    