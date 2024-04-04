from src.utils import StrEnum


class ParameterKeys(StrEnum):
    MODEL = "model"
    NAME = "name"
    STATE_DICT = "state_dict"
    PARAMS = "params"
    DATASET = "dataset"
    DATALOADER = "dataloader"
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    METRICS = "metrics"