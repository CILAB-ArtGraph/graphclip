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
    TOKENIZER = "tokenizer"
    BAR = "bar"
    DEVICE = "device"
    OUT_DIR = "out_dir"
    DESCRIPTION_SOURCE = "source"
    KEY = "key"
    CLASS_SOURCE = "class_source"
    CRITERION = "criterion"
    OPTIMIZER = "optimizer"
    SCHEDULER = "scheduler"
    EARLY_STOP = "early_stop"
    PATH = "path"
    CPU_DEVICE = "cpu"
    DEF_OUT_DIR = "./"
    WARMUP_EPOCHS = "warmup"
    DEF_WARMUP_EPOCHS = 0
    NUM_EPOCHS = "num_epochs"
    LOSS = "loss"
    HUGGINGFACE_COS_SCHEDULER = "HuggingFaceCosineScheduler"
    TRAINING = "training"
    VALIDATION = "validation"
    VISUAL = "visual"
    GNN = "gnn"
    CLEAN_OUT_DIR = "clean_out_dir"
    TASK = "task"
    DEF_TASK = "style"
    
