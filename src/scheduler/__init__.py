import torch.optim.lr_scheduler as schedulers
from transformers import get_cosine_schedule_with_warmup

scheduler_registry = schedulers.__dict__
scheduler_registry["HuggingFaceCosineScheduler"] = get_cosine_schedule_with_warmup