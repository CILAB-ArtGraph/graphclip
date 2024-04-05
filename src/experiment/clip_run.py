from typing import Any
from .run import Run
from src.models import CLIP
from open_clip import create_model, get_tokenizer, SimpleTokenizer
from .utils import ParameterKeys
from copy import deepcopy
import torch
from src.data import DataDict


class CLIPRun(Run):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.train_loader, self.val_loader, self.test_loader = self._init_dataloaders()
        self.model = self._init_model()
        self.metrics = self._init_metrics()
        self.tokenizer = self._init_tokenizer()

    def _init_model(self) -> CLIP:
        return create_model(**self.parameters.get(ParameterKeys.MODEL))

    def _init_tokenizer(self) -> SimpleTokenizer:
        return get_tokenizer(**self.parameters.get(ParameterKeys.TOKENIZER, {}))

    def get_class_maps(self, classes):
        classes = list(sorted(list(set(deepcopy(classes)))))
        idx2class = {ix: s for ix, s in enumerate(classes)}
        class2idx = {s: ix for ix, s in idx2class.items()}

        class_prompts = [f"This artwork is in {x} artistic style" for x in classes]

        return class_prompts, idx2class, class2idx

    def get_classes_for_test(self):
        dataset = self.test_loader.dataset
        return dataset.dataset[dataset.dataset.columns[-1]].unique().tolist()

    @torch.no_grad()
    def test(self) -> dict[str, float]:
        device = self.parameters.get(ParameterKeys.DEVICE, "cpu")
        self.model = self.model.to(device)
        classes = self.get_classes_for_test()
        class_prompts, idx2class, class2idx = self.get_class_maps(classes)
        
        print(f"Having {len(classes)} classes")
        class_tokens = self.tokenizer(class_prompts).to(device)

        class_feats = self.model.encode_text(class_tokens)
        class_feats /= class_feats.norm(dim=-1, keepdim=True)

        bar = self.get_bar(self.test_loader, desc="Test")
        
        metrics = deepcopy(self.metrics)
        
        for ix, data_dict in bar:
            imgs = data_dict[DataDict.IMAGE].to(device)
            txts = data_dict[DataDict.TEXT] 
            labels = torch.as_tensor(list(map(lambda x: class2idx[x], txts))).float()
            img_feats = self.model.encode_image(imgs)
            img_feats /= img_feats.norm(dim=-1, keepdim=True)
            
            out = img_feats @ class_feats.T
            out = out.detach().cpu()
            for m in metrics:
                metrics[m].update(out, labels)
        return {
            k: v.compute().cpu().item() for k, v in metrics.items()
        }

        