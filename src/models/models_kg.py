from typing import List, Dict
from torch import Tensor, cat
import torch.nn as nn
from torchvision import models
from timm import create_model

class ContextNetSingleTask(nn.Module):
    """
    The model propsed by Garcia et al (https://link.springer.com/article/10.1007/s13735-019-00189-4)
    """

    def __init__(self, emb_size: int, num_class:int):
        super().__init__()

        self.resnet = models.resnet50(pretrained=True)
        len_last = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.classifier = nn.Linear(len_last, num_class)

        self.encoder = nn.Linear(len_last, emb_size)


    def forward(self, img: Tensor) -> List[Tensor]:

        visual_features = self.resnet(img)
        visual_features = visual_features.view(visual_features.size(0), -1)

        out = self.classifier(visual_features)
        graph_proj = self.encoder(visual_features)

        return out, graph_proj

class ContextNetlMultiTask(nn.Module):
    """
    The model proposed by Garcia et al (https://link.springer.com/article/10.1007/s13735-019-00189-4)
    """
    def __init__(self, emb_size: int, num_class: Dict[str, int]):
        super().__init__()

        self.resnet = models.resnet50(pretrained=True)
        len_last = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.class_style = nn.Linear(len_last, num_class['style'])
        self.class_genre = nn.Linear(len_last, num_class['genre'])

        self.encoder = nn.Linear(len_last, emb_size)


    def forward(self, img: Tensor) -> List[Tensor]:

        visual_features = self.resnet(img)
        visual_features = visual_features.view(visual_features.size(0), -1)
        graph_proj = self.encoder(visual_features)

        out_style = self.class_style(visual_features)
        out_genre = self.class_genre(visual_features)
        outs = [out_style, out_genre]

        return outs, graph_proj

class MultiModalSingleTask(nn.Module):
    """
    The model proposed by Castellano et al, https://arxiv.org/abs/2105.15028
    """

    def __init__(self, emb_size: int, num_class:int):
        super().__init__()

        self.resnet = models.resnet50(pretrained=True)
        len_last = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(len_last + emb_size, num_class)
        )

        self.encoder = nn.Sequential(
            nn.Linear(len_last, emb_size),
            nn.Tanh(),
            nn.Linear(emb_size, emb_size),
            nn.Tanh()
        )

    def forward(self, img: Tensor) -> List[Tensor]:

        visual_features = self.resnet(img)
        graph_proj = self.encoder(visual_features)

        concat = cat((visual_features, graph_proj), 1)

        out = self.classifier(concat)

        return out, graph_proj

class MultiModalMultiTask(nn.Module):
    """
    The model proposed by Castellano et al, https://arxiv.org/abs/2105.15028
    """

    def __init__(self, emb_size: int, num_classes: Dict[str, int]):
        super().__init__()

        self.resnet = models.resnet50(pretrained=True)
        len_last = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.class_style = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(len_last + emb_size, num_classes['style'])
        )

        self.class_genre = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(len_last + emb_size, num_classes['genre'])
        )

        self.encoder = nn.Sequential(
            nn.Linear(len_last, emb_size),
            nn.Tanh(),
            nn.Linear(emb_size, emb_size),
            nn.Tanh()
        )

    def forward(self, img: Tensor) -> List[Tensor]:

        visual_features = self.resnet(img)
        graph_proj = self.encoder(visual_features)

        concat = cat((visual_features, graph_proj), 1)

        out_style = self.class_style(concat)
        out_genre = self.class_genre(concat)

        return [out_style, out_genre], graph_proj

class NewMultiModalSingleTask(nn.Module):

    def __init__(self, emb_size: int, num_class:int, dropout: float):
        super().__init__()

        self.resnet = models.resnet50(pretrained=True)
        len_last = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(len_last + emb_size, num_class))


    def forward(self, img: Tensor, embedding: Tensor) -> List[Tensor]:

        visual_features = self.resnet(img)
        visual_features = visual_features.view(visual_features.size(0), -1)

        comb = cat((visual_features, embedding), dim = 1)

        out = self.classifier(comb)

        return out

class NewMultiModalMultiTask(nn.Module):

    def __init__(self, emb_size: int, num_classes: Dict[str, int], dropout: float):
        super().__init__()

        self.resnet = models.resnet50(pretrained=True)
        len_last = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        
        self.class_style = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(len_last + emb_size, num_classes['style']))

        self.class_genre = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(len_last + emb_size, num_classes['genre']))

    def forward(self, img: Tensor, embedding_style: Tensor, embedding_genre: Tensor) -> List[Tensor]:

        visual_features = self.resnet(img)
        visual_features = visual_features.view(visual_features.size(0), -1)

        comb_style = cat((visual_features, embedding_style), dim = 1)
        comb_genre = cat((visual_features, embedding_genre), dim = 1)

        out_style = self.class_style(comb_style)
        out_genre = self.class_genre(comb_genre)

        return [out_style, out_genre]

class NewMultiModalSingleTaskVit(nn.Module):

    def __init__(self, emb_size: int, num_class:int, dropout: float):
        super().__init__()

        self.vit = create_model("vit_base_patch16_224", pretrained=True)
        for p in self.vit.parameters():
            p.requires_grad = False
        for p in self.vit.blocks[-1].parameters():
            p.requires_grad = True
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768 + emb_size, num_class))


    def forward(self, img: Tensor, embedding: Tensor) -> List[Tensor]:

        visual_features = self.vit.forward_features(img.squeeze())

        comb = cat((visual_features, embedding), dim = 1)

        out = self.classifier(comb)

        return out

class NewMultiModalMultiTaskViT(nn.Module):

    def __init__(self, emb_size: int, num_class: Dict[str, int], dropout: float):
        super().__init__()

        self.vit = create_model("vit_base_patch16_224", pretrained=True)
        len_last = self.vit.head.in_features
        for p in self.vit.parameters():
            p.requires_grad = False
        for p in self.vit.blocks[-1].parameters():
            p.requires_grad = True

        self.class_style = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768 + emb_size, num_class['style']))

        self.class_genre = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768 + emb_size, num_class['genre']))

    def forward(self, img: Tensor, embedding_style: Tensor, embedding_genre: Tensor) -> List[Tensor]:

        visual_features = self.vit.forward_features(img.squeeze())

        comb_style = cat((visual_features, embedding_style), dim = 1)
        comb_genre = cat((visual_features, embedding_genre), dim = 1)

        out_style = self.class_style(comb_style)
        out_genre = self.class_genre(comb_genre)

        return [out_style, out_genre]

class LabelProjector(nn.Module):

    def __init__(self, emb_size: int):
        super().__init__()

        self.resnet = models.resnet50(pretrained=True)
        len_last = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.encoder = nn.Linear(len_last, emb_size)

    def forward(self, img: Tensor) -> List[Tensor]:

        visual_features = self.resnet(img)
        visual_features = visual_features.view(visual_features.size(0), -1)

        out = self.encoder(visual_features)

        return out

class LabelProjectorVit(nn.Module):

    def __init__(self, emb_size: int):
        super().__init__()

        self.vit = create_model("vit_base_patch16_224", pretrained=True)
        for p in self.vit.parameters():
            p.requires_grad = False

        self.encoder = nn.Linear(768, emb_size)

    def forward(self, img: Tensor) -> List[Tensor]:

        visual_features = self.vit.forward_features(img.squeeze())

        out = self.encoder(visual_features)

        return out