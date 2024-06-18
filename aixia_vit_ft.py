from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
import pandas as pd
from tqdm import tqdm
from timm import create_model
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
import copy
import torch.optim as optim
from torch.optim import lr_scheduler
from src.data import AIxIADataset, DataDict
from src.utils.early_stop import EarlyStopping
import pickle
from sklearn.decomposition import PCA


if __name__ == "__main__":
    device = torch.device("cuda")
    model = create_model("vit_base_patch16_224", pretrained=True, num_classes=32).to(
        device
    )
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    data_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    batch_size = 512
    train_set = AIxIADataset(
        dataset="./data/processed/normal/artgraph_clip_style/train.csv",
        mapping="./data/external/artgraph2bestemotions/mapping/style_entidx2name.csv",
        mapping_kwargs={"names": ["idx", "name"]},
        img_dir="./data/raw/images-resized",
        preprocess=data_transforms,
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8
    )
    validation_set = AIxIADataset(
        dataset="./data/processed/normal/artgraph_clip_style/val.csv",
        mapping="./data/external/artgraph2bestemotions/mapping/style_entidx2name.csv",
        mapping_kwargs={"names": ["idx", "name"]},
        img_dir="./data/raw/images-resized",
        preprocess=data_transforms,
    )
    validation_loader = DataLoader(
        validation_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8
    )

    def fine_tune(model, train_loader, validation_loader, criterion, optimizer, scheduler, early_stop, num_epochs = 100):
        best_model = copy.deepcopy(model)
        best_acc = 0.0
        best_epoch=0
        
        stop = False
        for epoch in range(1, num_epochs + 1):
            if stop:
                break
            print(f'Epoch {epoch}/{num_epochs}')
            print('-'*120)

            data_loader = None
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                    data_loader = train_loader
                else:
                    model.eval()   # Set model to evaluate mode
                    data_loader = validation_loader

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for data_dict in tqdm(data_loader):
                    inputs = data_dict[DataDict.IMAGE].to(device)
                    labels = data_dict[DataDict.GTS].to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(nn.Softmax(dim = 1)(outputs), 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / (len(data_loader) * data_loader.batch_size)
                epoch_acc = running_corrects.double() / (len(data_loader) * data_loader.batch_size)

                if phase == 'val':
                    scheduler.step(epoch_loss)
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    best_model = copy.deepcopy(model)
                    
                    
                if phase == 'val':
                    early_stop(epoch_loss, model=model)
                    print('-'*120, end = '\n\n')
                    stop=early_stop.early_stop
                    
                    
        print(f'Best val Acc: {best_acc:4f}')
        print(f'Best epoch: {best_epoch:03d}')

        # load best model 
        return best_model

    #if the model fine tuned on the head layer is not present into the directory
    if 'vit_just_head.pt' not in os.listdir():
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr = 1e-6, verbose = True, factor = .1, patience = 1,
                                                threshold = 1e-3)
        early_stop= EarlyStopping(path="models/vit_just_head.pt", patience = 5)
        best_model_head=fine_tune(model, train_loader, validation_loader, criterion, optimizer, scheduler, early_stop, num_epochs = 5)

    for p in model.blocks[11].parameters():#last feature extraction layer
        p.requires_grad=True

    for p in model.norm.parameters():
        p.requires_grad=True

    if 'vit.pt' not in os.listdir(): 
        optimizer = optim.Adam(best_model_head.parameters(), lr=1e-3)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr = 1e-6, verbose = True, factor = .1, patience = 1,
                                                threshold = 1e-3)
        early_stop= EarlyStopping(path='models/vit.pt', patience = 5)
        best_model=fine_tune(best_model_head, train_loader, validation_loader, criterion, optimizer, scheduler, early_stop, num_epochs = 30)

    model.load_state_dict(torch.load('models/vit.pt'))

    for p in model.parameters():
        p.requires_grad=False

    model.reset_classifier(num_classes=0)

    #custom class which can manage a dataset and return just the input, since the task is a feature extraction
    class UnsupervisedDataSet(Dataset):
        def __init__(self, main_dir, list_files, transform):
            self.main_dir = main_dir
            self.transform = transform
            self.total_imgs = list_files

        def __len__(self):
            return len(self.total_imgs)

        def __getitem__(self, idx):
            img_path = f'{self.main_dir}/{self.total_imgs[idx]}'
            image = Image.open(img_path)
            if(image.mode != 'RGB'):
                image = image.convert('RGB')
            tensor_image = self.transform(image)
            return (tensor_image, self.total_imgs[idx])
        
    df = pd.read_csv('./data/external/artgraph2bestemotions/mapping/artwork_entidx2name.csv', names=["idx", "names"])
    list_images = df["names"].tolist()
    list_features = []

    batch_size = 32#you can change the batch size depending on the artwork
    extraction_dataset = UnsupervisedDataSet('./data/raw/images-resized', list_images, transform=data_transforms)
    train_loader = DataLoader(extraction_dataset, batch_size=batch_size, shuffle=False, 
                                drop_last=False)
    
    #extracting features for all the artworks
    model.eval()
    name_images = []
    with torch.no_grad():
        x = torch.zeros((len(list_images), 768))
        for idx, image in tqdm(enumerate(train_loader), total=len(train_loader)):
            img, img_name = image
            name_images.append(img_name)
            x[idx*batch_size : (idx+1)*batch_size] = model(img.to(device))
    
    x_num = x.detach().numpy()


    #rescaling vector dimension using PCA 
    pca = PCA(n_components=128)
    x_num_128 = pca.fit_transform(x_num)
    x_128 = torch.tensor(x_num_128)
    with open('data/interim/pca.pk','wb') as file:
        pickle.dump(pca, file)

    #saving
    node_feat_dir = './data/external/artgraph2bestemotions/raw/node-feat/artwork'
    os.makedirs(node_feat_dir, exist_ok=True)
    #node_feat_dir = os.path.join(fr'{root}/{graph}', 'raw', 'node-feat', 'artwork')
    if not os.path.exists(node_feat_dir):
        os.makedirs(node_feat_dir)

    x_df = pd.DataFrame(x_num)
    x_df.to_csv(f"{node_feat_dir}/node-feat-vit-fine-tuning.csv", index=False, header=False)