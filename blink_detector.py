#!/usr/bin/env python3

from transformers import ViTFeatureExtractor, AutoModelForImageClassification
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from tqdm import tqdm
import os
import cv2
from PIL import Image


class BlinkDataset(Dataset):
    def __init__(self, root_dir, feature_extractor=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.feature_extractor = feature_extractor
        self.images = os.listdir(root_dir)

        self.images=[]
        self.labels=[]

        open_dir = os.path.join(root_dir, 'open')
        for img_name in os.listdir(open_dir):
            if img_name.endswith('.png'):
                self.images.append(os.path.join(open_dir, img_name))
                self.labels.append(0)

        closed_dir = os.path.join(root_dir, 'closed')
        for img_name in os.listdir(closed_dir):
            if img_name.endswith('.png'):
                self.images.append(os.path.join(closed_dir, img_name))
                self.labels.append(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        label = self.labels[idx]

        features = []
        if self.feature_extractor:
            features = self.feature_extractor(images=image, return_tensors="pt")
            features = {k: v.squeeze(0) for k, v in features.items()}

        if self.transform:
            features["pixel_values"] = self.transform(features["pixel_values"])
        
        return features, label

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([
        transforms.Normalize(mean=0.4111, std=0.1085),
    ])

    model_name = "google/vit-base-patch16-224-in21k"
    model = AutoModelForImageClassification.from_pretrained(model_name)
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

    finetuneData = BlinkDataset(os.curdir + '/data', feature_extractor=feature_extractor, transform=transform)

    train_size = int(0.8 * len(finetuneData))
    test_size = len(finetuneData) - train_size

    train_data_finetune, test_data_finetune = torch.utils.data.random_split(
        finetuneData, [train_size, test_size]
    )

    train_loader_finetune = DataLoader(
        train_data_finetune,
        batch_size=64,
        shuffle=True
    )

    test_loader_finetune = DataLoader(
        test_data_finetune,
        batch_size=64,
        shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)

    model.classifier = torch.nn.Linear(model.config.hidden_size, 2)
    model.load_state_dict(torch.load('./model.pth'))
    model.train()
    model.to(device)

    summary(model, input_size=(64, 3, 224, 224))

    # for epoch in range(1):
    #     for images, labels in tqdm(train_loader_finetune):
    #         input = images["pixel_values"]
    #         input = input.to(device)
    #         labels = labels.to(device)
    #         outputs = model(input)
    #         loss = torch.nn.CrossEntropyLoss()(outputs.logits, labels)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     torch.save(model.state_dict(), './model.pth')

    model.eval()
    correct = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader_finetune):
            input = images["pixel_values"]
            input = input.to(device)
            labels = labels.to(device)
            outputs = model(input)

            predicted_class_idx = torch.argmax(outputs.logits, dim=-1)

            correct += (predicted_class_idx == labels).sum().item()

    print(f"Accuracy: {correct / test_size}")

if __name__=='__main__':
    main()