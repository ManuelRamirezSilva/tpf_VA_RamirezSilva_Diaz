import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CarsDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label = (self.data.iloc[idx, 5])-1
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformación estándar para ImageNet
imagenet_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_val_dataset_and_loader(csv_file, img_dir, batch_size=32, transform=imagenet_transform):
    dataset = CarsDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataset, loader
