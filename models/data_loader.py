from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
import numpy as np

BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001
IMG_SIZE = 224
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# ── Config ───────────────────────────────────────────────────────

#si tirra error de que no existe, seguramente le falte un ../
# puede ser que no se haya actualizado el path

DATA_DIR = "../data/stanford_cars"
TRAIN_CSV = "../data/stanford_cars/labels_train.csv"
VAL_CSV = "../data/stanford_cars/labels_val.csv"
TEST_CSV = "../data/stanford_cars/labels_test.csv"

NAMES_CSV = "../data/stanford_cars/names.csv"
TRAIN_IMG_DIR = "../data/stanford_cars/train"
VAL_IMG_DIR = "../data/stanford_cars/validation"
TEST_IMG_DIR = "../data/stanford_cars/test"

class CarsDataset(Dataset):
    def __init__(self, df_csv, img_dir, names_csv, transform=None, half_data=False):
        self.df = pd.read_csv(df_csv, header=None)
        if half_data:
            self.df = self.df.iloc[:len(self.df)//4]
        self.img_dir = img_dir
        self.transform = transform
        with open(names_csv) as f:
            self.class_names = [line.strip() for line in f]
        self.num_classes = len(self.class_names)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row[0]
        bbox = tuple(row[1:5])
        label = int(row[5]) - 1
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        # Optional: crop = image.crop(bbox)
        if self.transform:
            image = self.transform(image)
        return image, label

# ── Transforms ───────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist())
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist())
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist())
])

# ── Datasets y Loaders ───────────────────────────────────────────
train_ds = CarsDataset(TRAIN_CSV, TRAIN_IMG_DIR, NAMES_CSV, transform=train_transform)
val_ds = CarsDataset(VAL_CSV, VAL_IMG_DIR, NAMES_CSV, transform=val_transform)
test_ds = CarsDataset(TEST_CSV, TEST_IMG_DIR, NAMES_CSV, transform=test_transform)


train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    # num_workers=4 #descomentar para resnet18 y alexnet,
    pin_memory=True,
    persistent_workers=False #cambiar a true para resnet18 y alexnet
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    # num_workers=4, #descomentar para resnet18 y alexnet,
    pin_memory=True,
    persistent_workers=False #cambiar a true para resnet18 y alexnet
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

class_names = train_ds.class_names