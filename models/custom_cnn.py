import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()

        # Bloque 1: Convolución simple + BatchNorm + MaxPool
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bloque 2: Más filtros + BatchNorm + MaxPool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bloque 3: Aumentamos profundidad + BatchNorm
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Capa completamente conectada
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  # suponiendo entrada de 128x128
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def summary(self, input_size=(3, 128, 128)):
        """
        Imprime un resumen del modelo con el tamaño de entrada dado.
        """
        from torchsummary import summary
        device = next(self.parameters()).device
        summary(self, input_size, device=str(device))

# Hiperparámetros sugeridos para entrenar este modelo:
def get_custom_cnn_model(num_classes):
    model = CustomCNN(num_classes=num_classes)
    return model

# Hiperparámetros recomendados:
# - Optimizer: Adam (lr=0.001)
# - Loss: CrossEntropyLoss
# - Batch size: 64
# - Epochs: 30
# - Augmentaciones: RandomHorizontalFlip, RandomRotation, ColorJitter
# - Normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
