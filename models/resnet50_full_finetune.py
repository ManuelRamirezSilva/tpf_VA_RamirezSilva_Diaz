import torch.nn as nn
import torchvision.models as models

class ResNet50FullFineTune(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50FullFineTune, self).__init__()
        # Cargar ResNet50 preentrenada
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Descongelar todo el modelo (por defecto, todos los parámetros son entrenables)
        for param in self.resnet.parameters():
            param.requires_grad = True
        # Reemplazar la capa fully connected final por una nueva con BatchNorm
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

    def summary(self, input_size=(3, 128, 128)):
        """
        Imprime un resumen del modelo con el tamaño de entrada dado.
        """
        from torchsummary import summary
        device = next(self.parameters()).device
        summary(self, input_size, device=str(device))

# Hiperparámetros sugeridos para entrenar este modelo:
def get_resnet50_full_finetune(num_classes):
    model = ResNet50FullFineTune(num_classes=num_classes)
    return model

# Hiperparámetros recomendados:
# - Optimizer: Adam (lr=0.0001) o AdamW (mejor para modelos grandes)
# - Loss: CrossEntropyLoss
# - Batch size: 32 o 64 (según memoria GPU)
# - Epochs: 30-50
# - Augmentaciones: RandomResizedCrop(128), RandomHorizontalFlip, ColorJitter, RandomRotation
# - Normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# - EarlyStopping recomendado
# - Scheduler: CosineAnnealingLR, ReduceLROnPlateau o StepLR
# - Weight decay recomendado (ej: 1e-4)
# - Dropout opcional en la cabeza si hay overfitting
