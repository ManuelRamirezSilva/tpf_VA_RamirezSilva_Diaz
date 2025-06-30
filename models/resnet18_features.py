import torch.nn as nn
import torchvision.models as models

class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18FeatureExtractor, self).__init__()
        # Cargar ResNet18 preentrenada
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Congelar todas las capas
        for param in self.resnet.parameters():
            param.requires_grad = False
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
def get_resnet18_feature_extractor(num_classes):
    model = ResNet18FeatureExtractor(num_classes=num_classes)
    return model

# Hiperparámetros recomendados:
# - Optimizer: Adam (lr=0.001) SOLO para la cabeza (model.resnet.fc.parameters())
# - Loss: CrossEntropyLoss
# - Batch size: 64
# - Epochs: 30
# - Augmentaciones: RandomResizedCrop(128), RandomHorizontalFlip, ColorJitter
# - Normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# - EarlyStopping recomendado
# - Scheduler: ReduceLROnPlateau o StepLR
