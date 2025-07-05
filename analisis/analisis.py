import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def plot_gradcam_indices(model, dataset, indices, layer_name='layer4', mean=None, std=None):
    """
    Aplica Grad-CAM y plotea resultados para una lista de índices del dataset.
    model: modelo pytorch (eval mode)
    dataset: dataset de imágenes (getitem debe devolver (img_tensor, label))
    indices: lista de índices a analizar
    layer_name: nombre de la capa convolucional (por defecto 'layer4' para ResNet18)
    mean, std: listas numpy para desnormalizar (por defecto ImageNet)
    """
    
    if mean is None:
        mean = np.array([0.485, 0.456, 0.406])
    if std is None:
        std = np.array([0.229, 0.224, 0.225])
    
    model.eval()
    
    # Registrar hooks una sola vez
    activations = {}
    gradients = {}
    
    def save_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    def save_gradient(name):
        def hook(model, grad_input, grad_output):
            gradients[name] = grad_output[0].detach()
        return hook
    
    target_layer = dict([*model.named_modules()])[layer_name]
    forward_handle = target_layer.register_forward_hook(save_activation(layer_name))
    backward_handle = target_layer.register_backward_hook(save_gradient(layer_name))
    
    for idx in indices:
        img_tensor, label = dataset[idx]
        input_tensor = img_tensor.unsqueeze(0)
        
        output = model(input_tensor)
        pred_class = output.argmax().item()
        loss = output[0, pred_class]
        model.zero_grad()
        loss.backward()
        
        grads = gradients[layer_name]
        acts = activations[layer_name]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1).squeeze()
        cam = F.relu(cam)
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        cam = cam.cpu().numpy()
        
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        img_np = np.uint8(255 * img_np)
        
        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
        
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(img_np)
        plt.title(f'Original (idx={idx})')
        plt.axis('off')
        
        plt.subplot(1,3,2)
        plt.imshow(cam_resized, cmap='jet')
        plt.title('Grad-CAM')
        plt.axis('off')
        
        plt.subplot(1,3,3)
        plt.imshow(superimposed_img)
        plt.title(f'Superposición\nLabel: {label} | Pred: {pred_class}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # Limpiar hooks
    forward_handle.remove()
    backward_handle.remove()



def compare_gradcam_models(models, model_names, dataset, indices, layer_names=None, mean=None, std=None):
    """
    Plotea una tabla de Grad-CAMs para varios modelos y varios índices.
    models: lista de modelos PyTorch
    model_names: lista de nombres de modelos (str)
    dataset: dataset de imágenes
    indices: lista de índices a analizar
    layer_names: lista de nombres de capa para cada modelo (si None, usa 'layer4' para todos)
    mean, std: para desnormalizar
    """
    if mean is None:
        mean = np.array([0.485, 0.456, 0.406])
    if std is None:
        std = np.array([0.229, 0.224, 0.225])
    if layer_names is None:
        layer_names = ['layer4'] * len(models)

    n_rows = len(indices)
    n_cols = len(models)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))

    if n_rows == 1:
        axs = np.expand_dims(axs, 0)
    if n_cols == 1:
        axs = np.expand_dims(axs, 1)

    for col, (model, model_name, layer_name) in enumerate(zip(models, model_names, layer_names)):
        # Registrar hooks una sola vez por modelo
        activations = {}
        gradients = {}

        def save_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook

        def save_gradient(name):
            def hook(model, grad_input, grad_output):
                gradients[name] = grad_output[0].detach()
            return hook

        target_layer = dict([*model.named_modules()])[layer_name]
        forward_handle = target_layer.register_forward_hook(save_activation(layer_name))
        backward_handle = target_layer.register_backward_hook(save_gradient(layer_name))

        model.eval()
        for row, idx in enumerate(indices):
            img_tensor, label = dataset[idx]
            input_tensor = img_tensor.unsqueeze(0)

            output = model(input_tensor)
            pred_class = output.argmax().item()
            loss = output[0, pred_class]
            model.zero_grad()
            loss.backward()

            grads = gradients[layer_name]
            acts = activations[layer_name]
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * acts).sum(dim=1).squeeze()
            cam = F.relu(cam)
            cam -= cam.min()
            cam /= cam.max() + 1e-8
            cam = cam.cpu().numpy()

            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
            img_np = np.uint8(255 * img_np)

            cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

            axs[row, col].imshow(superimposed_img)
            axs[row, col].set_title(f"{model_name}\nidx={idx}\nLabel: {label} | Pred: {pred_class}")
            axs[row, col].axis('off')

        forward_handle.remove()
        backward_handle.remove()

    plt.tight_layout()
    plt.show()

def extract_features(model, dataloader, device="cpu"):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            # --- ResNet (18, 50, etc) ---
            if hasattr(model, 'layer4') and hasattr(model, 'avgpool'):
                feats = model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(imgs)))))))))
                feats = feats.view(feats.size(0), -1)
            # --- AlexNet ---
            elif hasattr(model, 'features') and hasattr(model, 'avgpool'):
                feats = model.features(imgs)
                feats = model.avgpool(feats)
                feats = feats.view(feats.size(0), -1)
            # --- VGG ---
            elif hasattr(model, 'features') and hasattr(model, 'classifier'):
                feats = model.features(imgs)
                feats = feats.view(feats.size(0), -1)
            else:
                raise ValueError("Modelo no soportado para extracción de features")
            features.append(feats.cpu().numpy())
            labels.append(lbls.cpu().numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

def get_metrics(y_true, y_pred):
    return [
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, average='weighted', zero_division=0),
        recall_score(y_true, y_pred, average='weighted', zero_division=0),
        f1_score(y_true, y_pred, average='weighted', zero_division=0)
    ]