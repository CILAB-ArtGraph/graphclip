import torch
import numpy as np
from torchvision.transforms import Resize
from torchvision.transforms import Compose
from PIL import Image
import cv2


class GradCAM:
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.gradients = None
        self.model.eval()

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x: torch.Tensor):
        self.model.zero_grad()
        x.requires_grad_()
        out = self.model(x)
        return out

    def generate_cam(
        self,
        image_tensor: torch.Tensor,
        target_class,
        target_dim: tuple[int] = (224, 224),
        device="cuda",
    ):
        output = self.forward(image_tensor)
        one_hot_output = torch.zeros((1, output.size()[-1]), dtype=torch.float)
        one_hot_output[0][target_class] = 1
        one_hot_output = one_hot_output.to(device)
        output.backward(gradient=one_hot_output)
        gradients = self.gradients.detach().cpu().numpy()
        feature_maps = self.model.feature_maps.detach().cpu().numpy()
        cam_weights = np.mean(gradients, axis=(2, 3))[0, :]
        cam = np.zeros(feature_maps.shape[2:], dtype=np.float32)

        for i, weight in enumerate(cam_weights):
            cam += weight * feature_maps[0, i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, target_dim)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam


def register_hooks(model, grad_cam):
    def forward_hook(module, input, output):
        grad_cam.model.feature_maps = output

    def backward_hook(module, grad_input, grad_output):
        grad_cam.save_gradient(grad_output[0])

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            target_module = module
    target_module.register_forward_hook(forward_hook)
    target_module.register_backward_hook(backward_hook)


def load_image(image_path):
    return Image.open(image_path).convert("RGB")


def apply_gradcam(
    image_path: str,
    preprocess: Compose,
    model: torch.nn.Module,
    gradcam: GradCAM,
    target,
    device,
    **kwargs
):
    register_hooks(model, gradcam)
    image_tensor = preprocess(load_image(image_path)).unsqueeze(0).to(device)

    cam = gradcam.generate_cam(image_tensor, target)
    return cam


def visualize_gradcam(image_path, gweights):
    image = cv2.imread(image_path)
    image = cv2.resize(image, tuple(gweights.shape))
    cam_heatmap = cv2.applyColorMap(np.uint8(255 * gweights), cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
    overlayed_image = cv2.addWeighted(image, 0.5, cam_heatmap, 0.5, 0)
    
    return overlayed_image