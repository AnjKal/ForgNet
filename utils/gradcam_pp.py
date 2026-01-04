import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

# Import config and model loader from inference
from utils.inference import load_model, CFG


# ============================================================
# GRAD-CAM++ IMPLEMENTATION
# ============================================================

class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor):
        self.model.zero_grad()

        output = self.model(input_tensor)  # [B, 1, H, W]
        score = output.mean()               # global score for CAM
        score.backward(retain_graph=True)

        grads = self.gradients              # [B, C, H, W]
        acts = self.activations             # [B, C, H, W]

        # ---- Grad-CAM++ weights ----
        grads_sq = grads ** 2
        grads_cube = grads ** 3

        eps = 1e-8
        alpha_num = grads_sq
        alpha_denom = 2 * grads_sq + (acts * grads_cube).sum(dim=(2, 3), keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alphas = alpha_num / (alpha_denom + eps)

        positive_grads = F.relu(grads)
        weights = (alphas * positive_grads).sum(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1)
        cam = F.relu(cam)

        # Normalize
        cam -= cam.min()
        cam /= (cam.max() + eps)

        return cam[0].cpu().numpy()


# ============================================================
# PREPROCESSING (MATCHES INFERENCE)
# ============================================================

def preprocess_for_cam(image: Image.Image):
    img = image.resize((CFG.img_size, CFG.img_size))
    img = np.array(img, np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(CFG.device)


# ============================================================
# HEATMAP OVERLAY
# ============================================================

def overlay_cam(image: Image.Image, cam: np.ndarray, alpha=0.5):
    cam = cv2.resize(cam, image.size)
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img = np.array(image)
    overlay = (1 - alpha) * img + alpha * heatmap

    return Image.fromarray(overlay.astype(np.uint8))


# ============================================================
# PUBLIC API — CALLED BY STREAMLIT
# ============================================================

def generate_gradcam(image: Image.Image, case_id: str):
    """
    Generates Grad-CAM++ visualization and saves it locally.

    Returns:
        PIL.Image (overlay)
    """
    model = load_model()

    cam_explainer = GradCAMPlusPlus(
        model=model,
        target_layer=model.seg_head.net[-1]
    )

    input_tensor = preprocess_for_cam(image)
    input_tensor.requires_grad_(True)

    cam = cam_explainer.generate(input_tensor)
    overlay = overlay_cam(image, cam)

    # Save output
    out_path = Path(f"{case_id}_gradcam.png")
    overlay.save(out_path)

    return overlay
