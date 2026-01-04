import streamlit as st
import torch
import numpy as np
from PIL import Image
import json
import cv2
import math
from pathlib import Path
from datetime import datetime
from transformers import AutoImageProcessor, AutoModel
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# CONFIGURATION
# ============================================================

class CFG:
    dino_path = "models/dinov2"
    model_weights = "models/model.pt"
    img_size = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    area_threshold = 150
    confidence_threshold = 0.25
    use_tta = True


# ============================================================
# MODEL DEFINITION (EXACT MATCH)
# ============================================================

class DinoTinyDecoder(nn.Module):
    def __init__(self, in_ch=768, out_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, out_ch, 1)
        )

    def forward(self, f, size):
        f = F.interpolate(f, size=size, mode="bilinear", align_corners=False)
        return self.net(f)


class DinoSegmenter(nn.Module):
    def __init__(self, encoder, processor):
        super().__init__()
        self.encoder = encoder
        self.processor = processor
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.seg_head = DinoTinyDecoder(768, 1)

    def forward_features(self, x):
        imgs = (x * 255).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        inputs = self.processor(images=list(imgs), return_tensors="pt").to(x.device)

        with torch.no_grad():
            feats = self.encoder(**inputs).last_hidden_state

        B, N, C = feats.shape
        fmap = feats[:, 1:, :].permute(0, 2, 1)
        s = int(math.sqrt(N - 1))
        return fmap.reshape(B, C, s, s)

    def forward(self, x):
        fmap = self.forward_features(x)
        return self.seg_head(fmap, (CFG.img_size, CFG.img_size))


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(CFG.dino_path, local_files_only=True)
    encoder = AutoModel.from_pretrained(CFG.dino_path, local_files_only=True).to(CFG.device).eval()

    model = DinoSegmenter(encoder, processor).to(CFG.device)
    model.load_state_dict(torch.load(CFG.model_weights, map_location=CFG.device))
    model.eval()

    return model


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

@torch.no_grad()
def predict(model, image):
    return torch.sigmoid(model(image))[0, 0].cpu().numpy()


@torch.no_grad()
def predict_with_tta(model, image):
    preds = []

    pred = torch.sigmoid(model(image))
    preds.append(pred)

    pred = torch.sigmoid(model(torch.flip(image, dims=[3])))
    preds.append(torch.flip(pred, dims=[3]))

    pred = torch.sigmoid(model(torch.flip(image, dims=[2])))
    preds.append(torch.flip(pred, dims=[2]))

    return torch.stack(preds).mean(0)[0, 0].cpu().numpy()


@torch.no_grad()
def get_embedding(image):
    model = load_model()
    # Resize to model's expected size for embedding extraction
    # Standard DINOv2 usually works well with 224 or multiples of 14. 
    # We'll use the configured img_size.
    
    img_resized = image.resize((CFG.img_size, CFG.img_size))
    img = np.array(img_resized, np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(CFG.device)
    
    # Forward pass through encoder only
    # The model.encoder is an AutoModel (ViT). We want the CLS token or pooled output.
    # self.processor does normaliztion, but we already did manual resize/norm.
    # Let's rely on the manual tensor preparation for consistency with inference.
    
    # We need to access the underlying encoder. 
    # DinoSegmenter has self.encoder.
    
    # Standard ViT output: last_hidden_state is (B, N, D)
    # CLS token is usually at index 0.
    outputs = model.encoder(tensor)
    last_hidden_state = outputs.last_hidden_state # (1, N, D)
    
    # Global Average Pooling usually works better than CLS for DINOv2 similarity
    # or just use CLS (index 0). DINOv2 CLS is strong.
    cls_token = last_hidden_state[0, 0, :].cpu().numpy()
    
    return cls_token.tolist()


# ============================================================
# POST-PROCESSING (EXACT MATCH)
# ============================================================

def postprocess(preds, original_size, alpha_grad=0.35):
    gx = cv2.Sobel(preds, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(preds, cv2.CV_32F, 0, 1, ksize=3)

    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_norm = grad_mag / (grad_mag.max() + 1e-6)

    enhanced = (1 - alpha_grad) * preds + alpha_grad * grad_norm
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    thr = np.mean(enhanced) + 0.3 * np.std(enhanced)
    mask = (enhanced > thr).astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    return cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)


# ============================================================
# RLE ENCODING (CORRECT FORMAT)
# ============================================================

def rle_encode(mask):
    pixels = mask.T.flatten()
    dots = np.where(pixels == 1)[0]

    if len(dots) == 0:
        return "authentic"

    runs, prev = [], -2
    for d in dots:
        if d > prev + 1:
            runs.extend((d + 1, 0))
        runs[-1] += 1
        prev = d

    return json.dumps([int(x) for x in runs])


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def infer_image(image):
    model = load_model()

    image_array = np.array(
        image.resize((CFG.img_size, CFG.img_size)),
        np.float32
    ) / 255.0

    image_tensor = torch.from_numpy(image_array) \
        .permute(2, 0, 1) \
        .unsqueeze(0) \
        .to(CFG.device)

    if CFG.use_tta:
        preds = predict_with_tta(model, image_tensor)
    else:
        preds = predict(model, image_tensor)

    mask = postprocess(preds, image.size)

    area = int(mask.sum())

    resized_mask = cv2.resize(
        mask,
        (CFG.img_size, CFG.img_size),
        interpolation=cv2.INTER_NEAREST
    )

    if area > 0:
        mean_inside = float(preds[resized_mask == 1].mean())
    else:
        mean_inside = 0.0

    # 🔴 EXACT DECISION LOGIC MATCH
    if area < 400 or mean_inside < 0.3:
        return "authentic", None

    return "forged", mask



# ============================================================
# HISTORY MANAGEMENT
# ============================================================

def save_prediction(case_id, image_path, label, mask, embedding=None):  # Add embedding
    history_file = Path("history.json")
    history = json.load(open(history_file)) if history_file.exists() else []
    
    annotation = "authentic" if mask is None else rle_encode(mask)
    
    record = {
        "case_id": case_id,
        "image_path": str(image_path),
        "label": label,
        "annotation": annotation,
        "timestamp": datetime.now().isoformat()
    }
    
    if embedding is not None:
        record["embedding"] = embedding
        
    history.append(record)
    
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    
    return annotation


def get_history():
    file = Path("history.json")
    return json.load(open(file)) if file.exists() else []
