"""
body_classifier.py — Kasa Tipi Siniflandirici (EfficientNet-B0)
================================================================
efficientb0arakasa.pth — 7 kasa tipi sinifi

Sinif eslemesi (egitimden dogrulanan class_to_idx):
  0: hatchback | 1: kamyon | 2: minibus | 3: panelvan
  4: pickup    | 5: sedan  | 6: suv

ÖNEMLI: Model torchvision.models.efficientnet_b0 ile egitilmistir.
  Anahtar format: features.*, classifier.1.*
  timm efficientnet_b0 DEGIL (conv_stem.* anahtar formati farklidir)

Gerekli kutuphaneler:
  - torch>=2.1.0
  - torchvision>=0.16.0
  - Pillow, numpy, opencv-python
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.error("PyTorch veya torchvision kurulu degil!")


# -------------------------------------------------------------------
# Sabitler — Egitimden dogrulanan sinif sirasi (class_to_idx)
# -------------------------------------------------------------------
BODY_CLASS_NAMES = {
    0: "hatchback",
    1: "kamyon",
    2: "minibus",
    3: "panelvan",
    4: "pickup",
    5: "sedan",
    6: "suv",
}

NUM_CLASSES = 7

# ImageNet normalize istatistikleri
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
INPUT_SIZE    = 224


# -------------------------------------------------------------------
# BodyClassifier
# -------------------------------------------------------------------
class BodyClassifier:
    """
    torchvision EfficientNet-B0 tabanli kasa tipi siniflandirici.

    Kullanim:
        classifier = BodyClassifier("/app/models/efficientb0arakasa.pth")
        label, confidence = classifier.predict(vehicle_crop_bgr)
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        assert TORCH_AVAILABLE, "PyTorch kurulu degil!"
        self.device     = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self._build_model()
        self._load_weights()
        self._build_transforms()
        self.model.eval()
        logger.info(f"BodyClassifier yuklendi: {model_path} -> device={self.device}")

    # ------------------------------------------------------------------
    def _build_model(self):
        """
        torchvision EfficientNet-B0 mimarisini olustur.
        efficientb0arakasa.pth 'features.*' ve 'classifier.1.*' key'leriyle
        torchvision formatiyla egitilmistir.
        """
        self.model = tv_models.efficientnet_b0(weights=None)
        # Son katmani 7 sinif icin degistir
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
        self.model = self.model.to(self.device)

    # ------------------------------------------------------------------
    def _load_weights(self):
        """
        .pth dosyasini yukle.
        Desteklenen checkpoint formatlari:
          - Dogrudan state_dict (features.* key'leri)
          - {'state_dict': ...}
          - {'model_state_dict': ...}
          - {'model': ...}
        """
        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False   # tam model veya state_dict olabilir
        )

        # Checkpoint tipini belirle
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                # Dogrudan state_dict (features.* key'leri)
                state_dict = checkpoint
        else:
            # torch.save(model) ile tam model kaydedilmis olabilir
            if hasattr(checkpoint, 'state_dict'):
                state_dict = checkpoint.state_dict()
            else:
                state_dict = checkpoint

        # "module." prefix varsa kaldir (DataParallel kaynakli)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Strict yuklemeyi dene, basarisizsa strict=False ile tekrar dene
        try:
            self.model.load_state_dict(state_dict, strict=True)
            logger.info("EfficientNet-B0 (torchvision) agirliklar yuklendi (strict=True).")
        except RuntimeError as e:
            logger.warning(f"Strict yukleme basarisiz, strict=False deneniyor: {str(e)[:200]}")
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            logger.info(
                f"EfficientNet-B0 agirliklar yuklendi (strict=False). "
                f"Eksik: {len(missing)}, Fazla: {len(unexpected)} anahtar."
            )

    # ------------------------------------------------------------------
    def _build_transforms(self):
        """Inference preprocessing pipeline."""
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((INPUT_SIZE, INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    # ------------------------------------------------------------------
    def predict(self, img_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Arac crop'undan kasa tipini tahmin eder.

        Args:
            img_bgr: BGR numpy array (arac crop)

        Returns:
            (label, confidence)
        """
        if img_bgr is None or img_bgr.size == 0:
            return "sedan", 0.0

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor  = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

        label = BODY_CLASS_NAMES.get(pred.item(), "sedan")
        return label, round(conf.item(), 4)

    # ------------------------------------------------------------------
    def predict_topk(self, img_bgr: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """Top-K tahmin doner."""
        if img_bgr is None or img_bgr.size == 0:
            return [("sedan", 0.0)]

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor  = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]

        topk_vals, topk_idxs = torch.topk(probs, k=min(k, NUM_CLASSES))
        return [
            (BODY_CLASS_NAMES.get(idx.item(), "sedan"), round(val.item(), 4))
            for val, idx in zip(topk_vals, topk_idxs)
        ]
