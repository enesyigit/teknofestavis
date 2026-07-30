"""
color_classifier.py — Araç Renk Sınıflandırıcı (MobileNetV3)
=============================================================
arac_renk_mobilenetv3.pth — 9 renk sınıfı

Sınıf eşlemesi (data.yaml'dan doğrulanan):
  0: beyaz | 1: siyah  | 2: gri      | 3: kirmizi | 4: mavi
  5: sari  | 6: yesil  | 7: turuncu  | 8: kahverengi

Gerekli kütüphaneler:
  - torch>=2.1.0
  - torchvision>=0.16.0
  - timm>=0.9.0
  - opencv-python-headless, numpy
"""

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torchvision.transforms as T
    import timm
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.error("PyTorch veya timm kurulu değil!")


# -------------------------------------------------------------------
# Sabitler — data.yaml'dan doğrulanan sınıf sırası
# -------------------------------------------------------------------
COLOR_CLASS_NAMES = {
    0: "beyaz",
    1: "siyah",
    2: "gri",
    3: "kirmizi",
    4: "mavi",
    5: "sari",
    6: "yesil",
    7: "turuncu",
    8: "kahverengi",
}

NUM_CLASSES   = 9
INPUT_SIZE    = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# -------------------------------------------------------------------
# ColorClassifier
# -------------------------------------------------------------------
class ColorClassifier:
    """
    MobileNetV3-Large tabanlı araç renk sınıflandırıcı.

    Kullanım:
        classifier = ColorClassifier("/app/models/araçrenktespit/.../arac_renk_mobilenetv3.pth")
        label, confidence = classifier.predict(vehicle_crop_bgr)

    NOT: Model eğitimi torchvision.models.mobilenet_v3_large veya
    timm'in mobilenetv3_large_100 mimarisiyle yapıldıysa doğru çalışır.
    Eğer 'unexpected key' hatası alırsanız aşağıdaki MODEL_ARCH'ı değiştirin:
      - "mobilenetv3_large_100" (timm, varsayılan)
      - "mobilenetv3_small_100" (timm, küçük varyant)
    """

    MODEL_ARCH = "mobilenetv3_large_100"

    def __init__(self, model_path: str, device: str = "cuda"):
        assert TORCH_AVAILABLE, "PyTorch kurulu değil!"
        self.device     = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self._build_model()
        self._load_weights()
        self._build_transforms()
        self.model.eval()
        logger.info(f"ColorClassifier yüklendi: {model_path} → device={self.device}")

    # ------------------------------------------------------------------
    def _build_model(self):
        """MobileNetV3-Large (9 çıkış sınıfı)."""
        self.model = timm.create_model(
            self.MODEL_ARCH,
            pretrained=False,
            num_classes=NUM_CLASSES
        )
        self.model = self.model.to(self.device)

    # ------------------------------------------------------------------
    def _load_weights(self):
        """
        .pth dosyasını yükle.
        Desteklenen formatlar: state_dict, {'state_dict':...}, tam model objesi.

        Eğer model torchvision.models ile eğitildiyse ve
        timm ağırlık boyutları uyuşmuyorsa:
          → strict=False ile deneyin (classifier katmanı yeniden init edilir)
        """
        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True
        )

        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # DataParallel prefix temizle
        state_dict = {k.replace("module.", ""): v
                      for k, v in state_dict.items()}

        try:
            self.model.load_state_dict(state_dict, strict=True)
            logger.info("MobileNetV3 (renk) ağırlıkları yüklendi (strict).")
        except RuntimeError as e:
            logger.warning(f"Strict yükleme başarısız, strict=False deneniyor: {e}")
            self.model.load_state_dict(state_dict, strict=False)
            logger.info("MobileNetV3 (renk) ağırlıkları yüklendi (strict=False).")

    # ------------------------------------------------------------------
    def _build_transforms(self):
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((INPUT_SIZE, INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    # ------------------------------------------------------------------
    def predict(self, img_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Araç crop'undan renk tahmin eder.

        Args:
            img_bgr: BGR numpy array (araç crop)

        Returns:
            (label, confidence)
        """
        if img_bgr is None or img_bgr.size == 0:
            return "beyaz", 0.0

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor  = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

        class_idx  = pred.item()
        confidence = conf.item()
        label      = COLOR_CLASS_NAMES.get(class_idx, "beyaz")

        return label, round(confidence, 4)

    # ------------------------------------------------------------------
    def predict_topk(self, img_bgr: np.ndarray, k: int = 3) -> list:
        """Top-K renk tahmini (debug/log için)."""
        if img_bgr is None or img_bgr.size == 0:
            return [("beyaz", 0.0)]

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor  = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]

        topk_vals, topk_idxs = torch.topk(probs, k=min(k, NUM_CLASSES))
        return [
            (COLOR_CLASS_NAMES.get(idx.item(), "beyaz"),
             round(val.item(), 4))
            for val, idx in zip(topk_vals, topk_idxs)
        ]
