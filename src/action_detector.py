"""
action_detector.py — Sürücü Eylemi Sınıflandırıcı (MobileNetV3)
================================================================
sofor_eylemi_mobilenetv3.pth — 7 eylem sınıfı

Sınıf eşlemesi (kullanıcı tarafından doğrulanan):
  0: su_icme
  1: telefonla_konusma
  2: arkaya_bakma
  3: sigara_icme
  4: emniyet_kemeri_ihlali
  5: etrafa_bakinma
  6: esneme

NOT — Slalom tespiti:
  Slalom hareketi bu modelde YOK.
  Slalom tespiti ilerleyen aşamada optik akış / piksel analizi ile
  ayrı bir modül (slalom_detector.py) olarak eklenecektir.

Gerekli kütüphaneler:
  - torch>=2.1.0
  - torchvision>=0.16.0
  - timm>=0.9.0
  - opencv-python-headless, numpy
"""

import logging
import os
import zipfile
import tempfile
from typing import Dict, List, Optional, Tuple

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

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


# -------------------------------------------------------------------
# Sabitler — Kullanıcı tarafından doğrulanan sınıf sırası
# -------------------------------------------------------------------
ACTION_CLASS_NAMES = {
    0: "su_icme",
    1: "telefonla_konusma",
    2: "arkaya_bakma",
    3: "sigara_icme",
    4: "emniyet_kemeri_ihlali",
    5: "etrafa_bakinma",
    6: "esneme",
    # "slalom" → ayrı modül (slalom_detector.py) — SONRA eklenecek
}

NUM_CLASSES   = 7
INPUT_SIZE    = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Düşük confidence eşiği — bu değerin altında eylem tespit edilmemiş sayılır
ACTION_CONF_THRESHOLD = 0.60


# -------------------------------------------------------------------
# ActionDetector
# -------------------------------------------------------------------
class ActionDetector:
    """
    MobileNetV3-Large tabanlı sürücü eylemi sınıflandırıcı.

    Giriş: Araç görüntüsünün sürücü bölgesi crop'u
    (vehicle_detector.get_driver_crop() çıkışı)

    Kullanım:
        detector = ActionDetector("/app/models/sofor_eylemi_mobilenetv3.pth")
        label, confidence = detector.predict(driver_crop_bgr)

    Model dosyası bulunamazsa None döner (opsiyonel bileşen).
    """

    MODEL_ARCH = "mobilenetv3_large_100"

    def __init__(self, model_path: str, device: str = "cuda"):
        assert TORCH_AVAILABLE, "PyTorch kurulu değil!"

        # Model dosyası yoksa uyarı ver ama çökme
        if not os.path.exists(model_path):
            logger.warning(
                f"Sürücü eylemi modeli bulunamadı: {model_path}\n"
                f"ActionDetector devre dışı. Tespitler results.json'a eklenmeyecek."
            )
            self.enabled = False
            return

        self.enabled    = True
        self.device     = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self._build_model()
        self._load_weights()
        self._build_transforms()
        self.model.eval()
        logger.info(f"ActionDetector yüklendi: {model_path} → device={self.device}")

    # ------------------------------------------------------------------
    def _build_model(self):
        """
        MobileNetV3-Large (7 cikis sinifi).
        Once torchvision dener (features.* key formati).
        Basarisizsa timm ile dener.
        """
        # torchvision MobileNetV3-Large
        self.model = tv_models.mobilenet_v3_large(weights=None)
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
        self.model = self.model.to(self.device)
        self._arch = "torchvision"

    # ------------------------------------------------------------------
    def _load_weights(self):
        """
        .pth dosyasini veya PyTorch zip-klasor formatini yukle.

        mobilenetv3_surucu_hareketleri bir KLASOR olabilir:
          PyTorch, torch.save() ciktisini bazen zip-directory olarak
          olusturur (data.pkl + data/ klasoru). Bu durumda klasoru
          gecici bir zip dosyasina paketleyip yukleriz.
        """
        model_path = self.model_path

        # --- Klasor formatini zip'e cevir ---
        if os.path.isdir(model_path):
            logger.info(
                f"Klasor formatinda PyTorch modeli tespit edildi: {model_path}\n"
                f"Gecici zip arsivinize donusturuluyor..."
            )
            tmp_zip = self._pack_dir_to_zip(model_path)
            try:
                checkpoint = torch.load(
                    tmp_zip,
                    map_location=self.device,
                    weights_only=False
                )
            finally:
                if os.path.exists(tmp_zip):
                    os.unlink(tmp_zip)
        else:
            checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=False
            )

        # --- State dict'i cikart ---
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        elif hasattr(checkpoint, 'state_dict'):
            # tam model nesnesi kaydedilmis
            state_dict = checkpoint.state_dict()
        else:
            state_dict = checkpoint

        # "module." prefix kaldir (DataParallel)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # --- Strict yuklemeyi dene ---
        try:
            self.model.load_state_dict(state_dict, strict=True)
            logger.info(f"MobileNetV3 (eylem/{self._arch}) agirliklar yuklendi (strict=True).")
            return
        except RuntimeError as e:
            logger.warning(f"Strict yukleme basarisiz ({self._arch}): {str(e)[:120]}")

        # --- torchvision ile olmadiysa timm dene ---
        if self._arch == "torchvision" and TIMM_AVAILABLE:
            logger.info("timm MobileNetV3-Large ile yeniden deneniyor...")
            self.model = timm.create_model(
                "mobilenetv3_large_100",
                pretrained=False,
                num_classes=NUM_CLASSES
            ).to(self.device)
            self._arch = "timm"
            try:
                self.model.load_state_dict(state_dict, strict=True)
                logger.info("MobileNetV3 (eylem/timm) agirliklar yuklendi (strict=True).")
                return
            except RuntimeError as e2:
                logger.warning(f"timm strict yukleme de basarisiz: {str(e2)[:120]}")

        # --- Son care: strict=False ---
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        logger.info(
            f"MobileNetV3 (eylem/{self._arch}) agirliklar yuklendi (strict=False). "
            f"Eksik: {len(missing)}, Fazla: {len(unexpected)} anahtar."
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _pack_dir_to_zip(dir_path: str) -> str:
        """
        PyTorch zip-klasor formatini gecici bir .pt dosyasina paketler.
        PyTorch'un beklendigi zip yapisini korur:
          <klasor_adi>/data.pkl
          <klasor_adi>/data/0
          ...
        """
        dir_name = os.path.basename(dir_path.rstrip("/\\"))
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pt")
        os.close(tmp_fd)

        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for root, dirs, files in os.walk(dir_path):
                # Gizli dosyalari da dahil et (.format_version, .storage_alignment vs)
                for fname in files:
                    fpath = os.path.join(root, fname)
                    # Zip icindeki yol: klasor_adi/... seklinde olmali
                    rel = os.path.relpath(fpath, os.path.dirname(dir_path))
                    arcname = rel.replace("\\", "/")
                    zf.write(fpath, arcname)

        logger.info(f"Gecici zip olusturuldu: {tmp_path}")
        return tmp_path

    # ------------------------------------------------------------------
    def _build_transforms(self):
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((INPUT_SIZE, INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    # ------------------------------------------------------------------
    def predict(self, img_bgr: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Sürücü bölgesi crop'undan eylem tahmin eder.

        Args:
            img_bgr: BGR numpy array (sürücü alanı crop)

        Returns:
            (label, confidence)
            Confidence ACTION_CONF_THRESHOLD altındaysa (None, 0.0) döner
            (results.json'a eklenmez).
        """
        if not self.enabled:
            return None, 0.0

        if img_bgr is None or img_bgr.size == 0:
            return None, 0.0

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor  = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

        class_idx  = pred.item()
        confidence = conf.item()

        # Eşiğin altındaysa tespit yok
        if confidence < ACTION_CONF_THRESHOLD:
            return None, 0.0

        label = ACTION_CLASS_NAMES.get(class_idx, None)
        return label, round(confidence, 4)

    # ------------------------------------------------------------------
    def predict_all_scores(self, img_bgr: np.ndarray) -> dict:
        """
        Tüm sınıfların confidence skorlarını döner.
        Debug ve VehicleCache oy birleştirme için kullanışlı.

        Returns:
            {"su_icme": 0.05, "telefonla_konusma": 0.87, ...}
        """
        if not self.enabled or img_bgr is None or img_bgr.size == 0:
            return {name: 0.0 for name in ACTION_CLASS_NAMES.values()}

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor  = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]

        return {
            ACTION_CLASS_NAMES[i]: round(probs[i].item(), 4)
            for i in range(NUM_CLASSES)
        }


# -------------------------------------------------------------------
# Slalom placeholder — ilerleyen aşamada doldurulacak
# -------------------------------------------------------------------
class SlalomDetector:
    """
    PLACEHOLDER — Slalom tespiti için optik akış / piksel analizi.

    Bu sınıf sonraki aşamada şu yöntemlerden biriyle implemente edilecek:
      1. Optik akış (Lucas-Kanade) ile yatay hareket analizi
      2. Lane detection + şerit değişim frekansı
      3. Araç bbox merkezi x-koordinat osilatör tespiti

    Şu an her zaman (None, 0.0) döner.
    """

    def __init__(self):
        logger.info("SlalomDetector: placeholder modu (henüz implemente edilmedi).")

    def update(self, track_id: int, vehicle_bbox: tuple,
               timestamp_sec: float) -> Tuple[Optional[str], float]:
        """
        Araç bbox geçmişinden slalom hareketi tespit eder.
        Returns:
            ("slalom", confidence) veya (None, 0.0)
        """
        # TODO: Optik akış / bbox x-koordinat analizi
        return None, 0.0
