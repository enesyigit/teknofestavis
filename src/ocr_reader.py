"""
ocr_reader.py — PaddleOCR ile Turk Plaka Metni Okuma
====================================================
Egitilmis rec model: /app/models/paddle_models/model.pdparams

Turk plaka formati: XXYYYY ZZZ
  XX   = 2 rakam (il kodu: 01-81)
  YYY  = 1-3 harf (A-Z)
  ZZZZ = 1-4 rakam
  Birlestik format (bosluksuz): 34ABC123

Paddle model klasoru yapisi:
  /app/models/paddle_models/
    model.pdparams   <- rec modeli agirliklar (68MB)
    model.pdopt      <- optimizer state (inference'da gerekli degil)

NOT: Bu dosyalar recognition (rec) modeli icin.
  Detection (det) ve orientation classification (cls)
  icin PP-OCRv4 on-demand indirilecek veya ayri klasore konulacak.
"""

import logging
import os
import re
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    logger.error("PaddleOCR kurulu degil! pip install paddleocr")


# -------------------------------------------------------------------
# Turk plaka regex — yarisma dokumanı formati
# -------------------------------------------------------------------
PLATE_REGEX = re.compile(r'^([0-9]{2})([A-Z]{1,3})([0-9]{1,4})$')

# -------------------------------------------------------------------
# Model yollari
# Docker icinde: /app/models/paddle_models/
# Kaggle icinde: /kaggle/working/models/paddle_models/
# -------------------------------------------------------------------
def _find_paddle_rec_dir() -> Optional[str]:
    """
    model.pdparams dosyasinin bulundugu rec klasorunu bulur.
    Yarısma Docker yolu: /app/models/paddle_models/
    """
    candidates = [
        "/app/models/paddle_models",                                          # Docker (yarisma)
        os.path.join(os.path.dirname(__file__), "..", "models", "paddle_models"),  # yerel
    ]
    for path in candidates:
        path = os.path.normpath(path)
        if os.path.exists(os.path.join(path, "model.pdparams")):
            logger.info(f"PaddleOCR rec model bulundu: {path}")
            return path
    logger.warning("PaddleOCR rec model bulunamadi — PP-OCRv4 kullanilacak.")
    return None


PADDLE_REC_DIR = _find_paddle_rec_dir()



# -------------------------------------------------------------------
# OCRReader
# -------------------------------------------------------------------
class OCRReader:
    """
    PaddleOCR tabanli plaka okuyucu.
    Egitilmis recognition modeli kullanir (model.pdparams).

    Kullanim:
        reader = OCRReader(use_gpu=True)
        plate_text, confidence = reader.read(plate_crop_bgr)
        # plate_text: "34ABC123" veya None
    """

    def __init__(self,
                 use_gpu: bool = True,
                 rec_model_dir: Optional[str] = None):
        """
        Args:
            use_gpu       : GPU kullan (Tesla T4 = True)
            rec_model_dir : Egitilmis rec model klasoru. None ise otomatik bulunur.
        """
        if not PADDLE_AVAILABLE:
            logger.error("PaddleOCR yuklu degil! OCR devre disi.")
            self.enabled = False
            return

        self.enabled = True

        # Recognition model dizinini belirle
        rec_dir = rec_model_dir or PADDLE_REC_DIR

        # PaddleOCR baslatma parametreleri
        ocr_kwargs = dict(
            use_angle_cls = True,   # donmus plaka destegi
            lang          = "en",   # Latin alfabe (Turk plaka)
            use_gpu       = use_gpu,
            show_log      = False,
        )

        if rec_dir and os.path.isdir(rec_dir):
            # Egitilmis rec model kullan
            ocr_kwargs["rec_model_dir"] = rec_dir
            logger.info(f"PaddleOCR: egitilmis rec model -> {rec_dir}")
            logger.info(f"  model.pdparams: {os.path.join(rec_dir, 'model.pdparams')}")
        else:
            # Rec model bulunamadi, PP-OCRv4 on-demand indir
            logger.warning(
                "PaddleOCR rec model bulunamadi! "
                "PP-OCRv4 modeli internet'ten indirilecek."
            )

        try:
            self.ocr = PaddleOCR(**ocr_kwargs)
            logger.info(f"OCRReader hazir (GPU={use_gpu}, rec={rec_dir or 'auto'}).")
        except Exception as e:
            logger.error(f"PaddleOCR baslatma hatasi: {e}")
            self.enabled = False

    # ------------------------------------------------------------------
    def _preprocess(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Plaka crop'unu OCR icin on isler:
          1. Minimum boyut garantisi (en az 32x64)
          2. CLAHE kontrast artirma
          3. BGR formati koru (PaddleOCR BGR/RGB kabul eder)
        """
        if img_bgr is None or img_bgr.size == 0:
            return None

        h, w = img_bgr.shape[:2]

        # Cok kucukse buyut
        if h < 32 or w < 64:
            scale = max(32 / max(h, 1), 64 / max(w, 1), 1.5)
            img_bgr = cv2.resize(
                img_bgr,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC
            )

        # CLAHE kontrast artirma
        gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray_eq  = clahe.apply(gray)

        return cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)

    # ------------------------------------------------------------------
    def _normalize(self, raw_text: str) -> Optional[str]:
        """
        OCR ham ciktisini Turk plaka formatina normalize eder.
        Ciktilar zaten regex formatinda geliyorsa direkt doner.

        Adimlar:
          1. Turkce karakter donusturme
          2. Bosluk ve ozel karakter temizle
          3. Buyuk harfe cevir
          4. Regex dogrulama: ^([0-9]{2})([A-Z]{1,3})([0-9]{1,4})$
          5. Kismi eslesme denemesi (OCR bazen karakter kacırır)
        """
        if not raw_text:
            return None

        # Turkce karakter donusturme
        tr_map = str.maketrans("ıİğĞüÜşŞöÖçÇ", "iIgGuUsSoOcC")
        text = raw_text.translate(tr_map)

        # Sadece alfanumerik, buyuk harf
        text = re.sub(r'[^A-Z0-9]', '', text.upper())

        if not text:
            return None

        # Tam regex eslesmesi
        if PLATE_REGEX.match(text):
            return text

        # Kismi eslesmesi — OCR bazen bosluk/cizgi ekler
        partial = re.search(r'([0-9]{2}[A-Z]{1,3}[0-9]{1,4})', text)
        if partial:
            candidate = partial.group(1)
            if PLATE_REGEX.match(candidate):
                return candidate

        logger.debug(f"Gecersiz plaka formati: '{text}'")
        return None

    # ------------------------------------------------------------------
    def _run_ocr(self, img_bgr: np.ndarray) -> Tuple[Optional[str], float]:
        """PaddleOCR calistir, en iyi sonucu don."""
        if not self.enabled:
            return None, 0.0

        try:
            result = self.ocr.ocr(img_bgr, cls=True)
        except Exception as e:
            logger.warning(f"PaddleOCR hatasi: {e}")
            return None, 0.0

        if not result or not result[0]:
            return None, 0.0

        best_text = None
        best_conf = 0.0

        for line in result[0]:
            if line is None or len(line) < 2:
                continue
            text_info = line[1]
            if not text_info:
                continue

            raw  = text_info[0]
            conf = float(text_info[1])

            norm = self._normalize(raw)
            if norm and conf > best_conf:
                best_text = norm
                best_conf = conf

        if best_text:
            logger.debug(f"Plaka: {best_text} (conf={best_conf:.3f})")

        return best_text, round(best_conf, 4)

    # ------------------------------------------------------------------
    def read(self, plate_crop_bgr: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Plaka crop'undan metin okur (tek deneme).

        Returns:
            ("34ABC123", 0.92) veya (None, 0.0)
        """
        preprocessed = self._preprocess(plate_crop_bgr)
        if preprocessed is None:
            return None, 0.0
        return self._run_ocr(preprocessed)

    # ------------------------------------------------------------------
    def read_with_retry(self,
                        plate_crop_bgr: np.ndarray,
                        retries: int = 2) -> Tuple[Optional[str], float]:
        """
        Birden fazla on-isleme stratejisi ile okuma dener.
        Ilk basarili sonucu doner.

        Strateji 1: CLAHE on-isleme (varsayilan)
        Strateji 2: Keskinlestirme filtresi + CLAHE
        Strateji 3: Otsu threshold + binary
        """
        # Strateji 1: Normal on islem
        text, conf = self.read(plate_crop_bgr)
        if text:
            return text, conf

        if plate_crop_bgr is None or plate_crop_bgr.size == 0:
            return None, 0.0

        # Strateji 2: Keskinlestirme
        if retries >= 1:
            kernel    = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened = cv2.filter2D(plate_crop_bgr, -1, kernel)
            text, conf = self.read(sharpened)
            if text:
                return text, conf

        # Strateji 3: Otsu threshold
        if retries >= 2:
            gray = cv2.cvtColor(plate_crop_bgr, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            text, conf = self.read(binary_bgr)
            if text:
                return text, conf

        return None, 0.0
