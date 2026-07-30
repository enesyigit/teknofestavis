"""
windshield_estimator.py — Ön Cam Bölgesi Tahmin Algoritması
===========================================================
Araç BBOX'u + Plaka BBOX'u kullanarak sürücünün görüldüğü
ön cam bölgesini geometrik olarak tahmin eder.

Kaynak: gemini-code-1782603664282.py
Değişiklikler:
  - Bizim projemizin (x1,y1,x2,y2) bbox formatına uyarlandı
  - EfficientNet-B0 kasa sınıf isimlerini WindshieldEstimator sınıflarına eşler
  - safe_crop ile frame üzerinden doğrudan crop üretir
  - Güven düşükse fallback bölge döner (asla None dönmez)

Kullanım:
    estimator = WindshieldEstimator(w_ref=300.0)
    ws_crop, is_confident = estimator.get_windshield_crop(
        frame=frame_bgr,
        vehicle_bbox=(x1, y1, x2, y2),     # xyxy
        plate_bbox=(px1, py1, px2, py2),    # xyxy veya None
        body_type="sedan"                   # EfficientNet-B0 çıkışı
    )
"""

import math
import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# EfficientNet-B0 sınıf isimlerini → WindshieldEstimator kategorilerine eşle
# EfficientNet sınıfları: hatchback, kamyon, minibus, panelvan, pickup, sedan, suv
# -------------------------------------------------------------------
BODY_TO_WS_CLASS = {
    "sedan":    "sedan_hatchback",
    "hatchback":"sedan_hatchback",
    "suv":      "suv_crossover",
    "pickup":   "kamyonet",
    "panelvan": "kamyonet",
    "minibus":  "otobus_tir",
    "kamyon":   "otobus_tir",
}


# -------------------------------------------------------------------
# Yardımcı: xyxy → xywh dönüşümü
# -------------------------------------------------------------------
def xyxy_to_xywh(x1: int, y1: int, x2: int, y2: int) -> Tuple[float, float, float, float]:
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)


# -------------------------------------------------------------------
# Yardımcı: xywh ön cam bbox'unu xyxy'ye çevir + frame sınırlarına kırp
# -------------------------------------------------------------------
def xywh_to_xyxy_clipped(xw: int, yw: int, ww: int, hw: int,
                          frame_h: int, frame_w: int) -> Tuple[int, int, int, int]:
    x1 = max(0, xw)
    y1 = max(0, yw)
    x2 = min(frame_w, xw + ww)
    y2 = min(frame_h, yw + hw)
    return x1, y1, x2, y2


# -------------------------------------------------------------------
# WindshieldEstimator — Ana sınıf (gemini-code-1782603664282.py tabanlı)
# -------------------------------------------------------------------
class WindshieldEstimator:
    """
    Araç ve plaka BBOX'larından ön cam bölgesini geometrik olarak tahmin eder.

    Katmanlar:
      1. Kasa sınıfına göre dinamik oranlar (r_top, r_h, r_w)
      2. Logaritmik perspektif düzeltmesi (araç uzaklığına göre)
      3. Geometrik yaw düzeltmesi (plaka X sapmasından ön cam kaydırma)
      4. Plaka Y konumuyla güven doğrulaması

    Args:
        w_ref: Referans araç genişliği (piksel). Kameraya ve sahneye göre ayarlayın.
               Tipik değer: 280-320px (normal yol kamerası için)
    """

    # Kasa sınıfı bazlı ön cam oranları
    # r_top : Araç üst kenarından ön cam üstüne kadar oranı  (araç H'ye göre)
    # r_h   : Ön cam yüksekliği oranı                        (araç H'ye göre)
    # r_w   : Ön cam genişliği oranı                         (araç W'ye göre)
    CLASS_RATIOS = {
        "sedan_hatchback": {"r_top": 0.13, "r_h": 0.32, "r_w": 0.74},
        "suv_crossover":   {"r_top": 0.16, "r_h": 0.28, "r_w": 0.72},
        "kamyonet":        {"r_top": 0.10, "r_h": 0.22, "r_w": 0.78},
        "otobus_tir":      {"r_top": 0.08, "r_h": 0.40, "r_w": 0.82},
    }

    def __init__(self, w_ref: float = 300.0):
        self.w_ref = w_ref

    # ------------------------------------------------------------------
    def estimate_xywh(self,
                      vehicle_box: Tuple[float, float, float, float],
                      vehicle_class: str,
                      plate_box: Optional[Tuple[float, float, float, float]] = None) -> Dict:
        """
        Orijinal algoritma (xywh formatı — gemini-code-1782603664282.py ile aynı).

        Args:
            vehicle_box  : (x, y, w, h) araç bbox
            vehicle_class: WindshieldEstimator kategorisi (sedan_hatchback, suv_crossover, ...)
            plate_box    : (x, y, w, h) plaka bbox veya None

        Returns:
            {
                "windshield_box"  : (x, y, w, h) int,
                "is_confident"    : bool,
                "error_margin_px" : float,
                "debug_info"      : {...}
            }
        """
        x, y, w, h = vehicle_box

        # Araç merkezi (X ekseni)
        Cx = x + w / 2.0

        # Plaka bilgisi varsa merkez ve sapma hesapla
        delta_x = 0.0
        yp = None
        if plate_box is not None:
            xp, yp_val, wp, hp = plate_box
            Px = xp + wp / 2.0
            delta_x = Px - Cx
            yp = yp_val

        # 1. Sınıf oranlarını al (bilinmiyorsa sedan_hatchback)
        ratios = self.CLASS_RATIOS.get(
            vehicle_class, self.CLASS_RATIOS["sedan_hatchback"]
        )
        r_top, r_h, r_w = ratios["r_top"], ratios["r_h"], ratios["r_w"]

        # 2. Perspektif Düzeltmesi (Logaritmik Ölçekleme)
        alpha = w / self.w_ref if self.w_ref > 0 else 1.0
        if alpha > 0:
            r_top_prime = r_top * (1 - 0.08 * math.log(alpha))
        else:
            r_top_prime = r_top

        # r_top_prime negatife düşmesin (çok yakın araçlar için)
        r_top_prime = max(0.03, r_top_prime)

        # 3. Geometrik Yaw Düzeltmesi
        delta_x_ws = delta_x * 0.38

        # 4. Final Ön Cam Koordinatlarının Hesaplanması
        ww = w * r_w
        hw = h * r_h
        yw = y + h * r_top_prime
        xw = (Cx + delta_x_ws) - ww / 2.0

        # 5. Plaka Yüksekliğinden Güven Skoru Doğrulaması
        is_confident = True
        confidence_error = 0.0

        if yp is not None:
            y_expected = yp - h * 0.48
            confidence_error = abs(yw - y_expected)
            if confidence_error > (h * 0.10):
                is_confident = False

        windshield_box_int = (int(xw), int(yw), int(ww), int(hw))

        return {
            "windshield_box":   windshield_box_int,
            "is_confident":     is_confident,
            "error_margin_px":  round(confidence_error, 2),
            "debug_info": {
                "alpha":        round(alpha, 3),
                "r_top_prime":  round(r_top_prime, 4),
                "delta_x_ws":   round(delta_x_ws, 2),
            },
        }

    # ------------------------------------------------------------------
    def get_windshield_crop(self,
                            frame: np.ndarray,
                            vehicle_bbox: Tuple[int, int, int, int],
                            plate_bbox:   Optional[Tuple[int, int, int, int]],
                            body_type:    str) -> Tuple[np.ndarray, bool]:
        """
        Proje ana arayüzü — frame'den direkt ön cam crop'u döner.

        Args:
            frame        : BGR numpy array (orijinal video frame)
            vehicle_bbox : (x1, y1, x2, y2) araç bbox — xyxy formatı
            plate_bbox   : (x1, y1, x2, y2) plaka bbox — xyxy formatı veya None
            body_type    : EfficientNet-B0 çıkışı ("sedan", "suv", "kamyon", ...)

        Returns:
            (windshield_crop, is_confident)
            windshield_crop: BGR numpy array  — action_detector.predict()'e gider
            is_confident   : bool — False ise bir önceki güvenilir frame crop'u önerilir
        """
        frame_h, frame_w = frame.shape[:2]

        vx1, vy1, vx2, vy2 = vehicle_bbox
        v_xywh = xyxy_to_xywh(vx1, vy1, vx2, vy2)

        # Plaka xyxy → xywh
        p_xywh = None
        if plate_bbox is not None:
            px1, py1, px2, py2 = plate_bbox
            p_xywh = xyxy_to_xywh(px1, py1, px2, py2)

        # WindshieldEstimator sınıf eşlemesi
        ws_class = BODY_TO_WS_CLASS.get(body_type, "sedan_hatchback")

        # Algoritma çalıştır
        result = self.estimate_xywh(
            vehicle_box   = v_xywh,
            vehicle_class = ws_class,
            plate_box     = p_xywh,
        )

        wx, wy, ww, wh = result["windshield_box"]
        is_confident    = result["is_confident"]

        if not is_confident:
            logger.debug(
                f"Ön cam tahmini düşük güven "
                f"(hata={result['error_margin_px']:.1f}px, "
                f"body={body_type}, ws_class={ws_class})"
            )

        # Frame sınırlarına kırp
        wx1, wy1, wx2, wy2 = xywh_to_xyxy_clipped(wx, wy, ww, wh, frame_h, frame_w)

        # Geçerli bir bölge var mı?
        if wx2 <= wx1 or wy2 <= wy1:
            logger.debug("Ön cam bbox geçersiz, araç üst yarısına fallback.")
            wx1, wy1 = vx1, vy1
            wx2 = vx2
            wy2 = vy1 + (vy2 - vy1) // 2
            is_confident = False

        crop = frame[wy1:wy2, wx1:wx2].copy()

        # Crop çok küçükse araç üst yarısına fallback
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            logger.debug("Ön cam crop çok küçük, fallback uygulanıyor.")
            fallback_y2 = vy1 + (vy2 - vy1) // 2
            crop = frame[vy1:max(vy1 + 20, fallback_y2), vx1:vx2].copy()
            is_confident = False

        return crop, is_confident

    # ------------------------------------------------------------------
    def get_windshield_bbox_xyxy(self,
                                  vehicle_bbox: Tuple[int, int, int, int],
                                  plate_bbox:   Optional[Tuple[int, int, int, int]],
                                  body_type:    str,
                                  frame_shape:  Tuple[int, int]) -> Tuple[Tuple[int,int,int,int], bool]:
        """
        Sadece ön cam bbox koordinatlarını (xyxy) döner (crop üretmez).
        Debug görselleştirme veya başka işlem için kullanılır.

        Returns:
            ((x1, y1, x2, y2), is_confident)
        """
        frame_h, frame_w = frame_shape
        vx1, vy1, vx2, vy2 = vehicle_bbox
        v_xywh = xyxy_to_xywh(vx1, vy1, vx2, vy2)

        p_xywh = None
        if plate_bbox is not None:
            px1, py1, px2, py2 = plate_bbox
            p_xywh = xyxy_to_xywh(px1, py1, px2, py2)

        ws_class = BODY_TO_WS_CLASS.get(body_type, "sedan_hatchback")
        result   = self.estimate_xywh(v_xywh, ws_class, p_xywh)

        wx, wy, ww, wh = result["windshield_box"]
        bbox_xyxy = xywh_to_xyxy_clipped(wx, wy, ww, wh, frame_h, frame_w)

        return bbox_xyxy, result["is_confident"]
