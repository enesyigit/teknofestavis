"""
vehicle_detector.py — Araç + Plaka Tespiti ve ByteTrack Takibi
==============================================================
aracplak.engine (TRT FP16) çıktısını alır, araçlara track_id atar
ve her araç için crop bilgisi döner.

Gerekli kütüphaneler:
  - supervision>=0.19.0    (ByteTrack implementasyonu)
  - opencv-python-headless
  - numpy
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# supervision — ByteTrack içerir
try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    logger.warning("supervision kurulu değil. ByteTrack devre dışı.")

from src.trt_engine import TRTEngine


# -------------------------------------------------------------------
# Veri yapıları
# -------------------------------------------------------------------
@dataclass
class TrackedVehicle:
    """
    Bir frame'de takip edilen tek bir araç.
    """
    track_id:        int
    vehicle_bbox:    Tuple[int, int, int, int]   # (x1, y1, x2, y2) — orijinal frame
    vehicle_conf:    float
    vehicle_crop:    np.ndarray                  # BGR araç crop

    plate_bbox:      Optional[Tuple[int, int, int, int]] = None
    plate_conf:      float = 0.0
    plate_crop:      Optional[np.ndarray] = None

    frame_idx:       int = 0
    timestamp_sec:   float = 0.0


# -------------------------------------------------------------------
# Yardımcı: bbox genişletme (araç bbox'ından sürücü bölgesi)
# -------------------------------------------------------------------
def get_driver_crop(vehicle_crop: np.ndarray) -> np.ndarray:
    """
    [DEPRECATED] — main.py artık WindshieldEstimator kullanıyor.
    Bu fonksiyon yalnızca fallback veya test amaçlı tutulmaktadır.

    Araç crop'unun sol ön bölümünü (sürücü alanı) döner.
    Sol yarı + üst %60 (Türkiye: sol direksiyon)
    """
    h, w = vehicle_crop.shape[:2]
    # Sol yarı (sürücü tarafı) + üst %60
    driver_crop = vehicle_crop[:int(h * 0.6), :int(w * 0.55)]
    if driver_crop.size == 0:
        return vehicle_crop  # fallback
    return driver_crop


# -------------------------------------------------------------------
# Yardımcı: plate bbox'ını araç içinde bul (IoU eşleşme)
# -------------------------------------------------------------------
def match_plate_to_vehicle(vehicle_bbox: tuple,
                            plates: list) -> Optional[tuple]:
    """
    Araç bbox'ının içinde kalan en yüksek confidence'lı plaka BBOX'ını döner.
    plates: [(x1,y1,x2,y2,conf), ...]
    """
    vx1, vy1, vx2, vy2 = vehicle_bbox
    best_plate  = None
    best_conf   = -1.0

    for px1, py1, px2, py2, pconf in plates:
        # Plaka merkezi araç içinde mi?
        pcx = (px1 + px2) / 2
        pcy = (py1 + py2) / 2
        if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
            if pconf > best_conf:
                best_conf  = pconf
                best_plate = (px1, py1, px2, py2, pconf)

    return best_plate


# -------------------------------------------------------------------
# Yardımcı: güvenli crop (sınır aşımı koruma)
# -------------------------------------------------------------------
def safe_crop(frame: np.ndarray, x1: int, y1: int,
              x2: int, y2: int) -> np.ndarray:
    H, W = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W, x2); y2 = min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    return frame[y1:y2, x1:x2].copy()


# -------------------------------------------------------------------
# VehicleDetector — Ana sınıf
# -------------------------------------------------------------------
class VehicleDetector:
    """
    TRT engine + ByteTrack ile araç tespiti ve takibi.

    Pipeline:
        frame → TRTEngine.detect() → {arac: [...], plak: [...]}
              → ByteTrack (araçlara track_id ata)
              → Her araç için plaka eşleştir
              → TrackedVehicle listesi döner
    """

    def __init__(self,
                 engine_path: str,
                 conf_threshold: float = 0.50,
                 iou_threshold:  float = 0.45,
                 fps:            float = 30.0):
        """
        Args:
            engine_path    : aracplak.engine dosya yolu
            conf_threshold : Minimum tespit güveni (araç + plaka)
            iou_threshold  : NMS IoU eşiği
            fps            : Video FPS (ByteTrack için)
        """
        self.engine = TRTEngine(
            engine_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )

        # ByteTrack başlatma
        if SUPERVISION_AVAILABLE:
            self.tracker = sv.ByteTrack(frame_rate=int(fps))
            logger.info(f"ByteTrack başlatıldı (fps={fps})")
        else:
            self.tracker = None
            logger.warning("ByteTrack yok — track_id yok.")

        self._frame_counter = 0

    # ------------------------------------------------------------------
    def detect(self,
               frame: np.ndarray,
               timestamp_sec: float = 0.0) -> List[TrackedVehicle]:
        """
        Tek bir frame'den araç tespiti + takip.

        Args:
            frame         : BGR numpy array (orijinal frame)
            timestamp_sec : Frame zaman damgası (saniye cinsinden)

        Returns:
            List[TrackedVehicle]
        """
        self._frame_counter += 1
        detections = self.engine.detect(frame)

        vehicles = detections.get("arac", [])
        plates   = detections.get("plak", [])

        if not vehicles:
            return []

        tracked_vehicles = self._apply_bytetrack(frame, vehicles,
                                                  plates, timestamp_sec)
        return tracked_vehicles

    # ------------------------------------------------------------------
    def _apply_bytetrack(self,
                          frame:         np.ndarray,
                          vehicles:      list,
                          plates:        list,
                          timestamp_sec: float) -> List[TrackedVehicle]:
        """ByteTrack ile track_id atar, TrackedVehicle listesi oluşturur."""

        if not vehicles:
            return []

        # supervision Detections formatına çevir
        bboxes = np.array([[v[0], v[1], v[2], v[3]] for v in vehicles],
                          dtype=np.float32)
        confs  = np.array([v[4] for v in vehicles], dtype=np.float32)

        sv_det = sv.Detections(
            xyxy=bboxes,
            confidence=confs,
            class_id=np.zeros(len(vehicles), dtype=int)   # hepsi class 0 = arac
        )

        # ByteTrack güncelle
        if self.tracker is not None:
            tracked = self.tracker.update_with_detections(sv_det)
            track_ids  = tracked.tracker_id
            bboxes_trk = tracked.xyxy
            confs_trk  = tracked.confidence
        else:
            # Tracker yoksa index'i track_id olarak kullan
            track_ids  = np.arange(len(vehicles))
            bboxes_trk = bboxes
            confs_trk  = confs

        result = []
        for i, (tid, bbox, conf) in enumerate(zip(track_ids,
                                                   bboxes_trk,
                                                   confs_trk)):
            x1, y1, x2, y2 = bbox.astype(int)
            vehicle_crop = safe_crop(frame, x1, y1, x2, y2)

            # Plaka eşleştir
            matched_plate = match_plate_to_vehicle(
                (x1, y1, x2, y2), plates
            )
            plate_crop = None
            plate_bbox = None
            plate_conf = 0.0
            if matched_plate is not None:
                px1, py1, px2, py2, pconf = matched_plate
                plate_crop = safe_crop(frame, px1, py1, px2, py2)
                plate_bbox = (px1, py1, px2, py2)
                plate_conf = pconf

            tv = TrackedVehicle(
                track_id      = int(tid),
                vehicle_bbox  = (x1, y1, x2, y2),
                vehicle_conf  = float(conf),
                vehicle_crop  = vehicle_crop,
                plate_bbox    = plate_bbox,
                plate_conf    = plate_conf,
                plate_crop    = plate_crop,
                frame_idx     = self._frame_counter,
                timestamp_sec = timestamp_sec,
            )
            result.append(tv)

        return result

    # ------------------------------------------------------------------
    def reset(self):
        """Tracker durumunu sıfırla (yeni video için)."""
        if SUPERVISION_AVAILABLE and self.tracker is not None:
            self.tracker = sv.ByteTrack(frame_rate=30)
        self._frame_counter = 0
