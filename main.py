"""
main.py — Teknofest AVIS Ana Pipeline Orkestratörü
===================================================
Giriş : /app/data/input/video.mp4
Çıkış : /app/data/output/results.json

Pipeline akışı:
  video.mp4
    → frame okuma (OpenCV)
    → VehicleDetector (aracplak.engine TRT FP16 + ByteTrack)
    → TrackedVehicle listesi
    → Paralel sınıflandırma:
         BodyClassifier    (EfficientNet-B0, 7 kasa tipi)
         ColorClassifier   (MobileNetV3, 9 renk)
         ActionDetector    (MobileNetV3, 7 sürücü eylemi)
         OCRReader         (PaddleOCR PP-OCRv4, plaka metni)
    → VehicleCache (oy birleştirme + güven kilidi ≥0.85)
    → OutputBuilder → results.json

Kısıtlamalar (yarışma dokümanı):
  - Max çalışma: 10 dakika
  - GPU: Tesla T4 (CUDA 12.1, TensorRT 8.6)
  - Değerlendirme ortamı tespiti → YASAK
"""

import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# -------------------------------------------------------------------
# Loglama yapılandırması
# -------------------------------------------------------------------
# Dinamik log klasörü oluşturma
LOG_DIR = os.path.dirname(os.getenv("OUTPUT_PATH", "/app/data/output/results.json"))
try:
    if LOG_DIR:
        os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")
except Exception:
    LOG_FILE = "pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ]
)
logger = logging.getLogger("main")


# -------------------------------------------------------------------
# Yol sabitleri (Docker içi yollar)
# -------------------------------------------------------------------
VIDEO_INPUT_PATH  = os.getenv("VIDEO_PATH",
                               "/app/data/input/video.mp4")
OUTPUT_JSON_PATH  = os.getenv("OUTPUT_PATH",
                               "/app/data/output/results.json")

# -------------------------------------------------------------------
# Model Yollari - Parametrik (Kaggle testi icin degistirilebilir)
# -------------------------------------------------------------------
MODEL_DIR          = Path("/app/models")
ENGINE_PATH        = MODEL_DIR / "aracplak.engine"
BODY_MODEL_PATH    = MODEL_DIR / "efficientb0arakasa.pth"
COLOR_MODEL_PATH   = (MODEL_DIR / "araçrenktespit"
                               / "araçrenktespit"
                               / "arac_renk_mobilenetv3.pth")
ACTION_MODEL_PATH  = MODEL_DIR.parent / "mobilenetv3_surucu_hareketleri"
# NOT: ACTION_MODEL_PATH torch.save() ile kaydedilmis zip formatli bir klasordur.

# Pipeline parametreleri
FRAME_SKIP         = int(os.getenv("FRAME_SKIP", "3"))    # Her N frame'de 1 analiz
VEHICLE_CONF_THR   = float(os.getenv("VEHICLE_CONF", "0.50"))
MAX_RUNTIME_SEC    = 570  # 9.5 dakika (guvenli margin)


# -------------------------------------------------------------------
# Model yükleme
# -------------------------------------------------------------------
def load_models(device: str = "cuda"):
    """Tüm modelleri GPU'ya yükler ve dict olarak döner."""

    from src.vehicle_detector import VehicleDetector
    from src.body_classifier  import BodyClassifier
    from src.color_classifier import ColorClassifier
    from src.action_detector  import ActionDetector
    from src.ocr_reader       import OCRReader
    from src.windshield_estimator import WindshieldEstimator

    logger.info("=" * 60)
    logger.info("MODEL YÜKLEME BAŞLIYOR")
    logger.info("=" * 60)

    t0 = time.time()

    models = {}

    # 1. Araç + Plaka Dedektörü (TRT FP16)
    logger.info(f"[1/4] VehicleDetector yükleniyor: {ENGINE_PATH}")
    assert ENGINE_PATH.exists(), f"Engine bulunamadı: {ENGINE_PATH}"
    models["detector"] = VehicleDetector(
        engine_path    = str(ENGINE_PATH),
        conf_threshold = VEHICLE_CONF_THR,
        fps            = 30.0
    )

    # 2. Kasa Tipi Sınıflandırıcı (EfficientNet-B0)
    logger.info(f"[2/4] BodyClassifier yükleniyor: {BODY_MODEL_PATH}")
    assert BODY_MODEL_PATH.exists(), f"Body model bulunamadı: {BODY_MODEL_PATH}"
    models["body"] = BodyClassifier(str(BODY_MODEL_PATH), device=device)

    # 3. Renk Sınıflandırıcı (MobileNetV3)
    logger.info(f"[3/4] ColorClassifier yükleniyor: {COLOR_MODEL_PATH}")
    assert COLOR_MODEL_PATH.exists(), f"Renk model bulunamadı: {COLOR_MODEL_PATH}"
    models["color"] = ColorClassifier(str(COLOR_MODEL_PATH), device=device)

    # 4. Sürücü Eylemi (MobileNetV3) — opsiyonel
    logger.info(f"[4/4] ActionDetector yükleniyor: {ACTION_MODEL_PATH}")
    models["action"] = ActionDetector(str(ACTION_MODEL_PATH), device=device)

    # 5. PaddleOCR (GPU)
    logger.info("[5/5] OCRReader (PaddleOCR) yükleniyor...")
    models["ocr"] = OCRReader(use_gpu=(device == "cuda"))

    # 6. Ön Cam Estimator (geometrik, model yok — her zaman hazır)
    logger.info("[6/6] WindshieldEstimator başlatılıyor...")
    # w_ref: kamera için ortalama araç genişliği (piksel).
    # Kendi kameranıza göre ayarlayın (tipik dashcam: 280-320)
    models["windshield"] = WindshieldEstimator(w_ref=300.0)

    elapsed = time.time() - t0
    logger.info(f"Tüm modeller yüklendi. ({elapsed:.1f}s)")
    logger.info("=" * 60)
    return models


# -------------------------------------------------------------------
# Video meta bilgisi
# -------------------------------------------------------------------
def get_video_info(cap: cv2.VideoCapture) -> dict:
    return {
        "fps":       cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "width":     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


# -------------------------------------------------------------------
# Birincil araç seçimi
# -------------------------------------------------------------------
def select_primary_vehicle(frame_counts: dict) -> int:
    """
    En uzun süre takip edilen (en çok frame gören) araç = birincil araç.
    frame_counts: {track_id: count}
    """
    if not frame_counts:
        return -1
    return max(frame_counts, key=frame_counts.get)


# -------------------------------------------------------------------
# Ana pipeline
# -------------------------------------------------------------------
def run_pipeline(video_path: str,
                 output_path: str,
                 models: dict,
                 device: str = "cuda"):
    """
    Ana video işleme döngüsü.
    """
    from src.vehicle_cache    import VehicleCache
    from src.output_builder   import OutputBuilder

    # Video aç
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Video açılamadı: {video_path}")
        sys.exit(1)

    info = get_video_info(cap)
    fps  = info["fps"]
    total_frames = info["frame_count"]
    logger.info(f"Video: {info['width']}x{info['height']} @ {fps:.1f}fps "
                f"({total_frames} frame)")

    # ByteTrack'e gerçek video FPS'ini ver (doğru hız tahmini için kritik)
    models["detector"].tracker = None  # eski tracker'ı sıfırla
    try:
        import supervision as sv
        models["detector"].tracker = sv.ByteTrack(frame_rate=int(fps))
        logger.info(f"ByteTrack yeniden başlatıldı: frame_rate={int(fps)}")
    except Exception as e:
        logger.warning(f"ByteTrack yeniden başlatma başarısız: {e}")

    cache   = VehicleCache(lock_threshold=0.85)
    builder = OutputBuilder()

    # Frame sayacı ve takip istatistikleri
    frame_idx      = 0
    track_frame_counts: dict = {}   # {track_id: frame_count}
    start_time     = time.time()

    logger.info("Pipeline başlıyor...")
    logger.info(f"Frame atlama: her {FRAME_SKIP} frame'de 1 analiz")

    # =================================================================
    # ANA DÖNGÜ
    # =================================================================
    while True:
        # Zaman aşımı kontrolü
        elapsed = time.time() - start_time
        if elapsed > MAX_RUNTIME_SEC:
            logger.warning(f"Zaman sınırına ulaşıldı ({elapsed:.0f}s). Döngü sonlandırılıyor.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        timestamp_sec = frame_idx / fps

        # Frame atlama (performans için)
        if frame_idx % FRAME_SKIP != 0:
            continue

        # ---- 1. Araç tespiti + ByteTrack ----
        tracked_vehicles = models["detector"].detect(frame, timestamp_sec)

        if not tracked_vehicles:
            continue

        # ---- 2. Her araç için sınıflandırma ----
        for tv in tracked_vehicles:
            tid = tv.track_id
            cache.mark_seen(tid, timestamp_sec)
            track_frame_counts[tid] = track_frame_counts.get(tid, 0) + 1

            # 2a. Kasa tipi — kilitli değilse hesapla
            if not cache.is_body_locked(tid) and tv.vehicle_crop is not None:
                body_label, body_conf = models["body"].predict(tv.vehicle_crop)
                cache.update_body(tid, body_label, body_conf)
                logger.debug(f"[{tid}] body={body_label} ({body_conf:.3f})")

            # 2b. Renk — kilitli değilse hesapla
            if not cache.is_color_locked(tid) and tv.vehicle_crop is not None:
                color_label, color_conf = models["color"].predict(tv.vehicle_crop)
                cache.update_color(tid, color_label, color_conf)
                logger.debug(f"[{tid}] color={color_label} ({color_conf:.3f})")

            # 2c. Plaka OCR — kilitli değilse ve plaka crop varsa
            if not cache.is_plate_locked(tid) and tv.plate_crop is not None:
                plate_text, plate_conf = models["ocr"].read_with_retry(
                    tv.plate_crop, retries=2
                )
                if plate_text:
                    cache.update_plate(tid, plate_text, plate_conf)
                    logger.debug(f"[{tid}] plate={plate_text} ({plate_conf:.3f})")

            # 2d. Sürücü eylemi — ön cam crop ile (WindshieldEstimator)
            if models["action"].enabled and tv.vehicle_crop is not None:
                # Kasa tipi: cache'deki son bilinen değeri kullan (varsa)
                body_cache = cache._body.get(tid)
                current_body = "sedan"  # default
                if body_cache and body_cache.votes:
                    current_body = (body_cache.locked_val
                                    if body_cache.locked
                                    else body_cache.votes[-1])

                # Ön cam crop'unu geometrik tahminle al
                ws_crop, ws_confident = models["windshield"].get_windshield_crop(
                    frame        = frame,
                    vehicle_bbox = tv.vehicle_bbox,
                    plate_bbox   = tv.plate_bbox,
                    body_type    = current_body,
                )

                # Düşük güvenli tahminleri logla ama yine de işle
                if not ws_confident:
                    logger.debug(f"[{tid}] Ön cam tahmini düşük güven (body={current_body})")

                action_label, action_conf = models["action"].predict(ws_crop)
                if action_label is not None:
                    cache.add_action(tid, action_label, action_conf, timestamp_sec)
                    logger.debug(f"[{tid}] action={action_label} ({action_conf:.3f}) "
                                 f"ws_conf={ws_confident}")

        # İlerleme logu (her 100 analiz edilen frame'de)
        analyzed = frame_idx // FRAME_SKIP
        if analyzed % 100 == 0 and analyzed > 0:
            elapsed_now = time.time() - start_time
            logger.info(
                f"Frame {frame_idx}/{total_frames} "
                f"({analyzed} analiz) "
                f"[{elapsed_now:.0f}s] "
                f"Araçlar: {len(track_frame_counts)}"
            )

    cap.release()

    # =================================================================
    # SONUÇ ÜRETİMİ
    # =================================================================
    elapsed_total = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Video işleme tamamlandı. ({elapsed_total:.1f}s)")
    logger.info(cache.summary())

    # Birincil araç seçimi
    primary_id = select_primary_vehicle(track_frame_counts)
    if primary_id == -1:
        logger.error("Hiç araç tespit edilemedi! Boş JSON üretiliyor.")
        builder.save(output_path)
        return

    logger.info(f"Birincil araç: track_id={primary_id} "
                f"({track_frame_counts[primary_id]} frame)")

    # OutputBuilder'a yükle ve kaydet
    builder.set_vehicle_from_cache(cache, primary_id)
    success = builder.save(output_path)

    if success:
        logger.info(f"✅ results.json başarıyla oluşturuldu: {output_path}")
    else:
        logger.error(f"❌ results.json oluşturulamadı!")
        sys.exit(1)


# -------------------------------------------------------------------
# Giriş noktası
# -------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("Teknofest AVIS — Akıllı Yol Güvenliği Pipeline")
    logger.info("=" * 60)

    # GPU kontrolü
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        device = "cuda"
    else:
        logger.warning("GPU bulunamadı! CPU modunda çalışılıyor (yavaş).")
        device = "cpu"

    # Giriş dosyası kontrolü
    if not os.path.exists(VIDEO_INPUT_PATH):
        logger.error(f"Giriş videosu bulunamadı: {VIDEO_INPUT_PATH}")
        sys.exit(1)

    # Çıkış dizini oluştur
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

    # Model yükleme
    models = load_models(device=device)

    # Pipeline çalıştır
    run_pipeline(
        video_path  = VIDEO_INPUT_PATH,
        output_path = OUTPUT_JSON_PATH,
        models      = models,
        device      = device,
    )


if __name__ == "__main__":
    main()
