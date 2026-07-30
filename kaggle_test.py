# ============================================================
# kaggle_test.py — Kaggle T4 Test Script
# ============================================================
# Bu dosya SADECE Kaggle'da test icin kullanilir.
# YARISMAYA GÖNDERILMEZ.
#
# Kaggle'da nasil kullanilir:
#   1. Tum proje dosyalarini Kaggle dataset olarak yukle
#   2. Bu notebook'u calistir
#   3. /kaggle/working/results.json ciktisini kontrol et
#
# Yarısma Docker'i icin: Dockerfile kullan (degistirme!)
# ============================================================

import os, sys

# ── Kaggle yol haritasi ─────────────────────────────────────
KAGGLE_INPUT   = "/kaggle/input"
KAGGLE_WORKING = "/kaggle/working"

# Dataset adini buraya yaz (Kaggle'a yukledikten sonra)
DATASET_SLUG   = "teknofest-avis"   # kaggle dataset slug

# Kaggle icindeki model yollari
MODEL_BASE     = f"{KAGGLE_INPUT}/{DATASET_SLUG}"
VIDEO_PATH     = f"{KAGGLE_INPUT}/{DATASET_SLUG}/video.mp4"
OUTPUT_PATH    = f"{KAGGLE_WORKING}/results.json"

# Proje src klasoru Kaggle working'e kopyalanmis olmali
PROJECT_ROOT   = KAGGLE_WORKING
sys.path.insert(0, PROJECT_ROOT)

# ── Ortam degiskenlerini ayarla (main.py bunları okur) ──────
os.environ["VIDEO_PATH"]   = VIDEO_PATH
os.environ["OUTPUT_PATH"]  = OUTPUT_PATH
os.environ["FRAME_SKIP"]   = "3"
os.environ["VEHICLE_CONF"] = "0.50"

print("="*55)
print(" Teknofest AVIS — Kaggle T4 Test")
print("="*55)
print(f"Video  : {VIDEO_PATH}")
print(f"Output : {OUTPUT_PATH}")
print(f"Models : {MODEL_BASE}")
print()

# ── GPU kontrol ─────────────────────────────────────────────
import torch
if torch.cuda.is_available():
    print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
    print(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("[WARN] GPU yok! CPU ile cok yavas olacak.")

# ── Kaggle model yollarini override et ──────────────────────
# Bu blok main.py'nin MODEL_DIR sabitini Kaggle yoluna yonlendirir
import main as pipeline_main
from pathlib import Path

pipeline_main.VIDEO_INPUT_PATH = VIDEO_PATH
pipeline_main.OUTPUT_JSON_PATH = OUTPUT_PATH
pipeline_main.MODEL_DIR        = Path(MODEL_BASE) / "models"
pipeline_main.ENGINE_PATH      = Path(MODEL_BASE) / "models" / "aracplak.engine"
pipeline_main.BODY_MODEL_PATH  = Path(MODEL_BASE) / "models" / "efficientb0arakasa.pth"
pipeline_main.COLOR_MODEL_PATH = (
    Path(MODEL_BASE) / "models" /
    "araçrenktespit" / "araçrenktespit" / "arac_renk_mobilenetv3.pth"
)
pipeline_main.ACTION_MODEL_PATH = Path(MODEL_BASE) / "mobilenetv3_surucu_hareketleri"

print()
print("Model yollari:")
for name, path in [
    ("Engine",  pipeline_main.ENGINE_PATH),
    ("Body",    pipeline_main.BODY_MODEL_PATH),
    ("Color",   pipeline_main.COLOR_MODEL_PATH),
    ("Action",  pipeline_main.ACTION_MODEL_PATH),
]:
    status = "[OK]" if path.exists() else "[MISSING]"
    print(f"  {status} {name}: {path}")

print()

# ── Paddle OCR rec model yolu ────────────────────────────────
PADDLE_REC = Path(MODEL_BASE) / "models" / "paddle_models"
print(f"PaddleOCR rec: {PADDLE_REC}")
print(f"  model.pdparams: {'[OK]' if (PADDLE_REC/'model.pdparams').exists() else '[MISSING]'}")

# ── Pipeline calistir ────────────────────────────────────────
if not Path(VIDEO_PATH).exists():
    print(f"\n[HATA] Video bulunamadi: {VIDEO_PATH}")
    print("Kaggle dataset'e video.mp4 ekleyin.")
    sys.exit(1)

print("\nPipeline basliyor...")
pipeline_main.main()

print(f"\n[TAMAM] Sonuc: {OUTPUT_PATH}")
import json
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        result = json.load(f)
    print(json.dumps(result, ensure_ascii=False, indent=2))
