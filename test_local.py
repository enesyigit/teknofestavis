# -*- coding: utf-8 -*-
"""
test_local.py - Yerel Windows Test Scripti
==========================================
TensorRT/Docker GEREKTIRMEYEN testler:
  [OK] Model yukleme (EfficientNet-B0, MobileNetV3 x2)
  [OK] Model inference (dummy goruntilerle)
  [OK] WindshieldEstimator geometri
  [OK] VehicleCache oy birlestirme + guven kilidi
  [OK] OutputBuilder JSON formati + dogrulama
  [SKIP] TRT Engine -> Docker'da test edilecek

Calistirma:
  cd C:\\Users\\enesk\\Desktop\\teknofest_avis
  python test_local.py
"""

import sys
import os
import io
import time
import json
from pathlib import Path

# Windows terminal UTF-8 zorla
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import cv2

# Proje kokunu Python path'ine ekle
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


# -------------------------------------------------
# Cikti yardimcilari (ASCII safe)
# -------------------------------------------------
def ok(msg):     print(f"  [PASS]  {msg}")
def fail(msg):   print(f"  [FAIL]  {msg}")
def skip(msg):   print(f"  [SKIP]  {msg}")
def info(msg):   print(f"  [INFO]  {msg}")
def header(msg): print(f"\n{'='*55}\n {msg}\n{'='*55}")

results = {"pass": 0, "fail": 0, "skip": 0}

def check(name, fn):
    try:
        fn()
        ok(name)
        results["pass"] += 1
    except AssertionError as e:
        fail(f"{name} -> AssertionError: {e}")
        results["fail"] += 1
    except Exception as e:
        fail(f"{name} -> {type(e).__name__}: {e}")
        results["fail"] += 1


# -------------------------------------------------
# Model dosya yollari
# -------------------------------------------------
BODY_MODEL   = PROJECT_ROOT / "models" / "efficientb0arakasa.pth"
COLOR_MODEL  = (PROJECT_ROOT / "models" / "aracrenktespit"
                             / "aracrenktespit" / "arac_renk_mobilenetv3.pth")
COLOR_MODEL2 = (PROJECT_ROOT / "models" / "\u00e7renktespit"
                             / "\u00e7renktespit" / "arac_renk_mobilenetv3.pth")
# Turkce klasor adini dene
_color_candidates = [
    PROJECT_ROOT / "models" / "ara\u00e7renktespit" / "ara\u00e7renktespit" / "arac_renk_mobilenetv3.pth",
    PROJECT_ROOT / "models" / "aracrenktespit"   / "aracrenktespit"   / "arac_renk_mobilenetv3.pth",
]
COLOR_MODEL = next((p for p in _color_candidates if p.exists()), _color_candidates[0])

ACTION_MODEL = PROJECT_ROOT / "mobilenetv3_surucu_hareketleri"
ENGINE_PATH  = PROJECT_ROOT / "models" / "aracplak.engine"

# Test icin sahte BGR goruntu (224x224)
def make_dummy_img(size=(224, 224), color=(128, 128, 128)):
    img = np.full((*size, 3), color, dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (180, 180), (255, 255, 255), -1)
    cv2.circle(img, (112, 112), 50, (0, 0, 200), -1)
    return img

DUMMY_IMG = make_dummy_img()


# =================================================
# BOLUM 1: Kutuphane Import Testleri
# =================================================
header("BOLUM 1: Kutuphane Import Testleri")

def test_torch():
    import torch
    info(f"PyTorch {torch.__version__}")
    cuda = torch.cuda.is_available()
    info(f"CUDA: {'OK - ' + torch.cuda.get_device_name(0) if cuda else 'Yok (CPU modu - yerel test icin normal)'}")

def test_timm():
    import timm
    info(f"timm {timm.__version__}")

def test_cv2():
    info(f"OpenCV {cv2.__version__}")

def test_supervision():
    import supervision as sv
    info(f"supervision {sv.__version__}")

def test_paddle():
    try:
        import paddle
        info(f"PaddlePaddle {paddle.__version__}")
    except ImportError:
        raise AssertionError("Kurulu degil - Docker'da test edilecek (normal)")

check("PyTorch import", test_torch)
check("timm import", test_timm)
check("OpenCV import", test_cv2)
check("supervision (ByteTrack)", test_supervision)
check("PaddlePaddle import", test_paddle)


# =================================================
# BOLUM 2: Model Dosyasi Varlik Kontrolu
# =================================================
header("BOLUM 2: Model Dosyasi Kontrolleri")

def check_body_model():
    assert BODY_MODEL.exists(), f"Bulunamadi: {BODY_MODEL}"
    size_mb = BODY_MODEL.stat().st_size / (1024**2)
    info(f"efficientb0arakasa.pth -> {size_mb:.1f} MB")

def check_color_model():
    assert COLOR_MODEL.exists(), f"Bulunamadi: {COLOR_MODEL}"
    size_mb = COLOR_MODEL.stat().st_size / (1024**2)
    info(f"arac_renk_mobilenetv3.pth -> {size_mb:.1f} MB")

def check_action_model():
    assert ACTION_MODEL.exists(), f"Bulunamadi: {ACTION_MODEL}"
    info(f"mobilenetv3_surucu_hareketleri -> mevcut")

def check_engine():
    assert ENGINE_PATH.exists(), f"Bulunamadi: {ENGINE_PATH}"
    size_mb = ENGINE_PATH.stat().st_size / (1024**2)
    info(f"aracplak.engine -> {size_mb:.1f} MB (TRT - Docker'da test edilecek)")

check("Body model dosyasi",   check_body_model)
check("Color model dosyasi",  check_color_model)
check("Action model dosyasi", check_action_model)
check("TRT engine dosyasi",   check_engine)


# =================================================
# BOLUM 3: BodyClassifier - EfficientNet-B0
# =================================================
header("BOLUM 3: BodyClassifier (EfficientNet-B0)")

body_clf = None

def test_body_load():
    global body_clf
    from src.body_classifier import BodyClassifier
    body_clf = BodyClassifier(str(BODY_MODEL), device="cpu")
    info("EfficientNet-B0 CPU'ya yuklendi")

def test_body_inference():
    assert body_clf is not None, "Once yukleme testi gecmeli"
    label, conf = body_clf.predict(DUMMY_IMG)
    info(f"Tahmin: {label}  confidence={conf:.4f}")
    VALID = {"sedan","suv","hatchback","pickup","minibus","panelvan","kamyon"}
    assert label in VALID, f"Gecersiz sinif: {label}"
    assert 0.0 <= conf <= 1.0, f"Gecersiz confidence: {conf}"

def test_body_topk():
    assert body_clf is not None
    topk = body_clf.predict_topk(DUMMY_IMG, k=3)
    assert len(topk) == 3
    info(f"Top-3: {topk}")

def test_body_speed():
    assert body_clf is not None
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        body_clf.predict(DUMMY_IMG)
        times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    info(f"CPU inference: ort={avg:.1f}ms  min={min(times):.1f}ms  max={max(times):.1f}ms")

check("BodyClassifier yukleme",    test_body_load)
check("BodyClassifier inference",  test_body_inference)
check("BodyClassifier top-k",      test_body_topk)
check("BodyClassifier hiz (CPU)", test_body_speed)


# =================================================
# BOLUM 4: ColorClassifier - MobileNetV3
# =================================================
header("BOLUM 4: ColorClassifier (MobileNetV3)")

color_clf = None

def test_color_load():
    global color_clf
    from src.color_classifier import ColorClassifier
    color_clf = ColorClassifier(str(COLOR_MODEL), device="cpu")
    info("MobileNetV3 (renk) CPU'ya yuklendi")

def test_color_inference():
    assert color_clf is not None
    label, conf = color_clf.predict(DUMMY_IMG)
    info(f"Tahmin: {label}  confidence={conf:.4f}")
    VALID = {"beyaz","siyah","gri","kirmizi","mavi","sari","yesil","turuncu","kahverengi"}
    assert label in VALID, f"Gecersiz renk: {label}"
    assert 0.0 <= conf <= 1.0

def test_color_white():
    assert color_clf is not None
    white = np.full((224, 224, 3), 240, dtype=np.uint8)
    label, conf = color_clf.predict(white)
    info(f"Beyaz goruntu tahmini: {label} ({conf:.4f})")

def test_color_topk():
    assert color_clf is not None
    topk = color_clf.predict_topk(DUMMY_IMG, k=3)
    assert len(topk) == 3
    info(f"Top-3: {topk}")

check("ColorClassifier yukleme",   test_color_load)
check("ColorClassifier inference", test_color_inference)
check("ColorClassifier beyaz img", test_color_white)
check("ColorClassifier top-k",     test_color_topk)


# =================================================
# BOLUM 5: ActionDetector - MobileNetV3
# =================================================
header("BOLUM 5: ActionDetector (MobileNetV3 - Surucu Eylemi)")

action_det = None

def test_action_load():
    global action_det
    from src.action_detector import ActionDetector
    action_det = ActionDetector(str(ACTION_MODEL), device="cpu")
    if not action_det.enabled:
        raise AssertionError("Model yuklenemedi (enabled=False)")
    info("MobileNetV3 (eylem) CPU'ya yuklendi")

def test_action_inference():
    assert action_det is not None and action_det.enabled
    label, conf = action_det.predict(DUMMY_IMG)
    if label is None:
        info("Tespit yok (confidence esigi altinda) - normal")
    else:
        info(f"Tahmin: {label}  confidence={conf:.4f}")
        VALID = {"su_icme","telefonla_konusma","arkaya_bakma","sigara_icme",
                 "emniyet_kemeri_ihlali","etrafa_bakinma","esneme"}
        assert label in VALID, f"Gecersiz eylem: {label}"

def test_action_scores():
    assert action_det is not None and action_det.enabled
    scores = action_det.predict_all_scores(DUMMY_IMG)
    assert len(scores) == 7
    total = sum(scores.values())
    assert abs(total - 1.0) < 0.01, f"Softmax toplami 1 olmali, alindi: {total:.4f}"
    info("Skor dagilimi:")
    for k, v in scores.items():
        info(f"  {k}: {v:.4f}")

check("ActionDetector yukleme",    test_action_load)
check("ActionDetector inference",  test_action_inference)
check("ActionDetector all_scores", test_action_scores)


# =================================================
# BOLUM 6: WindshieldEstimator
# =================================================
header("BOLUM 6: WindshieldEstimator (Geometrik)")

from src.windshield_estimator import WindshieldEstimator, BODY_TO_WS_CLASS

ws_est = WindshieldEstimator(w_ref=300.0)

def test_ws_mapping():
    expected = {
        "sedan":    "sedan_hatchback",
        "hatchback":"sedan_hatchback",
        "suv":      "suv_crossover",
        "pickup":   "kamyonet",
        "panelvan": "kamyonet",
        "minibus":  "otobus_tir",
        "kamyon":   "otobus_tir",
    }
    for body, ws in expected.items():
        mapped = BODY_TO_WS_CLASS.get(body)
        assert mapped == ws, f"{body} -> beklenen {ws}, alindi {mapped}"
    info("Tum kasa tipi esleme dogru")

def test_ws_sedan():
    vehicle_box = (100, 150, 350, 400)
    result = ws_est.estimate_xywh(vehicle_box, "sedan_hatchback", None)
    xw, yw, ww, hw = result["windshield_box"]
    info(f"Sedan on cam bbox: x={xw} y={yw} w={ww} h={hw}")
    assert ww > 0 and hw > 0
    assert yw >= 150

def test_ws_with_plate():
    vehicle_box = (100, 150, 350, 400)
    plate_box   = (250, 480, 50, 20)
    result = ws_est.estimate_xywh(vehicle_box, "suv_crossover", plate_box)
    info(f"Plaka + yaw: error={result['error_margin_px']:.1f}px  confident={result['is_confident']}")

def test_ws_frame_crop():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 200), (700, 600), (100, 100, 200), -1)
    crop, confident = ws_est.get_windshield_crop(
        frame=frame,
        vehicle_bbox=(200, 200, 700, 600),
        plate_bbox=(350, 520, 450, 550),
        body_type="sedan"
    )
    assert crop is not None and crop.size > 0
    assert crop.shape[2] == 3
    info(f"On cam crop: {crop.shape[1]}x{crop.shape[0]}px  confident={confident}")

def test_ws_all_types():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    for body in ["sedan","suv","hatchback","pickup","panelvan","minibus","kamyon"]:
        crop, _ = ws_est.get_windshield_crop(frame, (100,100,600,500), None, body)
        assert crop is not None and crop.size > 0
    info("Tum 7 kasa tipi icin crop uretildi")

check("WS kasa-sinif esleme",   test_ws_mapping)
check("WS sedan tahmini",       test_ws_sedan)
check("WS plaka + yaw",         test_ws_with_plate)
check("WS frame'den crop",      test_ws_frame_crop)
check("WS tum kasa tipleri",    test_ws_all_types)


# =================================================
# BOLUM 7: VehicleCache
# =================================================
header("BOLUM 7: VehicleCache (Oy Birlestirme + Guven Kilidi)")

from src.vehicle_cache import VehicleCache

def test_majority():
    cache = VehicleCache()
    for _ in range(5): cache.update_body(1, "sedan",  0.72)
    for _ in range(3): cache.update_body(1, "suv",    0.65)
    final = cache.get_final(1)
    assert final.tip == "sedan", f"Beklenen sedan, alindi {final.tip}"
    info(f"Majority vote: sedan (5 oy) > suv (3 oy) -> {final.tip}")

def test_lock():
    cache = VehicleCache(lock_threshold=0.85)
    cache.update_body(2, "kamyon", 0.91)   # kilitlenmeli
    cache.update_body(2, "sedan",  0.75)   # islenmemeli
    cache.update_body(2, "sedan",  0.80)   # islenmemeli
    final = cache.get_final(2)
    assert final.tip == "kamyon", f"Kilit calismadi: beklenen kamyon, alindi {final.tip}"
    info("Guven kilidi (0.91 >= 0.85): kamyon kilitlendi, sonraki oylar reddedildi")

def test_plate():
    cache = VehicleCache()
    cache.update_plate(3, "34ABC123", 0.82)
    cache.update_plate(3, "34ABC123", 0.91)   # kilitlenir
    cache.update_plate(3, "06XYZ999", 0.60)   # kilitli, islenmez
    final = cache.get_final(3)
    assert final.plaka == "34ABC123", f"Beklenen 34ABC123, alindi {final.plaka}"
    info("Plaka kilidi dogru calistiyor")

def test_action_rec():
    cache = VehicleCache()
    cache.add_action(4, "telefonla_konusma", 0.93, 1.5)
    cache.add_action(4, "telefonla_konusma", 0.88, 2.0)
    cache.add_action(4, "sigara_icme",       0.75, 5.0)
    actions = cache.get_actions(4)
    assert len(actions) == 3
    info(f"{len(actions)} eylem kaydedildi")

def test_conf_score():
    cache = VehicleCache()
    cache.update_body(5,  "sedan",   0.90)
    cache.update_color(5, "kirmizi", 0.85)
    cache.update_plate(5, "06TK123", 0.78)
    final = cache.get_final(5)
    expected = round((0.90 + 0.85 + 0.78) / 3, 4)
    info(f"Bilesik conf: {final.confidence_score:.4f} (beklenen ~{expected})")
    assert abs(final.confidence_score - expected) < 0.001

check("Cache majority vote",     test_majority)
check("Cache guven kilidi",      test_lock)
check("Cache plaka toplama",     test_plate)
check("Cache eylem kaydi",       test_action_rec)
check("Cache conf_score hesabi", test_conf_score)


# =================================================
# BOLUM 8: OutputBuilder - JSON Format
# =================================================
header("BOLUM 8: OutputBuilder (Yarisma JSON Formati)")

from src.output_builder import OutputBuilder

def test_format():
    cache = VehicleCache()
    cache.update_body(10,  "sedan",    0.91)
    cache.update_color(10, "kirmizi",  0.87)
    cache.update_plate(10, "34ABC123", 0.88)
    cache.add_action(10, "telefonla_konusma", 0.93, 1.5)
    cache.add_action(10, "sigara_icme",       0.78, 4.0)

    builder = OutputBuilder()
    builder.set_vehicle_from_cache(cache, 10)
    data = builder.build()

    ab = data["arac_bilgisi"]
    assert ab["tip"]   == "sedan"
    assert ab["renk"]  == "kirmizi"
    assert ab["plaka"] == "34ABC123"
    assert 0.0 <= ab["confidence_score"] <= 1.0
    info(f"arac_bilgisi: {json.dumps(ab, ensure_ascii=False)}")
    info(f"tespit sayisi: {len(data['tespitler'])}")
    for t in data["tespitler"]:
        info(f"  [{t['zaman_saniye']}s] {t['kategori']} -> {t['etiket']} ({t['confidence_score']})")

def test_dedup():
    cache = VehicleCache()
    cache.update_body(11, "suv", 0.80)
    cache.update_color(11, "siyah", 0.75)
    for ts in [1.0, 1.3, 1.6, 1.9, 2.2]:
        cache.add_action(11, "esneme", 0.82, ts)
    cache.add_action(11, "esneme", 0.85, 4.5)

    builder = OutputBuilder()
    builder.set_vehicle_from_cache(cache, 11)
    data = builder.build()
    count = sum(1 for t in data["tespitler"] if t["etiket"] == "esneme")
    info(f"6 giris -> deduplicate sonrasi: {count} tespit")
    assert count <= 2

def test_save_load():
    import tempfile
    cache = VehicleCache()
    cache.update_body(12,  "hatchback", 0.88)
    cache.update_color(12, "mavi",      0.91)
    cache.update_plate(12, "35BB456",   0.85)
    cache.add_action(12, "arkaya_bakma", 0.89, 7.2)

    builder = OutputBuilder()
    builder.set_vehicle_from_cache(cache, 12)

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w",
                                     delete=False, encoding="utf-8") as f:
        tmp_path = f.name
    try:
        success = builder.save(tmp_path)
        assert success
        with open(tmp_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["arac_bilgisi"]["tip"] == "hatchback"
        assert loaded["arac_bilgisi"]["plaka"] == "35BB456"
        info(f"JSON kayit/yukleme basarili")
        info(json.dumps(loaded, ensure_ascii=False, indent=2))
    finally:
        os.unlink(tmp_path)

def test_invalid():
    builder = OutputBuilder()
    builder.add_detection(1.0, "yanlis_kategori", "bilmiyorum", 0.99)
    data = builder.build()
    assert len(data.get("tespitler", [])) == 0
    info("Gecersiz kategori reddedildi")

check("OutputBuilder format",     test_format)
check("OutputBuilder dedup",      test_dedup)
check("OutputBuilder kayit/yukle",test_save_load)
check("OutputBuilder gecersiz",   test_invalid)


# =================================================
# BOLUM 9: Entegrasyon - Modeller Birlikte
# =================================================
header("BOLUM 9: Entegrasyon (Modeller + WS + Cache + JSON)")

def test_full_chain():
    if body_clf is None or color_clf is None or action_det is None:
        raise AssertionError("Onceki model yukleme testleri gecmeli")

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 200), (700, 550), (150, 100, 80), -1)

    vehicle_bbox = (200, 200, 700, 550)
    plate_bbox   = (330, 490, 430, 520)
    vehicle_crop = frame[200:550, 200:700].copy()

    body_label,  body_conf  = body_clf.predict(vehicle_crop)
    color_label, color_conf = color_clf.predict(vehicle_crop)

    ws_crop, ws_confident = ws_est.get_windshield_crop(
        frame=frame, vehicle_bbox=vehicle_bbox,
        plate_bbox=plate_bbox, body_type=body_label
    )

    action_label, action_conf = action_det.predict(ws_crop)

    info(f"Body  : {body_label} ({body_conf:.3f})")
    info(f"Color : {color_label} ({color_conf:.3f})")
    info(f"WS    : {ws_crop.shape[1]}x{ws_crop.shape[0]}px  confident={ws_confident}")
    info(f"Action: {action_label} ({action_conf:.3f})")

    cache = VehicleCache()
    tid = 42
    cache.update_body(tid,  body_label,  body_conf)
    cache.update_color(tid, color_label, color_conf)
    if action_label:
        cache.add_action(tid, action_label, action_conf, 1.5)

    builder = OutputBuilder()
    builder.set_vehicle_from_cache(cache, tid)
    data = builder.build()

    assert "arac_bilgisi" in data
    assert data["arac_bilgisi"]["tip"] in \
        {"sedan","suv","hatchback","pickup","minibus","panelvan","kamyon"}

    info("Nihai JSON:")
    info(json.dumps(data, ensure_ascii=False, indent=2))

check("Tam zincir entegrasyon", test_full_chain)


# =================================================
# SONUC
# =================================================
header("TEST SONUCLARI")
total = results["pass"] + results["fail"] + results["skip"]
print(f"\n  Toplam   : {total}")
print(f"  Gecti    : {results['pass']}")
print(f"  Basarisiz: {results['fail']}")
print(f"  Atlandi  : {results['skip']}")

if results["fail"] == 0:
    print("\n  *** Tum testler gecti! ***")
else:
    print(f"\n  *** {results['fail']} test basarisiz. ***")

print(f"""
{'='*55}
 DOCKER / TRT TESTLERI (Sonraki Asama)
{'='*55}
 TRT Engine'i test etmek icin:

   docker build -t teknofest-avis .

   docker run --gpus all ^
     -v "C:/test_video:/app/data/input" ^
     -v "C:/test_output:/app/data/output" ^
     teknofest-avis

 Cikti: C:\\test_output\\results.json
{'='*55}
""")
