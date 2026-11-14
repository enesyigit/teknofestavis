# Teknofest AVIS — 5G ve Yapay Zeka ile Akıllı Yol Güvenliği

Trafik videosundan **araç tipi, plaka, renk ve sürücü eylemlerini** tek geçişte çıkarıp yarışma formatında `results.json` üreten video analiz pipeline'ı.

> **EN:** Single-pass video analytics pipeline that extracts vehicle type, license plate, color and driver actions from traffic footage, built for the Teknofest "Smart Road Safety with 5G and AI" competition. Python · PyTorch · TensorRT · PaddleOCR · OpenCV.

## Ne yapıyor

| Alan | Çıktı |
| --- | --- |
| Araç tipi | sedan, suv, hatchback, pickup, minibus, panelvan, kamyon (7 sınıf) |
| Plaka | Türk plaka formatı, boşluksuz (`34ABC123`) |
| Renk | beyaz, siyah, gri, kırmızı, mavi, sarı, yeşil, turuncu, kahverengi (9 sınıf) |
| Sürücü eylemi | su içme, telefonla konuşma, arkaya bakma, sigara içme, emniyet kemeri ihlali, etrafa bakınma, slalom — frame bazlı, zaman damgalı |

## Pipeline

```
video.mp4
  │  OpenCV kare okuma — her 3. kare analiz edilir (FRAME_SKIP=3)
  ▼
VehicleDetector ── aracplak.engine (TensorRT FP16) + ByteTrack
  │  araç ve plaka kutuları, araç başına track_id
  ▼
WindshieldEstimator ── araç + plaka bbox'ından ön cam (sürücü) bölgesi
  ▼
┌─ BodyClassifier   EfficientNet-B0     7 kasa tipi
├─ ColorClassifier  MobileNetV3         9 renk
├─ ActionDetector   MobileNetV3         7 sürücü eylemi + slalom
└─ OCRReader        PaddleOCR PP-OCRv4  plaka metni
  ▼
VehicleCache ── oy birleştirme + 0.85 güven kilidi
  ▼
OutputBuilder ── şema doğrulama → results.json
```

## Modüller

| Dosya | Görev | Satır |
| --- | --- | ---: |
| `main.py` | Pipeline orkestrasyonu: video okuma, model yükleme, birincil araç seçimi, süre bütçesi (570 s), JSON üretimi | 313 |
| `src/trt_engine.py` | TensorRT FP16 engine sarmalayıcı: letterbox, NMS, çıktı decode, ONNX'ten engine derleme | 308 |
| `src/action_detector.py` | Sürücü eylemi sınıflandırıcı + `SlalomDetector` | 287 |
| `src/ocr_reader.py` | PaddleOCR ile plaka okuma, Türk plaka formatı doğrulama, retry | 238 |
| `src/vehicle_cache.py` | Track başına oy birleştirme (`majority_vote`, `weighted_majority_vote`), 0.85 güven kilidi | 235 |
| `src/windshield_estimator.py` | Araç/plaka geometrisinden ön cam bölgesi tahmini (fallback'li) | 233 |
| `src/output_builder.py` | Yarışma şemasında `results.json`, etiket doğrulama, tekrar eleme | 223 |
| `src/vehicle_detector.py` | Araç + plaka tespiti, ByteTrack ile ID atama, plaka–araç eşleştirme | 213 |
| `src/body_classifier.py` | Kasa tipi sınıflandırıcı (EfficientNet-B0) | 165 |
| `src/color_classifier.py` | Renk sınıflandırıcı (MobileNetV3) | 158 |

`src/` toplamı ≈ 2.060 satır; `test_local.py` 463, `main.py` 313 satır.

## Çıktı formatı

```json
{
  "arac_bilgisi": {
    "tip": "sedan",
    "plaka": "34ABC123",
    "renk": "beyaz",
    "confidence_score": 0.91
  },
  "tespitler": [
    {
      "zaman_saniye": 1.5,
      "kategori": "sofor_eylemi",
      "etiket": "telefonla_konusma",
      "confidence_score": 0.92
    }
  ]
}
```

Kurallar: tüm anahtarlar küçük harf ve Türkçe karakter içermez, plaka boşluksuz büyük harf, `confidence_score` 4, `zaman_saniye` 2 ondalık.

## Kurulum

### Docker (yarışma ortamı)

```bash
docker build -t teknofest-avis .
docker run --gpus all \
  -v /yol/girdi:/app/data/input \
  -v /yol/cikti:/app/data/output \
  teknofest-avis
```

Temel imaj `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`, hedef GPU Tesla T4 (TensorRT 8.6). Girdi/çıktı yolları yarışma dokümanında sabit: `/app/data/input/video.mp4` → `/app/data/output/results.json`.

### Yerel (Windows / CPU, Docker'sız)

```bash
pip install timm supervision opencv-python Pillow numpy scipy shapely
python test_local.py
```

TensorRT ve Paddle GPU yolları yalnızca Docker imajında çalışır; `test_local.py` bu kısımları atlar.

### Model ağırlıkları

Ağırlıklar Git LFS ile geliyor:

```bash
git lfs install
git lfs pull
```

| Dosya | Model |
| --- | --- |
| `models/aracplak.engine` | Araç + plaka dedektörü (TensorRT FP16) |
| `models/efficientb0arakasa.pth` | Kasa tipi, EfficientNet-B0 |
| `models/araçrenktespit/.../arac_renk_mobilenetv3.pth` | Renk, MobileNetV3 |
| `models/paddle_models/model.pdparams`, `model.pdopt` | PaddleOCR tanıma modeli |
| `mobilenetv3_surucu_hareketleri/` | Sürücü eylemi, MobileNetV3 (torch.save zip formatı) |

## Tasarım kararları

- **Kare atlama (`FRAME_SKIP=3`)** — 10 dakikalık video bütçesinde doğruluk/hız dengesi; aynı araç onlarca karede göründüğü için kayıp minimal.
- **0.85 güven kilidi** — `VehicleCache` bir alanı 0.85 güvenle kilitler ve o alan için model bir daha çalıştırılmaz; tekrar analiz maliyeti düşer.
- **Oy birleştirme** — kilitlenene kadar gelen tahminler ağırlıklı çoğunluk oyu ile birleştirilir; tek kareli yanlış sınıflandırma elenir.
- **ByteTrack** — tespit kutularını track_id'ye bağlar; araç bilgisi ve eylemler aynı araç altında toplanır.
- **Geometrik ön cam tahmini** — sürücü bölgesi için ayrı bir model yerine araç/plaka bbox oranları kullanılır; düşük güvende fallback bölge döner, asla `None` dönmez.
- **570 saniyelik süre bütçesi** — 10 dakikalık limit için güvenli marj; aşılırsa pipeline elindeki sonuçla kapanır.
- **Parametrik model yolları** — aynı kod hem Docker'da hem Kaggle T4 testinde (`kaggle_test.py`) çalışır.

## Test

| Komut | Ne yapar |
| --- | --- |
| `python test_local.py` | Docker/TensorRT gerektirmeyen testler: model yükleme, dummy inference, ön cam geometrisi, oy birleştirme + kilit, JSON şema doğrulama |
| `python kaggle_test.py` | Kaggle T4 üzerinde uçtan uca deneme (submission'a girmez) |

## Ekip ve katkı dağılımı

> Git geçmişine göre: 17 Temmuz 2026 tarihli `22f7981` "Initial commit" yalnızca `README.md` (tek satır) ve `.gitattributes` içeriyor. Uygulama kodunun tamamı, `979702e` "Add teknofest_avis project files" commit'i ile (362 dosya, +3.651 satır) eklenmiştir.

| Kişi | Katkı |
| --- | --- |
| **Enes Ketenci** — [@ketenci7enes](https://github.com/ketenci7enes) | Mimari ve uygulama kodunun tamamı: `main.py`, `src/` (9 modül, ~2.060 satır), `Dockerfile`, `requirements.txt`, `test_local.py`, `kaggle_test.py`, model entegrasyonu, yarışma çıktı şeması ve doğrulaması |
| **Enes Yiğit** — [@enesyigit](https://github.com/enesyigit) | Repo kurulumu, `.gitattributes`, Git LFS altyapısı ve büyük model dosyalarının LFS ile yüklenmesi, "remove hardcoded local path" düzeltmesi |

> Not: 30 Temmuz 2026'da repo, Git LFS'e geçiş sırasında yeniden başlatıldığı için önceki commit geçmişi bu repoda görünmüyor; kod içeriği aynıdır.

İletişim: [@ketenci7enes](https://github.com/ketenci7enes)
