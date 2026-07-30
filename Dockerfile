# ============================================================
# Dockerfile — Teknofest AVIS YARISMA SUBMISYONU
# ============================================================
# Base  : nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
# GPU   : NVIDIA Tesla T4 (CUDA 12.1, TensorRT 8.6)
# Limit : 10 dakika
#
# YARISMA GEREKSINIMLERI (degistirme!):
#   Giris : /app/data/input/video.mp4
#   Cikis : /app/data/output/results.json
#   Ortam : Degerlendirme ortami tespiti YASAK
# ============================================================

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

LABEL maintainer="Teknofest AVIS Team"
LABEL description="5G ve Yapay Zeka ile Akilli Yol Guvenligi - FTR Submisyon"

# ── Sabit ortam (yarısma gereksinimi) ───────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TZ=Europe/Istanbul
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Yarısma sabit yolları — DEGISTIRME
ENV VIDEO_PATH=/app/data/input/video.mp4
ENV OUTPUT_PATH=/app/data/output/results.json
ENV FRAME_SKIP=3
ENV VEHICLE_CONF=0.50

# ── Sistem bagımlıları ──────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-setuptools \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-dev \
    wget \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN pip install --upgrade pip setuptools wheel

# ── PyTorch CUDA 12.1 ───────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.1.2+cu121 \
    torchvision==0.16.2+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# ── PaddlePaddle GPU CUDA 12.1 ──────────────────────────────
RUN pip install --no-cache-dir \
    paddlepaddle-gpu==2.6.0.post120 \
    -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# ── PaddleOCR ───────────────────────────────────────────────
RUN pip install --no-cache-dir paddleocr==2.7.3

# ── Diger kutuphaneler ───────────────────────────────────────
RUN pip install --no-cache-dir \
    timm==0.9.12 \
    opencv-python-headless==4.9.0.80 \
    Pillow==10.2.0 \
    "numpy>=1.24,<2.0" \
    supervision==0.19.0 \
    lap==0.4.0 \
    scipy==1.11.4 \
    shapely==2.0.3 \
    filterpy==1.4.5 \
    python-dotenv==1.0.0

# ── PaddleOCR det+cls modellerini on-indir ──────────────────
# Rec icin egitilmis modelimiz var (model.pdparams).
# Det ve cls icin PP-OCRv4 build sirasinda indirilir.
RUN python3 -c "\
from paddleocr import PaddleOCR; \
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False); \
print('PaddleOCR det+cls hazir.') \
" || echo "Offline build — model indirme atlandı."

# ── Uygulama kopyala ────────────────────────────────────────
WORKDIR /app
COPY . /app/

# ── Veri dizinleri (yarışma sabit yolları) ──────────────────
RUN mkdir -p /app/data/input /app/data/output

# ── Model yol haritası (bilgi icin) ─────────────────────────
# /app/models/
#   aracplak.engine                          TRT FP16 arac+plaka detector (51MB)
#   efficientb0arakasa.pth                   EfficientNet-B0 kasa tipi    (16MB)
#   araçrenktespit/araçrenktespit/
#     arac_renk_mobilenetv3.pth              MobileNetV3-L renk           (6MB)
#   paddle_models/
#     model.pdparams                         PaddleOCR rec agirlikları    (65MB)
# /app/mobilenetv3_surucu_hareketleri/       MobileNetV3-L surucu eylemi (zip-dir)

# pdopt inference'da kullanilmaz, boyutu azalt
RUN rm -f /app/models/paddle_models/model.pdopt || true

# Derlenmiş dosyaları temizle
RUN find /app -name "*.pyc" -delete \
    && find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# ── Saglik kontrolü ─────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python3 -c "import torch; import cv2; print('OK')" || exit 1

# ── YARIŞMA GIRIS NOKTASI ────────────────────────────────────
ENTRYPOINT ["python3", "main.py"]
