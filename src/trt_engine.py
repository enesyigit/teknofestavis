"""
trt_engine.py — TensorRT FP16 Engine Wrapper
=============================================
Kullanım: aracplak.engine (araç + plaka tespiti, 2 sınıf)

Gerekli kütüphaneler (Docker'da mevcut olmalı):
  - tensorrt>=8.6.0        (TRT runtime)
  - pycuda>=2022.2         (CUDA memory yönetimi)
  - nvidia-cuda-runtime    (CUDA 12.1)
  - numpy, opencv-python

ÖNEMLİ — Engine taşınabilirlik:
  TensorRT engine'ler GPU mimarisine özgüdür.
  Bu engine Tesla T4 (Ampere/Turing) dışında bir GPU'da oluşturulduysa
  Docker build aşamasında ONNX'ten yeniden derlenmesi gerekebilir.
  Yeniden derleme için: build_engine_from_onnx() fonksiyonunu kullanın.
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# TensorRT import — sadece CUDA ortamında (Docker / Linux + GPU)
# -------------------------------------------------------------------
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # CUDA context'i otomatik başlatır
    TRT_AVAILABLE = True
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
except ImportError:
    TRT_AVAILABLE = False
    logger.warning("TensorRT veya PyCUDA bulunamadı. CPU fallback aktif.")


# -------------------------------------------------------------------
# Yardımcı: Letterbox (aspect-ratio korumalı resize + padding)
# -------------------------------------------------------------------
def letterbox(img: np.ndarray,
              new_shape: tuple = (640, 640),
              color: tuple = (114, 114, 114)) -> tuple:
    """
    Görüntüyü hedef boyuta letterbox ile yeniden boyutlandırır.
    Orijinal en-boy oranını korur, kenarları gri ile doldurur.

    Returns:
        img_padded  : Yeniden boyutlandırılmış ve doldurulmuş görüntü
        ratio       : Ölçekleme oranı (orijinal → yeni)
        (dw, dh)    : Sağ/sol ve üst/alt padding miktarları (piksel)
    """
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top    = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left   = int(round(dw - 0.1))
    right  = int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


# -------------------------------------------------------------------
# NMS — Non-Maximum Suppression (numpy tabanlı)
# -------------------------------------------------------------------
def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list:
    """
    Basit NMS implementasyonu.
    boxes: [N, 4] — (x1, y1, x2, y2)
    scores: [N]
    """
    if len(boxes) == 0:
        return []

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = ((boxes[order[1:], 2] - boxes[order[1:], 0]) *
                  (boxes[order[1:], 3] - boxes[order[1:], 1]))
        iou = inter / (area_i + area_j - inter + 1e-6)
        order = order[1:][iou <= iou_threshold]

    return keep


# -------------------------------------------------------------------
# Bbox ölçekleme: letterbox koordinatlarını orijinal frame'e çevir
# -------------------------------------------------------------------
def scale_boxes(boxes: np.ndarray,
                ratio: float,
                pad: tuple,
                orig_shape: tuple) -> np.ndarray:
    """
    Letterbox çıktı koordinatlarını orijinal görüntü piksellerine çevirir.
    orig_shape: (H, W)
    """
    dw, dh = pad
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes /= ratio

    # Görüntü sınırlarına kırp
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_shape[0])
    return boxes


# -------------------------------------------------------------------
# TRTEngine — Ana sınıf
# -------------------------------------------------------------------
class TRTEngine:
    """
    TensorRT FP16 inference engine wrapper.

    aracplak.engine için beklenen model çıktısı:
      Format A (YOLOv8/v9 raw, NMS'siz):
        output shape: [1, num_classes+4, 8400]
        İçerik: [cx, cy, w, h, cls0_score, cls1_score, ...]

      Format B (NMS uygulanmış):
        output shape: [1, N, 6]
        İçerik: [x1, y1, x2, y2, confidence, class_id]

    Kod her iki formatı da otomatik tanır.
    """

    # aracplak.engine sınıf isimleri
    CLASS_NAMES = {0: "arac", 1: "plak"}
    INPUT_SIZE  = (640, 640)   # Model giriş boyutu (H, W)
    NUM_CLASSES = 2            # arac, plak

    def __init__(self, engine_path: str,
                 conf_threshold: float = 0.50,
                 iou_threshold: float  = 0.45):
        assert TRT_AVAILABLE, "TensorRT kurulu değil! Docker ortamında çalıştırın."
        self.engine_path    = engine_path
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self._load_engine()

    # ------------------------------------------------------------------
    def _load_engine(self):
        logger.info(f"TRT Engine yükleniyor: {self.engine_path}")
        with open(self.engine_path, "rb") as f:
            serialized = f.read()

        runtime = trt.Runtime(TRT_LOGGER)
        self.engine  = runtime.deserialize_cuda_engine(serialized)
        self.context = self.engine.create_execution_context()

        self._inputs  = []
        self._outputs = []
        self._bindings = []
        self._stream   = cuda.Stream()

        for i in range(self.engine.num_bindings):
            name  = self.engine.get_binding_name(i)
            shape = tuple(self.engine.get_binding_shape(i))
            size  = int(np.prod(np.abs(shape)))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))

            host_mem   = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self._bindings.append(int(device_mem))

            entry = {"name": name, "shape": shape,
                     "host": host_mem, "device": device_mem}

            if self.engine.binding_is_input(i):
                self._inputs.append(entry)
                logger.info(f"  [INPUT]  {name} shape={shape} dtype={dtype}")
            else:
                self._outputs.append(entry)
                logger.info(f"  [OUTPUT] {name} shape={shape} dtype={dtype}")

        logger.info("TRT Engine başarıyla yüklendi.")

    # ------------------------------------------------------------------
    def _preprocess(self, img_bgr: np.ndarray):
        """BGR görüntüyü model girişi için hazırlar (FP16)."""
        img_lb, ratio, pad = letterbox(img_bgr, self.INPUT_SIZE)
        img_rgb = img_lb[:, :, ::-1]                      # BGR → RGB
        img_chw = img_rgb.transpose(2, 0, 1)              # HWC → CHW
        img_fp16 = np.ascontiguousarray(img_chw, dtype=np.float16) / 255.0
        return img_fp16, ratio, pad

    # ------------------------------------------------------------------
    def _run_inference(self, img_fp16: np.ndarray) -> list:
        """Ham TRT inference çalıştırır, output tensor listesi döner."""
        np.copyto(self._inputs[0]["host"], img_fp16.ravel())

        for inp in self._inputs:
            cuda.memcpy_htod_async(inp["device"], inp["host"], self._stream)

        self.context.execute_async_v2(
            bindings=self._bindings,
            stream_handle=self._stream.handle
        )

        for out in self._outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self._stream)

        self._stream.synchronize()

        return [out["host"].reshape(out["shape"]).copy()
                for out in self._outputs]

    # ------------------------------------------------------------------
    def _decode_output(self, raw_outputs: list,
                       ratio: float, pad: tuple,
                       orig_shape: tuple) -> dict:
        """
        TRT çıktısını decode ederek araç ve plaka tespitlerini ayırır.

        Returns:
            {
                "arac": [(x1,y1,x2,y2,conf), ...],
                "plak": [(x1,y1,x2,y2,conf), ...]
            }
        """
        out = raw_outputs[0].astype(np.float32)

        # ---- Format Tespiti ----
        if out.ndim == 3 and out.shape[1] == (self.NUM_CLASSES + 4):
            # Format A: [1, NC+4, 8400] — YOLOv8/v9 raw
            out = out[0].T  # [8400, NC+4]
            cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
            cls_scores   = out[:, 4:]                          # [8400, NC]
            class_ids    = cls_scores.argmax(axis=1)
            confidences  = cls_scores.max(axis=1)

            # xywh → xyxy
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes_raw = np.stack([x1, y1, x2, y2], axis=1)

        elif out.ndim == 3 and out.shape[-1] == 6:
            # Format B: [1, N, 6] — NMS uygulanmış
            out         = out[0]                               # [N, 6]
            boxes_raw   = out[:, :4]
            confidences = out[:, 4]
            class_ids   = out[:, 5].astype(int)

        else:
            logger.warning(f"Bilinmeyen TRT output shape: {out.shape}")
            return {"arac": [], "plak": []}

        # ---- Confidence filtresi ----
        mask        = confidences >= self.conf_threshold
        boxes_raw   = boxes_raw[mask]
        confidences = confidences[mask]
        class_ids   = class_ids[mask]

        if len(boxes_raw) == 0:
            return {"arac": [], "plak": []}

        # ---- Letterbox koordinatlarını orijinale çevir ----
        boxes_scaled = scale_boxes(boxes_raw.copy(), ratio, pad, orig_shape)

        # ---- NMS (sınıf başına) ----
        result = {"arac": [], "plak": []}
        for cls_id, cls_name in self.CLASS_NAMES.items():
            cls_mask = class_ids == cls_id
            if not cls_mask.any():
                continue
            b = boxes_scaled[cls_mask]
            c = confidences[cls_mask]
            keep = nms(b, c, self.iou_threshold)
            for idx in keep:
                x1, y1, x2, y2 = b[idx].astype(int)
                result[cls_name].append((x1, y1, x2, y2, float(c[idx])))

        return result

    # ------------------------------------------------------------------
    def detect(self, img_bgr: np.ndarray) -> dict:
        """
        Ana inference fonksiyonu.

        Args:
            img_bgr: BGR numpy array (frame)

        Returns:
            {
                "arac": [(x1,y1,x2,y2,conf), ...],
                "plak": [(x1,y1,x2,y2,conf), ...]
            }
        """
        orig_shape = img_bgr.shape[:2]  # (H, W)
        img_fp16, ratio, pad = self._preprocess(img_bgr)
        raw_outputs = self._run_inference(img_fp16)
        return self._decode_output(raw_outputs, ratio, pad, orig_shape)

    # ------------------------------------------------------------------
    def __del__(self):
        """CUDA kaynaklarını serbest bırak."""
        try:
            if hasattr(self, "_stream"):
                del self._stream
        except Exception:
            pass


# -------------------------------------------------------------------
# ONNX'ten Engine yeniden derleme (farklı GPU için)
# -------------------------------------------------------------------
def build_engine_from_onnx(onnx_path: str,
                            engine_path: str,
                            fp16: bool = True,
                            workspace_gb: int = 4) -> bool:
    """
    ONNX modelinden TensorRT FP16 engine derler.
    Tesla T4'te derleme için Docker build aşamasında çağrılabilir.

    Kullanım:
        build_engine_from_onnx("aracplak.onnx", "aracplak.engine", fp16=True)
    """
    if not TRT_AVAILABLE:
        logger.error("TensorRT kurulu değil.")
        return False

    import os
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for err in range(parser.num_errors):
                logger.error(f"ONNX parse hatası: {parser.get_error(err)}")
            return False

    config = builder.create_builder_config()
    config.max_workspace_size = workspace_gb * (1 << 30)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 modu aktif.")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        logger.error("Engine derlenemedi.")
        return False

    with open(engine_path, "wb") as f:
        f.write(serialized)

    logger.info(f"Engine kaydedildi: {engine_path}")
    return True
