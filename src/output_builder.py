"""
output_builder.py — Yarışma Formatında results.json Üreticisi
=============================================================
Yarışma dokümantasyonunda belirtilen tam çıktı formatı:

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

Kurallar:
  - Tüm anahtarlar küçük harf, Türkçe karakter YOK
  - Plaka: boşuksuz, büyük harf (34ABC123)
  - confidence_score: [0.0, 1.0], 4 ondalık
  - zaman_saniye: 2 ondalık
"""

import json
import logging
import os
from typing import List, Optional

from src.vehicle_cache import VehicleCache, FinalVehicleData, ActionObservation

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Geçerli yarışma kategori ve etiketleri
# -------------------------------------------------------------------
VALID_KATEGORILER = {"sofor_eylemi", "nesneler", "yolcular"}

VALID_ETIKETLER = {
    "sofor_eylemi": {
        "arkaya_bakma", "esneme", "sigara_icme", "su_icme",
        "telefonla_konusma", "slalom", "etrafa_bakinma",
        "emniyet_kemeri_ihlali"
    },
    "nesneler": {"teknocan", "bilgisayar"},
    "yolcular": {"arka_koltuk_1", "arka_koltuk_2", "on_koltuk"},
}

VALID_TIP = {"sedan", "suv", "hatchback", "pickup", "minibus", "panelvan", "kamyon"}
VALID_RENK = {"beyaz", "siyah", "gri", "kirmizi", "mavi", "sari", "yesil", "turuncu", "kahverengi"}

# Eylem tekrarlama minimum süresi (saniye) — aynı eylemi sürekli eklememek için
ACTION_DEDUP_SECONDS = 2.0


# -------------------------------------------------------------------
# OutputBuilder
# -------------------------------------------------------------------
class OutputBuilder:
    """
    VehicleCache ve eylem gözlemlerinden yarışma formatında JSON üretir.

    Kullanım:
        builder = OutputBuilder()
        builder.set_vehicle(cache, primary_track_id)
        builder.add_external_detections(extra_tespitler)
        builder.save("/app/data/output/results.json")
    """

    def __init__(self):
        self._arac_bilgisi = None
        self._tespitler: List[dict] = []

    # ------------------------------------------------------------------
    def set_vehicle_from_cache(self,
                                cache: VehicleCache,
                                primary_track_id: int):
        """
        VehicleCache'den birincil araç bilgisini ve eylem tespitlerini yükler.

        primary_track_id: En uzun süre takip edilen araç (main.py'de seçilir)
        """
        final: FinalVehicleData = cache.get_final(primary_track_id)

        # --- arac_bilgisi ---
        tip  = final.tip  if final.tip  in VALID_TIP  else "sedan"
        renk = final.renk if final.renk in VALID_RENK else "beyaz"
        plaka = self._validate_plaka(final.plaka)

        self._arac_bilgisi = {
            "tip":              tip,
            "plaka":            plaka,
            "renk":             renk,
            "confidence_score": self._clamp_conf(final.confidence_score),
        }

        # --- sofor_eylemi tespitlerini ekle ---
        actions: List[ActionObservation] = cache.get_actions(primary_track_id)
        self._add_action_detections(actions)

        logger.info(f"arac_bilgisi: {self._arac_bilgisi}")
        logger.info(f"Toplam tespit: {len(self._tespitler)}")

    # ------------------------------------------------------------------
    def add_detection(self,
                       zaman_saniye:    float,
                       kategori:        str,
                       etiket:          str,
                       confidence_score: float):
        """
        Manuel olarak tek bir tespit ekle.
        (teknocan, bilgisayar, yolcu gibi ek modeller için)
        """
        if kategori not in VALID_KATEGORILER:
            logger.warning(f"Geçersiz kategori: {kategori}")
            return

        valid_etiketler = VALID_ETIKETLER.get(kategori, set())
        if etiket not in valid_etiketler:
            logger.warning(f"Geçersiz etiket: {etiket} (kategori={kategori})")
            return

        self._tespitler.append({
            "zaman_saniye":    round(float(zaman_saniye), 2),
            "kategori":        kategori,
            "etiket":          etiket,
            "confidence_score": self._clamp_conf(confidence_score),
        })

    # ------------------------------------------------------------------
    def _add_action_detections(self, actions: List[ActionObservation]):
        """
        Eylem gözlemlerini tespitler listesine ekle.
        Aynı eylem için tekrarlı tespitleri deduplicate eder.
        """
        last_seen: dict = {}  # {label: timestamp}

        # Zaman damgasına göre sırala
        sorted_actions = sorted(actions, key=lambda a: a.timestamp_sec)

        for obs in sorted_actions:
            label = obs.label
            ts    = obs.timestamp_sec
            conf  = obs.confidence

            # Deduplicate: aynı eylem ACTION_DEDUP_SECONDS içinde tekrar ekleme
            if label in last_seen:
                if ts - last_seen[label] < ACTION_DEDUP_SECONDS:
                    continue

            # Geçerli etiket kontrolü
            if label not in VALID_ETIKETLER.get("sofor_eylemi", set()):
                logger.warning(f"Bilinmeyen eylem etiketi atlandı: {label}")
                continue

            self._tespitler.append({
                "zaman_saniye":    round(ts, 2),
                "kategori":        "sofor_eylemi",
                "etiket":          label,
                "confidence_score": self._clamp_conf(conf),
            })
            last_seen[label] = ts

    # ------------------------------------------------------------------
    def build(self) -> dict:
        """
        Tam results.json sözlüğünü oluşturur.

        Returns:
            Yarışma formatında Python dict
        """
        if self._arac_bilgisi is None:
            logger.warning("arac_bilgisi henüz set edilmedi! Varsayılan kullanılıyor.")
            self._arac_bilgisi = {
                "tip":              "sedan",
                "plaka":            None,
                "renk":             "beyaz",
                "confidence_score": 0.0,
            }

        # Tespitleri zaman sırasına diz
        tespitler_sorted = sorted(
            self._tespitler,
            key=lambda x: x["zaman_saniye"]
        )

        return {
            "arac_bilgisi": self._arac_bilgisi,
            "tespitler":    tespitler_sorted,
        }

    # ------------------------------------------------------------------
    def save(self, output_path: str) -> bool:
        """
        results.json dosyasını belirtilen yola kaydeder.
        Klasör yoksa otomatik oluşturur.

        Args:
            output_path: /app/data/output/results.json

        Returns:
            True başarılı, False hata
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            data = self.build()

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Kayıt sonrası doğrulama
            self._validate(data)
            logger.info(f"results.json kaydedildi: {output_path}")
            logger.info(f"  → arac_bilgisi: {data['arac_bilgisi']}")
            logger.info(f"  → tespit sayısı: {len(data['tespitler'])}")
            return True

        except Exception as e:
            logger.error(f"results.json kayıt hatası: {e}")
            return False

    # ------------------------------------------------------------------
    def _validate(self, data: dict):
        """Üretilen JSON'ı yarışma kurallarına göre doğrula."""
        ab = data.get("arac_bilgisi", {})
        errors = []

        if ab.get("tip") not in VALID_TIP:
            errors.append(f"Geçersiz tip: {ab.get('tip')}")

        if ab.get("renk") not in VALID_RENK:
            errors.append(f"Geçersiz renk: {ab.get('renk')}")

        conf = ab.get("confidence_score", -1)
        if not (0.0 <= conf <= 1.0):
            errors.append(f"Geçersiz confidence_score: {conf}")

        for t in data.get("tespitler", []):
            if t.get("kategori") not in VALID_KATEGORILER:
                errors.append(f"Geçersiz kategori: {t.get('kategori')}")
            kat = t.get("kategori", "")
            if t.get("etiket") not in VALID_ETIKETLER.get(kat, set()):
                errors.append(f"Geçersiz etiket: {t.get('etiket')} in {kat}")

        if errors:
            for err in errors:
                logger.error(f"[DOĞRULAMA HATASI] {err}")
        else:
            logger.info("results.json doğrulaması geçti ✓")

    # ------------------------------------------------------------------
    @staticmethod
    def _validate_plaka(plaka: Optional[str]) -> Optional[str]:
        """Plaka metninin null veya geçerli format olduğunu garantile."""
        if plaka is None:
            return None
        import re
        pattern = re.compile(r'^[0-9]{2}[A-Z]{1,3}[0-9]{1,4}$')
        return plaka if pattern.match(plaka) else None

    @staticmethod
    def _clamp_conf(conf: float) -> float:
        """Confidence değerini [0.0, 1.0] aralığına kırp."""
        return round(max(0.0, min(1.0, float(conf))), 4)
