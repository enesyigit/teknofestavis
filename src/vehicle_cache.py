"""
vehicle_cache.py — VehicleCache: Oy Birleştirme + Güven Kilidi
===============================================================
Her track_id için birden fazla frame'den gelen tahminleri biriktirir.
Yeterli güven sağlandığında (≥ 0.85) ilgili alanı kilitler.

Yarışma gereksinimleri:
  - Araç başına tek bir nihai arac_bilgisi (tip, plaka, renk, conf)
  - Frame bazlı tespitler (sofor_eylemi kategorisi)
  - Confidence kilidi: aynı veriyi tekrar tekrar analiz etme
"""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Güven kilidi eşiği
# -------------------------------------------------------------------
LOCK_THRESHOLD = 0.85   # Bu değer veya üzeri → alan kilitlenir


# -------------------------------------------------------------------
# Veri yapıları
# -------------------------------------------------------------------
@dataclass
class FieldState:
    """Tek bir alan (tip/renk/plaka) için oy + kilit durumu."""
    votes:      List[str]   = field(default_factory=list)
    confs:      List[float] = field(default_factory=list)
    locked:     bool        = False
    locked_val: Optional[str]   = None
    locked_conf: float           = 0.0


@dataclass
class ActionObservation:
    """Tek bir eylem gözlemi (frame bazlı)."""
    label:         str
    confidence:    float
    timestamp_sec: float


@dataclass
class FinalVehicleData:
    """VehicleCache.get_final() çıkışı — yarışma JSON'una doğrudan girer."""
    # arac_bilgisi
    tip:            str
    plaka:          Optional[str]
    renk:           str
    confidence_score: float   # (tip_conf + renk_conf + plaka_conf) / N

    # Gözlem sayıları (debug)
    tip_conf:       float
    renk_conf:      float
    plaka_conf:     float
    frame_count:    int


# -------------------------------------------------------------------
# Yardımcı fonksiyonlar
# -------------------------------------------------------------------
def majority_vote(votes: list) -> Optional[str]:
    """En çok oy alan sınıf etiketini döner."""
    if not votes:
        return None
    return Counter(votes).most_common(1)[0][0]


def weighted_majority_vote(votes: list, confs: list) -> Optional[str]:
    """Confidence ağırlıklı oy birleştirme."""
    if not votes:
        return None
    score_map: Dict[str, float] = defaultdict(float)
    for v, c in zip(votes, confs):
        score_map[v] += c
    return max(score_map, key=score_map.get)


def best_plate_text(votes: list, confs: list) -> Tuple[Optional[str], float]:
    """
    Plaka için en güvenilir okunuşu seç.
    Önce en çok tekrar eden, eşit tekrarda en yüksek confidence.
    """
    if not votes:
        return None, 0.0

    count_map: Dict[str, int]   = Counter(votes)
    conf_map:  Dict[str, float] = defaultdict(float)
    for v, c in zip(votes, confs):
        conf_map[v] = max(conf_map[v], c)

    # Öncelik: tekrar sayısı, sonra confidence
    best = max(count_map, key=lambda x: (count_map[x], conf_map[x]))
    return best, conf_map[best]


def avg_conf(confs: list) -> float:
    return round(sum(confs) / len(confs), 4) if confs else 0.0

def max_conf(confs: list) -> float:
    return round(max(confs), 4) if confs else 0.0


# -------------------------------------------------------------------
# VehicleCache
# -------------------------------------------------------------------
class VehicleCache:
    """
    Araç başına tahmin geçmişi tutan ve nihai sonucu üreten cache.

    Kullanım:
        cache = VehicleCache()

        # Her frame'de:
        cache.update_body(track_id, "sedan", 0.91)
        cache.update_color(track_id, "kirmizi", 0.87)
        cache.update_plate(track_id, "34ABC123", 0.79)
        cache.add_action(track_id, "telefonla_konusma", 0.93, ts)

        # Video sonunda:
        final = cache.get_final(track_id)
        actions = cache.get_actions(track_id)
    """

    def __init__(self, lock_threshold: float = LOCK_THRESHOLD):
        self.lock_threshold = lock_threshold

        # Sınıflandırma alanları
        self._body:  Dict[int, FieldState] = defaultdict(FieldState)
        self._color: Dict[int, FieldState] = defaultdict(FieldState)
        self._plate: Dict[int, FieldState] = defaultdict(FieldState)

        # Eylem gözlemleri (frame bazlı)
        self._actions: Dict[int, List[ActionObservation]] = defaultdict(list)

        # İlk frame zamanı (track_id başına)
        self._first_seen: Dict[int, float] = {}

        # Toplam frame sayısı
        self._frame_counts: Dict[int, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Güncelleme fonksiyonları
    # ------------------------------------------------------------------
    def _update_field(self, state: FieldState, label: str, conf: float):
        """Genel alan güncelleme (kilitli alanları atla)."""
        if state.locked:
            return  # Zaten kilitli, güncelleme yok

        state.votes.append(label)
        state.confs.append(conf)

        # Güven kilidi kontrolü
        if conf >= self.lock_threshold:
            state.locked      = True
            state.locked_val  = label
            state.locked_conf = conf
            logger.debug(f"Alan kilitlendi: {label} (conf={conf:.3f})")

    def update_body(self, track_id: int, label: str, conf: float):
        """Kasa tipi tahmini ekle."""
        self._update_field(self._body[track_id], label, conf)
        self._frame_counts[track_id] += 1

    def update_color(self, track_id: int, label: str, conf: float):
        """Araç rengi tahmini ekle."""
        self._update_field(self._color[track_id], label, conf)

    def update_plate(self, track_id: int,
                     plate_text: Optional[str], conf: float):
        """Plaka metni ekle (None ise atla)."""
        if plate_text is None:
            return
        self._update_field(self._plate[track_id], plate_text, conf)

    def add_action(self, track_id: int, label: str,
                   conf: float, timestamp_sec: float):
        """
        Sürücü eylemi gözlemi ekle (frame bazlı, kilit YOK).
        Her gözlem ayrı bir tespit olarak results.json'a girer.
        """
        if label is None or conf <= 0:
            return
        self._actions[track_id].append(
            ActionObservation(label=label,
                              confidence=round(conf, 4),
                              timestamp_sec=round(timestamp_sec, 2))
        )

    def mark_seen(self, track_id: int, timestamp_sec: float):
        """Track_id'yi ilk görülme zamanıyla kaydet."""
        if track_id not in self._first_seen:
            self._first_seen[track_id] = timestamp_sec

    # ------------------------------------------------------------------
    # Sorgulama fonksiyonları
    # ------------------------------------------------------------------
    def is_body_locked(self, track_id: int) -> bool:
        return self._body[track_id].locked

    def is_color_locked(self, track_id: int) -> bool:
        return self._color[track_id].locked

    def is_plate_locked(self, track_id: int) -> bool:
        return self._plate[track_id].locked

    # ------------------------------------------------------------------
    def get_final(self, track_id: int) -> FinalVehicleData:
        """
        Track_id için nihai arac_bilgisi üret.
        Kilitli alanlar direkt kullanılır, kilitli değiller majority vote ile.
        """
        body_state  = self._body[track_id]
        color_state = self._color[track_id]
        plate_state = self._plate[track_id]

        # --- Kasa tipi ---
        if body_state.locked:
            tip      = body_state.locked_val
            tip_conf = body_state.locked_conf
        else:
            tip      = weighted_majority_vote(body_state.votes, body_state.confs) or "sedan"
            tip_conf = max_conf(body_state.confs)

        # --- Renk ---
        if color_state.locked:
            renk      = color_state.locked_val
            renk_conf = color_state.locked_conf
        else:
            renk      = weighted_majority_vote(color_state.votes, color_state.confs) or "beyaz"
            renk_conf = max_conf(color_state.confs)

        # --- Plaka ---
        if plate_state.locked:
            plaka      = plate_state.locked_val
            plaka_conf = plate_state.locked_conf
        else:
            plaka, plaka_conf = best_plate_text(plate_state.votes, plate_state.confs)

        # --- Birleşik confidence ---
        contributing = [c for c in [tip_conf, renk_conf, plaka_conf] if c > 0]
        agg_conf = round(sum(contributing) / len(contributing), 4) if contributing else 0.0

        return FinalVehicleData(
            tip             = tip,
            plaka           = plaka,
            renk            = renk,
            confidence_score = agg_conf,
            tip_conf        = tip_conf,
            renk_conf       = renk_conf,
            plaka_conf      = plaka_conf,
            frame_count     = self._frame_counts[track_id],
        )

    # ------------------------------------------------------------------
    def get_actions(self, track_id: int) -> List[ActionObservation]:
        """Frame bazlı eylem gözlem listesini döner."""
        return self._actions.get(track_id, [])

    def get_all_track_ids(self) -> List[int]:
        """Kayıtlı tüm track_id'leri döner."""
        all_ids = set()
        all_ids.update(self._body.keys())
        all_ids.update(self._color.keys())
        return list(all_ids)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Log için özet string."""
        lines = [f"VehicleCache — {len(self.get_all_track_ids())} araç:"]
        for tid in self.get_all_track_ids():
            b = self._body[tid]
            c = self._color[tid]
            p = self._plate[tid]
            a = len(self._actions.get(tid, []))
            lines.append(
                f"  track_id={tid}: "
                f"body={'🔒' if b.locked else '🔄'}({len(b.votes)} oy) "
                f"color={'🔒' if c.locked else '🔄'}({len(c.votes)} oy) "
                f"plate={'🔒' if p.locked else '🔄'}({len(p.votes)} oy) "
                f"actions={a}"
            )
        return "\n".join(lines)
