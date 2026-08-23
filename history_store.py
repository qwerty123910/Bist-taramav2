"""
Tarama Geçmişi ve Kriter Katkı Analizi
=========================================
ÖNEMLİ: Streamlit Cloud'un ücretsiz katmanında dosya sistemi KALICI DEĞİLDİR.
Uygulama "uyku moduna" girip yeniden başladığında veya yeniden deploy
edildiğinde bu dosyadaki geçmiş SİLİNEBİLİR. Bu yüzden:
  1) Her taramadan sonra "Geçmişi İndir" ile JSON dosyanızı telefonunuza
     kaydetmenizi,
  2) Uygulama açıldığında "Geçmiş Yükle" ile o dosyayı geri yüklemenizi
     öneririm. Bu, kalıcılığı sizin elinizde tutar.
"""

import os
import json
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "scan_history.json")

SCORE_COLUMNS_OF_INTEREST = [
    "score_pe_ratio", "score_pb_ratio", "score_roe", "score_debt_to_equity",
    "score_profit_margin", "score_profit_growth",
    "score_momentum_3m", "score_sma_trend", "score_rsi_score", "score_volume_ratio",
]


def load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_history(history: list):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2, default=str)


def append_scan(selected_df: pd.DataFrame, meta: Optional[dict] = None) -> list:
    """Bir taramanın sonucunu (seçilen top_n hisseler + skor kırılımları) geçmişe ekler."""
    history = load_history()

    record = {
        "timestamp": datetime.now().isoformat(),
        "meta": meta or {},
        "holdings": [],
    }
    for ticker, row in selected_df.iterrows():
        holding = {"ticker": ticker, "final_score": _safe_float(row.get("final_score"))}
        for col in SCORE_COLUMNS_OF_INTEREST:
            if col in row:
                holding[col] = _safe_float(row.get(col))
        holding["last_price"] = _safe_float(row.get("last_price"))
        record["holdings"].append(holding)

    history.append(record)
    save_history(history)
    return history


def _safe_float(v) -> Optional[float]:
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def history_to_dataframe(history: list) -> pd.DataFrame:
    """Geçmişi tek satır = bir taramadaki bir hisse olacak şekilde düz tabloya çevirir."""
    rows = []
    for scan in history:
        for h in scan["holdings"]:
            row = {"scan_date": scan["timestamp"], **h}
            rows.append(row)
    return pd.DataFrame(rows)


def merge_uploaded_history(uploaded_json_bytes: bytes) -> list:
    """Kullanıcının telefonundan geri yüklediği bir geçmiş dosyasını mevcutla birleştirir
    (aynı timestamp'li kayıtları tekrar eklemez)."""
    try:
        uploaded = json.loads(uploaded_json_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Geçersiz geçmiş dosyası: {e}")

    current = load_history()
    existing_ts = {r["timestamp"] for r in current}
    for r in uploaded:
        if r.get("timestamp") not in existing_ts:
            current.append(r)
            existing_ts.add(r.get("timestamp"))

    current.sort(key=lambda r: r["timestamp"])
    save_history(current)
    return current


def compute_forward_returns(history: list, fetch_price_fn) -> pd.DataFrame:
    """
    Ardışık iki tarama arasında, o taramada seçilen her hissenin gerçekleşen
    getirisini hesaplar (fetch_price_fn: (ticker, date) -> fiyat döndüren fonksiyon).
    En az 2 tarama gerektirir; ilk tarama "başlangıç", sonrakiyle karşılaştırılır.
    """
    if len(history) < 2:
        return pd.DataFrame()

    sorted_history = sorted(history, key=lambda r: r["timestamp"])
    rows = []
    for i in range(len(sorted_history) - 1):
        scan = sorted_history[i]
        next_scan = sorted_history[i + 1]
        scan_date = scan["timestamp"][:10]
        next_date = next_scan["timestamp"][:10]

        for h in scan["holdings"]:
            ticker = h["ticker"]
            price_then = h.get("last_price")
            price_next = fetch_price_fn(ticker, next_date)
            if price_then is None or price_next is None or price_then == 0:
                continue
            forward_return = (price_next / price_then - 1) * 100
            row = {"ticker": ticker, "scan_date": scan_date, "forward_return": forward_return}
            for col in SCORE_COLUMNS_OF_INTEREST:
                row[col] = h.get(col)
            rows.append(row)

    return pd.DataFrame(rows)


def criterion_contribution(forward_returns_df: pd.DataFrame) -> pd.Series:
    """Her skor bileşeni ile gerçekleşen ileri getiri arasındaki korelasyonu hesaplar.
    Pozitif değer: o kriterde yüksek puan alan hisseler gerçekten daha çok kazandırmış.
    En az birkaç ay/tarama birikmeden bu sayılar anlamlı değildir — az örnekte
    tesadüfle de yüksek çıkabilir, temkinli yorumlayın."""
    if forward_returns_df.empty:
        return pd.Series(dtype=float)

    correlations = {}
    for col in SCORE_COLUMNS_OF_INTEREST:
        if col not in forward_returns_df.columns:
            continue
        pair = forward_returns_df[[col, "forward_return"]].dropna()
        if len(pair) >= 5:  # çok az örnekle korelasyon anlamsız olur
            correlations[col] = pair[col].corr(pair["forward_return"])

    return pd.Series(correlations).sort_values(ascending=False)
