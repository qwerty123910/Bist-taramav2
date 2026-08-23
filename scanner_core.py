"""
Çekirdek Tarama ve Puanlama Motoru
====================================
Veri çekme, teknik/temel gösterge hesaplama, sektör-farkında puanlama,
çeşitlilik kısıtı, piyasa rejimi tespiti ve portföy risk hesaplarını içerir.

Veri kaynağı: Yahoo Finance (yfinance kütüphanesi, API anahtarı gerekmez)
"""

import time
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("scanner_core")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TICKERS_FILE = os.path.join(BASE_DIR, "bist100_tickers.txt")
INDEX_TICKER = "XU100.IS"   # BIST100 endeksi (Yahoo Finance'te doğrulanmış sembol)


# ============================================================
# KONFİGÜRASYON
# ============================================================

DEFAULT_TOP_N = 10
DEFAULT_MIN_AVG_VOLUME_TRY = 5_000_000
DEFAULT_MAX_PER_SECTOR = 2
HISTORY_PERIOD = "1y"
REQUEST_DELAY_SEC = 0.5

# Her kriter için: weight (ağırlık), direction (high=yüksek iyi / low=düşük iyi),
# group_by_sector (True ise sektör içi yüzdelik alınır — bankayı bankayla kıyaslar)
#
# NOT: Büyüme ve temettü kriterleri eklenince ağırlıkları yeniden dengeledim.
# Toplamları 1.0. Bu benim önerdiğim bir başlangıç noktası — "doğru" ağırlık
# diye bir şey yok, test edilmesi gerekir (bkz. Backtest sekmesi).
WEIGHTS = {
    "pe_ratio":        {"weight": 0.12, "direction": "low",  "group_by_sector": True},
    "pb_ratio":        {"weight": 0.08, "direction": "low",  "group_by_sector": True},
    "roe":             {"weight": 0.13, "direction": "high", "group_by_sector": False},
    "debt_to_equity":  {"weight": 0.08, "direction": "low",  "group_by_sector": False},
    "profit_margin":   {"weight": 0.08, "direction": "high", "group_by_sector": False},
    "profit_growth":   {"weight": 0.14, "direction": "high", "group_by_sector": False},
    "momentum_3m":     {"weight": 0.13, "direction": "high", "group_by_sector": False},
    "sma_trend":       {"weight": 0.09, "direction": "high", "group_by_sector": False},
    "rsi_score":       {"weight": 0.10, "direction": "high", "group_by_sector": False},
    "volume_ratio":    {"weight": 0.05, "direction": "high", "group_by_sector": False},
}

REGIME_WEIGHT_MULTIPLIERS = {
    "bullish": {"momentum_3m": 1.3, "sma_trend": 1.2, "rsi_score": 1.1},
    "bearish": {"roe": 1.3, "debt_to_equity": 1.4,
                "profit_margin": 1.2, "momentum_3m": 0.5, "rsi_score": 0.7},
    "neutral": {},
}


# ============================================================
# TICKER LİSTESİ
# ============================================================

def load_tickers(path: str = TICKERS_FILE) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ticker dosyası bulunamadı: {path}")
    tickers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            code = line.upper()
            if not code.endswith(".IS"):
                code += ".IS"
            tickers.append(code)
    return tickers


# ============================================================
# VERİ ÇEKME
# ============================================================

@dataclass
class StockData:
    ticker: str
    hist: Optional[pd.DataFrame] = None
    info: dict = field(default_factory=dict)
    error: Optional[str] = None


def fetch_stock_data(ticker: str, period: str = HISTORY_PERIOD) -> StockData:
    data = StockData(ticker=ticker)
    if yf is None:
        data.error = "yfinance kurulu değil"
        return data
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, auto_adjust=True)
        if hist.empty:
            data.error = "Fiyat verisi boş döndü"
            return data
        data.hist = hist
        data.info = t.info or {}
    except Exception as e:
        data.error = str(e)
    return data


def fetch_universe(
    tickers: list[str],
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    period: str = HISTORY_PERIOD,
) -> dict[str, StockData]:
    results = {}
    total = len(tickers)
    for i, ticker in enumerate(tickers, 1):
        results[ticker] = fetch_stock_data(ticker, period=period)
        if progress_callback:
            progress_callback(i, total, ticker)
        time.sleep(REQUEST_DELAY_SEC)
    return results


def fetch_index_history(index_ticker: str = INDEX_TICKER, period: str = HISTORY_PERIOD) -> Optional[pd.DataFrame]:
    """XU100 endeks geçmişini çeker. Başarısız olursa None döner."""
    if yf is None:
        return None
    try:
        hist = yf.Ticker(index_ticker).history(period=period, auto_adjust=True)
        return hist if not hist.empty else None
    except Exception as e:
        log.warning(f"Endeks verisi çekilemedi ({index_ticker}): {e}")
        return None


# ============================================================
# TEKNİK GÖSTERGELER
# ============================================================

def compute_rsi(closes: pd.Series, period: int = 14) -> float:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty and not np.isnan(rsi.iloc[-1]) else np.nan


def rsi_to_score(rsi: float) -> float:
    if np.isnan(rsi):
        return np.nan
    ideal = 57.5
    return max(0.0, 100 - abs(rsi - ideal) * 2)


def compute_technical_metrics(hist: pd.DataFrame) -> dict:
    closes = hist["Close"]
    volumes = hist["Volume"]

    sma50 = closes.rolling(50).mean().iloc[-1]
    sma200 = closes.rolling(200).mean().iloc[-1] if len(closes) >= 200 else np.nan
    last_close = closes.iloc[-1]

    sma_trend = np.nan
    if not np.isnan(sma50):
        sma_trend = ((last_close / sma50) - 1) * 100
        if not np.isnan(sma200):
            sma_trend += ((last_close / sma200) - 1) * 100

    momentum_3m = np.nan
    if len(closes) >= 63:
        momentum_3m = (last_close / closes.iloc[-63] - 1) * 100

    avg_vol_recent = volumes.tail(20).mean()
    avg_vol_long = volumes.tail(120).mean() if len(volumes) >= 120 else volumes.mean()
    volume_ratio = (avg_vol_recent / avg_vol_long) if avg_vol_long else np.nan

    rsi = compute_rsi(closes)
    avg_liquidity_try = avg_vol_recent * last_close

    daily_returns = closes.pct_change().dropna()
    annualized_vol = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 20 else np.nan

    return {
        "last_price": last_close,
        "sma_trend": sma_trend,
        "momentum_3m": momentum_3m,
        "volume_ratio": volume_ratio,
        "rsi": rsi,
        "rsi_score": rsi_to_score(rsi),
        "avg_liquidity_try": avg_liquidity_try,
        "annualized_vol": annualized_vol,
    }


def extract_fundamental_metrics(info: dict) -> dict:
    earnings_growth = info.get("earningsQuarterlyGrowth")
    return {
        "sector": info.get("sector") or "Bilinmiyor",
        "pe_ratio": info.get("trailingPE", np.nan),
        "pb_ratio": info.get("priceToBook", np.nan),
        "roe": (info.get("returnOnEquity") or np.nan) * 100 if info.get("returnOnEquity") is not None else np.nan,
        "debt_to_equity": info.get("debtToEquity", np.nan),
        "profit_margin": (info.get("profitMargins") or np.nan) * 100 if info.get("profitMargins") is not None else np.nan,
        "profit_growth": earnings_growth * 100 if earnings_growth is not None else np.nan,
    }


# ============================================================
# PUANLAMA
# ============================================================

# Puanlamada kullanılan ve veri kalitesi hesabına dahil edilen alanlar
QUALITY_CHECK_FIELDS = [
    "pe_ratio", "pb_ratio", "roe", "debt_to_equity", "profit_margin",
    "profit_growth", "momentum_3m", "sma_trend", "rsi_score", "volume_ratio",
]


def compute_data_quality(row: dict) -> float:
    """0-100 arası: kaç kritik alanın gerçek (NaN olmayan) veriyle geldiğini gösterir."""
    present = sum(1 for f in QUALITY_CHECK_FIELDS if not pd.isna(row.get(f, np.nan)))
    return round(100 * present / len(QUALITY_CHECK_FIELDS), 1)


def build_metrics_table(universe: dict[str, StockData]) -> pd.DataFrame:
    rows = []
    for ticker, data in universe.items():
        if data.error or data.hist is None:
            log.warning(f"{ticker} atlandı: {data.error}")
            continue
        row = {"ticker": ticker}
        row.update(compute_technical_metrics(data.hist))
        row.update(extract_fundamental_metrics(data.info))
        row["data_quality"] = compute_data_quality(row)
        rows.append(row)
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()


def apply_liquidity_filter(df: pd.DataFrame, min_volume_try: float) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["avg_liquidity_try"] >= min_volume_try]


def percentile_score(series: pd.Series, direction: str) -> pd.Series:
    ranked = series.rank(pct=True, na_option="keep") * 100
    if direction == "low":
        ranked = 100 - ranked
    return ranked


def percentile_score_grouped(df: pd.DataFrame, column: str, direction: str, group_col: str = "sector") -> pd.Series:
    """Sektör içi yüzdelik puan. Sektörde yeterli hisse yoksa global yüzdeliğe düşer."""
    result = pd.Series(index=df.index, dtype=float)
    for sector, group in df.groupby(group_col):
        if len(group) >= 3:
            result.loc[group.index] = percentile_score(group[column], direction)
        else:
            result.loc[group.index] = np.nan
    missing = result.isna()
    if missing.any():
        global_scores = percentile_score(df[column], direction)
        result.loc[missing] = global_scores.loc[missing]
    return result


def get_effective_weights(base_weights: dict, regime: str = "neutral") -> dict:
    multipliers = REGIME_WEIGHT_MULTIPLIERS.get(regime, {})
    adjusted = {}
    for metric, cfg in base_weights.items():
        mult = multipliers.get(metric, 1.0)
        adjusted[metric] = {**cfg, "weight": cfg["weight"] * mult}
    total = sum(c["weight"] for c in adjusted.values())
    for metric in adjusted:
        adjusted[metric]["weight"] /= total
    return adjusted


def compute_final_scores(df: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    if df.empty:
        return df
    weights = weights or WEIGHTS
    scored = df.copy()

    score_cols = []
    for metric, cfg in weights.items():
        if metric not in scored.columns:
            continue
        col_name = f"score_{metric}"
        if cfg.get("group_by_sector") and "sector" in scored.columns:
            scored[col_name] = percentile_score_grouped(scored, metric, cfg["direction"])
        else:
            scored[col_name] = percentile_score(scored[metric], cfg["direction"])
        score_cols.append((col_name, cfg["weight"]))

    def weighted_row_score(row):
        available = [(c, w) for c, w in score_cols if not pd.isna(row[c])]
        if not available:
            return np.nan
        w_sum = sum(w for _, w in available)
        return sum(row[c] * w for c, w in available) / w_sum

    scored["final_score"] = scored.apply(weighted_row_score, axis=1)
    return scored.sort_values("final_score", ascending=False)


def select_diversified(scored_df: pd.DataFrame, top_n: int, max_per_sector: int = DEFAULT_MAX_PER_SECTOR):
    """En yüksek puanlıdan başlayıp sektör başına en fazla max_per_sector hisse
    alacak şekilde top_n seçer. Yetmezse kısıtı gevşetip bunu bildirir."""
    if scored_df.empty:
        return scored_df, False

    selected_idx = []
    sector_counts = {}
    for ticker, row in scored_df.iterrows():
        sector = row.get("sector", "Bilinmiyor")
        if sector_counts.get(sector, 0) >= max_per_sector:
            continue
        selected_idx.append(ticker)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(selected_idx) >= top_n:
            break

    relaxed = False
    if len(selected_idx) < top_n:
        relaxed = True
        remaining = [t for t in scored_df.index if t not in selected_idx]
        needed = top_n - len(selected_idx)
        selected_idx.extend(remaining[:needed])

    return scored_df.loc[selected_idx], relaxed


# ============================================================
# PİYASA REJİMİ TESPİTİ
# ============================================================

def detect_market_regime(index_hist: Optional[pd.DataFrame] = None) -> dict:
    if index_hist is None:
        index_hist = fetch_index_history()

    if index_hist is None or index_hist.empty or len(index_hist) < 60:
        return {"regime": "neutral", "note": "Endeks verisi alınamadı, rejim ayarlaması uygulanmadı."}

    closes = index_hist["Close"]
    last = closes.iloc[-1]
    sma50 = closes.rolling(50).mean().iloc[-1]
    sma200 = closes.rolling(200).mean().iloc[-1] if len(closes) >= 200 else np.nan

    if np.isnan(sma50):
        return {"regime": "neutral", "note": "Yeterli endeks verisi yok."}

    above50 = last > sma50
    above200 = (last > sma200) if not np.isnan(sma200) else above50

    if above50 and above200:
        regime, note = "bullish", "XU100, 50 ve 200 günlük ortalamaların üzerinde — yükseliş eğilimi."
    elif not above50 and not above200:
        regime, note = "bearish", "XU100, 50 ve 200 günlük ortalamaların altında — düşüş eğilimi."
    else:
        regime, note = "neutral", "XU100 karışık sinyal veriyor — net bir trend yok."

    return {"regime": regime, "note": note, "last": last, "sma50": sma50, "sma200": sma200}


# ============================================================
# PORTFÖY RİSK SKORU
# ============================================================

def compute_portfolio_risk(universe: dict, selected_tickers: list,
                             index_hist: Optional[pd.DataFrame] = None) -> dict:
    vols = []
    stock_returns = {}
    for ticker in selected_tickers:
        data = universe.get(ticker)
        if data is None or data.hist is None:
            continue
        closes = data.hist["Close"]
        rets = closes.pct_change().dropna()
        if len(rets) > 20:
            vols.append(rets.std() * np.sqrt(252) * 100)
            stock_returns[ticker] = rets

    avg_vol = float(np.mean(vols)) if vols else np.nan

    beta = np.nan
    if index_hist is not None and not index_hist.empty and stock_returns:
        index_rets = index_hist["Close"].pct_change().dropna()
        betas = []
        for ticker, rets in stock_returns.items():
            aligned = pd.concat([rets, index_rets], axis=1, join="inner")
            aligned.columns = ["stock", "index"]
            if len(aligned) > 20 and aligned["index"].var() > 0:
                cov = aligned["stock"].cov(aligned["index"])
                betas.append(cov / aligned["index"].var())
        if betas:
            beta = float(np.mean(betas))

    return {
        "avg_annualized_volatility_pct": avg_vol,
        "avg_beta_vs_xu100": beta,
        "n_stocks": len(vols),
    }


# ============================================================
# TAM TARAMA AKIŞI
# ============================================================

def run_scan(
    top_n: int = DEFAULT_TOP_N,
    min_avg_volume_try: float = DEFAULT_MIN_AVG_VOLUME_TRY,
    tickers: Optional[list[str]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    apply_sector_diversity: bool = True,
    max_per_sector: int = DEFAULT_MAX_PER_SECTOR,
    apply_regime_adjustment: bool = False,
) -> dict:
    if yf is None:
        raise RuntimeError("yfinance kurulu değil. 'pip install -r requirements.txt' çalıştırın.")

    tickers = tickers or load_tickers()
    universe = fetch_universe(tickers, progress_callback=progress_callback)

    df = build_metrics_table(universe)
    if df.empty:
        raise RuntimeError("Hiçbir hisse için veri alınamadı. İnternet bağlantınızı kontrol edin.")

    df = apply_liquidity_filter(df, min_avg_volume_try)
    if df.empty:
        raise RuntimeError("Likidite filtresinden geçen hisse kalmadı. Minimum hacim eşiğini düşürün.")

    index_hist = fetch_index_history()
    regime_info = None
    weights = WEIGHTS
    if apply_regime_adjustment:
        regime_info = detect_market_regime(index_hist)
        weights = get_effective_weights(WEIGHTS, regime_info["regime"])

    scored = compute_final_scores(df, weights=weights)

    if apply_sector_diversity:
        selected, relaxed = select_diversified(scored, top_n, max_per_sector)
    else:
        selected, relaxed = scored.head(top_n), False

    return {
        "full_ranked": scored,
        "selected": selected,
        "diversity_relaxed": relaxed,
        "regime": regime_info,
        "universe": universe,
        "index_hist": index_hist,
    }
