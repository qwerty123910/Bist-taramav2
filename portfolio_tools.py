"""
Portföy Araçları
==================
Kullanıcının kendi girdiği pozisyonlar üzerinde:
- Stop-loss ihlali kontrolü
- Sektör yoğunlaşması
- Önerilen listeyle örtüşme
- Toplam portföy risk özeti (scanner_core.compute_portfolio_risk'i kullanır)
"""

from typing import Optional
import numpy as np
import pandas as pd

import scanner_core as core


def normalize_ticker(code: str) -> str:
    code = code.strip().upper()
    if not code.endswith(".IS"):
        code += ".IS"
    return code


def fetch_current_prices(tickers: list[str]) -> dict[str, float]:
    """Sadece güncel fiyat için hafif bir çekim (tam geçmiş gerekmez)."""
    prices = {}
    for t in tickers:
        data = core.fetch_stock_data(t, period="5d")
        if data.hist is not None and not data.hist.empty:
            prices[t] = float(data.hist["Close"].iloc[-1])
    return prices


def check_positions(positions: pd.DataFrame) -> pd.DataFrame:
    """
    positions sütunları: ticker, adet, maliyet, stop_loss_yuzde (opsiyonel)
    Döndürür: aynı tabloya güncel_fiyat, kar_zarar_yuzde, stop_ihlali sütunları eklenmiş hali.
    """
    if positions.empty:
        return positions

    tickers = [normalize_ticker(t) for t in positions["ticker"]]
    prices = fetch_current_prices(tickers)

    result = positions.copy()
    result["ticker"] = tickers
    result["guncel_fiyat"] = result["ticker"].map(prices)
    result["kar_zarar_yuzde"] = np.where(
        result["guncel_fiyat"].notna() & (result["maliyet"] > 0),
        (result["guncel_fiyat"] / result["maliyet"] - 1) * 100,
        np.nan,
    )

    if "stop_loss_yuzde" in result.columns:
        result["stop_ihlali"] = (
            result["stop_loss_yuzde"].notna()
            & result["kar_zarar_yuzde"].notna()
            & (result["kar_zarar_yuzde"] <= -result["stop_loss_yuzde"].abs())
        )
    else:
        result["stop_ihlali"] = False

    return result


def portfolio_sector_breakdown(positions_df: pd.DataFrame) -> pd.Series:
    """Her hisse için sektörü çeker (yavaş olabilir, .info gerektirir) ve
    pozisyon büyüklüğüne (adet * güncel_fiyat) göre sektör dağılımı verir."""
    sectors = {}
    for ticker in positions_df["ticker"]:
        data = core.fetch_stock_data(ticker, period="5d")
        sectors[ticker] = (data.info or {}).get("sector") or "Bilinmiyor"

    df = positions_df.copy()
    df["sector"] = df["ticker"].map(sectors)
    df["deger"] = df["adet"] * df["guncel_fiyat"].fillna(df["maliyet"])
    return df.groupby("sector")["deger"].sum().sort_values(ascending=False)


def overlap_with_recommendations(positions_df: pd.DataFrame, recommended_tickers: list[str]) -> dict:
    owned = set(positions_df["ticker"])
    recommended = set(recommended_tickers)
    return {
        "ortak": sorted(owned & recommended),
        "sadece_portfoyde": sorted(owned - recommended),
        "sadece_onerilerde": sorted(recommended - owned),
    }
