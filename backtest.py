"""
Backtest Modülü — ÖNEMLİ SINIRLAMA
=====================================
Bu backtest SADECE TEKNİK kriterleri (momentum, hareketli ortalama trendi,
RSI, hacim) kullanır. Temel kriterleri (F/K, ROE, borç/özkaynak vb.)
DAHİL ETMEZ.

Neden: yfinance sadece "şu an geçerli" F/K, ROE gibi değerleri verir;
"2 yıl önce bu hissenin F/K'sı neydi" sorusunun cevabını (point-in-time
temel veri) ücretsiz ve güvenilir şekilde veren bir kaynak bulamadım.
Bunu simüle etmeye çalışmak (örn. bugünkü F/K'yı geçmişe yansıtmak),
"ileriye bakma yanlılığı" (lookahead bias) yaratır ve sonucu yapay
şekilde iyi gösterir — bu yüzden bunu YAPMIYORUM.

Sonuç: Buradaki backtest, gerçek uygulamada kullandığınız TAM puanlama
stratejisinin (temel + teknik) geçmişte nasıl performans göstereceğinin
BİR TAHMİNİ DEĞİLDİR. Sadece teknik bacağın tek başına nasıl çalıştığını
gösterir. Bunu bu şekilde sunmazsam yanıltıcı olur.
"""

import logging
from typing import Optional, Callable

import numpy as np
import pandas as pd

import scanner_core as core

log = logging.getLogger("backtest")

# Teknik alt küme — sadece bunlar backtest'te kullanılabilir
TECHNICAL_WEIGHTS = {
    "momentum_3m":  {"weight": 0.40, "direction": "high"},
    "sma_trend":    {"weight": 0.30, "direction": "high"},
    "rsi_score":    {"weight": 0.20, "direction": "high"},
    "volume_ratio": {"weight": 0.10, "direction": "high"},
}


def _technical_score_asof(hist_slice: pd.DataFrame) -> Optional[dict]:
    """Belirli bir tarihe kadar olan veriyle (ileriye bakmadan) teknik metrikleri hesaplar."""
    if len(hist_slice) < 70:  # momentum_3m ve sma50 için minimum veri
        return None
    return core.compute_technical_metrics(hist_slice)


def fetch_backtest_histories(
    tickers: list[str],
    period: str = "3y",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict[str, pd.DataFrame]:
    """Her ticker için TEK SEFER tam geçmişi çeker (sonra tarih tarih dilimlenecek)."""
    histories = {}
    total = len(tickers)
    for i, t in enumerate(tickers, 1):
        data = core.fetch_stock_data(t, period=period)
        if data.hist is not None and len(data.hist) > 100:
            histories[t] = data.hist
        if progress_callback:
            progress_callback(i, total, t)
    return histories


def _percentile_score(values: dict, direction: str) -> dict:
    s = pd.Series(values)
    ranked = s.rank(pct=True) * 100
    if direction == "low":
        ranked = 100 - ranked
    return ranked.to_dict()


def run_backtest(
    histories: dict[str, pd.DataFrame],
    index_hist: Optional[pd.DataFrame],
    top_n: int = 10,
    rebalance_freq: str = "M",  # 'M' aylık, 'Q' çeyreklik
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """
    Her rebalance tarihinde SADECE O TARİHE KADARKİ veriyle teknik puan
    hesaplar, en iyi top_n hisseyi eşit ağırlıkla seçer, bir sonraki
    rebalance tarihine kadar tutar. İleriye bakma yanlılığı yoktur
    (teknik kriterler için) ama temel kriterleri içermez (yukarıya bakın).
    """
    if not histories:
        raise RuntimeError("Geçmiş veri bulunamadı.")

    # Tüm tickerlardaki en uzun/tam tarih aralığını referans al
    all_dates = sorted(set().union(*[set(h.index) for h in histories.values()]))
    all_dates = pd.DatetimeIndex(all_dates)

    # Rebalance tarihleri: her ayın/çeyreğin son işlem günü
    period_labels = all_dates.to_period(rebalance_freq)
    rebalance_dates = [all_dates[period_labels == p][-1] for p in sorted(set(period_labels))]
    # İlk birkaç tarihte yeterli geçmiş olmayabilir; ilk 70+ işlem günü sonrasından başla
    rebalance_dates = [d for d in rebalance_dates if (all_dates <= d).sum() >= 70]

    if len(rebalance_dates) < 2:
        raise RuntimeError("Backtest için yeterli geçmiş veri yok (en az ~4 ay gerekir).")

    portfolio_returns = []
    holdings_log = []
    total_steps = len(rebalance_dates) - 1

    for i in range(total_steps):
        current_date = rebalance_dates[i]
        next_date = rebalance_dates[i + 1]

        # O ana kadarki veriyle teknik skor hesapla (ileri veri KULLANILMAZ)
        scores = {}
        for ticker, hist in histories.items():
            hist_slice = hist[hist.index <= current_date]
            metrics = _technical_score_asof(hist_slice)
            if metrics is None:
                continue
            scores[ticker] = metrics

        if len(scores) < top_n:
            continue

        metrics_df = pd.DataFrame(scores).T
        score_cols = []
        for metric, cfg in TECHNICAL_WEIGHTS.items():
            if metric not in metrics_df.columns:
                continue
            col = f"score_{metric}"
            metrics_df[col] = _percentile_score(metrics_df[metric].to_dict(), cfg["direction"])
            metrics_df[col] = pd.Series(metrics_df[col])
            score_cols.append((col, cfg["weight"]))

        def weighted(row):
            avail = [(c, w) for c, w in score_cols if not pd.isna(row[c])]
            if not avail:
                return np.nan
            wsum = sum(w for _, w in avail)
            return sum(row[c] * w for c, w in avail) / wsum

        metrics_df["tech_score"] = metrics_df.apply(weighted, axis=1)
        selected = metrics_df["tech_score"].dropna().sort_values(ascending=False).head(top_n).index.tolist()

        # Bir sonraki rebalance tarihine kadarki getiriyi hesapla (eşit ağırlık)
        period_returns = []
        for ticker in selected:
            hist = histories[ticker]
            price_then = hist[hist.index <= current_date]["Close"].iloc[-1]
            future = hist[(hist.index > current_date) & (hist.index <= next_date)]
            if future.empty:
                continue
            price_next = future["Close"].iloc[-1]
            period_returns.append(price_next / price_then - 1)

        if not period_returns:
            continue

        avg_return = float(np.mean(period_returns))
        portfolio_returns.append({"date": next_date, "period_return": avg_return, "n_holdings": len(period_returns)})
        holdings_log.append({"date": current_date, "holdings": selected})

        if progress_callback:
            progress_callback(i + 1, total_steps)

    if not portfolio_returns:
        raise RuntimeError("Backtest hiçbir dönem için sonuç üretemedi.")

    returns_df = pd.DataFrame(portfolio_returns).set_index("date")
    returns_df["cum_return"] = (1 + returns_df["period_return"]).cumprod() - 1

    # Benchmark: XU100 buy & hold — backtest'in FİİLEN BAŞLADIĞI tarihten
    # (ilk rebalance tarihi, rebalance_dates[0]) itibaren, aynı baz noktasıyla.
    bench_df = None
    if index_hist is not None and not index_hist.empty:
        backtest_start = rebalance_dates[0]
        idx_closes = index_hist["Close"]
        idx_closes = idx_closes[idx_closes.index >= backtest_start]
        if not idx_closes.empty:
            start_price = idx_closes.iloc[0]
            bench_series = idx_closes / start_price - 1
            # Portföyün getiri tarihleriyle hizala (o tarihe kadarki en son bilinen değer)
            bench_df = pd.DataFrame({"cum_return": bench_series})
            bench_df = bench_df.reindex(bench_df.index.union(returns_df.index)).ffill().reindex(returns_df.index)

    return {
        "returns": returns_df,
        "benchmark": bench_df,
        "holdings_log": holdings_log,
        "total_return_pct": returns_df["cum_return"].iloc[-1] * 100,
        "benchmark_return_pct": (bench_df["cum_return"].iloc[-1] * 100) if bench_df is not None else None,
        "n_periods": len(returns_df),
    }
