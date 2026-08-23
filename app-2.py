"""
BIST100 Hisse Tarama ve Puanlama — Mobil Uyumlu Web Uygulaması
=================================================================
Çalıştırma: streamlit run app.py

Sekmeler:
  📊 Tarama         — ana tarama + sektör çeşitliliği + rejim ayarı
  📈 Geçmiş Performans — TEKNİK kriterlerin backtest'i (bkz. backtest.py sınırlaması)
  💼 Portföyüm      — kendi pozisyonlarınız, stop-loss kontrolü, risk özeti
  🕒 Tarama Geçmişi — geçmiş taramalarınızı kaydeder, kriter katkı analizi yapar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

import scanner_core as core
import portfolio_tools as ptools
import backtest as bt
import history_store as hs

st.set_page_config(page_title="BIST100 Tarama", page_icon="📊", layout="centered")

st.markdown(
    """
    <style>
    div.stButton > button { width: 100%; padding: 0.7em; font-size: 1.03em; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 BIST100 Hisse Tarama")

tab_scan, tab_backtest, tab_portfolio, tab_history = st.tabs(
    ["📊 Tarama", "📈 Geçmiş Perf.", "💼 Portföyüm", "🕒 Geçmişim"]
)


def fmt(v, decimals=1, suffix=""):
    try:
        if v is None or pd.isna(v):
            return "—"
        return f"{v:.{decimals}f}{suffix}"
    except Exception:
        return "—"


# ====================================================================
# SEKME 1: TARAMA
# ====================================================================
with tab_scan:
    st.caption("Teknik + temel kriterlere göre puanlanmış en iyi hisseler")

    with st.expander("⚙️ Ayarlar", expanded=False):
        top_n = st.slider("Kaç hisse listelensin", 5, 10, core.DEFAULT_TOP_N, key="scan_top_n")
        min_vol = st.number_input(
            "Minimum günlük hacim (TL)", min_value=0,
            value=core.DEFAULT_MIN_AVG_VOLUME_TRY, step=500_000, key="scan_min_vol",
        )
        apply_diversity = st.checkbox("Sektör çeşitliliği uygula (sektör başına max hisse)", value=True)
        max_per_sector = st.slider("Sektör başına maksimum hisse", 1, 5, core.DEFAULT_MAX_PER_SECTOR,
                                    disabled=not apply_diversity)
        apply_regime = st.checkbox(
            "Piyasa rejimine göre ağırlıkları otomatik ayarla",
            value=False,
            help="Açıksa: XU100 yükseliş trendindeyken momentum ağırlığı artar, "
                 "düşüş trendindeyken kalite/temettü ağırlığı artar.",
        )

    if st.button("▶ Taramayı Başlat", type="primary"):
        try:
            tickers = core.load_tickers()
        except Exception as e:
            st.error(f"Ticker listesi yüklenemedi: {e}")
            st.stop()

        progress_bar = st.progress(0, text="Başlatılıyor...")

        def progress_cb(i, total, ticker):
            progress_bar.progress(i / total, text=f"Çekiliyor: {ticker} ({i}/{total})")

        try:
            with st.spinner("Veriler çekiliyor, 1-2 dakika sürebilir..."):
                result = core.run_scan(
                    top_n=top_n,
                    min_avg_volume_try=min_vol,
                    tickers=tickers,
                    progress_callback=progress_cb,
                    apply_sector_diversity=apply_diversity,
                    max_per_sector=max_per_sector,
                    apply_regime_adjustment=apply_regime,
                )
            progress_bar.empty()
            st.session_state["scan_result"] = result
            st.session_state["scan_time"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.session_state["scan_top_n_used"] = top_n

            # Otomatik olarak geçmişe kaydet
            hs.append_scan(result["selected"], meta={
                "top_n": top_n, "min_vol": min_vol,
                "diversity": apply_diversity, "regime_adjusted": apply_regime,
            })
        except Exception as e:
            progress_bar.empty()
            st.error(f"Tarama başarısız oldu: {e}")
            st.stop()

    if "scan_result" in st.session_state:
        result = st.session_state["scan_result"]
        selected = result["selected"]
        top_n_used = st.session_state["scan_top_n_used"]

        st.success(f"Tamamlandı ({st.session_state['scan_time']}) — "
                   f"{len(result['full_ranked'])} hisse tarandı")

        if result.get("regime"):
            regime = result["regime"]
            icon = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(regime["regime"], "⚪")
            st.info(f"{icon} Piyasa rejimi: **{regime['regime']}** — {regime['note']}")

        if result.get("diversity_relaxed"):
            st.warning("Sektör çeşitliliği kısıtı tam uygulanamadı (yeterli farklı sektörde "
                       "hisse yok), bazı slotlar kısıt gevşetilerek dolduruldu.")

        for rank, (ticker, row) in enumerate(selected.iterrows(), 1):
            quality = row.get("data_quality", np.nan)
            quality_badge = ""
            if not pd.isna(quality) and quality < 60:
                quality_badge = " ⚠️ düşük veri güvenilirliği"

            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"**{rank}. {ticker.replace('.IS', '')}** · {row.get('sector', '')}"
                            f"{quality_badge}")
                c2.markdown(f"**{fmt(row.get('final_score'))}/100**")

                m1, m2, m3 = st.columns(3)
                m1.metric("Fiyat", fmt(row.get("last_price"), 2, " TL"))
                m2.metric("F/K", fmt(row.get("pe_ratio")))
                m3.metric("ROE", fmt(row.get("roe"), 1, "%"))

                m4, m5, m6 = st.columns(3)
                m4.metric("Kâr Büyüme", fmt(row.get("profit_growth"), 1, "%"))
                m5.metric("Borç/Özkaynak", fmt(row.get("debt_to_equity"), 1))
                m6.metric("3A Momentum", fmt(row.get("momentum_3m"), 1, "%"))

                if not pd.isna(quality):
                    st.caption(f"Veri güvenilirliği: %{quality:.0f}")

        st.divider()

        # Seçilen hisselerin son 1 yıllık fiyat performansı vs XU100
        st.subheader("📉 Son 1 Yıl: Seçilenler vs XU100")
        st.caption(
            "Bu, seçilen hisselerin GEÇMİŞTEKİ fiyat performansıdır — stratejinin "
            "geleceği tahmin ettiği anlamına gelmez. Stratejinin geçmiş test sonucu "
            "için 'Geçmiş Performans' sekmesine bakın."
        )
        try:
            chart_data = {}
            for ticker in selected.index:
                udata = result["universe"].get(ticker)
                if udata and udata.hist is not None and len(udata.hist) > 5:
                    closes = udata.hist["Close"]
                    chart_data[ticker.replace(".IS", "")] = closes / closes.iloc[0] * 100
            if result.get("index_hist") is not None and not result["index_hist"].empty:
                idx_closes = result["index_hist"]["Close"]
                chart_data["XU100"] = idx_closes / idx_closes.iloc[0] * 100
            if chart_data:
                chart_df = pd.DataFrame(chart_data)
                st.line_chart(chart_df)
            else:
                st.caption("Grafik için yeterli veri yok.")
        except Exception as e:
            st.caption(f"Grafik oluşturulamadı: {e}")

        with st.expander("Tüm taranan hisseler (tam tablo)"):
            st.dataframe(result["full_ranked"], use_container_width=True)

        csv = result["full_ranked"].to_csv(encoding="utf-8-sig")
        st.download_button("⬇ CSV Olarak İndir", data=csv,
                            file_name=f"bist_tarama_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv")
    else:
        st.info("Taramayı başlatmak için yukarıdaki butona basın.")


# ====================================================================
# SEKME 2: GEÇMİŞ PERFORMANS (BACKTEST)
# ====================================================================
with tab_backtest:
    st.subheader("📈 Geçmiş Performans Testi (Backtest)")
    st.warning(
        "**Önemli sınırlama:** Bu backtest SADECE teknik kriterleri (momentum, "
        "trend, RSI, hacim) kullanır — F/K, ROE gibi temel kriterler dahil DEĞİLDİR. "
        "Neden: geçmişteki 'o günkü' F/K, ROE gibi verileri ücretsiz ve güvenilir "
        "şekilde çekebileceğim bir kaynak yok; bugünkü değerleri geçmişe uygulamak "
        "yapay olarak iyi sonuç gösterir (ileriye bakma yanlılığı). Yani bu, "
        "gerçek uygulamada kullandığınız tam stratejinin geçmiş testi DEĞİL, "
        "sadece teknik bacağın testi."
    )

    bt_period = st.selectbox("Test edilecek geçmiş süre", ["2y", "3y", "5y"], index=1)
    bt_top_n = st.slider("Dönem başına kaç hisse tutulsun", 5, 10, 10, key="bt_top_n")
    bt_freq = st.radio("Rebalance sıklığı", ["Aylık", "Çeyreklik"], horizontal=True)
    freq_code = "M" if bt_freq == "Aylık" else "Q"

    if st.button("▶ Backtest'i Çalıştır", type="primary"):
        try:
            tickers = core.load_tickers()
        except Exception as e:
            st.error(f"Ticker listesi yüklenemedi: {e}")
            st.stop()

        progress_bar = st.progress(0, text="Geçmiş veriler çekiliyor...")

        def bt_progress_cb(i, total, ticker):
            progress_bar.progress(i / total, text=f"Çekiliyor: {ticker} ({i}/{total})")

        try:
            with st.spinner(f"Son {bt_period} için veri çekiliyor, birkaç dakika sürebilir..."):
                histories = bt.fetch_backtest_histories(tickers, period=bt_period,
                                                          progress_callback=bt_progress_cb)
                index_hist = core.fetch_index_history(period=bt_period)
            progress_bar.progress(1.0, text="Backtest hesaplanıyor...")

            result_bt = bt.run_backtest(histories, index_hist, top_n=bt_top_n, rebalance_freq=freq_code)
            progress_bar.empty()
            st.session_state["bt_result"] = result_bt
        except Exception as e:
            progress_bar.empty()
            st.error(f"Backtest başarısız oldu: {e}")
            st.stop()

    if "bt_result" in st.session_state:
        r = st.session_state["bt_result"]
        c1, c2 = st.columns(2)
        c1.metric("Strateji (teknik) toplam getiri", fmt(r["total_return_pct"], 1, "%"))
        c2.metric("XU100 (al-tut) toplam getiri", fmt(r.get("benchmark_return_pct"), 1, "%")
                  if r.get("benchmark_return_pct") is not None else "—")

        chart_df = pd.DataFrame({"Strateji (teknik)": r["returns"]["cum_return"] * 100})
        if r.get("benchmark") is not None:
            chart_df["XU100"] = r["benchmark"]["cum_return"] * 100
        st.line_chart(chart_df)

        with st.expander("Dönem dönem tutulan hisseler"):
            for h in r["holdings_log"]:
                st.caption(f"{h['date'].date()}: {', '.join(t.replace('.IS','') for t in h['holdings'])}")


# ====================================================================
# SEKME 3: PORTFÖYÜM
# ====================================================================
with tab_portfolio:
    st.subheader("💼 Portföyüm — Risk ve Stop-Loss Kontrolü")
    st.caption("Kendi pozisyonlarınızı girin; güncel fiyatla stop-loss ihlali ve "
               "sektör yoğunlaşmasını kontrol edelim.")

    if "positions_df" not in st.session_state:
        st.session_state["positions_df"] = pd.DataFrame({
            "ticker": ["THYAO", "GARAN"],
            "adet": [10, 20],
            "maliyet": [250.0, 90.0],
            "stop_loss_yuzde": [10.0, 10.0],
        })

    edited = st.data_editor(
        st.session_state["positions_df"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "ticker": st.column_config.TextColumn("Hisse Kodu"),
            "adet": st.column_config.NumberColumn("Adet", min_value=0),
            "maliyet": st.column_config.NumberColumn("Maliyet (TL)", min_value=0.0),
            "stop_loss_yuzde": st.column_config.NumberColumn("Stop-Loss %", min_value=0.0),
        },
    )
    st.session_state["positions_df"] = edited

    if st.button("🔍 Portföyü Analiz Et", type="primary"):
        positions = edited.dropna(subset=["ticker"])
        positions = positions[positions["ticker"].str.strip() != ""]
        if positions.empty:
            st.warning("En az bir pozisyon girin.")
        else:
            with st.spinner("Güncel fiyatlar çekiliyor..."):
                try:
                    checked = ptools.check_positions(positions)
                    st.session_state["checked_positions"] = checked
                except Exception as e:
                    st.error(f"Analiz başarısız: {e}")

    if "checked_positions" in st.session_state:
        checked = st.session_state["checked_positions"]

        breaches = checked[checked["stop_ihlali"] == True]
        if not breaches.empty:
            st.error(f"🚨 {len(breaches)} pozisyonda stop-loss ihlali: "
                     f"{', '.join(breaches['ticker'])}")
        else:
            st.success("Stop-loss ihlali yok.")

        for _, row in checked.iterrows():
            with st.container(border=True):
                st.markdown(f"**{row['ticker'].replace('.IS', '')}**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Güncel Fiyat", fmt(row.get("guncel_fiyat"), 2, " TL"))
                c2.metric("Maliyet", fmt(row.get("maliyet"), 2, " TL"))
                pnl = row.get("kar_zarar_yuzde")
                c3.metric("Kâr/Zarar", fmt(pnl, 1, "%"),
                          delta=fmt(pnl, 1, "%") if pnl is not None and not pd.isna(pnl) else None)
                if row.get("stop_ihlali"):
                    st.caption("🚨 Stop-loss seviyesi aşıldı")

        st.divider()
        st.subheader("Sektör Dağılımı")
        try:
            with st.spinner("Sektör bilgileri çekiliyor..."):
                sector_dist = ptools.portfolio_sector_breakdown(checked)
            st.bar_chart(sector_dist)
        except Exception as e:
            st.caption(f"Sektör dağılımı hesaplanamadı: {e}")

        if "scan_result" in st.session_state:
            recommended = list(st.session_state["scan_result"]["selected"].index)
            overlap = ptools.overlap_with_recommendations(checked, recommended)
            st.subheader("Güncel Tarama Önerileriyle Örtüşme")
            st.write(f"**Ortak:** {', '.join(t.replace('.IS','') for t in overlap['ortak']) or '—'}")
            st.write(f"**Sadece portföyünüzde:** "
                     f"{', '.join(t.replace('.IS','') for t in overlap['sadece_portfoyde']) or '—'}")
            st.write(f"**Sadece önerilerde:** "
                     f"{', '.join(t.replace('.IS','') for t in overlap['sadece_onerilerde']) or '—'}")


# ====================================================================
# SEKME 4: TARAMA GEÇMİŞİ
# ====================================================================
with tab_history:
    st.subheader("🕒 Tarama Geçmişi ve Kriter Katkı Analizi")
    st.warning(
        "Bu uygulama ücretsiz bulutta çalışıyorsa dosya sistemi **kalıcı değildir** "
        "— uygulama yeniden başladığında geçmiş silinebilir. Düzenli olarak "
        "**'Geçmişi İndir'** ile yedek alıp, açılışta **'Geçmiş Yükle'** ile geri "
        "yüklemenizi öneririm."
    )

    history = hs.load_history()

    uploaded = st.file_uploader("Geçmiş Yükle (.json)", type=["json"])
    if uploaded is not None:
        try:
            history = hs.merge_uploaded_history(uploaded.read())
            st.success(f"Geçmiş birleştirildi. Toplam {len(history)} tarama kaydı var.")
        except Exception as e:
            st.error(f"Yükleme başarısız: {e}")

    if history:
        st.write(f"Kayıtlı tarama sayısı: **{len(history)}**")
        flat = hs.history_to_dataframe(history)
        with st.expander("Tüm geçmiş kayıtlar"):
            st.dataframe(flat, use_container_width=True)

        import json as _json
        history_json = _json.dumps(history, ensure_ascii=False, indent=2, default=str)
        st.download_button("⬇ Geçmişi İndir (.json)", data=history_json,
                            file_name=f"tarama_gecmisi_{datetime.now().strftime('%Y%m%d')}.json",
                            mime="application/json")

        st.divider()
        st.subheader("Kriter Katkı Analizi")
        st.caption(
            "Hangi kriterin gerçekten ileri getiriyle ilişkili olduğunu gösterir. "
            "**En az birkaç ay/tarama birikmeden bu sayılar güvenilir değildir** — "
            "az örneklemde tesadüfen yüksek/düşük çıkabilir."
        )

        if len(history) < 2:
            st.info("Analiz için en az 2 tarama gerekir (farklı tarihlerde).")
        else:
            if st.button("Katkı Analizini Çalıştır"):
                def price_fetcher(ticker, date):
                    data = core.fetch_stock_data(ticker, period="1y")
                    if data.hist is None or data.hist.empty:
                        return None
                    try:
                        target = pd.Timestamp(date)
                        sub = data.hist[data.hist.index <= target]
                        if sub.empty:
                            return None
                        return float(sub["Close"].iloc[-1])
                    except Exception:
                        return None

                with st.spinner("Geçmiş fiyatlar çekilip ilişki hesaplanıyor..."):
                    fwd = hs.compute_forward_returns(history, price_fetcher)
                    corr = hs.criterion_contribution(fwd)

                if corr.empty:
                    st.info("Yeterli örneklem yok (her kriter için en az 5 gözlem gerekir).")
                else:
                    st.bar_chart(corr)
                    st.caption("Pozitif = o kriterde yüksek puan alanlar daha çok kazanmış. "
                               "Negatif = tam tersi (kriter işe yaramıyor veya ters etki ediyor olabilir).")
    else:
        st.info("Henüz kayıtlı tarama yok. 'Tarama' sekmesinde bir tarama çalıştırınca "
                "otomatik olarak buraya kaydedilecek.")
