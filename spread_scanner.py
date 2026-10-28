import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date
from scipy.stats import norm

st.set_page_config(page_title="Credit Spread Scanner", layout="wide")

# ---------------------------------------------------------
# BLACK-SCHOLES HELPER (FOR ACCURATE DELTA ESTIMATION)
# ---------------------------------------------------------
def calculate_put_delta(S, K, T, r, sigma):
    """Calculate Black-Scholes Delta for European put."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1) - 1.0)

# ---------------------------------------------------------
# TECHNICAL & VOLATILITY ANALYTICS
# ---------------------------------------------------------
def get_underlying_metrics(ticker_symbol):
    t = yf.Ticker(ticker_symbol)
    df = t.history(period="1y")
    if df.empty or len(df) < 50:
        return None, None

    close = df['Close']
    current_price = close.iloc[-1]

    # Realized Volatility (20-day annualized log returns)
    log_ret = np.log(close / close.shift(1))
    rv_20 = float(log_ret.tail(20).std() * np.sqrt(252))

    # Trend Indicators
    ema_21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
    ema_50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    ema_200 = close.ewm(span=200, adjust=False).mean().iloc[-1] if len(df) >= 200 else ema_50

    # ADX Calculation (14-period)
    high = df['High']
    low = df['Low']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).mean() / atr)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx_14 = float(dx.rolling(14).mean().iloc[-1])

    # Trend Alignment Score (ThinkScript logic)
    bull_macro = current_price > ema_200
    bull_swing = ema_21 > ema_50
    trend_score = 100 if (bull_macro and bull_swing) else (60 if bull_macro else 20)

    metrics = {
        "price": current_price,
        "rv_20": rv_20,
        "ema_21": ema_21,
        "ema_50": ema_50,
        "ema_200": ema_200,
        "adx_14": adx_14,
        "trend_score": trend_score,
        "ticker_obj": t
    }
    return metrics, df

# ---------------------------------------------------------
# SPREAD PRICING & SCANNER ENGINE
# ---------------------------------------------------------
def scan_ticker_spreads(ticker, min_dte, max_dte, target_delta, wing_width, min_credit):
    metrics, df = get_underlying_metrics(ticker)
    if not metrics:
        return []

    t = metrics["ticker_obj"]
    S = metrics["price"]
    rv = metrics["rv_20"]
    r = 0.045  # Benchmark risk-free interest rate

    expirations = t.options
    today = date.today()
    candidate_spreads = []

    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days

        if not (min_dte <= dte <= max_dte):
            continue

        try:
            chain = t.option_chain(exp_str)
            puts = chain.puts
        except Exception:
            continue

        if puts.empty:
            continue

        # Filter active liquid puts
        liquid_puts = puts[puts['openInterest'] > 50].copy()
        if liquid_puts.empty:
            continue

        # Estimate IV & Put Deltas
        T = dte / 365.0
        deltas = []
        for _, row in liquid_puts.iterrows():
            sigma = row['impliedVolatility']
            d = calculate_put_delta(S, row['strike'], T, r, sigma) if sigma > 0 else 0
            deltas.append(d)
        liquid_puts['delta'] = deltas

        # Identify candidate short puts near target Delta (e.g., -0.15 to -0.25)
        short_candidates = liquid_puts[
            (liquid_puts['delta'] <= -(target_delta - 0.05)) & 
            (liquid_puts['delta'] >= -(target_delta + 0.06)) &
            (liquid_puts['strike'] < S)
        ]

        for _, short_leg in short_candidates.iterrows():
            short_k = short_leg['strike']
            long_k = short_k - wing_width

            # Find matching long leg
            long_leg_match = liquid_puts[liquid_puts['strike'] == long_k]
            if long_leg_match.empty:
                continue
            long_leg = long_leg_match.iloc[0]

            # Midpoint pricing
            short_mid = (short_leg['bid'] + short_leg['ask']) / 2.0 if short_leg['bid'] > 0 else short_leg['lastPrice']
            long_mid = (long_leg['bid'] + long_leg['ask']) / 2.0 if long_leg['bid'] > 0 else long_leg['lastPrice']
            net_credit = round(short_mid - long_mid, 2)

            if net_credit < min_credit:
                continue

            max_risk = round(wing_width - net_credit, 2)
            roc = round((net_credit / max_risk) * 100, 1) if max_risk > 0 else 0
            cushion_pct = round(((S - short_k) / S) * 100, 1)

            # IV/RV Edge
            iv_avg = (short_leg['impliedVolatility'] + long_leg['impliedVolatility']) / 2.0
            iv_rv_ratio = round(iv_avg / rv, 2) if rv > 0 else 1.0

            candidate_spreads.append({
                "Ticker": ticker,
                "Price": round(S, 2),
                "Exp": exp_str,
                "DTE": dte,
                "Short Put": short_k,
                "Long Put": long_k,
                "Short Delta": round(short_leg['delta'], 2),
                "Credit ($)": net_credit,
                "Max Risk ($)": max_risk,
                "ROC (%)": roc,
                "Cushion (%)": cushion_pct,
                "IV/RV Ratio": iv_rv_ratio,
                "ADX": round(metrics["adx_14"], 1),
                "Trend": "Bullish" if metrics["trend_score"] == 100 else "Neutral/Chop"
            })

    return candidate_spreads

# ---------------------------------------------------------
# USER INTERFACE
# ---------------------------------------------------------
st.title("🛡️ Put Credit Spread Yield Scanner")
st.caption("Screens high-beta liquid underlyings for $10–$20 wide spreads meeting yield and delta thresholds.")

with st.sidebar:
    st.header("Scan Parameters")
    default_tickers = "AMD, SPY, QQQ, NVDA, TSLA, AMZN, META, AAPL, MSFT, IWM"
    ticker_input = st.text_area("Watchlist (comma-separated)", default_tickers)
    
    col_d1, col_d2 = st.columns(2)
    min_dte = col_d1.slider("Min DTE", 7, 25, 10)
    max_dte = col_d2.slider("Max DTE", 15, 60, 24)

    col_w1, col_w2 = st.columns(2)
    wing_width = col_w1.selectbox("Wing Width ($)", [5.0, 10.0, 15.0, 20.0], index=1)
    target_delta = col_w2.slider("Target Short Delta", 0.10, 0.35, 0.20, step=0.01)

    min_credit = st.number_input("Min Credit Required ($)", value=1.20, step=0.10)
    run_scan = st.button("Run Spread Scanner", type="primary")

if run_scan:
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    all_results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, ticker in enumerate(tickers):
        status_text.text(f"Scanning option chains for {ticker}...")
        try:
            res = scan_ticker_spreads(ticker, min_dte, max_dte, target_delta, wing_width, min_credit)
            all_results.extend(res)
        except Exception as e:
            st.warning(f"Error scanning {ticker}: {e}")
        progress_bar.progress((idx + 1) / len(tickers))

    status_text.empty()
    progress_bar.empty()

    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Sort by best ROC
        results_df = results_df.sort_values(by="ROC (%)", ascending=False).reset_index(drop=True)

        st.subheader(f"Found {len(results_df)} Setups Matching Yield Criteria")

        # Color-coded display styling
        def highlight_metrics(row):
            styles = [''] * len(row)
            if row['Credit ($)'] >= 2.00:
                styles[results_df.columns.get_loc('Credit ($)')] = 'background-color: #1b4d3e; color: #a3e635; font-weight: bold;'
            if row['ROC (%)'] >= 20.0:
                styles[results_df.columns.get_loc('ROC (%)')] = 'background-color: #1b4d3e; color: #a3e635; font-weight: bold;'
            if row['IV/RV Ratio'] >= 1.15:
                styles[results_df.columns.get_loc('IV/RV Ratio')] = 'color: #38bdf8;'
            return styles

        st.dataframe(
            results_df.style.apply(highlight_metrics, axis=1),
            use_container_width=True,
            column_config={
                "Price": st.column_config.NumberColumn(format="$%.2f"),
                "Short Put": st.column_config.NumberColumn(format="$%.1f"),
                "Long Put": st.column_config.NumberColumn(format="$%.1f"),
                "Credit ($)": st.column_config.NumberColumn(format="$%.2f"),
                "Max Risk ($)": st.column_config.NumberColumn(format="$%.2f"),
                "ROC (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "Cushion (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "Short Delta": st.column_config.NumberColumn(format="%.2f"),
            }
        )
    else:
        st.warning("No spreads met your minimum credit and delta requirements in the selected DTE range. Consider widening wings or adjusting delta.")
