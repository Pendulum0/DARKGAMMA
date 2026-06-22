"""
DARKGAMMA — Gamma Exposure / Options Greeks dashboard
Data: Yahoo Finance (yfinance). Greeks: Black-Scholes (computed locally).
Single-file Streamlit app. Replace your existing app.py with this.

Run:  streamlit run app.py
Deps: pip install streamlit yfinance plotly pandas numpy scipy
"""

import datetime as dt
import math
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG / THEME
# ----------------------------------------------------------------------------
st.set_page_config(page_title="DARKGAMMA", page_icon="", layout="wide",
                   initial_sidebar_state="collapsed")

RISK_FREE = 0.043           # annualized risk-free rate assumption (edit as needed)
CONTRACT_MULT = 100         # shares per option contract
MAX_DTE_BUTTONS = 5         # 0DTE .. 5DTE
ET = dt.timezone(dt.timedelta(hours=-4))  # Eastern (EDT). Switch to -5 for EST.

# ----------------------------------------------------------------------------
# THEME SYSTEM — user-customizable from the Customize tab (stored in session)
# ----------------------------------------------------------------------------
def _hx(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
def _lighten(h, f):
    r, g, b = _hx(h)
    return f"#{int(r+(255-r)*f):02x}{int(g+(255-g)*f):02x}{int(b+(255-b)*f):02x}"
def _rgba(h, a):
    r, g, b = _hx(h)
    return f"rgba({r},{g},{b},{a})"

THEMES = {
    "Midnight":  dict(g="#00e08a", r="#ff4d5e", y="#ffb01f", bg="#000000", panel="#0d0d0d", grid="#222222", txt="#e6e6e6", dim="#7a7a7a", accent="#e6e6e6"),
    "Synthwave": dict(g="#2ee6c8", r="#ff3d8b", y="#ffe14d", bg="#160d2b", panel="#211543", grid="#3a2566", txt="#efe6ff", dim="#9a82c8", accent="#ff3d8b"),
    "Bubblegum": dict(g="#5fe0b0", r="#ff7eb6", y="#ffd56b", bg="#1c1430", panel="#2a1f45", grid="#3e2f5e", txt="#fdeef8", dim="#ad93cf", accent="#82e9ff"),
    "Ember":     dict(g="#9ad84f", r="#ff5c7a", y="#ffb347", bg="#120c0a", panel="#1d1512", grid="#33231c", txt="#f6e7d6", dim="#8d7565", accent="#ff8a3c"),
    "Solar":     dict(g="#9aa800", r="#e0483f", y="#caa400", bg="#04303b", panel="#0a3d49", grid="#13525f", txt="#eee8d5", dim="#6f8a92", accent="#2aa6d6"),
    "Mono":      dict(g="#f2f2f2", r="#7c7c7c", y="#b8b8b8", bg="#000000", panel="#0d0d0d", grid="#222222", txt="#e6e6e6", dim="#7a7a7a", accent="#bdbdbd"),
}
FONTS = {
    "Mono":    "'JetBrains Mono','SF Mono','Menlo',monospace",
    "Rounded": "'Quicksand',system-ui,sans-serif",
    "Grotesk": "'Space Grotesk',system-ui,sans-serif",
    "Techno":  "'Orbitron',system-ui,sans-serif",
}
BOX_STYLES = ["Sharp", "Bubbly", "Glass"]

st.session_state.setdefault("dg_theme", "Midnight")
st.session_state.setdefault("dg_font", "Mono")
st.session_state.setdefault("dg_box", "Sharp")
_t = THEMES.get(st.session_state["dg_theme"], THEMES["Midnight"])
_font = FONTS.get(st.session_state["dg_font"], FONTS["Mono"])
_box = st.session_state["dg_box"] if st.session_state["dg_box"] in BOX_STYLES else "Sharp"

C_GREEN, C_RED, C_YELL = _t["g"], _t["r"], _t["y"]
C_BG, C_PANEL, C_GRID = _t["bg"], _t["panel"], _t["grid"]
C_TXT, C_DIM, C_ACCENT = _t["txt"], _t["dim"], _t["accent"]

if _box == "Bubbly":
    CARD = (f"background:linear-gradient(145deg,{_lighten(C_PANEL,0.07)},{C_PANEL});"
            f"border:1px solid {_rgba(C_ACCENT,0.20)};border-radius:20px;"
            f"box-shadow:0 6px 22px {_rgba(C_ACCENT,0.10)},inset 0 1px 0 {_rgba('#ffffff',0.05)};")
    TAB_RADIUS = "16px"
elif _box == "Glass":
    CARD = (f"background:{_rgba(C_PANEL,0.45)};border:1px solid {_rgba(C_TXT,0.12)};"
            f"border-radius:16px;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);"
            f"box-shadow:0 4px 18px rgba(0,0,0,0.35);")
    TAB_RADIUS = "14px"
else:
    CARD = f"background:{C_PANEL};border:1px solid {C_GRID};border-radius:8px;box-shadow:none;"
    TAB_RADIUS = "6px"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;500;700&family=Space+Grotesk:wght@400;500;700&family=Orbitron:wght@500;700&display=swap');
.stApp {{ background:{C_BG}; }}
#MainMenu, header, footer {{ visibility:hidden; }}
.block-container {{ padding-top:0.5rem; padding-bottom:3rem; padding-left:3rem; padding-right:3rem; max-width:1900px; }}
* {{ font-family:{_font}; }}
code, pre, [data-testid="stCode"], [data-testid="stCode"] * {{ font-family:'JetBrains Mono','SF Mono',monospace !important; }}

.gx-top {{ display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid {C_GRID}; padding:6px 10px 10px 4px; }}
.gx-logo {{ font-size:20px; font-weight:700; letter-spacing:2px; color:{C_TXT}; }}
.gx-logo span {{ color:{C_ACCENT}; }}
.gx-live {{ color:{C_ACCENT}; font-size:12px; font-weight:700; letter-spacing:1px; }}
.gx-clock {{ color:{C_DIM}; font-size:12px; border:1px solid {C_GRID}; padding:3px 8px; border-radius:4px; margin-left:8px; }}
.gx-disc {{ color:{C_DIM}; opacity:0.55; font-size:9px; text-align:center; letter-spacing:1px; padding:3px 0; border-bottom:1px solid {C_GRID}; }}
.gx-news {{ color:{C_DIM}; font-size:11px; padding:6px 4px; border-bottom:1px solid {C_GRID}; white-space:nowrap; overflow:hidden; }}
.gx-news b {{ color:{C_TXT}; }}
.gx-news .tag {{ color:{C_ACCENT}; border:1px solid {C_GRID}; padding:1px 5px; border-radius:3px; }}

.gx-cardrow {{ display:flex; gap:12px; margin:12px 0; }}
.gx-card, .dg-box {{ {CARD} }}
.gx-card {{ flex:1; padding:14px 18px; }}
.gx-card .lbl {{ font-size:10px; letter-spacing:1.5px; color:{C_DIM}; }}
.gx-card .dot {{ font-size:9px; margin-right:5px; }}
.gx-card .val {{ font-size:26px; font-weight:700; margin:4px 0 2px; }}
.gx-card .sub {{ font-size:10px; color:{C_DIM}; }}
.dg-box {{ padding:9px 14px; }}

[data-baseweb="tab-list"] {{ gap:5px; justify-content:center; background:transparent; }}
[data-baseweb="tab"] {{ background:{C_PANEL}; border:1px solid {C_GRID}; border-radius:{TAB_RADIUS}; color:{C_DIM}; font-size:13px; padding:6px 16px; }}
[aria-selected="true"][data-baseweb="tab"] {{ color:{C_ACCENT}; border-color:{_rgba(C_ACCENT,0.4)}; }}

div[data-testid="stMetric"] {{ {CARD} padding:10px 16px; }}
.stSelectbox label, .stRadio label {{ color:{C_DIM}; font-size:11px; }}
hr {{ border-color:{C_GRID}; }}
.panel-title {{ font-size:11px; letter-spacing:1px; color:{C_DIM}; margin-bottom:2px; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor=C_BG, plot_bgcolor=C_BG,
    font=dict(color=C_TXT, size=11, family="monospace"),
    margin=dict(l=12, r=12, t=44, b=12),
    xaxis=dict(gridcolor=C_GRID, zerolinecolor=C_GRID, color=C_DIM),
    yaxis=dict(gridcolor=C_GRID, zerolinecolor=C_GRID, color=C_DIM),
    showlegend=False,
    hovermode="closest",
    hoverlabel=dict(bgcolor=C_PANEL, bordercolor=C_GRID,
                    font=dict(color=C_TXT, size=12, family="monospace")),
)

# ----------------------------------------------------------------------------
# BLACK-SCHOLES GREEKS (vectorized)
# ----------------------------------------------------------------------------
def bs_greeks(S, K, T, r, sigma, is_call):
    """Return per-share greeks. T in years. Arrays supported."""
    K = np.asarray(K, float); sigma = np.asarray(sigma, float)
    T = np.maximum(np.asarray(T, float), 1.0 / (365 * 24 * 60))   # >= 1 min
    sigma = np.clip(sigma, 1e-4, 5.0)
    sqT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT
    pdf = norm.pdf(d1)
    Nd1, Nd2 = norm.cdf(d1), norm.cdf(d2)

    gamma = pdf / (S * sigma * sqT)
    vega  = S * pdf * sqT                      # per 1.00 vol
    speed = -gamma / S * (d1 / (sigma * sqT) + 1)
    zomma = gamma * (d1 * d2 - 1) / sigma
    vanna = -pdf * d2 / sigma
    color = -pdf / (2 * S * T * sigma * sqT) * (1 + d1 * (2 * r * T - d2 * sigma * sqT) / (sigma * sqT))

    call = np.asarray(is_call, bool)
    delta = np.where(call, Nd1, Nd1 - 1.0)
    theta = np.where(
        call,
        -S * pdf * sigma / (2 * sqT) - r * K * np.exp(-r * T) * Nd2,
        -S * pdf * sigma / (2 * sqT) + r * K * np.exp(-r * T) * norm.cdf(-d2),
    )
    rho = np.where(call,
                   K * T * np.exp(-r * T) * Nd2,
                   -K * T * np.exp(-r * T) * norm.cdf(-d2))
    charm = np.where(
        call,
        -pdf * (2 * r * T - d2 * sigma * sqT) / (2 * T * sigma * sqT),
        -pdf * (2 * r * T - d2 * sigma * sqT) / (2 * T * sigma * sqT),
    )
    return dict(delta=delta, gamma=gamma, vega=vega / 100, theta=theta / 365,
                rho=rho / 100, speed=speed, zomma=zomma, vanna=vanna,
                charm=charm / 365, color=color / 365)

# greek -> column key used for exposure
GREEK_MAP = {
    "GEX": "gamma", "DEX": "delta", "VEX": "vega", "CHEX": "charm",
    "THETA": "theta", "VEGA": "vega", "RHO": "rho",
    "SPEED": "speed", "ZONNA": "zomma", "COLOR": "color",
}
# greeks that scale by S^2 (dollar-gamma family) vs S (dollar-delta family)
SECOND_ORDER = {"gamma", "speed", "zomma", "color"}

# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
@st.cache_data(ttl=120, show_spinner=False)
def get_spot(ticker):
    tk = yf.Ticker(ticker)
    try:
        p = tk.fast_info["last_price"]
        if p: return float(p)
    except Exception:
        pass
    h = tk.history(period="1d", interval="1m")
    return float(h["Close"].iloc[-1])

@st.cache_data(ttl=120, show_spinner=False)
def get_expirations(ticker):
    return list(yf.Ticker(ticker).options)

@st.cache_data(ttl=120, show_spinner=False)
def get_chain(ticker, expiry):
    """Raw calls+puts for one expiry as a tagged DataFrame."""
    ch = yf.Ticker(ticker).option_chain(expiry)
    calls = ch.calls.copy(); calls["type"] = "C"
    puts  = ch.puts.copy();  puts["type"]  = "P"
    df = pd.concat([calls, puts], ignore_index=True)
    df["expiry"] = expiry
    return df

def build_dataset(ticker, expiries, spot):
    """Combine expiries, compute greeks + exposures per strike/type."""
    frames = []
    today = dt.datetime.now(ET).date()
    for exp in expiries:
        try:
            df = get_chain(ticker, exp)
        except Exception:
            continue
        if df.empty:
            continue
        exp_date = dt.datetime.strptime(exp, "%Y-%m-%d").date()
        dte = max((exp_date - today).days, 0)
        T = max(dte, 0.5) / 365.0
        iv = df["impliedVolatility"].fillna(0).clip(lower=0).values
        iv = np.where(iv <= 0.01, 0.30, iv)          # fallback IV
        is_call = (df["type"] == "C").values
        g = bs_greeks(spot, df["strike"].values, T, RISK_FREE, iv, is_call)
        for k, v in g.items():
            df[k] = v
        df["dte"] = dte
        df["oi"] = df["openInterest"].fillna(0).values
        df["vol"] = df["volume"].fillna(0).values
        df["mid"] = ((df["bid"].fillna(0) + df["ask"].fillna(0)) / 2).where(
            (df["bid"] > 0) & (df["ask"] > 0), df["lastPrice"]).fillna(0)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def exposure(df, greek_key, spot, basis="oi"):
    """Signed dollar exposure per row. Calls +, Puts - (dealer convention)."""
    qty = df[basis].values
    sign = np.where(df["type"].values == "C", 1.0, -1.0)
    scale = spot * spot * 0.01 if greek_key in SECOND_ORDER else spot
    return df[greek_key].values * qty * CONTRACT_MULT * scale * sign

def per_strike(df, greek_key, spot, basis="oi"):
    e = exposure(df, greek_key, spot, basis)
    tmp = df.assign(_exp=e)
    g = tmp.groupby("strike").apply(
        lambda x: pd.Series({
            "call": x.loc[x["type"] == "C", "_exp"].sum(),
            "put":  x.loc[x["type"] == "P", "_exp"].sum(),
        })
    ).reset_index()
    g["net"] = g["call"] + g["put"]
    return g.sort_values("strike")

def find_levels(df, spot):
    g = per_strike(df, "gamma", spot)
    if g.empty:
        return dict(call_wall=spot, put_wall=spot, flip=spot, max_pain=spot)
    call_wall = g.loc[g["call"].idxmax(), "strike"] if g["call"].max() > 0 else spot
    put_wall  = g.loc[g["put"].idxmin(),  "strike"] if g["put"].min() < 0 else spot
    # gamma flip: cumulative net gex zero-cross
    g2 = g.sort_values("strike")
    cum = g2["net"].cumsum().values
    flip = spot
    for i in range(1, len(cum)):
        if cum[i - 1] <= 0 < cum[i] or cum[i - 1] >= 0 > cum[i]:
            flip = g2["strike"].values[i]; break
    # max pain
    strikes = np.sort(df["strike"].unique())
    oi_c = df[df["type"] == "C"].groupby("strike")["oi"].sum()
    oi_p = df[df["type"] == "P"].groupby("strike")["oi"].sum()
    pains = []
    for K in strikes:
        c = sum(oi_c.get(s, 0) * max(K - s, 0) for s in strikes)
        p = sum(oi_p.get(s, 0) * max(s - K, 0) for s in strikes)
        pains.append(c + p)
    max_pain = strikes[int(np.argmin(pains))] if len(strikes) else spot
    return dict(call_wall=call_wall, put_wall=put_wall, flip=flip, max_pain=max_pain)

def fmt_b(x):
    a = abs(x)
    if a >= 1e9:  return f"{x/1e9:+.2f}B"
    if a >= 1e6:  return f"{x/1e6:+.0f}M"
    if a >= 1e3:  return f"{x/1e3:+.0f}K"
    return f"{x:+.0f}"

# ----------------------------------------------------------------------------
# TOP BAR
# ----------------------------------------------------------------------------
now = dt.datetime.now(ET).strftime("%H:%M:%S ET")
c1, c2, c3 = st.columns([3, 5, 3])
with c1:
    st.markdown('<div class="gx-logo">🔳 DARK<span>GAMMA</span></div>', unsafe_allow_html=True)
with c2:
    ticker = st.text_input("ticker", "QQQ", label_visibility="collapsed").upper().strip()
with c3:
    st.markdown(f'<div style="text-align:right;"><span class="gx-live">● LIVE</span>'
                f'<span class="gx-clock">{now}</span></div>', unsafe_allow_html=True)

st.markdown('<div class="gx-disc">FOR INFORMATIONAL AND EDUCATIONAL PURPOSES ONLY · NOT '
            'FINANCIAL ADVICE · PAST PERFORMANCE DOES NOT GUARANTEE FUTURE RESULTS · '
            'ALWAYS DO YOUR OWN RESEARCH</div>', unsafe_allow_html=True)

# DTE selector
dte_choice = st.radio("dte", ["0DTE", "1DTE", "2DTE", "3DTE", "4DTE", "5DTE", "ADTE"],
                      horizontal=True, label_visibility="collapsed", index=0)

# ----------------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------------
try:
    spot = get_spot(ticker)
    all_exp = get_expirations(ticker)
except Exception as e:
    st.error(f"Could not load {ticker} from Yahoo Finance: {e}")
    st.stop()

if not all_exp:
    st.error(f"No options listed for {ticker}.")
    st.stop()

if dte_choice == "ADTE":
    use_exp = all_exp[:MAX_DTE_BUTTONS + 1]
    exp_label = f"AGGREGATE ({len(use_exp)} EXP)"
else:
    idx = int(dte_choice[0])
    use_exp = [all_exp[min(idx, len(all_exp) - 1)]]
    exp_label = dte_choice

with st.spinner("Loading chain…"):
    data = build_dataset(ticker, use_exp, spot)

if data.empty:
    st.error("No option data returned for this selection.")
    st.stop()

lv = find_levels(data, spot)
net_gex = per_strike(data, "gamma", spot)["net"].sum()
total_prem = (data["mid"] * data["oi"] * CONTRACT_MULT).sum()

# ----------------------------------------------------------------------------
# METRIC CARDS
# ----------------------------------------------------------------------------
st.markdown(f'<div style="color:{C_DIM};font-size:12px;letter-spacing:1px;margin-top:8px;">'
            f'{ticker} · {exp_label}</div>', unsafe_allow_html=True)

cards = [
    ("● SPOT", f"${spot:,.2f}", "Current reference level", C_TXT, C_DIM),
    ("● CALL WALL", f"${lv['call_wall']:,.2f}", "Upside gamma structure", C_GREEN, C_GREEN),
    ("● PUT WALL", f"${lv['put_wall']:,.2f}", "Downside gamma structure", C_RED, C_RED),
    ("● NET GEX", fmt_b(net_gex), "Aggregate dealer exposure",
     C_GREEN if net_gex >= 0 else C_RED, C_GREEN if net_gex >= 0 else C_RED),
    ("● GAMMA FLIP", f"${lv['flip']:,.2f}", "Regime transition level", C_YELL, C_YELL),
]
html = '<div class="gx-cardrow">'
for lbl, val, sub, vc, dc in cards:
    html += (f'<div class="gx-card"><div class="lbl"><span class="dot" style="color:{dc}">●</span>'
             f'{lbl[2:]}</div><div class="val" style="color:{vc}">{val}</div>'
             f'<div class="sub">{sub}</div></div>')
html += '</div>'
st.markdown(html, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# CONFLUENCE ENGINE — score every strike across all signals, rank strongest first
# ----------------------------------------------------------------------------
def confluence_table(data, spot, lv, top_n=8):
    g  = per_strike(data, "gamma", spot).set_index("strike")
    d  = per_strike(data, "delta", spot).set_index("strike")
    vn = per_strike(data, "vanna", spot).set_index("strike")
    ch = per_strike(data, "charm", spot).set_index("strike")
    oi   = data.groupby("strike")["oi"].sum()
    vol  = data.groupby("strike")["vol"].sum()
    tmp  = data.assign(_p=data["mid"] * data["oi"] * CONTRACT_MULT)
    prem = tmp.groupby("strike")["_p"].sum()

    f = pd.DataFrame(index=g.index)
    f["gex"]   = g["net"].abs()
    f["call_g"] = g["call"]
    f["put_g"]  = g["put"].abs()
    f["dex"]   = d["net"].abs()
    f["vanna"] = vn["net"].abs()
    f["charm"] = ch["net"].abs()
    f["oi"]    = oi
    f["vol"]   = vol
    f["prem"]  = prem
    f = f.fillna(0.0)
    if f.empty:
        return pd.DataFrame()

    def norm(col):
        m = f[col].max()
        return f[col] / m if m > 0 else f[col] * 0.0

    # normalized signal components (0..1)
    comp = pd.DataFrame({
        "γ":    norm("gex"),
        "OI":   norm("oi"),
        "VOL":  norm("vol"),
        "PREM": norm("prem"),
        "DEX":  norm("dex"),
        "VAN":  norm("vanna"),
        "CHM":  norm("charm"),
    }, index=f.index)

    weights = {"γ": .30, "OI": .20, "VOL": .15, "PREM": .12, "DEX": .10, "VAN": .07, "CHM": .06}
    base = sum(comp[k] * w for k, w in weights.items())

    # ALIGNMENT: reward strikes where many independent signals are simultaneously hot
    hot_mask = comp[list(weights)] > 0.45
    breadth = hot_mask.sum(axis=1)
    align_mult = 1.0 + 0.14 * breadth          # the "everything lines up" multiplier

    # mild proximity preference (actionable intraday), capped
    dist = (f.index.to_series() - spot).abs()
    prox = 1.0 - (dist / (0.06 * spot)).clip(0, 1) * 0.30

    f["score"] = base * align_mult * prox
    f["breadth"] = breadth
    f["hot"] = [[k for k in weights if hot_mask.loc[s, k]] for s in f.index]

    # structural tags
    def tags(s):
        t = []
        if abs(s - lv["call_wall"]) < 1e-6: t.append("Call Wall")
        if abs(s - lv["put_wall"])  < 1e-6: t.append("Put Wall")
        if abs(s - lv["flip"])      < 1e-6: t.append("γ Flip")
        if abs(s - lv["max_pain"])  < 1e-6: t.append("Max Pain")
        if s == f["oi"].idxmax():  t.append("Peak OI")
        if s == f["vol"].idxmax(): t.append("Peak Vol")
        return t
    f["tags"] = [tags(s) for s in f.index]
    # structural levels get a flat strength bump so they always surface
    f["score"] += f["tags"].apply(lambda t: 0.18 if t else 0.0)

    f = f.sort_values("score", ascending=False)

    # cluster adjacent strikes into one "zone" (keep the strongest representative)
    cluster = max(spot * 0.003, 0.5)
    picked, used = [], []
    for s in f.index:
        if any(abs(s - u) <= cluster for u in used):
            continue
        used.append(s); picked.append(s)
        if len(picked) >= top_n:
            break
    out = f.loc[picked].copy()
    out["strike"] = out.index
    # role
    def role(s):
        if s > spot + 0.25 * cluster: return "RESISTANCE"
        if s < spot - 0.25 * cluster: return "SUPPORT"
        return "PIVOT"
    out["role"] = [role(s) for s in out.index]
    # rescale score to 0..100 for display
    mx = out["score"].max()
    out["pct"] = (out["score"] / mx * 100).round(0) if mx > 0 else 0
    return out.reset_index(drop=True)

# ----------------------------------------------------------------------------
# REGIME ENGINE — realized vol, GARCH, trend/momentum/vol topology, structure
# ----------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def get_price_history(ticker, period="2y", interval="1d"):
    h = yf.Ticker(ticker).history(period=period, interval=interval)
    return h[["Close"]].dropna()

def hurst_exponent(ts):
    """0.5=random walk, >0.5 trending/persistent, <0.5 mean-reverting."""
    ts = np.asarray(ts, float)
    if len(ts) < 40:
        return 0.5
    lags = range(2, min(20, len(ts) // 2))
    tau = []
    for lag in lags:
        diff = ts[lag:] - ts[:-lag]
        tau.append(np.sqrt(np.std(diff)) if np.std(diff) > 0 else 1e-9)
    try:
        m = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return float(np.clip(m[0] * 2.0, 0.0, 1.0))
    except Exception:
        return 0.5

def zscore_last(series, win=252):
    s = series.dropna()
    if len(s) < 20:
        return 0.0
    s = s.tail(win)
    mu, sd = s.mean(), s.std()
    return float((s.iloc[-1] - mu) / sd) if sd > 0 else 0.0

def compute_regime(hist, net_gex, flip, spot):
    px = hist["Close"].astype(float)
    ret = np.log(px / px.shift(1)).dropna()
    out = {}

    # --- realized volatility (annualized) + percentile rank ---
    rv = ret.rolling(21).std() * np.sqrt(252) * 100        # % annualized, time series
    rv_now = float(rv.iloc[-1]) if rv.notna().any() else 0.0
    rv_hist = rv.dropna().tail(252)
    rv_pct = float((rv_hist < rv_now).mean() * 100) if len(rv_hist) else 50.0
    rv_z = zscore_last(rv)
    out["rv"] = rv
    out["rv_now"] = rv_now
    out["rv_pct"] = rv_pct
    out["rv_z"] = rv_z

    # --- GARCH(1,1) forecast (fallback to EWMA if arch unavailable/fails) ---
    garch_now = rv_now
    garch_fcast = rv_now
    persistence = np.nan
    try:
        from arch import arch_model
        am = arch_model(ret.tail(750) * 100, vol="GARCH", p=1, q=1, dist="t")
        res = am.fit(disp="off", show_warning=False)
        cv = res.conditional_volatility
        garch_now = float(cv.iloc[-1]) * np.sqrt(252)
        f = res.forecast(horizon=5, reindex=False)
        garch_fcast = float(np.sqrt(f.variance.values[-1].mean())) * np.sqrt(252)
        persistence = float(res.params.get("alpha[1]", 0) + res.params.get("beta[1]", 0))
    except Exception:
        ew = ret.ewm(span=20).std() * np.sqrt(252) * 100
        garch_now = float(ew.iloc[-1])
        garch_fcast = garch_now
    out["garch_now"] = garch_now
    out["garch_fcast"] = garch_fcast
    out["persistence"] = persistence
    vol_path_pct = (garch_fcast - garch_now) / garch_now * 100 if garch_now else 0.0
    out["vol_path"] = vol_path_pct

    # --- topology factors (z-scored) : trend / momentum / volatility ---
    logpx = np.log(px)
    look = 60
    def slope_z(s, w):
        vals = []
        arr = s.values
        for i in range(w, len(arr)):
            y = arr[i - w:i]
            x = np.arange(w)
            b = np.polyfit(x, y, 1)[0]
            vals.append(b)
        return pd.Series(vals, index=s.index[w:])
    trend_series = slope_z(logpx, 20) * 1e4
    mom_series = (px / px.shift(20) - 1) * 100
    trend = zscore_last(trend_series)
    momentum = zscore_last(mom_series)
    out["trend"] = trend
    out["momentum"] = momentum
    out["vol_factor"] = rv_z
    out["trend_series"] = trend_series.tail(look)
    out["mom_series"] = mom_series.tail(look)
    out["volz_series"] = ((rv - rv.tail(252).mean()) / (rv.tail(252).std() or 1)).tail(look)

    # --- structure / character ---
    hurst = hurst_exponent(px.tail(150).values)
    out["hurst"] = hurst
    structure = "trending / persistent" if hurst > 0.55 else ("mean-revert" if hurst < 0.45 else "random walk")
    out["structure"] = structure

    # --- distance from center (z of price vs 50d mean) ---
    ma = px.rolling(50).mean()
    sd = px.rolling(50).std()
    dist_center = float((px.iloc[-1] - ma.iloc[-1]) / sd.iloc[-1]) if sd.iloc[-1] else 0.0
    out["dist_center"] = dist_center

    # --- return distribution stats ---
    from scipy.stats import skew, kurtosis
    r = ret.tail(252).values
    out["ret_sample"] = r
    out["skew"] = float(skew(r)) if len(r) > 3 else 0.0
    out["kurt"] = float(kurtosis(r)) if len(r) > 3 else 0.0

    # --- regime label ---
    high_vol = rv_pct >= 65
    low_vol = rv_pct <= 35
    strong_trend = abs(trend) >= 0.8 or abs(momentum) >= 0.8
    extended = abs(dist_center) >= 2.0
    if extended:
        label = "EXTENDED"
    elif high_vol and strong_trend:
        label = "VOLATILE / EXPANSION"
    elif high_vol:
        label = "VOLATILE / CHOP"
    elif low_vol and strong_trend:
        label = "QUIET TREND"
    elif low_vol:
        label = "COMPRESSION / RANGE"
    elif strong_trend:
        label = "TRENDING"
    else:
        label = "NEUTRAL / TRANSITION"
    out["label"] = label

    # trend-vs-revert character for trading guidance
    out["expansion"] = high_vol or strong_trend or extended

    # --- plain-English read ---
    if extended:
        read = ("Price is stretched from its mean — the move is extended. Caution chasing here; "
                "watch for a snapback toward the center rather than initiating fresh trend trades.")
    elif label.startswith("VOLATILE"):
        read = ("High realized vol with momentum present. Trend-following / breakout strategies are favored; "
                "fading moves is dangerous. Respect vol, size down, don't fade blindly.")
    elif label == "COMPRESSION / RANGE":
        read = ("Low vol, range-bound. Mean-reversion is favored — fade the edges, expect chop. "
                "Compression eventually resolves into expansion, so watch for the break.")
    elif label == "QUIET TREND":
        read = ("Low vol but directional — a steady grind. Trend continuation favored with shallow pullbacks; "
                "calm enough to hold, but don't expect violent expansion yet.")
    elif label == "TRENDING":
        read = ("Directional bias present at moderate vol. Lean with the trend on pullbacks rather than fading.")
    else:
        read = ("Mixed signals — no dominant regime. Lower conviction; let the tape pick a side before committing.")
    out["read"] = read

    # --- cross-reference with Net GEX gamma regime ---
    gex_trend = net_gex < 0          # negative gamma → trend/expansion
    statreg_trend = out["expansion"]
    if statreg_trend and gex_trend:
        xref = ("AGREE — both engines flag a trending / expansion regime. "
                "High-conviction momentum environment; do not fade.")
        xref_ok = True
    elif (not statreg_trend) and (not gex_trend):
        xref = ("AGREE — both engines flag a rotational / contained regime. "
                "High-conviction fade-the-edges environment.")
        xref_ok = True
    else:
        a = "trend/expansion" if statreg_trend else "rotation/range"
        b = "trend (neg γ)" if gex_trend else "rotation (pos γ)"
        xref = (f"DISAGREE — vol-stats say {a}, gamma says {b}. Lower conviction; "
                f"often a transition. Wait for them to align before sizing up.")
        xref_ok = False
    out["xref"] = xref
    out["xref_ok"] = xref_ok
    return out

# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
tab_conf, tab_regime, tab_greeks, tab_heat, tab_flow, tab_custom, tab_more = st.tabs(
    ["◎ Confluence", "◴ Regime", "📊 Greeks", "▦ Heatmap", "〰 Flow", "🎨 Customize", "··· More"])

# ---------- GREEKS ----------
with tab_greeks:
    left, right = st.columns([1, 6])
    with left:
        st.markdown(f'<div style="color:{C_ACCENT};font-size:11px;letter-spacing:1px;">EXPOSURE</div>',
                    unsafe_allow_html=True)
        exp_sel = st.radio("e", ["GEX", "DEX", "VEX", "CHEX"], label_visibility="collapsed", key="exp")
        st.markdown(f'<div style="color:{C_DIM};font-size:11px;letter-spacing:1px;margin-top:8px;">DECAY</div>',
                    unsafe_allow_html=True)
        dec_sel = st.radio("d", ["—", "THETA", "VEGA", "RHO"], label_visibility="collapsed", key="dec")
        st.markdown(f'<div style="color:{C_ACCENT};font-size:11px;letter-spacing:1px;margin-top:8px;">SENSITIVITY</div>',
                    unsafe_allow_html=True)
        sen_sel = st.radio("s", ["—", "SPEED", "ZONNA", "COLOR"], label_visibility="collapsed", key="sen")
        basis = st.radio("basis", ["OI", "VOL"], horizontal=True, key="basis").lower()

    active = exp_sel
    if dec_sel != "—": active = dec_sel
    if sen_sel != "—": active = sen_sel
    gkey = GREEK_MAP[active]

    with right:
        g = per_strike(data, gkey, spot, basis=basis)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=g["strike"], x=g["call"], orientation="h",
                             marker_color=C_GREEN, name="Call", width=0.6,
                             hovertemplate="$%{y:.2f}<br>Call %{x:.3s}<extra></extra>"))
        fig.add_trace(go.Bar(y=g["strike"], x=g["put"], orientation="h",
                             marker_color=C_RED, name="Put", width=0.6,
                             hovertemplate="$%{y:.2f}<br>Put %{x:.3s}<extra></extra>"))
        for val, col, txt, dsh in [(lv["call_wall"], C_GREEN, "Call Wall", "dash"),
                              (lv["put_wall"], C_RED, "Put Wall", "dot"),
                              (lv["max_pain"], C_YELL, "Max Pain", "dashdot"),
                              (spot, "#ffffff", "Spot", "solid"),
                              (lv["flip"], C_DIM, "γ Flip", "longdash")]:
            fig.add_hline(y=val, line=dict(color=col, width=1, dash=dsh),
                          annotation_text=txt, annotation_position="right",
                          annotation_font=dict(size=9, color=col))
        fig.update_layout(**PLOTLY_LAYOUT, barmode="relative", height=560,
                          title=dict(text=f"{active} Exposure — strike-level structure",
                                     font=dict(size=13, color=C_TXT), x=0.01, y=0.98))
        fig.update_yaxes(title="Strike")
        st.plotly_chart(fig, use_container_width=True, theme=None)

    # ---- three bottom panels ----
    p1, p2, p3 = st.columns(3)
    # P/C SKEW (call vol - put vol per strike)
    with p1:
        st.markdown('<div class="panel-title">P/C SKEW</div>', unsafe_allow_html=True)
        cv = data[data["type"] == "C"].groupby("strike")["vol"].sum()
        pv = data[data["type"] == "P"].groupby("strike")["vol"].sum()
        sk = pd.DataFrame({"call": cv, "put": -pv}).fillna(0).reset_index()
        f1 = go.Figure()
        f1.add_trace(go.Bar(y=sk["strike"], x=sk["call"], orientation="h", marker_color=C_GREEN, width=0.6, hovertemplate="$%{y:.2f}<br>Call vol %{x:.3s}<extra></extra>"))
        f1.add_trace(go.Bar(y=sk["strike"], x=sk["put"], orientation="h", marker_color=C_RED, width=0.6, hovertemplate="$%{y:.2f}<br>Put vol %{x:.3s}<extra></extra>"))
        for v, c in [(lv["call_wall"], C_GREEN), (spot, "#fff"), (lv["put_wall"], C_RED)]:
            f1.add_hline(y=v, line=dict(color=c, width=1, dash="dash"))
        f1.update_layout(**PLOTLY_LAYOUT, barmode="relative", height=260)
        st.plotly_chart(f1, use_container_width=True, theme=None)
    # OI DISTRIBUTION
    with p2:
        st.markdown('<div class="panel-title">OI DISTRIBUTION</div>', unsafe_allow_html=True)
        oi = data.groupby("strike")["oi"].sum().reset_index()
        oi["pct"] = oi["oi"] / oi["oi"].max() * 100 if oi["oi"].max() else 0
        cmax = oi.loc[oi["oi"].idxmax(), "strike"] if not oi.empty else spot
        colors = ["#f2f2f2" if s == cmax else "#484848" for s in oi["strike"]]
        f2 = go.Figure(go.Bar(y=oi["strike"], x=oi["pct"], orientation="h", hovertemplate="$%{y:.2f}<br>OI %{x:.0f}%<extra></extra>",
                              marker_color=colors, width=0.6))
        for v, c in [(lv["call_wall"], C_GREEN), (spot, "#fff"), (lv["put_wall"], C_RED)]:
            f2.add_hline(y=v, line=dict(color=c, width=1, dash="dash"))
        f2.update_layout(**PLOTLY_LAYOUT, height=260)
        st.plotly_chart(f2, use_container_width=True, theme=None)
    # OPTIONS PREMIUM
    with p3:
        st.markdown(f'<div class="panel-title">OPTIONS PREMIUM &nbsp; '
                    f'<span style="color:{C_TXT}">Total: {fmt_b(total_prem).lstrip("+")}</span></div>',
                    unsafe_allow_html=True)
        data["_prem"] = data["mid"] * data["oi"] * CONTRACT_MULT
        cp = data[data["type"] == "C"].groupby("strike")["_prem"].sum()
        pp = -data[data["type"] == "P"].groupby("strike")["_prem"].sum()
        pr = pd.DataFrame({"call": cp, "put": pp}).fillna(0).reset_index()
        f3 = go.Figure()
        f3.add_trace(go.Bar(y=pr["strike"], x=pr["call"], orientation="h", marker_color=C_GREEN, width=0.6, hovertemplate="$%{y:.2f}<br>Call $%{x:.3s}<extra></extra>"))
        f3.add_trace(go.Bar(y=pr["strike"], x=pr["put"], orientation="h", marker_color=C_RED, width=0.6, hovertemplate="$%{y:.2f}<br>Put $%{x:.3s}<extra></extra>"))
        for v, c in [(lv["call_wall"], C_GREEN), (spot, "#fff"), (lv["put_wall"], C_RED)]:
            f3.add_hline(y=v, line=dict(color=c, width=1, dash="dash"))
        f3.update_layout(**PLOTLY_LAYOUT, barmode="relative", height=260)
        st.plotly_chart(f3, use_container_width=True, theme=None)

# ---------- HEATMAP ----------
with tab_heat:
    st.markdown('<div class="panel-title">GEX HEATMAP — strike × expiry</div>', unsafe_allow_html=True)
    if dte_choice != "ADTE":
        st.info("Switch to ADTE to populate the multi-expiry heatmap.")
    rows = []
    for exp in (use_exp if dte_choice == "ADTE" else all_exp[:MAX_DTE_BUTTONS + 1]):
        try:
            d = build_dataset(ticker, [exp], spot)
            if d.empty: continue
            ps = per_strike(d, "gamma", spot)
            ps["expiry"] = exp
            rows.append(ps[["strike", "net", "expiry"]])
        except Exception:
            continue
    if rows:
        hm = pd.concat(rows)
        pivot = hm.pivot_table(index="strike", columns="expiry", values="net", aggfunc="sum").fillna(0)
        # focus around spot
        pivot = pivot.loc[(pivot.index > spot * 0.92) & (pivot.index < spot * 1.08)]
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            colorscale=[[0, C_RED], [0.5, "#000000"], [1, C_GREEN]], zmid=0,
            hovertemplate="$%{y:.2f}<br>%{x}<br>Net GEX %{z:.3s}<extra></extra>",
            colorbar=dict(title="Net GEX")))
        fig.add_hline(y=spot, line=dict(color="#fff", width=1, dash="dash"))
        fig.update_layout(**PLOTLY_LAYOUT, height=600)
        fig.update_yaxes(title="Strike"); fig.update_xaxes(title="Expiry")
        st.plotly_chart(fig, use_container_width=True, theme=None)
    else:
        st.warning("No data for heatmap.")

# ---------- FLOW ----------
with tab_flow:
    st.markdown('<div class="panel-title">VOLUME & PREMIUM FLOW</div>', unsafe_allow_html=True)
    fa, fb = st.columns(2)
    with fa:
        cv = data[data["type"] == "C"].groupby("strike")["vol"].sum()
        pv = data[data["type"] == "P"].groupby("strike")["vol"].sum()
        vf = pd.DataFrame({"call": cv, "put": -pv}).fillna(0).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(y=vf["strike"], x=vf["call"], orientation="h", marker_color=C_GREEN, width=0.6, hovertemplate="$%{y:.2f}<br>Call vol %{x:.3s}<extra></extra>"))
        fig.add_trace(go.Bar(y=vf["strike"], x=vf["put"], orientation="h", marker_color=C_RED, width=0.6, hovertemplate="$%{y:.2f}<br>Put vol %{x:.3s}<extra></extra>"))
        fig.add_hline(y=spot, line=dict(color="#fff", width=1, dash="dash"))
        fig.update_layout(**PLOTLY_LAYOUT, barmode="relative", height=520,
                          title=dict(text="Call vs Put Volume", font=dict(size=12, color=C_TXT), x=0.01))
        st.plotly_chart(fig, use_container_width=True, theme=None)
    with fb:
        cc = data.sort_values("vol", ascending=False).head(15)[
            ["type", "strike", "expiry", "vol", "oi", "impliedVolatility", "mid"]].copy()
        cc.columns = ["C/P", "Strike", "Expiry", "Volume", "OI", "IV", "Mid"]
        cc["IV"] = (cc["IV"] * 100).round(1)
        st.markdown('<div class="panel-title">TOP VOLUME CONTRACTS</div>', unsafe_allow_html=True)
        st.dataframe(cc, use_container_width=True, height=480, hide_index=True)

# ---------- REGIME ----------
with tab_regime:
    try:
        hist = get_price_history(ticker, period="2y")
    except Exception as e:
        hist = pd.DataFrame()
    if hist.empty or len(hist) < 60:
        st.warning("Not enough price history to compute regime.")
    else:
        with st.spinner("Modeling regime…"):
            rg = compute_regime(hist, net_gex, lv["flip"], spot)

        lab = rg["label"]
        lab_col = C_RED if rg["expansion"] else C_GREEN
        if lab == "EXTENDED":
            lab_col = C_YELL

        # ---- banner ----
        st.markdown(
            f'<div style="border:1px solid {C_GRID};border-left:4px solid {lab_col};border-radius:10px;'
            f'padding:16px 20px;margin-bottom:14px;background:{C_PANEL};">'
            f'<div style="color:{C_DIM};font-size:11px;letter-spacing:2px;">MARKET REGIME · {ticker}</div>'
            f'<div style="color:{lab_col};font-size:30px;font-weight:800;letter-spacing:1px;margin:2px 0;">{lab}</div>'
            f'<div style="color:{C_TXT};font-size:12px;">{rg["read"]}</div></div>', unsafe_allow_html=True)

        # ---- cross-reference vs gamma ----
        xcol = C_GREEN if rg["xref_ok"] else C_YELL
        st.markdown(
            f'<div style="border:1px solid {xcol}44;border-radius:8px;padding:10px 14px;margin-bottom:14px;'
            f'background:{C_PANEL};"><span style="color:{xcol};font-size:11px;letter-spacing:1px;font-weight:700;">'
            f'VOL-STATS × GAMMA</span><div style="color:{C_TXT};font-size:12px;margin-top:3px;">{rg["xref"]}</div></div>',
            unsafe_allow_html=True)

        # ---- status grid ----
        def vol_path_txt(v):
            if v > 3:  return f"rising {v:+.0f}%"
            if v < -3: return f"easing {v:+.0f}%"
            return "flat"
        mem = "—"
        if not np.isnan(rg["persistence"]):
            p = rg["persistence"]
            mem = "sticky" if p > 0.9 else ("moderate" if p > 0.7 else "fast decay")
        rows = [
            ("STATUS", lab, lab_col),
            ("STRUCTURE", rg["structure"], C_TXT),
            ("VOL LEVEL", f'{rg["rv_pct"]:.0f}th pct', C_RED if rg["rv_pct"] >= 65 else (C_GREEN if rg["rv_pct"] <= 35 else C_TXT)),
            ("VOL NOW (ann)", f'{rg["rv_now"]:.0f}%', C_TXT),
            ("VOL FORECAST", f'{rg["garch_fcast"]:.0f}%', C_TXT),
            ("VOL PATH", vol_path_txt(rg["vol_path"]), C_RED if rg["vol_path"] > 3 else (C_GREEN if rg["vol_path"] < -3 else C_DIM)),
            ("VOL MEMORY", mem, C_TXT),
            ("HURST", f'{rg["hurst"]:.2f}', C_TXT),
            ("TREND z", f'{rg["trend"]:+.2f}', C_GREEN if rg["trend"] > 0 else C_RED),
            ("MOMENTUM z", f'{rg["momentum"]:+.2f}', C_GREEN if rg["momentum"] > 0 else C_RED),
            ("VOLATILITY z", f'{rg["vol_factor"]:+.2f}', C_RED if rg["vol_factor"] > 0 else C_GREEN),
            ("DIST CENTER", f'{rg["dist_center"]:+.2f}σ', C_YELL if abs(rg["dist_center"]) >= 2 else C_TXT),
        ]
        cells = "".join(
            f'<div class="dg-box" style="flex:1 1 22%;min-width:150px;margin:4px;">'
            f'<div style="color:{C_DIM};font-size:9px;letter-spacing:1px;">{lbl}</div>'
            f'<div style="color:{col};font-size:15px;font-weight:700;">{val}</div></div>' for lbl, val, col in rows)
        st.markdown(f'<div style="display:flex;flex-wrap:wrap;margin-bottom:6px;">{cells}</div>', unsafe_allow_html=True)

        # ---- TOPOLOGY: trend / momentum / volatility factor histories ----
        st.markdown('<div class="panel-title" style="margin-top:14px;">TOPOLOGY — trend · momentum · volatility</div>',
                    unsafe_allow_html=True)
        t1, t2, t3 = st.columns(3)
        for col, series, name, curval in [
            (t1, rg["trend_series"], "Trend", rg["trend"]),
            (t2, rg["mom_series"], "Momentum", rg["momentum"]),
            (t3, rg["volz_series"], "Volatility", rg["vol_factor"])]:
            with col:
                vals = series.fillna(0)
                colors = [C_GREEN if v >= 0 else C_RED for v in vals]
                fig = go.Figure(go.Bar(x=list(range(len(vals))), y=vals.values, marker_color=colors,
                                       hovertemplate="%{y:+.2f}<extra></extra>"))
                fig.add_hline(y=0, line=dict(color=C_GRID, width=1))
                fig.update_layout(**PLOTLY_LAYOUT, height=210,
                                  title=dict(text=f"{name}  {curval:+.2f}", font=dict(size=12, color=C_TXT),
                                             x=0.02, y=0.96, yanchor="top"))
                fig.update_xaxes(showticklabels=False)
                st.plotly_chart(fig, use_container_width=True, theme=None)

        # ---- price + realized vol + return distribution ----
        b1, b2 = st.columns(2)
        with b1:
            st.markdown('<div class="panel-title">PRICE & REALIZED VOL</div>', unsafe_allow_html=True)
            px = hist["Close"].tail(252)
            rvv = rg["rv"].tail(252)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(px))), y=px.values, line=dict(color=C_TXT, width=1.5),
                                     name="Price", yaxis="y1", hovertemplate="Price $%{y:.2f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=list(range(len(rvv))), y=rvv.values, line=dict(color=C_YELL, width=1),
                                     name="RV%", yaxis="y2", hovertemplate="RV %{y:.1f}%<extra></extra>"))
            lay = dict(PLOTLY_LAYOUT)
            lay.pop("yaxis", None)
            fig.update_layout(**lay, height=300,
                              yaxis=dict(title="Price", gridcolor=C_GRID, color=C_DIM),
                              yaxis2=dict(title="RV%", overlaying="y", side="right", color=C_YELL, showgrid=False))
            fig.update_xaxes(showticklabels=False)
            st.plotly_chart(fig, use_container_width=True, theme=None)
        with b2:
            st.markdown('<div class="panel-title">RETURN DISTRIBUTION vs NORMAL</div>', unsafe_allow_html=True)
            r = rg["ret_sample"] * 100
            counts, edges = np.histogram(r, bins=40, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            from scipy.stats import norm as _norm
            xline = np.linspace(r.min(), r.max(), 200)
            pdf = _norm.pdf(xline, r.mean(), r.std())
            fig = go.Figure()
            fig.add_trace(go.Bar(x=centers, y=counts, marker_color="#3a6ea5", opacity=0.8, name="Actual",
                                 hovertemplate="ret %{x:.2f}%<br>density %{y:.2f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=xline, y=pdf, line=dict(color=C_YELL, width=2), name="Normal",
                                     hovertemplate="normal %{y:.2f}<extra></extra>"))
            fig.update_layout(**PLOTLY_LAYOUT, height=300,
                              title=dict(text=f"skew {rg['skew']:+.2f} · excess kurt {rg['kurt']:+.2f}",
                                         font=dict(size=11, color=C_DIM), x=0.02))
            st.plotly_chart(fig, use_container_width=True, theme=None)

        st.caption("All metrics derived from free daily price history. Vol annualized. Topology factors are "
                   "z-scored (how far each sits from its ~1yr norm). Hurst >0.55 trending, <0.45 mean-reverting. "
                   "GARCH(1,1) drives the vol forecast/path; falls back to EWMA if unavailable. 'Dist center' is "
                   "price's z-score vs its 50-day mean — beyond ±2σ flags an extended/stretched move.")

# ---------- CUSTOMIZE ----------
with tab_custom:
    st.markdown(
        f'<div style="border:1px solid {C_GRID};border-left:4px solid {C_ACCENT};border-radius:10px;'
        f'padding:16px 20px;margin-bottom:16px;background:{C_PANEL};">'
        f'<div style="color:{C_ACCENT};font-size:13px;letter-spacing:2px;">CUSTOMIZE</div>'
        f'<div style="color:{C_DIM};font-size:12px;margin-top:3px;">Pick a theme, font, and box style. '
        f'Changes apply instantly across the whole app.</div></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.selectbox("Theme", list(THEMES.keys()), key="dg_theme")
    with c2:
        st.selectbox("Font", list(FONTS.keys()), key="dg_font")
    with c3:
        st.selectbox("Box style", BOX_STYLES, key="dg_box")

    # ---- theme swatches ----
    st.markdown('<div class="panel-title" style="margin-top:14px;">THEME PREVIEW</div>', unsafe_allow_html=True)
    sw = ""
    for nm, th in THEMES.items():
        active = nm == st.session_state["dg_theme"]
        chips = "".join(f'<span style="display:inline-block;width:16px;height:16px;border-radius:50%;'
                        f'background:{th[k]};margin-right:3px;border:1px solid {_rgba(th["txt"],0.2)};"></span>'
                        for k in ["g", "r", "y", "accent"])
        sw += (f'<div class="dg-box" style="display:inline-block;margin:5px;vertical-align:top;'
               f'{"outline:2px solid "+C_ACCENT+";" if active else ""}background:{th["bg"]};">'
               f'<div style="color:{th["txt"]};font-size:12px;font-weight:700;margin-bottom:5px;">{nm}'
               f'{"  ●" if active else ""}</div>{chips}</div>')
    st.markdown(f'<div>{sw}</div>', unsafe_allow_html=True)

    # ---- live sample of the display boxes ----
    st.markdown('<div class="panel-title" style="margin-top:16px;">LIVE SAMPLE</div>', unsafe_allow_html=True)
    sample = (
        f'<div class="gx-cardrow">'
        f'<div class="gx-card"><div class="lbl"><span class="dot" style="color:{C_GREEN}">●</span>CALL WALL</div>'
        f'<div class="val" style="color:{C_GREEN}">$740.00</div><div class="sub">Upside gamma structure</div></div>'
        f'<div class="gx-card"><div class="lbl"><span class="dot" style="color:{C_RED}">●</span>PUT WALL</div>'
        f'<div class="val" style="color:{C_RED}">$724.00</div><div class="sub">Downside gamma structure</div></div>'
        f'<div class="gx-card"><div class="lbl"><span class="dot" style="color:{C_YELL}">●</span>GAMMA FLIP</div>'
        f'<div class="val" style="color:{C_YELL}">$727.96</div><div class="sub">Regime transition level</div></div>'
        f'</div>')
    st.markdown(sample, unsafe_allow_html=True)

    st.caption("Themes change every color (calls stay the bright/up color, puts the down color, so reads stay "
               "consistent). 'Bubbly' rounds and adds a soft glow; 'Glass' is translucent with blur. Fonts: Mono "
               "(terminal), Rounded (soft), Grotesk (modern), Techno (sci-fi). Charts and copy boxes stay monospace "
               "for number legibility. Settings persist for this session.")

# ---------- MORE ----------
with tab_more:
    sub_tut, sub_lvl = st.tabs(["📖 Tutorial", "🎯 Levels & Settings"])

    # ===================== TUTORIAL =====================
    with sub_tut:
        st.markdown(f"""
        <div style="background:{C_PANEL};border:1px solid {C_ACCENT}33;border-radius:10px;
        padding:16px 20px;margin-bottom:14px;">
        <div style="color:{C_ACCENT};font-size:13px;letter-spacing:2px;">DARKGAMMA — FULL TUTORIAL</div>
        <div style="color:{C_DIM};font-size:12px;margin-top:4px;">
        Start at the top if you've never touched options. Each section opens when you click it.
        By the end you'll be able to use every part of this software.</div></div>
        """, unsafe_allow_html=True)

        # ---- 0. The 30-second version ----
        with st.expander("🟢  THE 30-SECOND VERSION (read this first)", expanded=True):
            st.markdown(f"""
**What this app does, in one breath:** Big banks/market-makers ("dealers") are forced to
buy and sell the underlying (QQQ, the NQ proxy) to stay hedged on the options they hold.
Their hedging creates **invisible walls of buying and selling pressure** at specific prices.
This app finds those walls so you know **where price is likely to stall, bounce, or accelerate.**

**The only 3 numbers that matter at a glance:**

| Level | Plain meaning | How you use it |
|---|---|---|
| 🟩 **Call Wall** | Strong ceiling above price | Price tends to get *rejected* here. Fade longs / take profit. |
| 🟥 **Put Wall** | Strong floor below price | Price tends to *bounce* here. Look for longs. |
| 🟨 **Gamma Flip** | The "calm vs chaos" line | **Above it = calm, range-y.** **Below it = violent, trending.** |

**The single rule that does 80% of the work:**
- Price **above** the Gamma Flip → market is *pinned and choppy* → **fade the edges** (sell call wall, buy put wall).
- Price **below** the Gamma Flip → market is *unstable and trends hard* → **don't fade, ride momentum.**

That's it. Everything below just explains *why*.
""")

        # ---- 1. Options 101 ----
        with st.expander("1️⃣  OPTIONS 101 — zero knowledge assumed"):
            st.markdown("""
**An option is a contract.** It gives the buyer the *right* (not obligation) to buy or sell
100 shares of something at a fixed price, before a deadline.

- **Call** = a bet/right that price goes **UP**. (Right to *buy* at the strike.)
- **Put** = a bet/right that price goes **DOWN**. (Right to *sell* at the strike.)
- **Strike** = the fixed price in the contract (e.g. the $740 call).
- **Expiry** = the deadline. Options lose all "time value" at expiry.
- **Premium** = what the option costs to buy.
- **Open Interest (OI)** = how many of that exact contract currently *exist* and are open.
  → OI shows where the *big standing positions* are. **This is what builds the walls.**
- **Volume** = how many traded *today*. → Volume shows *fresh, live activity right now.*
- **Implied Volatility (IV)** = the market's guess at how much the stock will move.
  Higher IV = bigger expected swings = pricier options.

**Why you care:** every option that exists has to be *hedged* by the dealer who sold it.
Millions of contracts = a mountain of forced hedging. That mountain is what you're reading.
""")

        # ---- 2. The Greeks ----
        with st.expander("2️⃣  THE GREEKS — what the sidebar buttons mean"):
            st.markdown("""
"Greeks" measure how an option's price reacts to things. You don't need the math —
just what each one tells you on the chart.

**EXPOSURE group (the main ones):**
- **GEX (Gamma)** ⭐ — *the headliner.* Gamma measures how fast dealer hedging *accelerates*.
  High gamma at a strike = a strong magnet/wall. **This is your default view.**
- **DEX (Delta)** — directional pressure. Shows where dealers must buy/sell to stay flat.
- **VEX (Vega)** — sensitivity to volatility changes. Useful around events/IV spikes.
- **CHEX (Charm)** — how delta decays as time passes. Drives the classic "0DTE afternoon drift."

**DECAY group:**
- **Theta** — time decay; how much value bleeds out per day.
- **Vega** — same as VEX, the volatility sensitivity.
- **Rho** — interest-rate sensitivity (rarely matters intraday).

**SENSITIVITY group (advanced 3rd-order):**
- **Speed** — how fast gamma itself changes. Spikes warn that walls will move quickly.
- **Zonna (Zomma)** — how gamma reacts to IV changes.
- **Color** — how gamma decays over time (gamma's version of theta).

**90% of the time you stay on GEX.** The rest are for when you want a deeper read.
""")

        # ---- 3. Dealer gamma ----
        with st.expander("3️⃣  THE BIG IDEA — dealer gamma & why walls exist"):
            st.markdown(f"""
This is the engine of the whole app. Read it twice.

**Dealers (market-makers) must stay neutral.** When they sell you a call, they're now short it,
so they buy shares to hedge. As price moves, their hedge needs *constant adjusting*. **That
adjusting is forced buying and selling that you can predict.**

There are two worlds, decided by the **Gamma Flip** line:

**🟢 POSITIVE GAMMA (price ABOVE the flip) — "the calm world"**
- Dealers hedge by **buying dips and selling rips.**
- This *fights* every move → volatility gets crushed, price gets **pinned** and chops sideways.
- **Net GEX reads positive (green).**
- 👉 Your play: *mean-reversion.* Sell near the Call Wall, buy near the Put Wall. Fade extremes.

**🔴 NEGATIVE GAMMA (price BELOW the flip) — "the chaos world"**
- Dealers hedge by **selling dips and buying rips.**
- This *feeds* every move → volatility explodes, price **trends and gaps hard.**
- **Net GEX reads negative (red).**
- 👉 Your play: *momentum.* Don't fade. Breakouts run. Stops get hunted. Size down.

**The Net GEX card and the Confluence tab tell you which world you're in right now.**
""")

        # ---- 4. Key levels ----
        with st.expander("4️⃣  THE KEY LEVELS — and exactly how to trade each"):
            st.markdown("""
These are the dashed lines on the main chart and the top 5 cards.

**🟩 CALL WALL** — the strike with the largest positive (call) gamma.
- Acts as a **ceiling / magnet from below.** In a positive-gamma day price struggles to break above it.
- *Trade:* take profit on longs into it; short rejections off it; if it *breaks* with volume, expect acceleration.

**🟥 PUT WALL** — the strike with the largest put gamma.
- Acts as a **floor / magnet from above.** Strong support in positive gamma.
- *Trade:* look for long bounces here; if it *breaks*, you've likely flipped into negative-gamma chaos → trend down.

**🟨 GAMMA FLIP** — the price where dealer gamma flips from positive to negative.
- **The most important level in the app.** It's the dividing line between calm and chaos.
- *Trade:* it's your regime switch. Above = fade. Below = chase. Reclaiming or losing it intraday is a major signal.

**🟧 MAX PAIN** — the strike where the most options expire worthless.
- Price often **drifts toward it into expiry** (esp. on 0DTE afternoons / OPEX days).
- *Trade:* a soft magnet, not a hard wall. Good context, weak standalone signal.

**⚪ SPOT** — current price. Everything is read *relative to spot.*
""")

        # ---- 5. Reading the charts ----
        with st.expander("5️⃣  HOW TO READ EVERY CHART & PANEL"):
            st.markdown("""
**Main GEX chart (Greeks tab):** strikes go up the side, exposure goes left/right.
- **Green bars to the right** = call gamma (resistance stacking above).
- **Red bars to the left** = put gamma (support stacking below).
- The *longest* green bar = Call Wall. The *longest* red bar = Put Wall.
- Dashed lines mark all your key levels. Where price sits vs those lines is the whole read.

**OI vs VOL toggle:**
- **OI** = standing structure (where the big positions already are). Use for *levels/walls.*
- **VOL** = today's fresh flow (what's being bought/sold *now*). Use for *intraday shifts.*

**Bottom three panels:**
- **P/C Skew** — call volume vs put volume per strike. Heavy red = bearish positioning, heavy green = bullish.
- **OI Distribution** — where open interest is concentrated (yellow bar = the biggest single strike).
- **Options Premium** — where the actual *dollars* are parked. Big premium = big dealer hedging obligation = stronger wall.

**Heatmap tab:** strike (vertical) × expiry (horizontal), colored by net GEX
(green = positive/support-y, red = negative). Best viewed on **ADTE** so you see how walls
line up across multiple days — strikes that stay green/red across *several* expiries are the sturdiest.

**Flow tab:** live call-vs-put volume + a table of the 15 highest-volume contracts right now.
This is your "what just lit up" radar.

**Confluence tab:** every strike scored and ranked by how many signals align there
(gamma, OI, volume, premium, delta, vanna, charm), strongest level first, with the current
regime banner up top. **Read this first thing every morning** — it's your shortlist of levels to watch.
""")

        # ---- 6. DTE ----
        with st.expander("6️⃣  DTE SELECTOR — 0DTE through 5DTE & ADTE"):
            st.markdown("""
**DTE = Days To Expiration.** The buttons at the top pick *which* expiry's options you're reading.

- **0DTE** — options expiring **today.** Gamma is *highest* near expiry, so **0DTE walls are the
  strongest and most reactive intraday levels.** This is your primary intraday view.
- **1DTE–5DTE** — tomorrow through next week. Walls are softer but reflect bigger-picture positioning.
- **ADTE ("All DTE")** — aggregates 0–5DTE into one combined picture. Use it to find **levels that
  stack across multiple expiries** — those are the heavyweight walls that hold all week.

**Rule of thumb:** scalping/intraday → **0DTE**. Swing context / strongest confluence → **ADTE.**
""")

        # ---- 7. Suggested settings ----
        with st.expander("7️⃣  ⭐ SUGGESTED SETTINGS — INTRADAY & NQ TRADING"):
            st.markdown("""
**General intraday scalping (any index):**
- DTE: **0DTE**
- Basis: **OI** to set your levels pre-market, then watch **VOL** for live shifts
- Exposure: **GEX**
- Workflow: read Confluence → mark the top-ranked levels on your chart → trade *to/from* those levels.

**NQ / MNQ futures trading (your setup):**
- You trade NQ, but options live on **QQQ** — so read QQQ here and **convert with ×41.4**
  (the Levels tab does this for you automatically).
- DTE: **0DTE** for the day's battle lines; cross-check with **ADTE** for the levels that hold all week.
- Map QQQ Call Wall / Put Wall / Flip → NQ price, drop them on MotiveWave/TradingView as horizontal lines.
- **Regime first:** check Net GEX + where NQ sits vs the Flip *before* you take a single trade.
  - NQ above flip + positive GEX → **fade the walls, expect chop, smaller targets.**
  - NQ below flip + negative GEX → **trade momentum, expect range expansion, trail wider, size down.**
- Best windows: the **9:30 open**, the **gamma-pin grind into 3–4pm** (charm/0DTE drift toward Max Pain),
  and **reclaim/loss of the Flip** as your highest-conviction intraday signal.

**Pre-market routine (2 minutes):**
1. Open **Confluence** → note the regime banner + the top-ranked levels.
2. Open **Levels tab** → grab the NQ-converted numbers.
3. Mark Flip, Call Wall, Put Wall, Max Pain on your trading chart.
4. Decide *fade or chase* based on the Flip. Done.
""")

        # ---- 8. How it works under the hood ----
        with st.expander("8️⃣  HOW THE SOFTWARE WORKS (data & math)"):
            st.markdown(f"""
- **Data source:** Yahoo Finance options chains (free). Spot, strikes, OI, volume, bid/ask, IV.
- **Greeks:** Yahoo doesn't give gamma, so DARKGAMMA computes every Greek itself using the
  **Black-Scholes model** from each option's IV. Risk-free rate is set to **{RISK_FREE*100:.1f}%**
  (editable at the top of `app.py`).
- **Exposure math:** for each strike, exposure = Greek × Open Interest × 100 × spot factor.
  **Dealer convention: calls count positive (+γ), puts negative (−γ)** — the standard assumption
  that dealers are long calls / short puts.
- **Walls:** Call Wall = max positive call gamma. Put Wall = max put gamma. **Gamma Flip** = the
  strike where cumulative net gamma crosses zero. **Max Pain** = strike minimizing total option value.
- **Refresh:** data is cached ~2 min. Reload the page for a fresh pull.

**Honest limitations (so you trust it correctly):**
- Yahoo's IV is occasionally missing/zero on deep wings → those get floored to 0.30 so gamma
  doesn't blow up. A weird wall is usually one junk-IV strike.
- QQQ is a *proxy* for NQ, not a perfect 1:1 — the 41.4× ratio drifts slightly over time.
- This is a **structural/probabilistic** tool, not a crystal ball. Walls bend and break.
  It tells you *where the pressure is*, not *what will happen.*
""")

        # ---- 9. Glossary ----
        with st.expander("9️⃣  QUICK GLOSSARY"):
            st.markdown("""
- **GEX** — Gamma Exposure. Total dealer gamma; the core metric.
- **Net GEX** — the sum across all strikes. Positive = calm regime, negative = volatile regime.
- **Dealer / Market-Maker** — the firm on the other side of options trades; hedges constantly.
- **Pin / Pinning** — price getting stuck near a high-gamma strike.
- **Wall** — a strike with so much gamma it acts as support/resistance.
- **Flip** — the price where net gamma changes sign.
- **0DTE** — expiring today; strongest intraday gamma.
- **ITM/OTM** — In/Out of The Money (option has/doesn't have intrinsic value).
- **OPEX** — monthly options expiration (3rd Friday); big positioning resets.
- **IV** — Implied Volatility; market's expected movement.
""")

        st.markdown(f"""
        <div style="background:{C_PANEL};border:1px solid {C_GRID};border-radius:8px;
        padding:12px 16px;margin-top:10px;color:{C_DIM};font-size:11px;">
        ⚠️ <b style="color:{C_TXT}">Disclaimer:</b> DARKGAMMA is an educational/informational tool.
        It is not financial advice and does not predict the future. Options/futures trading carries
        substantial risk. Always do your own research and manage risk.</div>
        """, unsafe_allow_html=True)

    # ===================== LEVELS & SETTINGS =====================
    with sub_lvl:
        st.markdown('<div class="panel-title">KEY LEVELS & SETTINGS</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            st.metric("Spot", f"${spot:,.2f}")
            st.metric("Call Wall", f"${lv['call_wall']:,.2f}")
            st.metric("Put Wall", f"${lv['put_wall']:,.2f}")
            st.metric("Gamma Flip", f"${lv['flip']:,.2f}")
            st.metric("Max Pain", f"${lv['max_pain']:,.2f}")
        with b:
            st.metric("Net GEX", fmt_b(net_gex))
            st.metric("Total Premium", fmt_b(total_prem).lstrip("+"))
            st.metric("Contracts (rows)", f"{len(data):,}")
            st.metric("Total OI", f"{int(data['oi'].sum()):,}")
            st.metric("Risk-free rate", f"{RISK_FREE*100:.1f}%")
        st.caption("Greeks computed via Black-Scholes from Yahoo Finance IV. "
                   "Dealer convention: calls +γ, puts −γ. Edit RISK_FREE / ET at top of file.")
        # NQ conversion helper (your ~41.4x QQQ->NQ ratio)
        if ticker == "QQQ":
            st.markdown("---")
            st.markdown('<div class="panel-title">QQQ → NQ (×41.4)</div>', unsafe_allow_html=True)
            r = 41.4
            st.write(f"Call Wall ≈ **{lv['call_wall']*r:,.0f}**  ·  "
                     f"Put Wall ≈ **{lv['put_wall']*r:,.0f}**  ·  "
                     f"Flip ≈ **{lv['flip']*r:,.0f}**  ·  "
                     f"Spot ≈ **{spot*r:,.0f}**")

# ---------- CONFLUENCE ----------
with tab_conf:
    regime = "POSITIVE γ — fade walls, expect chop" if net_gex >= 0 else "NEGATIVE γ — ride momentum, expect trends"
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'border:1px solid {C_GRID};border-radius:8px;padding:10px 16px;margin-bottom:12px;'
        f'background:{C_PANEL};">'
        f'<div><span style="color:{C_DIM};font-size:11px;letter-spacing:1px;">REGIME</span>'
        f'<div style="color:{C_TXT};font-size:14px;font-weight:700;">{regime}</div></div>'
        f'<div style="text-align:right;"><span style="color:{C_DIM};font-size:11px;">NET GEX</span>'
        f'<div style="color:{C_GREEN if net_gex>=0 else C_RED};font-size:16px;font-weight:700;">{fmt_b(net_gex)}</div></div>'
        f'<div style="text-align:right;"><span style="color:{C_DIM};font-size:11px;">SPOT</span>'
        f'<div style="color:{C_TXT};font-size:16px;font-weight:700;">${spot:,.2f}</div></div>'
        f'</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="panel-title">KEY LEVELS TO WATCH — confluence ranked, strongest first</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div style="color:{C_DIM};font-size:10px;margin-bottom:8px;">'
                f'Score blends gamma · OI · volume · premium · delta · vanna · charm, '
                f'then boosts strikes where multiple signals align at once.</div>', unsafe_allow_html=True)

    conf = confluence_table(data, spot, lv, top_n=8)
    if conf.empty:
        st.warning("Not enough data to rank levels.")
    else:
        is_qqq = (ticker == "QQQ")
        for i, row in conf.iterrows():
            rank = i + 1
            strike = row["strike"]
            pct = int(row["pct"])
            role = row["role"]
            tags = row["tags"]
            hot = row["hot"]
            dist_pct = (strike - spot) / spot * 100
            arrow = "▲" if strike > spot else ("▼" if strike < spot else "●")
            nq_txt = f' &nbsp;<span style="color:{C_DIM};">→ NQ {strike*41.4:,.0f}</span>' if is_qqq else ""
            # strength bar
            filled = int(round(pct / 100 * 16))
            bar = "█" * filled + "░" * (16 - filled)
            # brightness of rank by score
            bright = "#ffffff" if pct >= 75 else (C_TXT if pct >= 45 else C_DIM)
            role_col = C_GREEN if role == "RESISTANCE" else (C_RED if role == "SUPPORT" else C_YELL)
            tag_html = "".join(
                f'<span style="border:1px solid {C_GRID};color:{C_TXT};font-size:9px;'
                f'padding:1px 6px;border-radius:3px;margin-right:4px;">{t}</span>' for t in tags)
            hot_html = " ".join(f'<span style="color:{C_TXT};">{h}</span>' for h in hot) or \
                       f'<span style="color:{C_DIM};">—</span>'

            st.markdown(
                f'<div style="border:1px solid {C_GRID};border-left:3px solid {role_col};'
                f'border-radius:8px;padding:10px 14px;margin-bottom:7px;background:{C_PANEL};">'
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
                f'<div><span style="color:{C_DIM};font-size:12px;">#{rank}</span> &nbsp;'
                f'<span style="color:{bright};font-size:20px;font-weight:700;">{arrow} ${strike:,.2f}</span>'
                f'{nq_txt} &nbsp;<span style="color:{role_col};font-size:10px;letter-spacing:1px;">{role}</span></div>'
                f'<div style="text-align:right;"><span style="font-family:monospace;color:{bright};">{bar}</span> '
                f'<span style="color:{bright};font-weight:700;">{pct}</span>'
                f'<span style="color:{C_DIM};font-size:10px;"> &nbsp;{dist_pct:+.1f}%</span></div></div>'
                f'<div style="margin-top:6px;font-size:10px;color:{C_DIM};">'
                f'{tag_html}{"&nbsp;" if tags else ""}'
                f'<span style="margin-left:6px;">aligned: {hot_html}</span></div>'
                f'</div>', unsafe_allow_html=True)

        st.caption("Bar/score are relative to the strongest level in view. ▲ above spot (resistance), "
                   "▼ below (support). 'Aligned' shows which signals are simultaneously hot at that strike "
                   "(γ gamma · OI · VOL volume · PREM premium · DEX delta · VAN vanna · CHM charm). "
                   "Switch to ADTE for the levels that hold across the whole week.")

        # ---- COPY LEVELS → TRADINGVIEW ----
        copy_block = "\n".join(f'{row["strike"]:.2f}' for _, row in conf.iterrows())

        st.markdown(f'<div class="panel-title" style="margin-top:16px;">COPY LEVELS → TRADINGVIEW</div>',
                    unsafe_allow_html=True)
        st.caption("Hover the box and click the copy icon (top-right), then paste into the DARKGAMMA "
                   "indicator's \"Paste Levels\" box on TradingView. Format is price=name, one per line. "
                   "Prices are QQQ strikes — the indicator converts them to NQ automatically.")
        st.code(copy_block, language="text")
