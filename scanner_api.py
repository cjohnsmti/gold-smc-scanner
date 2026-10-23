import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import os
import matplotlib.pyplot as plt
import mplfinance as mpf
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

app = FastAPI()
ET_TZ = pytz.timezone('US/Eastern')
GOLD_TICKER = "GC=F"

# --- CONFIGURATION (ใส่ค่าจริงของคุณตรงนี้) ---
TELEGRAM_BOT_TOKEN = os.getenv("8673492871:AAFxKvEcPuyarRUkVy5Wix8vxDGgo6SNCos", "8673492871:AAFxKvEcPuyarRUkVy5Wix8vxDGgo6SNCos")
TELEGRAM_CHAT_ID = os.getenv("8298446670", "8298446670")

# --- UTILITY FUNCTIONS ---
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try: requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=10)
    except: pass

def send_telegram_photo(img_path, caption):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(img_path, 'rb') as photo:
            requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"}, files={'photo': photo}, timeout=15)
    except: pass

def safe_get_yf_data(ticker, period, interval):
    try:
        data = yf.Ticker(ticker).history(period=period, interval=interval)
        return data if not data.empty else pd.DataFrame()
    except: return pd.DataFrame()

# --- SMC ENGINE ---
def find_trading_range_and_pda(df):
    down_idx = None
    for i in range(len(df)-1, -1, -1):
        if df['Close'].iloc[i] < df['Open'].iloc[i]:
            down_idx = i; break
    if down_idx is None or down_idx == 0: return None
    down_open = df['Open'].iloc[down_idx]
    up_idx = None
    for i in range(down_idx - 1, -1, -1):
        if df['Close'].iloc[i] > df['Open'].iloc[i] and df['Low'].iloc[i] > down_open:
            up_idx = i; break
    if up_idx is None: return None
    range_high = df['High'].iloc[up_idx:down_idx].max()
    range_low = df['Low'].iloc[up_idx:].min()
    equilibrium = (range_high + range_low) / 2
    current_price = df['Close'].iloc[-1]
    zone = "PREMIUM" if current_price > equilibrium else "DISCOUNT"
    return {"high": range_high, "low": range_low, "equilibrium": equilibrium, "zone": zone, "erl_top": range_high, "erl_bottom": range_low}

def check_clean_price_action(df_1h):
    last_candle = df_1h.iloc[-1]
    body = abs(last_candle['Close'] - last_candle['Open'])
    total_range = last_candle['High'] - last_candle['Low']
    if total_range == 0: return False
    return (total_range - body) / total_range <= 0.70

def find_institutional_ob(df, htf_equilibrium, direction="BULLISH"):
    obs = []
    for i in range(1, len(df) - 1):
        candle = df.iloc[i-1]
        is_down = candle['Close'] < candle['Open']
        is_up = candle['Close'] > candle['Open']
        if direction == "BULLISH" and is_down and abs(candle['Close'] - htf_equilibrium) / htf_equilibrium < 0.015:
            if df['High'].iloc[i:].max() > candle['High']:
                obs.append({"type": "BULLISH_OB", "top": candle['Open'], "bottom": candle['Low'], "poi_entry": candle['Open'], "sl": candle['Low']})
        elif direction == "BEARISH" and is_up and abs(candle['Close'] - htf_equilibrium) / htf_equilibrium < 0.015:
            if df['Low'].iloc[i:].min() < candle['Low']:
                obs.append({"type": "BEARISH_OB", "top": candle['High'], "bottom": candle['Close'], "poi_entry": candle['Close'], "sl": candle['High']})
    return obs[-1] if obs else None

def find_overlap_fvg(df, old_zone):
    for i in range(1, len(df) - 1):
        if df['Low'].iloc[i+1] > df['High'].iloc[i-1]:
            if old_zone and (df['High'].iloc[i-1] <= old_zone <= df['Low'].iloc[i+1]):
                return {"type": "BULLISH_FVG_OVERLAP", "top": df['Low'].iloc[i+1], "bottom": df['High'].iloc[i-1]}
        if df['High'].iloc[i+1] < df['Low'].iloc[i-1]:
            if old_zone and (df['High'].iloc[i+1] <= old_zone <= df['Low'].iloc[i-1]):
                return {"type": "BEARISH_FVG_OVERLAP", "top": df['Low'].iloc[i-1], "bottom": df['High'].iloc[i+1]}
    return None

def calculate_rr(entry, sl, tp):
    risk = abs(entry - sl)
    return abs(tp - entry) / risk if risk > 0 else 0

def check_macro_correlation():
    dxy = safe_get_yf_data("DX-Y.NYB", "5d", "1h")
    tnx = safe_get_yf_data("^TNX", "5d", "1h")
    msg = ""
    if not dxy.empty and len(dxy)>12 and dxy['Close'].iloc[-1] > dxy['High'].iloc[-12:-1].max(): msg += "⚠️ DXY พุ่งขึ้น\n"
    if not tnx.empty and len(tnx)>12 and tnx['Close'].iloc[-1] > tnx['High'].iloc[-12:-1].max(): msg += "⚠️ US10Y พุ่งขึ้น\n"
    return msg if msg else "✅ Macro สนับสนุน/เป็นกลาง\n"

# --- AI LSTM ---
AI_MODEL, AI_SCALER_X, AI_SCALER_Y = None, None, None
 # def initialize_ai_system():
    global AI_MODEL, AI_SCALER_X, AI_SCALER_Y
    if not AI_AVAILABLE: return
    print("🧠 กำลังสอน AI...")
    df_g = safe_get_yf_data(GOLD_TICKER, "2y", "1h")
    df_d = safe_get_yf_data("DX-Y.NYB", "2y", "1h")
    df_t = safe_get_yf_data("^TNX", "2y", "1h")
    if df_g.empty or df_d.empty or df_t.empty: return
    df = pd.DataFrame(index=df_g.index)
    df['Gold'], df['Vol'], df['DXY'], df['TNX'] = df_g['Close'], df_g['Volume'], df_d['Close'].reindex(df_g.index, method='ffill'), df_t['Close'].reindex(df_g.index, method='ffill')
    df.dropna(inplace=True)
    X, Y = df[['Gold','Vol','DXY','TNX']].values, df['Gold'].shift(-4).dropna().values
    X = X[:-4]
    AI_SCALER_X, AI_SCALER_Y = MinMaxScaler(), MinMaxScaler()
    X_s, Y_s = AI_SCALER_X.fit_transform(X), AI_SCALER_Y.fit_transform(Y.reshape(-1, 1))
    X_seq, Y_seq = [], []
    for i in range(24, len(X_s)-4): X_seq.append(X_s[i-24:i]); Y_seq.append(Y_s[i])
    AI_MODEL = Sequential([LSTM(64, return_sequences=True, input_shape=(24, 4)), Dropout(0.2), LSTM(32), Dropout(0.2), Dense(16, activation='relu'), Dense(1)])
    AI_MODEL.compile(optimizer='adam', loss='mse')
    AI_MODEL.fit(np.array(X_seq), np.array(Y_seq), epochs=10, batch_size=32, verbose=0)
    print("✅ AI พร้อมใช้งาน!")

def predict_ai_target(df_1h):
    return None
    if not AI_MODEL: return None
    try:
        live = df_1h.tail(24)[['Close','Volume']].copy()
        dxy = safe_get_yf_data("DX-Y.NYB", "5d", "1h")['Close'].tail(24)
        tnx = safe_get_yf_data("^TNX", "5d", "1h")['Close'].tail(24)
        live['DXY'], live['TNX'] = dxy.values, tnx.values
        pred = AI_SCALER_Y.inverse_transform(AI_MODEL.predict(AI_SCALER_X.transform(live.values).reshape(1, 24, 4), verbose=0))[0][0]
        return round(pred, 2)
    except: return None

# --- CHART ---
def generate_chart(df_1h, ob, fvg, tr):
    plot_df = df_1h.tail(48).copy()
    apds = []
    if ob: plot_df['OB_T'], plot_df['OB_B'] = ob['top'], ob['bottom']; apds.extend([mpf.make_addplot(plot_df['OB_T'], color='green', linestyle='--'), mpf.make_addplot(plot_df['OB_B'], color='green', linestyle='--')])
    if fvg: plot_df['FVG_T'], plot_df['FVG_B'] = fvg['top'], fvg['bottom']; apds.extend([mpf.make_addplot(plot_df['FVG_T'], color='blue', linestyle='-.'), mpf.make_addplot(plot_df['FVG_B'], color='blue', linestyle='-.')])
    if tr: plot_df['EQ'] = tr['equilibrium']; apds.append(mpf.make_addplot(plot_df['EQ'], color='gray', linestyle=':'))
    img = "/tmp/gold_ultimate.png"
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', inherit=True)
    mpf.plot(plot_df, type='candle', style=mpf.make_mpf_style(marketcolors=mc, facecolor='white'), addplot=apds if apds else None, figsize=(10, 6), savefig=img)
    return img

# --- MAIN SCANNER LOGIC (For n8n) ---
@app.get("/scan-gold")
def run_ultimate_scanner():
    """Endpoint สำหรับ n8n ยิงมาเรียก"""
    print(f"\n[{datetime.now(ET_TZ).strftime('%H:%M:%S')}] n8n Triggered Scan...")
    
    df_d = safe_get_yf_data(GOLD_TICKER, "60d", "1d")
    df_4h = safe_get_yf_data(GOLD_TICKER, "60d", "4h")
    df_1h = safe_get_yf_data(GOLD_TICKER, "30d", "1h")
    if df_d.empty or df_1h.empty or df_4h.empty: 
        return JSONResponse(content={"status": "no_signal", "reason": "Data Error"})

    if not check_clean_price_action(df_1h): 
        return JSONResponse(content={"status": "no_signal", "reason": "Noisy Chart"})

    tr = find_trading_range_and_pda(df_4h)
    if not tr: 
        return JSONResponse(content={"status": "no_signal", "reason": "No Range"})
    
    current_price = df_1h['Close'].iloc[-1]
    bias = "BULLISH" if tr['zone'] == "DISCOUNT" else "BEARISH"
    
    macro_msg = check_macro_correlation()
    if "US10Y พุ่งขึ้น" in macro_msg and bias == "BULLISH": 
        return JSONResponse(content={"status": "no_signal", "reason": "Macro Conflict"})

    ob = find_institutional_ob(df_1h, tr['equilibrium'], direction=bias)
    fvg = find_overlap_fvg(df_1h, tr['equilibrium'])
    
    if not ob and not fvg: 
        return JSONResponse(content={"status": "no_signal", "reason": "No OB/FVG Overlap"})

    entry_point = fvg['bottom'] if (fvg and "BULLISH" in fvg['type']) else (ob['poi_entry'] if ob else None)
    sl_point = ob['sl'] if ob else (fvg['bottom'] * 0.995 if fvg else None)
    tp_point = tr['erl_top'] if bias == "BULLISH" else tr['erl_bottom']
    
    if not entry_point or not sl_point or not tp_point: 
        return JSONResponse(content={"status": "no_signal", "reason": "Calculation Error"})

    rr = calculate_rr(entry_point, sl_point, tp_point)
    if rr < 2.0: # ปรับ RR เป็น 2:1 ตามที่คุณต้องการ
        return JSONResponse(content={"status": "no_signal", "reason": f"RR {rr:.2f} too low"})

    ai_target = predict_ai_target(df_1h)

    # --- ถ้าผ่านทุกด่าน ส่ง Telegram ทันที ---
    msg = f"<b>[🏆 INSTITUTIONAL SETUP DETECTED]</b>\n"
    msg += f"💰 <b>ราคาปัจจุบัน:</b> ${current_price:.2f}\n"
    msg += f"📊 <b>Bias:</b> {'ขาขึ้น (Buy)' if bias=='BULLISH' else 'ขาลง (Sell)'}\n"
    msg += f"⚖️ <b>Equilibrium:</b> ${tr['equilibrium']:.2f} ({tr['zone']})\n\n"
    msg += f"🎯 <b>Entry Limit:</b> ${entry_point:.2f}\n"
    msg += f"🛡️ <b>Stop Loss:</b> ${sl_point:.2f}\n"
    msg += f"🏆 <b>Take Profit:</b> ${tp_point:.2f}\n"
    msg += f"📈 <b>Risk:Reward:</b> <b>{rr:.1f}:1</b> (ผ่านเกณฑ์ 2:1)\n\n"
    if ob: msg += f"🧱 <b>Order Block:</b> ตรวจพบ\n"
    if fvg: msg += f"⚡ <b>FVG Overlap:</b> ตรวจพบ\n"
    msg += f"\n<b>--- Macro & AI ---</b>\n{macro_msg}"
    if ai_target: msg += f"🤖 <b>AI คาดการณ์:</b> ${ai_target:.2f}\n"

    send_telegram_message(msg)
    
    img_path = generate_chart(df_1h, ob, fvg, tr)
    if os.path.exists(img_path):
        send_telegram_photo(img_path, "📊 แผนที่สนามรบ (OB/FVG)")
        os.remove(img_path)

    return JSONResponse(content={
        "status": "signal_found", 
        "data": {"price": current_price, "entry": entry_point, "sl": sl_point, "tp": tp_point, "rr": round(rr, 2)}
    })

# --- STARTUP ---
initialize_ai_system()
