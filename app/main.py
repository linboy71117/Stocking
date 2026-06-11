import json
import logging
import os
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
import requests

# ==========================================
# 📊 系統日誌與防禦型環境初始化
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("QuantEngine")

from fastapi_websocket_pubsub import PubSubEndpoint

app = FastAPI(
    title="台股量化交易技術指標分析核心引擎 API",
    description="提供 K 線、高階量化技術指標 (MA, RSI, MACD, KDJ, Bollinger Bands) 及 AI 動態波段策略決策支援",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 📁 股票資料庫加載 (防呆、防崩潰)
# ==========================================
STOCK_CSV_PATH = "app/stocks.csv"
if os.path.exists(STOCK_CSV_PATH):
    try:
        stock_df = pd.read_csv(STOCK_CSV_PATH, dtype=str)
        logger.info(f"成功加載本地股票資料庫，共計 {len(stock_df)} 檔標的。")
    except Exception as e:
        logger.error(f"讀取 stocks.csv 失敗: {str(e)}")
        stock_df = pd.DataFrame(columns=["code", "name", "alias"])
else:
    logger.warning("未找到 app/stocks.csv，自動啟用記憶體備援防線。")
    stock_df = pd.DataFrame(columns=["code", "name", "alias"])

stock_df = stock_df.fillna("")

# ==========================================
# 🛠️ 核心量化技術指標演算法 (全手寫高精度 Pandas 引擎)
# ==========================================
class QuantIndicators:
    """專業級量化技術指標計算引擎，嚴格處理 NaN、Inf 異常值。"""
    
    @staticmethod
    def calculate_ma(df: pd.DataFrame, windows: List[int] = [5, 10, 20, 60, 120]) -> pd.DataFrame:
        """計算移動平均線 (Moving Average)"""
        for w in windows:
            df[f"MA{w}"] = df["Close"].rolling(window=w, min_periods=1).mean()
        return df

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """計算相對強弱指標 (RSI)"""
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=window).mean()
        rs = gain / loss
        df[f"RSI{window}"] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """計算指數平滑異同移動平均線 (MACD)"""
        df["EMA_fast"] = df["Close"].ewm(span=fast, adjust=False).mean()
        df["EMA_slow"] = df["Close"].ewm(span=slow, adjust=False).mean()
        df["DIF"] = df["EMA_fast"] - df["EMA_slow"]
        df["DEA"] = df["DIF"].ewm(span=signal, adjust=False).mean()
        df["MACD_Hist"] = (df["DIF"] - df["DEA"]) * 2
        return df

    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """計算布林通道 (Bollinger Bands)"""
        ma = df["Close"].rolling(window=window, min_periods=1).mean()
        std = df["Close"].rolling(window=window, min_periods=1).std()
        df["BB_Middle"] = ma
        df["BB_Upper"] = ma + (num_std * std)
        df["BB_Lower"] = ma - (num_std * std)
        return df

    @staticmethod
    def calculate_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
        """計算隨機指標 (KDJ 操盤線)"""
        low_list = df["Low"].rolling(window=n, min_periods=1).min()
        high_list = df["High"].rolling(window=n, min_periods=1).max()
        rsv = (df["Close"] - low_list) / (high_list - low_list) * 100
        rsv = rsv.fillna(50)  # 防呆處理
        
        k = np.zeros(len(df))
        d = np.zeros(len(df))
        j = np.zeros(len(df))
        
        current_k, current_d = 50.0, 50.0
        for i in range(len(df)):
            current_k = (1 / m1) * rsv.iloc[i] + ((m1 - 1) / m1) * current_k
            current_d = (1 / m2) * current_k + ((m2 - 1) / m2) * current_d
            k[i] = current_k
            d[i] = current_d
            j[i] = 3 * current_k - 2 * current_d
            
        df["KDJ_K"] = k
        df["KDJ_D"] = d
        df["KDJ_J"] = j
        return df

# ==========================================
# 🌐 WebSocket 支持
# ==========================================
endpoint = PubSubEndpoint()
app.include_router(endpoint.router, prefix='/ws')

# ==========================================
# 📡 HTTP 請求防鎖死安全裝甲頭
# ==========================================
def get_secure_ticker(stock_id: str) -> yf.Ticker:
    ticker_str = f"{stock_id}.TW" if stock_id.isdigit() else stock_id
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7'
    })
    return yf.Ticker(ticker_str, session=session)

# ==========================================
# 🔍 路由一：模糊搜尋股票代碼與別名
# ==========================================
@app.get("/api/search", summary="股票模糊搜尋")
def search_stock(q: str = Query(..., description="請輸入股票代碼、名稱或拼音別名")):
    if not q:
        return []
    q_clean = q.strip()
    result = stock_df[
        stock_df["name"].str.contains(q_clean, case=False, na=False) |
        stock_df["code"].str.contains(q_clean, na=False) |
        stock_df["alias"].str.contains(q_clean, case=False, na=False)
    ]
    return result.head(15).to_dict(orient="records")

# ==========================================
# 📈 路由二：超高集成度技術指標主端點 (K線、MA、RSI、MACD、布林、KDJ)
# ==========================================
@app.get("/api/stock/{stock_id}", summary="股票完全體技術指標整合數據")
def get_stock_data(stock_id: str, period: str = "3mo", ma_windows: List[int] = [5, 10, 20, 60, 120], rsi_window: int = 14, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9, bollinger_window: int = 20, bollinger_num_std: float = 2.0):
    try:
        stock_name = stock_id
        match = stock_df[stock_df["code"].str.upper() == stock_id.upper()]
        if not match.empty:
            stock_name = match.iloc[0]["name"]

        stock = get_secure_ticker(stock_id)
        
        # 動態間隔適配器
        interval_map = {"5y": "1wk", "max": "1mo"}
        interval = interval_map.get(period, "1d")
        
        df = stock.history(period=period, interval=interval)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"無法取得該標的歷史數據: {stock_id}")

        # 核心計算鏈
        df = QuantIndicators.calculate_ma(df)
        df = QuantIndicators.calculate_rsi(df)
        df = QuantIndicators.calculate_macd(df)
        df = QuantIndicators.calculate_bollinger_bands(df)
        df = QuantIndicators.calculate_kdj(df)

        # 清理並置換無效值，全面阻斷 JSON NaN 崩潰
        df = df.replace({np.nan: None, np.inf: None, -np.inf: None})

        # 構建毫秒級 Timestamp 的序列陣列
        candlestick, volume, rsi, macd_list, bollinger, kdj_list = [], [], [], [], [], []
        ma_data = {"ma5": [], "ma10": [], "ma20": [], "ma60": [], "ma120": []}

        for idx, row in df.iterrows():
            ts = int(idx.timestamp() * 1000)
            if row["Open"] is None or row["Close"] is None:
                continue

            # OHLC 數據
            candlestick.append([ts, round(row["Open"], 2), round(row["High"], 2), round(row["Low"], 2), round(row["Close"], 2)])
            
            # 成交量
            if row["Volume"] is not None:
                volume.append([ts, int(row["Volume"])])

            # 各均線分配
            for w in [5, 10, 20, 60, 120]:
                if row[f"MA{w}"] is not None:
                    ma_data[f"ma{w}"].append([ts, round(row[f"MA{w}"], 2)])

            # RSI
            if row["RSI14"] is not None:
                rsi.append([ts, round(row["RSI14"], 2)])

            # MACD
            if row["DIF"] is not None and row["DEA"] is not None and row["MACD_Hist"] is not None:
                macd_list.append({
                    "time": ts,
                    "dif": round(row["DIF"], 2),
                    "dea": round(row["DEA"], 2),
                    "hist": round(row["MACD_Hist"], 2)
                })

            # 布林通道
            if row["BB_Middle"] is not None:
                bollinger.append({
                    "time": ts,
                    "upper": round(row["BB_Upper"], 2),
                    "middle": round(row["BB_Middle"], 2),
                    "lower": round(row["BB_Lower"], 2)
                })

            # KDJ
            if row["KDJ_K"] is not None:
                kdj_list.append({
                    "time": ts,
                    "k": round(row["KDJ_K"], 2),
                    "d": round(row["KDJ_D"], 2),
                    "j": round(row["KDJ_J"], 2)
                })

        # 基本資料卡抓取與防禦
        info_card = {"longName": stock_name, "industry": "未分類", "marketCap": "暫無資料", "trailingPE": "N/A", "dividendYield": "0%"}
        try:
            raw_info = stock.info
            if raw_info:
                info_card["longName"] = raw_info.get("longName", stock_name)
                info_card["industry"] = raw_info.get("industry", "未分類")
                cap = raw_info.get("marketCap")
                info_card["marketCap"] = f"{round(cap / 100000000, 2)} 億" if cap else "暫無資料"
                pe = raw_info.get("trailingPE")
                info_card["trailingPE"] = round(pe, 2) if pe else "N/A"
                dy = raw_info.get("dividendYield")
                info_card["dividendYield"] = f"{round(dy * 100, 2)}%" if dy else "0%"
        except Exception:
            pass

        return {
            "status": "success",
            "stock_id": stock_id,
            "stock_name": stock_name,
            "period": period,
            "candlestick": candlestick,
            "volume": volume,
            "ma": ma_data,
            "rsi": rsi,
            "macd": macd_list,
            "bollinger": bollinger,
            "kdj": kdj_list,
            "info_card": info_card
        }
    except Exception as e:
        logger.error(f"獲取技術指標異常: {str(e)}")
        return {"status": "error", "message": f"後端引擎計算失敗: {str(e)}"}

# ==========================================
# 🧠 路由三：多重時間跨度量化策略 AI 股評端點
# ==========================================
@app.get("/api/ai_analysis/{stock_id}", summary="量化回測策略分析")
def get_ai_analysis(stock_id: str, period: str = "3mo"):
    try:
        stock = get_secure_ticker(stock_id)
        df = stock.history(period=period)
        
        if df.empty or len(df) < 20:
            return {"status": "error", "analysis": "數據樣本量不足 (少於 20 個交易日)，無法進行多因子量化評估。"}
            
        # 計算輔助決策特徵
        df = QuantIndicators.calculate_ma(df)
        df = QuantIndicators.calculate_rsi(df)
        df = QuantIndicators.calculate_macd(df)
        df = QuantIndicators.calculate_bollinger_bands(df)
        
        current_price = round(df["Close"].iloc[-1], 2)
        price_change = round(((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100, 2)
        
        latest_rsi = df["RSI14"].iloc[-1]
        latest_macd_hist = df["MACD_Hist"].iloc[-1]
        latest_close = df["Close"].iloc[-1]
        latest_bbu = df["BB_Upper"].iloc[-1]
        latest_bbl = df["BB_Lower"].iloc[-1]

        # 策略標籤診斷機制
        period_titles = {
            "1mo": "【超短線極速動能分析】",
            "3mo": "【短線波段策略趨勢評估】",
            "6mo": f"【{stock_id} 中期籌碼結構報告】",
            "1y": "【跨年度歷史日線大總檢】",
            "5y": "【跨景氣大週期配置策略】"
        }
        title_tag = period_titles.get(period, "【量化智能看盤策略】")

        # 1. 型態動能診斷
        if price_change > 15:
            pattern = "呈現強烈噴發多頭型態，主力追價意願極高"
        elif 0 < price_change <= 15:
            pattern = "穩步築底盤堅，屬於多方控盤的緩步上升軌道"
        elif -15 <= price_change < 0:
            pattern = "處於弱勢修正箱型或空頭波段，上方解套賣壓沈重"
        else:
            pattern = "遭遇斷崖式修正，籌碼大機率出現多殺多恐慌潮"

        # 2. 多因子技術指標量化打分
        signals = []
        if latest_rsi and latest_rsi > 75: signals.append("RSI 進入超買過熱區，追高需防回檔")
        elif latest_rsi and latest_rsi < 25: signals.append("RSI 進入超賣恐慌區，短線具備乖離反彈契機")
        
        if latest_macd_hist and latest_macd_hist > 0: signals.append("MACD 柱狀體位於正值，多頭波段動能尚未止熄")
        else: signals.append("MACD 柱狀體收縮或翻負，多方攻勢暫歇，全面防禦空方摜壓")

        if latest_close and latest_bbu and latest_close >= latest_bbu: signals.append("股價強行突破布林上軌，進入異常超漲波動，注意爆量收長上影線")
        elif latest_close and latest_bbl and latest_close <= latest_bbl: signals.append("股價摜破布林下軌，極端超跌，等待量縮收腳的右側交易信號")

        signal_str = "、".join(signals) if signals else "目前各技術指標處於中性平衡區，無極端超漲超跌信號。"

        # 3. 操盤大腦風控策略輸出
        if period in ["1mo", "3mo"]:
            advice = f"超短線波段波動劇烈（震幅達 {price_change}%）。操作上應嚴格以 MA5 均線作為移動停利防線。不追高，若帶量強勢突破近期阻力平台，可嘗試極小部位的順勢追擊，但必須隨時做好快進快出的撤退準備。"
        elif period in ["6mo", "1y"]:
            advice = f"中期報酬為 {price_change}%，籌碼正在進行箱型整理與大洗盤。目前適合採取『拉回不破前低、或回測季線支撐有守』時分批建倉。耐心等待中期均線（MA20/MA60）形成黃金交叉的右側攻擊波訊號。"
        else:
            advice = f"這是一份跨越景氣大循環的量化報告。當前的短期波動皆是雜訊。長期存股與配置族群應重點核對該股的產業護城河、歷年股利發放穩定度以及 trailing PE 是否落在歷史本益比河流圖的下緣合理評價區。"

        analysis_text = f"{title_tag}\n\n" \
                        f"🤖 該股當前收盤價為 {current_price} 元。\n\n" \
                        f"📈 區間型態觀察：在此時間切片下，走勢呈現『{pattern}』。\n\n" \
                        f"📊 多因子即時體檢：{signal_str}。\n\n" \
                        f"💡 導航大腦操盤指引：{advice}\n\n" \
                        f"⚠️ 警示：本分析為量化引擎根據多因子平均線、RSI、MACD 與布林通道之歷史智慧資料回測模擬，不構成實質投資建議。請務必控管好資金部位與風控防線。"

        return {"status": "success", "analysis": analysis_text}
    except Exception as e:
        return {"status": "error", "analysis": f"量化策略引擎生成異常: {str(e)}"}

# ==========================================
# 📊 路由四：全自動量化選股與突破監測端點 (量化核心精髓)
# ==========================================
@app.get("/api/screener/breakout", summary="布林通道與均線突破量化選股器")
def scan_breakout_stocks(strategy: str = "bollinger_upper"):
    """
    全自動高階量化選股器，掃描 CSV 資料庫中符合特定技術面突破型態的潛力標的。
    """
    if stock_df.empty:
        return {"status": "error", "message": "股票資料庫為空，無法執行選股掃描。"}
        
    scan_targets = stock_df.head(20).to_dict(orient="records") # 為避免掃描上千檔導致超時，先鎖定前 20 檔熱門股進行展示計算
    results = []
    
    for item in scan_targets:
        code = item["code"]
        name = item["name"]
        try:
            stock = get_secure_ticker(code)
            df = stock.history(period="1mo")
            if df.empty or len(df) < 20: continue
            
            df = QuantIndicators.calculate_ma(df)
            df = QuantIndicators.calculate_bollinger_bands(df)
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            if strategy == "bollinger_upper":
                # 策略 1：強勢股突破布林上軌
                if prev["Close"] < prev["BB_Upper"] and latest["Close"] >= latest["BB_Upper"]:
                    results.append({"code": code, "name": name, "reason": "股價今日強勢帶量突破布林通道上軌，多頭動能爆發"})
            elif strategy == "golden_cross":
                # 策略 2：MA5 黃金交叉 MA20
                if prev["MA5"] <= prev["MA20"] and latest["MA5"] > latest["MA20"]:
                    results.append({"code": code, "name": name, "reason": "MA5 短天期均線向上黃金交叉 MA20 月線，多頭結構確立"})
        except Exception:
            continue
            
    return {
        "status": "success",
        "strategy_applied": strategy,
        "scan_count": len(scan_targets),
        "match_count": len(results),
        "matches": results
    }
