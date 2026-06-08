from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np

# =========================
# 股票資料庫
# =========================
try:
    stock_df = pd.read_csv("app/stocks.csv", dtype=str)
except Exception:
    # 防呆：如果讀取不到 csv，建立一個空 DataFrame 避免伺服器崩潰
    stock_df = pd.DataFrame(columns=["code", "name", "alias"])

stock_df = stock_df.fillna("")

# =========================
# FastAPI
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 搜尋股票
# =========================
@app.get("/api/search")
def search_stock(q: str):
    if not q:
        return []
    result = stock_df[
        stock_df["name"].str.contains(q, case=False, na=False) |
        stock_df["code"].str.contains(q, na=False) |
        stock_df["alias"].str.contains(q, case=False, na=False)
    ]
    return result.head(10).to_dict(orient="records")

# =========================
# 股票K線 + 技術指標 + 基本資料卡
# =========================
@app.get("/api/stock/{stock_id}")
def get_stock_data(stock_id: str, period: str = "3mo"):
    try:
        stock_name = stock_id
        match = stock_df[stock_df["code"].str.upper() == stock_id.upper()]
        if not match.empty:
            stock_name = match.iloc[0]["name"]

        ticker = f"{stock_id}.TW" if stock_id.isdigit() else stock_id
        stock = yf.Ticker(ticker)

        # 根據 period 抓取歷史資料
        # 日線資料預設已包含一整年開盤日的 Open, High, Low, Close, Volume
        if period == "1mo":
            df = stock.history(period="1mo")
        elif period == "3mo":
            df = stock.history(period="3mo")
        elif period == "6mo":
            df = stock.history(period="6mo")
        elif period == "1y":
            df = stock.history(period="1y")
        elif period == "5y":
            df = stock.history(period="5y", interval="1wk")
        elif period == "max":
            df = stock.history(period="max", interval="1mo")
        else:
            df = stock.history(period="3mo")

        if df.empty:
            return {"status": "error", "message": f"找不到股票代碼: {ticker}"}

        # 1. 均線計算 (MA5 / MA20)
        df["MA5"] = df["Close"].rolling(window=5).mean()
        df["MA20"] = df["Close"].rolling(window=20).mean()

        # 2. 技術指標：手寫 RSI 計算 (14天區間)
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI14"] = 100 - (100 / (1 + rs))

        # 替換空值避免 JSON 解析失敗
        df = df.replace({np.nan: None, np.inf: None, -np.inf: None})

        # 整理圖表陣列
        chart_data = []
        ma5_data = []
        ma20_data = []
        rsi_data = []
        volume_data = []  # 新增：盤中成交量

        for index, row in df.iterrows():
            timestamp = int(index.timestamp() * 1000)
            if row["Open"] is None or row["Close"] is None:
                continue

            # OHLC 數據 (開盤、最高、最低、收盤)
            chart_data.append([
                timestamp,
                round(float(row["Open"]), 2),
                round(float(row["High"]), 2),
                round(float(row["Low"]), 2),
                round(float(row["Close"]), 2)
            ])

            # 成交量數據
            if row["Volume"] is not None:
                volume_data.append([timestamp, int(row["Volume"])])

            if row["MA5"] is not None:
                ma5_data.append([timestamp, round(float(row["MA5"]), 2)])

            if row["MA20"] is not None:
                ma20_data.append([timestamp, round(float(row["MA20"]), 2)])

            if row["RSI14"] is not None:
                rsi_data.append([timestamp, round(float(row["RSI14"]), 2)])

        # 3. 股票基本資料卡數據抓取 (防呆處理，避免 yfinance info 速度慢或回傳空值)
        info_card = {
            "longName": stock_name,
            "industry": "未分類",
            "marketCap": "暫無資料",
            "trailingPE": "N/A",
            "dividendYield": "0%"
        }
        try:
            raw_info = stock.info
            if raw_info:
                info_card["longName"] = raw_info.get("longName", stock_name)
                info_card["industry"] = raw_info.get("industry", "未分類")
                
                # 市值換算成億元更直覺
                cap = raw_info.get("marketCap")
                info_card["marketCap"] = f"{round(cap / 100000000, 2)} 億" if cap else "暫無資料"
                
                pe = raw_info.get("trailingPE")
                info_card["trailingPE"] = round(pe, 2) if pe else "N/A"
                
                dy = raw_info.get("dividendYield")
                info_card["dividendYield"] = f"{round(dy * 100, 2)}%" if dy else "0%"
        except Exception:
            pass # 抓取 info 失敗時採用預設值

        return {
            "status": "success",
            "stock_id": stock_id,
            "stock_name": stock_name,
            "period": period,
            "count": len(chart_data),
            "candlestick": chart_data,
            "volume": volume_data,
            "ma5": ma5_data,
            "ma20": ma20_data,
            "rsi": rsi_data,
            "info_card": info_card
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

# =========================
# AI 推薦分析端點
# =========================
# =========================
# 升級版：動態時間跨度 AI 推薦分析端點
# =========================
@app.get("/api/ai_analysis/{stock_id}")
def get_ai_analysis(stock_id: str, period: str = "3mo"):
    """
    根據前端傳入的 period (1mo, 3mo, 1y, max...)
    自動調整分析的資料範圍，給出對應時間跨度的專業操盤建議。
    """
    try:
        ticker = f"{stock_id}.TW" if stock_id.isdigit() else stock_id
        
        # 加上我們上一波寫的防封鎖 session 裝甲
        import requests
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        stock = yf.Ticker(ticker, session=session)
        
        # 🎯 關鍵：讓 AI 抓取的歷史資料長度與前端完全一致
        if period in ["1mo", "3mo", "6mo", "1y", "5y", "max"]:
            df = stock.history(period=period)
        else:
            df = stock.history(period="3mo")
            period = "3mo"
        
        if df.empty:
            return {"status": "error", "analysis": "無法取得足夠的歷史數據進行該區間的 AI 分析。"}
            
        current_price = round(df["Close"].iloc[-1], 2)
        price_change = round(((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100, 2)
        
        # 定義時間跨度的中文描述
        period_names = {
            "1mo": "【近 1 個月 - 超短線極速分析】",
            "3mo": "【近 3 個月 - 短線波段策略趨勢】",
            "6mo": " {stock_id} 【近 半年 - 中期籌碼整理報告】",
            "1y": "【近 1 年 - 跨年度歷史日線總結】",
            "5y": "【近 5 年 - 跨越景氣循環週線大布局】",
            "max": "【過去20年 - 史詩級最大歷史週期總體檢】"
        }
        title_tag = period_names.get(period, "【智慧綜合看盤分析】")

        # 根據不同的跨度與漲跌幅，給出截然不同的操盤靈魂
        if period in ["1mo", "3mo"]:
            # 短線邏輯：聚焦在動量、爆發力與停損支撐
            trend = "短線爆發力強勁，動能十足" if price_change > 0 else "短線面臨賣壓宣洩，技術面修正中"
            strategy = f"對於短線投資者，目前價格波動激烈（區間內漲跌幅 {price_change}%）。建議密切監控 MA5 均線防線，若跌破應果斷嚴格執行停損；若帶量突破前高，可進行極短線順勢加碼。"
        elif period in ["6mo", "1y"]:
            # 中線邏輯：聚焦在波段築底、季線半年線支撐與基本面搭配
            trend = "波段多頭排列，築底完成向上" if price_change > 0 else "中線陷入箱型整理，或處於空頭通道"
            strategy = f"從中長線波段視角來看，該股近期的核心支撐已經浮現（區間累計報酬率為 {price_change}%）。適合採取『拉回不破底』時分批佈局的策略，耐心等待中期均線黃金交叉的波段行情。"
        else:
            # 長線邏輯 (5y, max)：聚焦在20年大歷史循環、公司護城河與資產配置
            trend = "長期價值持續增長，具備產業護城河" if price_change > 0 else "長期走勢處於大週期循環低谷"
            strategy = f"這是一份跨越數年週期的史詩級長線體檢報告（大週期報酬率為 {price_change}%）。此時的日常震盪皆為雜訊，長線投資人應關注該股的長期本益比（PE）是否在歷史合理評價區間，適合透過定期定額或價值型長期存股進行配置。"

        analysis_text = f"{title_tag}\n\n" \
                        f"🤖 該股目前最新收盤價為 {current_price} 元。\n\n" \
                        f"📊 區間型態觀察：在此時間跨度下，整體走勢呈現『{trend}』。\n\n" \
                        f"💡 導航大腦建議：{strategy}\n\n" \
                        f"⚠️ 警示：以上內容為 AI 根據歷史數據與移動平均線之智慧模擬，不代表絕對投資承諾，操作時請務必控管好資金部位。"

        return {"status": "success", "analysis": analysis_text}
    except Exception as e:
        return {"status": "error", "analysis": f"AI 分析生成失敗: {str(e)}"}

    """
    此處可直接放入大模型 API 串接密碼。
    目前先以精密的邏輯規則引擎，根據股票當前狀態動態生成專業級 AI 股評。
    """
    try:
        ticker = f"{stock_id}.TW" if stock_id.isdigit() else stock_id
        stock = yf.Ticker(ticker)
        df = stock.history(period="1mo")
        
        if df.empty:
            return {"analysis": "無法取得足夠數據進行 AI 分析。"}
            
        current_price = round(df["Close"].iloc[-1], 2)
        price_change = round(((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100, 2)
        
        trend = "多頭排列，表現強勢" if price_change > 0 else "空頭回檔修正中"
        action = "建議拉回找買點，分批佈局" if price_change > 0 else "目前量縮整理，建議觀望或靜待突破"

        analysis_text = f"【AI 智能綜合看盤分析 - {stock_id}】\n\n" \
                        f"🤖 該股當前收盤價為 {current_price} 元。經過近一個月的籌碼與價格追蹤，" \
                        f"整體價格走勢呈現 {trend}（近一個月漲跌幅為 {price_change}%）。\n\n" \
                        f"📊 技術指標觀點：短線 K 線型態顯示其在波段支撐位表現出韌性。配合成交量變化，" \
                        f"多方力道相對穩定。{action}。\n\n" \
                        f"💡 投資策略警示：短期支撐位參考前波低點，若跌破需注意停損風險；若帶量向上突破阻力區，則有望重啟波段攻勢。"

        return {"status": "success", "analysis": analysis_text}
    except Exception as e:
        return {"status": "error", "analysis": f"AI 分析生成失敗: {str(e)}"}
