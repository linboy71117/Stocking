from app.stocks import stocks
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
import pandas as pd
#from app.stocks import stocks

df = pd.read_csv(
    "app/stocks.csv",
    dtype=str
)

app = FastAPI()

# 允許前端跨網域請求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/search")
def search_stock(q: str):

    result = df[
        df["name"].str.contains(
            q,
            case=False,
            na=False
        )
        |
        df["code"].str.contains(
            q,
            na=False
        )
        |
        df["alias"].str.contains(
            q,
            case=False,
            na=False
        )
    ]

    return (
        result
        .head(10)
        .to_dict(
            orient="records"
        )
    )

@app.get("/api/stock/{stock_id}")
def get_stock_data(stock_id: str):
    try:
        # 轉換台股代碼
        ticker = f"{stock_id}.TW" if stock_id.isdigit() else stock_id
        
        # 抓取最近 3 個月的資料
        stock = yf.Ticker(ticker)
        df = stock.history(period="3mo")
        
        if df.empty:
            return {"status": "error", "message": f"找不到股票代碼: {ticker}"}
        
        # 計算均線
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # 🎯 關鍵修正：將所有 NaN 或 Inf 轉為 None，否則 FastAPI 轉 JSON 會出錯
        df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
        
        chart_data = []
        ma5_data = []
        ma20_data = []
        
        for index, row in df.iterrows():
            # 確保時間戳轉換正確
            timestamp = int(index.timestamp() * 1000)
            
            # 如果價格是 None 就跳過防呆
            if row['Open'] is None or row['Close'] is None:
                continue
                
            chart_data.append([
                timestamp,
                round(float(row['Open']), 2),
                round(float(row['High']), 2),
                round(float(row['Low']), 2),
                round(float(row['Close']), 2)
            ])
            
            # 均線如果還沒算出來 (前幾天) 就給 None
            m5 = round(float(row['MA5']), 2) if row['MA5'] is not None else None
            m20 = round(float(row['MA20']), 2) if row['MA20'] is not None else None
            
            if m5 is not None:
                ma5_data.append([timestamp, m5])
            if m20 is not None:
                ma20_data.append([timestamp, m20])
            
        return {
            "status": "success",
            "stock_id": stock_id,
            "stock_name": stock_id, # 先直接用 id 當名字，避開 yfinance 的 info 阻擋
            "candlestick": chart_data,
            "ma5": ma5_data,
            "ma20": ma20_data
        }
    except Exception as e:
        # 如果真的出錯，強制回傳錯誤訊息，不要變成 404
        return {"status": "error", "message": f"後端計算錯誤: {str(e)}"}
