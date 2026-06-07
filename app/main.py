from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np

# =========================
# 股票資料庫
# =========================
stock_df = pd.read_csv(
    "app/stocks.csv",
    dtype=str
)

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

    result = stock_df[
        stock_df["name"].str.contains(
            q,
            case=False,
            na=False
        )
        |
        stock_df["code"].str.contains(
            q,
            na=False
        )
        |
        stock_df["alias"].str.contains(
            q,
            case=False,
            na=False
        )
    ]

    return (
        result
        .head(10)
        .to_dict(orient="records")
    )

# =========================
# 股票K線
# =========================
@app.get("/api/stock/{stock_id}")
def get_stock_data(
    stock_id: str,
    period: str = "3mo"
):

    try:

        stock_name = stock_id

        match = stock_df[
            stock_df["code"].str.upper()
            ==
            stock_id.upper()
        ]

        if not match.empty:
            stock_name = match.iloc[0]["name"]

        ticker = (
            f"{stock_id}.TW"
            if stock_id.isdigit()
            else stock_id
        )

        stock = yf.Ticker(ticker)

        # =========================
        # 依時間區間抓資料
        # =========================

        if period == "1mo":

            df = stock.history(
                period="1mo"
            )

        elif period == "3mo":

            df = stock.history(
                period="3mo"
            )

        elif period == "6mo":

            df = stock.history(
                period="6mo"
            )

        elif period == "1y":

            df = stock.history(
                period="1y"
            )

        elif period == "5y":

            df = stock.history(
                period="5y",
                interval="1wk"
            )

        elif period == "max":

            df = stock.history(
                period="max",
                interval="1mo"
            )

        else:

            df = stock.history(
                period="3mo"
            )

        if df.empty:

            return {
                "status": "error",
                "message": f"找不到股票代碼: {ticker}"
            }

        # =========================
        # 均線
        # =========================

        df["MA5"] = (
            df["Close"]
            .rolling(window=5)
            .mean()
        )

        df["MA20"] = (
            df["Close"]
            .rolling(window=20)
            .mean()
        )

        df = df.replace({
            np.nan: None,
            np.inf: None,
            -np.inf: None
        })

        chart_data = []
        ma5_data = []
        ma20_data = []

        for index, row in df.iterrows():

            timestamp = int(
                index.timestamp() * 1000
            )

            if (
                row["Open"] is None
                or
                row["Close"] is None
            ):
                continue

            chart_data.append([
                timestamp,
                round(float(row["Open"]), 2),
                round(float(row["High"]), 2),
                round(float(row["Low"]), 2),
                round(float(row["Close"]), 2)
            ])

            if row["MA5"] is not None:

                ma5_data.append([
                    timestamp,
                    round(float(row["MA5"]), 2)
                ])

            if row["MA20"] is not None:

                ma20_data.append([
                    timestamp,
                    round(float(row["MA20"]), 2)
                ])

        return {
            "status": "success",
            "stock_id": stock_id,
            "stock_name": stock_name,
            "period": period,
            "count": len(chart_data),
            "candlestick": chart_data,
            "ma5": ma5_data,
            "ma20": ma20_data
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }