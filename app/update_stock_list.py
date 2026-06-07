import pandas as pd

stocks = [

# 台股熱門
["2330","台積電","tsmc","TW"],
["2317","鴻海","foxconn","TW"],
["2454","聯發科","mediatek","TW"],
["2303","聯電","umc","TW"],
["2344","華邦電","winbond","TW"],
["2408","南亞科","nanya","TW"],
["2379","瑞昱","realtek","TW"],
["3034","聯詠","novatek","TW"],

# ETF
["0050","元大台灣50","0050","ETF"],
["0056","元大高股息","0056","ETF"],
["00878","國泰永續高股息","00878","ETF"],
["00919","群益台灣精選高息","00919","ETF"],

# 美股
["AAPL","Apple","蘋果","US"],
["NVDA","NVIDIA","輝達","US"],
["TSLA","Tesla","特斯拉","US"],
["AMD","AMD","超微","US"],
["MSFT","Microsoft","微軟","US"],
["GOOGL","Google","谷歌","US"],
["META","Meta","臉書","US"],
["AMZN","Amazon","亞馬遜","US"]


]

df = pd.DataFrame(
stocks,
columns=[
"code",
"name",
"alias",
"market"
]
)

df.to_csv(
"app/stocks.csv",
index=False,
encoding="utf-8-sig"
)

print(f"建立完成，共 {len(df)} 筆")
