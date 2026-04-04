def save_data(df):
    df.to_csv("aapl.csv", index = False)

import yfinance as yf
data = yf.download("AAPL",
                   start = "2000-01-01",
                   end = "2026-01-01",
                   auto_adjust = False)

data.columns = data.columns.get_level_values(0) 
data.reset_index(inplace = True)
data = data[[
    "Date", "Open", "High", "Low", "Close", "Volume"
]]
save_data(data)

print(data.shape) 
