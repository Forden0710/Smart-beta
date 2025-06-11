import pandas as pd

# 讀取CSV（假設檔名為 e_index.csv）
e_index = pd.read_csv('e_index.csv', header=0)

# 檢查欄位名稱
print(e_index.columns)
# Index(['日期', '價格指數值'], dtype='object')

# 轉換日期格式
e_index['日期'] = pd.to_datetime(e_index['日期'], format='%Y/%m/%d')
# 將指數值轉為數值型態
e_index['價格指數值'] = pd.to_numeric(e_index['價格指數值'], errors='coerce')

# 建立季欄位
e_index['Quarter'] = e_index['日期'].dt.to_period('Q')

# 計算每季平均指數
quarterly_avg_index = e_index.groupby('Quarter')['價格指數值'].mean().reset_index()
quarterly_avg_index = quarterly_avg_index.sort_values('Quarter')

# 計算每季報酬率
quarterly_avg_index['Return'] = quarterly_avg_index['價格指數值'].pct_change()
print(quarterly_avg_index)
