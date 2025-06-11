import pandas as pd

# --- 參數設定 ---
# 1. 檔案路徑
INPUT_PATH = 'TEJ_data.xlsx'
OUTPUT_PATH = 'data_fill.xlsx' # 最終輸出檔名

# 2. 篩選條件
REQUIRED_DATA_POINTS = 1216

# 3. 欄位名稱 (中文 -> 英文)
# 原始中文欄位名
ORIGINAL_COMPANY_COLUMN_NAME = '證券代碼'
ORIGINAL_DATE_COLUMN_NAME = '年月日'
# 目標英文欄位名
NEW_COMPANY_COLUMN_NAME = 'StockID'
NEW_DATE_COLUMN_NAME = 'Date'


print(">>> 腳本開始執行 <<<")

# ==================================================================
# 步驟 1: 讀取原始 Excel 檔案
# ==================================================================
try:
    df = pd.read_excel(INPUT_PATH)
    print(f"✅ 成功讀取檔案: '{INPUT_PATH}'")
except FileNotFoundError:
    print(f"❌ 錯誤：找不到檔案 '{INPUT_PATH}'。請確認檔案路徑和名稱是否正確。")
    exit()

# ==================================================================
# 步驟 2: 依據資料筆數 (1216筆) 篩選公司
# ==================================================================
print(f"\n>>> 步驟 2: 正在篩選資料筆數剛好為 {REQUIRED_DATA_POINTS} 筆的公司...")

group_counts = df.groupby(ORIGINAL_COMPANY_COLUMN_NAME)[ORIGINAL_COMPANY_COLUMN_NAME].transform('size')
mask = (group_counts == REQUIRED_DATA_POINTS)
df_filtered = df[mask].copy() # 使用 .copy() 避免 SettingWithCopyWarning

# 檢查是否有符合條件的資料
if df_filtered.empty:
    print(f"⚠️ 警告：在 '{INPUT_PATH}' 中，找不到任何一家公司的資料筆數剛好是 {REQUIRED_DATA_POINTS} 筆。")
    print(">>> 腳本結束 <<<")
    exit()
else:
    original_company_count = len(df_filtered[ORIGINAL_COMPANY_COLUMN_NAME].unique())
    print(f"✅ 篩選完成！找到 {original_company_count} 家符合條件的公司。")

# ==================================================================
# 步驟 3: 將欄位名稱從中文替換為英文
# ==================================================================
print("\n>>> 步驟 3: 正在將欄位名稱轉換為英文...")

column_rename_map = {
    '證券代碼': NEW_COMPANY_COLUMN_NAME,
    '股票代碼': NEW_COMPANY_COLUMN_NAME, # 增加彈性
    '年月日': NEW_DATE_COLUMN_NAME,
    '流通在外股數(百萬股)': 'Outstanding_Share_Million',
    '季底普通股市值': 'Market_Cap',
    '股利殖利率': 'Dividend',
    '稅前息前折舊前淨利率': 'EBITDA_Margin',
    '營收成長率': 'Revenue_Growth',
    '總資產成長率': 'Total_Asset_Growth',
    '淨值成長率': 'Net_Worth_Growth',
    '負債比率': 'Debt_Ratio',
    'ROA－綜合損益': 'ROA',
    'ROE－綜合損益': 'ROE',
    '最高價(元)': 'High',
    '最低價(元)': 'Low',
    '收盤價(元)': 'Close',
    '成交值(千元)': 'Value',
    '報酬率％': 'Return',
    '週轉率％': 'Turnover',
    '股價淨值比-TEJ': 'PB'
}

# 執行更名
df_filtered.rename(columns=column_rename_map, inplace=True)
print("✅ 欄位名稱已成功轉換為英文。")
# print("新的欄位列表:", df_filtered.columns.tolist()) # 若想確認可取消註解

# ==================================================================
# 步驟 3.5: 移除收盤價有空值的公司
# ==================================================================
close_col = 'Close'  # 英文欄位名已經轉換
if df_filtered[close_col].isnull().any():
    nan_companies = df_filtered.loc[df_filtered[close_col].isnull(), NEW_COMPANY_COLUMN_NAME].unique()
    print(f"⚠️ 發現 {len(nan_companies)} 家公司有收盤價缺失，將全部移除：{nan_companies}")
    df_filtered = df_filtered[~df_filtered[NEW_COMPANY_COLUMN_NAME].isin(nan_companies)]
    print(f"✅ 已移除收盤價有缺失的公司，剩餘公司數：{df_filtered[NEW_COMPANY_COLUMN_NAME].nunique()}")
else:
    print("✅ 所有公司收盤價皆無缺失。")

# ==================================================================
# 步驟 4: 依公司代碼分組，填充缺失值 (bfill)
# ==================================================================
print("\n>>> 步驟 4: 正在依公司代碼進行資料向後填充 (bfill)...")

# 找出除了關鍵欄位(公司代碼、日期)以外，所有需要被填充的資料欄位
key_columns = [NEW_COMPANY_COLUMN_NAME, NEW_DATE_COLUMN_NAME]
columns_to_fill = df_filtered.columns.difference(key_columns)

# 執行分組填充
# 注意：這裡直接在 df_filtered 上操作，因為之前的步驟已經處理完畢
df_filtered[columns_to_fill] = df_filtered.groupby(NEW_COMPANY_COLUMN_NAME)[columns_to_fill].bfill()
print("✅ 資料填充完成。")

# ==================================================================
# 步驟 5: 儲存最終處理結果
# ==================================================================
print("\n>>> 步驟 5: 正在儲存最終結果...")
try:
    df_filtered.to_excel(OUTPUT_PATH, index=False, engine='xlsxwriter')
    print("-" * 40)
    print(f"🎉 全部處理完成！")
    print(f"最終結果已儲存至 '{OUTPUT_PATH}'")
    print("-" * 40)
except Exception as e:
    print(f"❌ 儲存檔案時發生錯誤: {e}")

print(">>> 程式結束 <<<")