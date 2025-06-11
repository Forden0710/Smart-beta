import pandas as pd
import numpy as np

def calculate_and_standardize_final(lookback_period=30):
    """
    (最終修正版)
    支援滾動視窗與標準化，避免因 NaN 導致整家公司被刪除。
    """
    # --- 參數設定 ---
    INPUT_FILE = 'data_fill.xlsx'
    OUTPUT_FILE = 'data_final.xlsx'
    COMPANY_COL = 'StockID'
    DATE_COL = 'Date'
    PRICE_COL = 'Close'

    # --- 步驟 1: 讀取並設定索引 ---
    print(f">>> 步驟 1: 正在讀取資料檔案: {INPUT_FILE}...")
    try:
        df = pd.read_excel(INPUT_FILE)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df.set_index([COMPANY_COL, DATE_COL], inplace=True)
        df.sort_index(inplace=True)
        print("✅ 檔案讀取成功。")
    except Exception as e:
        print(f"❌ 讀取或設定索引時出錯: {e}")
        return

    # --- 步驟 2: 計算每日報酬率 ---
    print("\n>>> 步驟 2: 正在計算每日報酬率...")
    df['Daily_Log_Return'] = df.groupby(level=COMPANY_COL)[PRICE_COL].transform(
        lambda x: np.log(x / x.shift(1))
    )
    print("✅ 每日報酬率計算完成。")

    # --- 步驟 3: 計算滾動動能與標準差 ---
    print(f"\n>>> 步驟 3: 正在計算 {lookback_period} 日滾動指標...")
    df['temp_simple_return'] = np.exp(df['Daily_Log_Return']) - 1

    momentum_series = (
        df.groupby(level=COMPANY_COL)['temp_simple_return']
        .rolling(window=lookback_period, min_periods=10)
        .apply(lambda x: np.prod(1 + x) - 1, raw=False)
        .reset_index(level=0, drop=True)
    )

    std_dev_series = (
        df.groupby(level=COMPANY_COL)['Daily_Log_Return']
        .rolling(window=lookback_period, min_periods=10)
        .std()
        .reset_index(level=0, drop=True)
    )

    df['Momentum'] = momentum_series
    df['Std_Dev'] = std_dev_series
    df.drop(columns=['Daily_Log_Return', 'temp_simple_return'], inplace=True)
    print("✅ 日動能與日標準差計算完成。")

    # --- 步驟 3.5: 標準化特徵 ---
    print("\n>>> 步驟 3.5: 正在標準化數值特徵...")
    cols_to_standardize = [PRICE_COL, 'Momentum', 'Std_Dev']
    standardized_df = pd.DataFrame(index=df.index)

    for col in cols_to_standardize:
        standardized_df[f"{col}_std"] = df.groupby(level=COMPANY_COL)[col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else np.nan
        )

    df = pd.concat([df, standardized_df], axis=1)
    print("✅ 特徵標準化完成。")

    # --- 步驟 4: 儲存最終結果 ---
    print(f"\n>>> 步驟 4: 正在儲存最終結果至 {OUTPUT_FILE}...")
    try:
        df.reset_index(inplace=True)

        # 只針對標準化欄位 NaN 做處理，不清空整筆公司資料
        important_cols = [f"{col}_std" for col in cols_to_standardize]
        df.dropna(subset=important_cols, inplace=True)

        if df.empty:
            print("❌ 警告：移除 NaN 後資料為空！")
            return

        # 移除不想輸出的標準化欄位
        df.drop(columns=['Close_std', 'Momentum_std', 'Std_Dev_std'], inplace=True, errors='ignore')

        df.to_excel(OUTPUT_FILE, index=False, engine='xlsxwriter')
        print("-" * 40)
        print(f"🎉 全部處理完成！")
        print(f"已將結果儲存至 '{OUTPUT_FILE}'")
        print(f"最終檔案的公司數量: {df[COMPANY_COL].nunique()}")
        print("-" * 40)
    except Exception as e:
        print(f"❌ 儲存最終檔案時發生錯誤: {e}")

# --- 主函數執行入口 ---
if __name__ == "__main__":
    calculate_and_standardize_final(lookback_period=30)
