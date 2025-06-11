# =============================================================================
# FINAL SCRIPT V32.1 - Code Style Refactoring for Readability
# =============================================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import warnings
import time
from math import floor

# --- 環境設定 ---
# 作用: 設定圖形顯示的字體與忽略不必要的警告，提升使用者體驗。
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False 
except Exception as e:
    print(f"字體設定警告: {e}。圖表中的中文可能無法正常顯示。")
warnings.filterwarnings('ignore', category=FutureWarning)


# --- 1. 組態設定 (Config Class) ---
# 作用: 將所有可調整的參數、檔案路徑、欄位名稱等集中管理，方便使用者快速修改與實驗，無需更動核心程式碼。
class Config:
    # --- 檔案與路徑設定 ---
    FILE_PATH = "data_final.xlsx"
    BENCHMARK_FILE_PATH = "e_index.xlsx"
    REPORT_FILE_PATH = 'strategy_reportcor5M.xlsx'

    # --- 資料欄位名稱設定 ---
    COMPANY_COL = 'StockID'
    DATE_COL = 'Date'
    PRICE_COL = 'Close'
    BENCHMARK_NAME = '電子工業類指數'

    # --- 特徵因子設定 ---
    # --- 特徵因子設定 ---
    # 作用: 設定用於模型訓練的特徵欄位名稱清單，請確保名稱與資料欄位一致。
    # FACTOR_COLS = [
    #     'Outstanding_Share_Million', 'Market_Cap', 'PB', 'Value', 'EBITDA_Margin',
    #     'ROA', 'ROE', 'Revenue_Growth', 'Total_Asset_Growth', 'Net_Worth_Growth',
    #     'Debt_Ratio', 'Dividend', 'Momentum', 'Std_Dev', 'Return',
    #     'Turnover','High', 'Low'
    # ]
    FACTOR_COLS = [
        'Market_Cap', 'PB', 'Value', 'EBITDA_Margin',
        'ROE', 'Revenue_Growth', 'Total_Asset_Growth', 'Net_Worth_Growth',
        'Debt_Ratio', 'Dividend', 'Momentum', 'Std_Dev', 'Return',
        'Turnover','Low'
    ]
    # --- 預測目標設定 ---
    # 作用: 設定機器學習模型要預測的目標欄位名稱、預測未來幾日的報酬、以及原始與標準化目標欄位。
    TARGET_COL = 'F_Return_20D_Z'  # 標準化後的未來20日報酬 (Z分數)
    FORWARD_RETURN_DAYS = 20       # 預測未來幾日的報酬 (此處為20日)
    RAW_TARGET_COL = f'F_Return_{FORWARD_RETURN_DAYS}D_Raw'  # 未標準化的未來20日報酬

    # --- 資料集時間範圍設定 ---
    DATA_START_DATE = '2020-02-01'  # 資料集起始日期
    TRAIN_END_DATE = '2023-12-31'   # 訓練集結束日期
    TEST_START_DATE = '2024-01-01'  # 測試集開始日期

    # --- 機器學習模型設定 ---
    RANDOM_STATE = 42  # 隨機種子，確保實驗可重現
    CV_SPLITS = 5      # 交叉驗證分割數
    N_ITER_SEARCH = 100  # 隨機搜尋超參數的次數
    N_FEATURES_TO_SELECT = 5  # 特徵選擇數量
    RANDOM_SEARCH_PARAM_DIST = {  # LightGBM超參數搜尋空間
        'n_estimators': [100, 200, 300, 500],         # 樹的數量
        'learning_rate': [0.01, 0.05, 0.1],           # 學習率
        'num_leaves': [20, 31, 40, 50],               # 每棵樹的最大葉子數
        'max_depth': [5, 10, 15, -1],                 # 樹的最大深度
        'reg_alpha': [0, 0.1, 0.5],                   # L1正則化
        'reg_lambda': [0, 0.1, 0.5],                  # L2正則化
        'colsample_bytree': [0.7, 0.8, 0.9]           # 每棵樹隨機採樣的特徵比例
    }

    # --- 回測策略設定 ---
    INITIAL_CAPITAL = 100000.0 # 初始資金
    TRANSACTION_FEE_RATE = 0.001425 # 交易手續費率 (0.1425%)
    TRANSACTION_TAX_RATE = 0.003 # 交易稅率 (0.3%)
    TOP_N_STOCKS = 10 # 每期選擇的股票數量
    REBALANCE_FREQUENCY = 'M' # 換倉頻率 ('M' = 每月, 'Q' = 每季, 'Y' = 每年)
    INDIVIDUAL_STOP_LOSS_PCT = 0.20 # 個股移動停損百分比 (20%)
1

# --- 2. 輔助繪圖與計算函數 (Helpers) ---

# 作用: 繪製特徵因子之間的相關係數熱力圖。
def plot_correlation_heatmap(df, columns):
    print("\n>>> 繪製特徵相關係數熱力圖...")
    valid_columns = [col for col in columns if col in df.columns]
    correlation_matrix = df[valid_columns].corr()
    
    plt.figure(figsize=(18, 14))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('特徵相關係數熱力圖 (訓練集)', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    print("✅ 圖表已產生。")


# 作用: 繪製模型訓練後，各特徵的重要性長條圖。
def plot_feature_importance(importance, names, model_type):
    print(f"\n>>> 繪製 {model_type} 特徵重要性圖表...")
    fi_df = pd.DataFrame({
        'feature_names': names, 
        'feature_importance': importance
    }).sort_values(by='feature_importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    plt.barh(fi_df['feature_names'], fi_df['feature_importance'])
    plt.xlabel('特徵重要性')
    plt.ylabel('特徵名稱')
    plt.title(f'{model_type} - 特徵重要性', fontsize=16)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    print("✅ 圖表已產生。")


# 作用: 繪製最終的績效圖，並在圖上標註每月資產數字。
def plot_backtest_results(performance_df, benchmark_perf_df=None):
    print("\n>>> 繪製回測績效圖表...")
    fig, ax = plt.subplots(figsize=(16, 9))
    
    ax.stackplot(
        performance_df.index, 
        performance_df['Holdings_Value'], 
        performance_df['Cash'], 
        labels=['策略持股市值', '策略現金'], 
        colors=['#3498db', '#bdc3c7'], 
        alpha=0.7
    )
    
    ax.plot(performance_df.index, performance_df['Total_Assets'], color='red', linewidth=2.5, label='策略總資產', marker='o', markersize=4)
    for date, row in performance_df.iterrows():
        total_assets = row['Total_Assets']
        label = f"{total_assets/1e6:.2f}M" if total_assets >= 1e6 else f"{total_assets/1e3:.0f}k"
        ax.annotate(label, (date, total_assets), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='blue')
    
    if benchmark_perf_df is not None and not benchmark_perf_df.empty:
        plot_benchmark_df = benchmark_perf_df[benchmark_perf_df.index.isin(performance_df.index)]
        ax.plot(plot_benchmark_df.index, plot_benchmark_df['Total_Assets'], color='orange', linestyle='--', linewidth=2.5, label=f'{Config.BENCHMARK_NAME} (買入並持有)')
        for date, row in plot_benchmark_df.iterrows():
            total_assets = row['Total_Assets']
            label = f"{total_assets/1e6:.2f}M" if total_assets >= 1e6 else f"{total_assets/1e3:.0f}k"
            ax.annotate(label, (date, total_assets), textcoords="offset points", xytext=(0, -20), ha='center', fontsize=8, color='dimgray')

    ax.set_title('策略績效 vs. 基準指數', fontsize=16)
    ax.set_xlabel('日期')
    ax.set_ylabel('資產價值 (元)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_xlim(performance_df.index.min(), performance_df.index.max())
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper left')
    plt.tight_layout()
    print("✅ 圖表已產生。")


# 作用: 從績效 DataFrame 中，計算標準化的績效指標。
def calculate_performance_metrics(perf_df, initial_capital, annualization_factor):
    final_assets = perf_df['Total_Assets'].iloc[-1]
    total_return = (final_assets / initial_capital) - 1
    
    perf_df['Period_Return'] = perf_df['Total_Assets'].pct_change().fillna(0)
    num_periods = len(perf_df)
    
    annualized_return = (1 + total_return) ** (annualization_factor / num_periods) - 1 if num_periods > 1 else total_return
    annualized_volatility = perf_df['Period_Return'].std() * np.sqrt(annualization_factor)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
    
    rolling_max = perf_df['Total_Assets'].cummax()
    drawdown = perf_df['Total_Assets'] / rolling_max - 1
    max_drawdown = drawdown.min()
    
    return {
        "最終資產": final_assets, "總報酬率": total_return, "年化報酬率": annualized_return,
        "年化波動率": annualized_volatility, "夏普比率": sharpe_ratio, "最大回撤": max_drawdown
    }


# 作用: 顯示依股票代碼彙總的總交易摘要，並將結果 DataFrame 回傳。
def display_and_get_transaction_summary(transaction_log):
    if not transaction_log:
        print("\n--- 無交易紀錄可供彙總 ---")
        return pd.DataFrame()
        
    trans_df = pd.DataFrame(transaction_log)
    
    buy_summary = trans_df[trans_df['Type'] == 'Buy'].groupby('StockID').agg(
        Buy_Trades=('Type', 'count'), Total_Shares_Bought=('Shares', 'sum'),
        Avg_Buy_Price=('Price', 'mean'), Total_Buy_Value=('Value', 'sum')
    )
    
    sell_summary = trans_df[trans_df['Type'].str.contains('Sell')].groupby('StockID').agg(
        Sell_Trades=('Type', 'count'), Total_Shares_Sold=('Shares', 'sum'),
        Avg_Sell_Price=('Price', 'mean'), Total_Sell_Value=('Value', 'sum')
    )
    
    summary_df = pd.concat([buy_summary, sell_summary], axis=1).fillna(0)
    summary_df['Realized_PnL'] = summary_df['Total_Sell_Value'] - summary_df['Total_Buy_Value']
    summary_df['ROI_%'] = (summary_df['Realized_PnL'] / (summary_df['Total_Buy_Value'] + 1e-6)) * 100
    summary_df = summary_df.sort_values(by='Realized_PnL', ascending=False)
    
    summary_df = summary_df.astype({
        'Buy_Trades': int, 'Total_Shares_Bought': int, 
        'Sell_Trades': int, 'Total_Shares_Sold': int
    })
    
    final_columns = [
        'Buy_Trades', 'Total_Shares_Bought', 'Avg_Buy_Price', 'Total_Buy_Value',
        'Sell_Trades', 'Total_Shares_Sold', 'Avg_Sell_Price', 'Total_Sell_Value',
        'Realized_PnL', 'ROI_%'
    ]
    summary_df = summary_df[final_columns]
    
    formatters = {
        'Avg_Buy_Price': '{:,.2f}'.format, 'Total_Buy_Value': '{:,.0f}'.format,
        'Avg_Sell_Price': '{:,.2f}'.format, 'Total_Sell_Value': '{:,.0f}'.format,
        'Realized_PnL': '{:,.0f}'.format, 'ROI_%': '{:,.2f}%'.format
    }
    
    print("\n--- 回測總交易彙總表 (依獲利排序) ---")
    print(summary_df.to_string(formatters=formatters))
    
    return summary_df


# --- 3. 核心流程函式 (Core Functions) ---

# 作用: 載入並處理市場基準指數的資料。
def load_benchmark_data(file_path):
    print(f"\n>>> 載入基準指數資料: {file_path}")
    try:
        benchmark_df = pd.read_excel(file_path)
        date_col_found, close_col_found = None, None
        possible_date_cols = ['Date', '日期', '年月日', '交易日']
        possible_close_cols = ['Close', '收盤價', '收盤指數', '價格指數值']
        
        for col in possible_date_cols:
            if col in benchmark_df.columns:
                date_col_found = col
                break
        for col in possible_close_cols:
            if col in benchmark_df.columns:
                close_col_found = col
                break
                
        if date_col_found and close_col_found:
            print(f"    - 偵測到 日期欄位:'{date_col_found}', 收盤價欄位:'{close_col_found}'")
            benchmark_df = benchmark_df.rename(columns={date_col_found: 'Date', close_col_found: 'Close'})
            benchmark_df['Date'] = pd.to_datetime(benchmark_df['Date'], errors='coerce')
            benchmark_df = benchmark_df[['Date', 'Close']].sort_values('Date').dropna().reset_index(drop=True)
            print("✅ 基準資料載入成功。")
            return benchmark_df
        else:
            raise ValueError("在指數檔案中找不到可辨識的日期或收盤價欄位。")
    except Exception as e:
        print(f"❌ 錯誤：無法載入基準檔案。錯誤訊息: {e}")
        return None


# 作用: 執行完整的機器學習流程。
def run_ml_pipeline():
    print("--- 開始執行【機器學習流程】 ---")
    try:
        df = pd.read_excel(Config.FILE_PATH)
        df[Config.DATE_COL] = pd.to_datetime(df[Config.DATE_COL])
        df.sort_values(by=[Config.COMPANY_COL, Config.DATE_COL], inplace=True)
        print("✅ 步驟 1: 資料讀取成功。")
    except Exception as e:
        print(f"❌ 步驟 1 失敗: {e}")
        return None, None, None, None, None

    print(f"\n>>> 步驟 2: 計算預測目標...")
    df[f'Future_Price_{Config.FORWARD_RETURN_DAYS}D'] = df.groupby(Config.COMPANY_COL)[Config.PRICE_COL].shift(-Config.FORWARD_RETURN_DAYS)
    df[Config.RAW_TARGET_COL] = (df[f'Future_Price_{Config.FORWARD_RETURN_DAYS}D'] / df[Config.PRICE_COL]) - 1
    df[Config.TARGET_COL] = df.groupby(Config.DATE_COL)[Config.RAW_TARGET_COL].transform(lambda x: (x - x.mean()) / x.std())
    df = df.drop(columns=[f'Future_Price_{Config.FORWARD_RETURN_DAYS}D'])
    print("✅ 預測目標計算完成。")

    print("\n>>> 步驟 3: 切分資料集...")
    df_filtered = df[df[Config.DATE_COL] >= Config.DATA_START_DATE].copy()
    train_df = df_filtered[df_filtered[Config.DATE_COL] <= Config.TRAIN_END_DATE].copy()
    test_df_for_backtest = df_filtered[df_filtered[Config.DATE_COL] >= Config.TEST_START_DATE].copy()
    test_df_for_model = test_df_for_backtest.copy()
    all_cols_to_check = Config.FACTOR_COLS + [Config.TARGET_COL, Config.RAW_TARGET_COL]
    train_df.dropna(subset=all_cols_to_check, inplace=True)
    test_df_for_model.dropna(subset=all_cols_to_check, inplace=True)
    if train_df.empty or test_df_for_model.empty:
        print("❌ 錯誤: 切分後資料集為空。")
        return None, None, None, None, None
    print(f"✅ 資料切分完成: 訓練集 {len(train_df)} 筆, 測試集(回測用) {len(test_df_for_backtest)} 筆, 測試集(模型評估用) {len(test_df_for_model)} 筆。")
    X_train, y_train = train_df[Config.FACTOR_COLS], train_df[Config.TARGET_COL]
    
    plot_correlation_heatmap(train_df, Config.FACTOR_COLS)

    print(f"\n>>> 步驟 4: 超參數調校...")
    tscv = TimeSeriesSplit(n_splits=Config.CV_SPLITS)
    random_search = RandomizedSearchCV(
        estimator=LGBMRegressor(random_state=Config.RANDOM_STATE, n_jobs=-1, verbosity=-1),
        param_distributions=Config.RANDOM_SEARCH_PARAM_DIST, 
        n_iter=Config.N_ITER_SEARCH, cv=tscv, n_jobs=-1, 
        scoring='r2', verbose=1, random_state=Config.RANDOM_STATE
    )
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    print(f"✅ 調校完成。最佳參數: {best_params}")

    print(f"\n>>> 步驟 5: 因子篩選...")
    base_model = LGBMRegressor(**best_params, random_state=Config.RANDOM_STATE, n_jobs=-1, verbosity=-1)
    selector = SelectFromModel(base_model, max_features=Config.N_FEATURES_TO_SELECT, threshold=-np.inf)
    selector.fit(X_train, y_train)
    selected_features = X_train.columns[selector.get_support()].tolist()
    print(f"✅ 篩選完成，選出特徵: {selected_features}")

    print("\n>>> 步驟 6: 訓練最終模型...")
    X_train_selected = X_train[selected_features]
    final_model = LGBMRegressor(**best_params, random_state=Config.RANDOM_STATE, n_jobs=-1, verbosity=-1)
    final_model.fit(X_train_selected, y_train)
    print("✅ 最終模型訓練完成。")

    print("\n>>> 步驟 7: 評估模型能力...")
    X_test_selected = test_df_for_model[selected_features]
    y_test = test_df_for_model[Config.TARGET_COL]
    pred_train = final_model.predict(X_train_selected)
    pred_test = final_model.predict(X_test_selected)
    r2_train, r2_test = r2_score(y_train, pred_train), r2_score(y_test, pred_test)
    ic_train, _ = spearmanr(y_train, pred_train)
    ic_test, _ = spearmanr(y_test, pred_test)
    print("--- 模型能力評估結果 ---")
    print(f"  - 訓練集 R-squared: {r2_train:.4f}, 測試集 R-squared: {r2_test:.4f}")
    print(f"  - 訓練集 IC: {ic_train:.4f}, 測試集 IC: {ic_test:.4f}")
    print("-" * 30)
    
    feature_importance_df = pd.DataFrame({
        'Factor': selected_features,
        'Importance': final_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    plot_feature_importance(feature_importance_df['Importance'], feature_importance_df['Factor'], 'LightGBM Regressor')
    print("\n--- 機器學習流程結束 ---")
    return final_model, selected_features, df, test_df_for_backtest, feature_importance_df


# 作用: 獨立計算基準指數的回測績效。
def calculate_benchmark_performance(benchmark_df, strategy_perf_df, initial_capital, annualization_factor):
    print("\n>>> 計算基準績效...")
    print(f"    - 策略回測日期: {strategy_perf_df.index.min().date()} 到 {strategy_perf_df.index.max().date()}")
    print(f"    - 基準指數日期: {benchmark_df[Config.DATE_COL].min().date()} 到 {benchmark_df[Config.DATE_COL].max().date()}")
    
    benchmark_perf_df = benchmark_df[benchmark_df[Config.DATE_COL].isin(strategy_perf_df.index)].copy()
    
    if benchmark_perf_df.empty:
        print("    - ⚠️ 警告：日期範圍沒有重疊，無法比較。")
        return None, None
        
    print(f"    - ✅ 日期成功對齊！找到 {len(benchmark_perf_df)} 筆重疊資料。")
    first_day_index_val = benchmark_perf_df['Close'].iloc[0]
    benchmark_perf_df['Total_Assets'] = (benchmark_perf_df['Close'] / first_day_index_val) * initial_capital
    benchmark_metrics = calculate_performance_metrics(benchmark_perf_df.copy(), initial_capital, annualization_factor)
    benchmark_perf_df.set_index('Date', inplace=True)
    return benchmark_perf_df, benchmark_metrics


# 作用: 執行「贏家續抱+移動停損」策略的回測。
def run_backtesting_strategy(model, features, full_df, test_df, benchmark_df):
    print("\n--- 開始執行【V28.1 回測流程 (贏家續抱 + 移動停損)】---")
    if test_df.empty:
        print("❌ 回測失敗：測試集為空。")
        return {}
    
    cash = Config.INITIAL_CAPITAL
    holdings = {}
    performance_log = []
    transaction_log = []
    rebalancing_selections_log = []
    price_lookup = test_df.set_index([Config.DATE_COL, Config.COMPANY_COL])[Config.PRICE_COL]
    rebalance_periods = sorted(test_df[Config.DATE_COL].dt.to_period(Config.REBALANCE_FREQUENCY).unique())
    print(f"策略將以「{Config.REBALANCE_FREQUENCY}」為週期進行換倉，共 {len(rebalance_periods)} 次。")

    for period in rebalance_periods:
        start_of_period_data = test_df[test_df[Config.DATE_COL].dt.to_period(Config.REBALANCE_FREQUENCY) == period]
        if start_of_period_data.empty:
            continue
            
        current_date = start_of_period_data[Config.DATE_COL].min()
        print(f"\n-+-+-+- 📅 {period} (執行日: {current_date.date()}) 換倉開始 -+-+-+-")
        
        # ... (後續的停損、盤點、選股、交易邏輯與 V27/V28 版本相同) ...
        print("  [風控] 檢查個股移動停損...")
        stopped_out_stocks = []
        for stock_id, data in list(holdings.items()):
            current_price = price_lookup.get((current_date, stock_id))
            if current_price is None: continue
            new_high_water_mark = max(data['high_water_mark'], current_price)
            holdings[stock_id]['high_water_mark'] = new_high_water_mark
            stop_loss_price = new_high_water_mark * (1 - Config.INDIVIDUAL_STOP_LOSS_PCT)
            if current_price < stop_loss_price:
                print(f"    🚨 {stock_id} 觸發移動停損！(現價 {current_price:.2f} < 停損價 {stop_loss_price:.2f})")
                value = data['shares'] * current_price
                cost = value * (Config.TRANSACTION_FEE_RATE + Config.TRANSACTION_TAX_RATE)
                cash += value - cost
                transaction_log.append({'Date': current_date, 'Type': 'StopLoss Sell', 'StockID': stock_id, 'Shares': data['shares'], 'Price': current_price, 'Value': value, 'Cost': cost, 'Cash_After_Trade': cash})
                stopped_out_stocks.append(stock_id)
        for stock_id in stopped_out_stocks:
            del holdings[stock_id]
        
        holdings_value = sum(data['shares'] * price_lookup.get((current_date, stock_id), 0) for stock_id, data in holdings.items())
        total_assets = holdings_value + cash
        performance_log.append({'Date': current_date, 'Total_Assets': total_assets, 'Holdings_Value': holdings_value, 'Cash': cash, 'Transaction_Costs': 0})
        print(f"  [盤點] 總資產: {total_assets:,.0f}, 持股市值: {holdings_value:,.0f}, 現金: {cash:,.0f}")
        
        prediction_df = start_of_period_data[start_of_period_data[Config.DATE_COL] == current_date].copy()
        X_pred = prediction_df[features]
        predictions = model.predict(X_pred)
        results_df = prediction_df[[Config.COMPANY_COL]].copy()
        results_df['prediction_score'] = predictions
        target_stocks_df = results_df.sort_values(by='prediction_score', ascending=False).head(Config.TOP_N_STOCKS)
        target_stock_ids = set(target_stocks_df[Config.COMPANY_COL].tolist())
        print(f"  [決策] 本期新目標名單: {list(target_stock_ids)}")
        rebalancing_selections_log.append({'Date': current_date, 'Selected_Stocks': ", ".join(list(target_stock_ids))})

        current_stock_ids = set(holdings.keys())
        stocks_to_sell = current_stock_ids - target_stock_ids
        stocks_to_buy = target_stock_ids - current_stock_ids
        stocks_to_hold = current_stock_ids & target_stock_ids
        print(f"    - 持續持有: {list(stocks_to_hold) if stocks_to_hold else '無'}")
        print(f"    - 計畫賣出: {list(stocks_to_sell) if stocks_to_sell else '無'}")
        print(f"    - 計畫買入: {list(stocks_to_buy) if stocks_to_buy else '無'}")
        
        sell_cost, buy_cost = 0, 0
        if stocks_to_sell:
            print("  --- 當期交易詳情: 賣出 ---")
            for stock_id in list(stocks_to_sell):
                price = price_lookup.get((current_date, stock_id)); shares = holdings[stock_id]['shares']
                if price:
                    value = shares * price; cost = value * (Config.TRANSACTION_FEE_RATE + Config.TRANSACTION_TAX_RATE); cash += value - cost; sell_cost += cost
                    transaction_log.append({'Date': current_date, 'Type': 'Sell', 'StockID': stock_id, 'Shares': shares, 'Price': price, 'Value': value, 'Cost': cost, 'Cash_After_Trade': cash})
                    print(f"      -> {stock_id}: 賣出 {shares:,} 股 @ {price:.2f}元, 金額 {value:,.0f}元")
                else:
                    print(f"      -> ⚠️ [強制平倉] 找不到 {stock_id} 當日價格，該部位將從持股中移除，價值視為0。")
                    transaction_log.append({'Date': current_date, 'Type': 'Forced Sell', 'StockID': stock_id, 'Shares': shares, 'Price': 0, 'Value': 0, 'Cost': 0, 'Cash_After_Trade': cash})
                del holdings[stock_id]
        if sell_cost > 0: print(f"    - 賣出完成後，現金更新為: {cash:,.0f} 元")
        
        if stocks_to_buy:
            print("  --- 當期交易詳情: 買入 ---")
            capital_per_stock = cash / len(stocks_to_buy) if len(stocks_to_buy) > 0 else 0
            print(f"    - 可用現金 {cash:,.0f} 元，預計為每支新股分配 {capital_per_stock:,.0f} 元")
            for stock_id in stocks_to_buy:
                price = price_lookup.get((current_date, stock_id))
                if price:
                    cost_per_share = price * (1 + Config.TRANSACTION_FEE_RATE)
                    shares_to_buy = floor(capital_per_stock / cost_per_share) if cost_per_share > 0 else 0
                    if shares_to_buy > 0:
                        value = shares_to_buy * price; fee = value * Config.TRANSACTION_FEE_RATE
                        if cash >= value + fee:
                            cash -= (value + fee); holdings[stock_id] = {'shares': shares_to_buy, 'high_water_mark': price}; buy_cost += fee
                            transaction_log.append({'Date': current_date, 'Type': 'Buy', 'StockID': stock_id, 'Shares': shares_to_buy, 'Price': price, 'Value': value, 'Cost': fee, 'Cash_After_Trade': cash})
                            print(f"      -> {stock_id}: 買入 {shares_to_buy:,} 股 @ {price:.2f}元, 金額 {value:,.0f}元")
                        else: print(f"      - [警告] 現金不足，無法買入 {shares_to_buy:,} 股 {stock_id}")
        
        performance_log[-1]['Transaction_Costs'] = sell_cost + buy_cost
        print(f"  [執行] 換倉完成。剩餘現金: {cash:,.0f}")
    
    if holdings:
        last_trading_day = test_df[test_df[Config.DATE_COL].dt.year == test_df[Config.DATE_COL].max().year][Config.DATE_COL].max()
        print(f"\n-+-+-+- 📅 {last_trading_day.date()} (年終強制平倉) -+-+-+-")
        print(f"  [結算] 正在以年底收盤價賣出所有剩餘持股...")
        final_sell_cost = 0
        for stock_id, data in list(holdings.items()):
            price = price_lookup.get((last_trading_day, stock_id))
            if price:
                value = data['shares'] * price; cost = value * (Config.TRANSACTION_FEE_RATE + Config.TRANSACTION_TAX_RATE); cash += value - cost; final_sell_cost += cost
                transaction_log.append({'Date': last_trading_day, 'Type': 'Year-End Sell', 'StockID': stock_id, 'Shares': data['shares'], 'Price': price, 'Value': value, 'Cost': cost, 'Cash_After_Trade': cash})
                print(f"    -> 賣出: {stock_id}, {data['shares']:,} 股 @ {price:.2f} 元")
        holdings.clear()
        final_assets = cash
        performance_log.append({'Date': last_trading_day, 'Total_Assets': final_assets, 'Holdings_Value': 0, 'Cash': final_assets, 'Transaction_Costs': final_sell_cost})
        print(f"    - 平倉完成。最終現金餘額: {final_assets:,.0f} 元")

    if not performance_log:
        print("\n❌ 回測結束，無任何紀錄。")
        return {}
    
    strategy_perf_df = pd.DataFrame(performance_log).set_index('Date')
    transaction_summary_df = display_and_get_transaction_summary(transaction_log)
    rebalancing_selections_df = pd.DataFrame(rebalancing_selections_log)
    full_transaction_log_df = pd.DataFrame(transaction_log)
    
    return {
        "strategy_perf": strategy_perf_df, "benchmark_data": benchmark_df,
        "transaction_summary": transaction_summary_df, "rebalancing_selections": rebalancing_selections_df,
        "full_transaction_log": full_transaction_log_df
    }

# --- 4. 主執行函式 (Main) ---
def main():
    print("--- 專案執行啟動 ---")
    
    benchmark_df = load_benchmark_data(Config.BENCHMARK_FILE_PATH)
    ml_results = run_ml_pipeline()
    
    if not ml_results:
        print("\n❌ 因機器學習流程失敗，已終止執行。")
        return

    final_model, selected_features, full_df, test_df, feature_importance_df = ml_results
    backtest_results = run_backtesting_strategy(final_model, selected_features, full_df, test_df, benchmark_df)
    
    if not backtest_results:
        print("\n❌ 因回測流程失敗，已終止執行。")
        return
        
    print("\n>>> 正在將所有分析結果寫入單一 Excel 檔案...")
    try:
        with pd.ExcelWriter(Config.REPORT_FILE_PATH, engine='xlsxwriter') as writer:
            periods_per_year = {'M': 12, 'Q': 4}
            annualization_factor = periods_per_year.get(Config.REBALANCE_FREQUENCY, 12)
            strategy_metrics = calculate_performance_metrics(backtest_results["strategy_perf"], Config.INITIAL_CAPITAL, annualization_factor)
            
            benchmark_perf_df, benchmark_metrics = None, None
            if backtest_results["benchmark_data"] is not None:
                benchmark_perf_df, benchmark_metrics = calculate_benchmark_performance(backtest_results["benchmark_data"], backtest_results["strategy_perf"], Config.INITIAL_CAPITAL, annualization_factor)

            comparison_data = {"策略": strategy_metrics}
            if benchmark_metrics:
                comparison_data[Config.BENCHMARK_NAME] = benchmark_metrics
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_excel(writer, sheet_name='績效總覽 (Performance_Summary)')
            # 在它的前面，加入以下計算：
            total_importance = feature_importance_df['Importance'].sum()
            feature_importance_df['Normalized_Importance_%'] = (feature_importance_df['Importance'] / total_importance) * 100

            # 修改寫入 Excel 的程式碼，將新欄位也寫入
            feature_importance_df.to_excel(writer, sheet_name='最佳因子與權重 (Feature_Importance)', index=False)
            backtest_results["rebalancing_selections"].to_excel(writer, sheet_name='每期持股選擇 (Periodic_Selections)', index=False)
            backtest_results["transaction_summary"].to_excel(writer, sheet_name='總交易彙總 (Transaction_Summary)')
            backtest_results["full_transaction_log"].to_excel(writer, sheet_name='詳細交易紀錄 (Full_Log)', index=False)
            backtest_results["strategy_perf"].to_excel(writer, sheet_name='策略權益曲線 (Performance_Curve)')
        
        print(f"✅ 所有報告已成功儲存至: {Config.REPORT_FILE_PATH}")
    except Exception as e:
        print(f"\n❌ 錯誤：報表儲存失敗。原因: {e}")

    plot_backtest_results(backtest_results["strategy_perf"], benchmark_perf_df)
    print("\n>>> 所有流程執行完畢，正在顯示圖表...")
    plt.show()


# --- 5. 程式進入點 (Entry Point) ---
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"\n--- 總執行時間 ---")
    print(f"⏳ 本次完整流程總共耗時: {minutes} 分 {seconds} 秒 ({elapsed_time:.2f} 秒)")
