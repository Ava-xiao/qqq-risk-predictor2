#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
下载 QQQ 股票历史数据 (OHLCV) 并保存为 CSV 文件
使用 yfinance 库，增强了稳定性与错误处理
"""

import sys
import pandas as pd
import yfinance as yf
from datetime import datetime

def check_python_version():
    """检查 Python 版本，如果低于 3.9 则发出警告（避免类型注解问题）"""
    if sys.version_info < (3, 9):
        print("警告：当前 Python 版本低于 3.9，可能遇到类型注解兼容性问题。")
        print("建议升级到 Python 3.9 或更高版本。")

def fetch_qqq_data(start_date="2023-01-01", end_date=None, save_csv=True):
    """
    下载 QQQ 股票数据
    
    参数:
        start_date (str): 开始日期，格式 'YYYY-MM-DD'
        end_date (str): 结束日期，默认当天 (None 表示当前日期)
        save_csv (bool): 是否将数据保存为 CSV 文件
    
    返回:
        pd.DataFrame: 包含 OHLCV 数据的 DataFrame，失败时返回 None
    """
    ticker_symbol = "QQQ"
    
    # 如果未指定结束日期，则使用今天
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"正在下载 {ticker_symbol} 数据...")
    print(f"时间范围: {start_date} 至 {end_date}")
    
    try:
        # 使用 yf.download 下载数据，设置超时和进度条隐藏
        df = yf.download(
            ticker_symbol,
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False,
            timeout=10,          # 网络超时（秒）
            auto_adjust=False,    # 不自动调整 OHLCV（保留原始数据）
            prepost=False         # 不包含盘前盘后数据
        )
        
        # 检查是否获取到数据
        if df.empty:
            print("错误：未获取到任何数据。请检查：")
            print("  - 网络连接是否正常")
            print("  - 日期范围是否合理")
            print("  - Yahoo Finance 是否可访问")
            return None
        
        print("数据下载成功，正在处理...")
        
        # 重置索引，将日期从索引变为列
        df.reset_index(inplace=True)
        # 确保列名为 'Date'（有些版本返回 'Datetime'，统一为 'Date'）
        if 'Date' not in df.columns:
            # 如果索引名是 'Datetime'，重命名
            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        
        # 只保留必要的 OHLCV 列（yfinance 默认返回 Open, High, Low, Close, Adj Close, Volume）
        # 这里我们保留 Adj Close 以备后续使用，但为了简化也可以只保留 Close
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        # 检查是否存在 Adj Close，若有则保留（可选）
        if 'Adj Close' in df.columns:
            required_columns.append('Adj Close')
        
        # 确保所有需要的列都存在
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            print(f"警告：缺少列 {missing}，将只保留现有列。")
            # 只保留存在的列
            df = df[[col for col in required_columns if col in df.columns]]
        else:
            df = df[required_columns]
        
        # 将 Date 列重命名为 Datetime（与原脚本一致）
        df.rename(columns={'Date': 'Datetime'}, inplace=True)
        
        print(f"成功获取 {len(df)} 条记录。")
        print("\n前 5 行数据预览:")
        print(df.head())
        print("\n后 5 行数据预览:")
        print(df.tail())
        
        # 保存为 CSV 文件
        if save_csv:
            output_filename = f"{ticker_symbol}_{start_date}_to_{end_date}.csv"
            df.to_csv(output_filename, index=False)
            print(f"\n数据已保存至: {output_filename}")
        
        return df
    
    except Exception as e:
        print(f"发生错误: {e}")
        return None

def test_connection():
    """测试与 Yahoo Finance 的基础连接"""
    import requests
    test_url = "https://query1.finance.yahoo.com/v7/finance/download/QQQ"
    try:
        r = requests.get(test_url, timeout=5)
        if r.status_code == 200:
            print("网络连接正常，Yahoo Finance 可访问。")
            return True
        else:
            print(f"网络连接异常，状态码: {r.status_code}")
            return False
    except Exception as e:
        print(f"网络连接失败: {e}")
        return False

if __name__ == "__main__":
    # 检查 Python 版本
    check_python_version()
    
    # 可选：先测试网络连接
    print("正在检查网络...")
    if not test_connection():
        print("网络连接可能有问题，尝试继续下载...")
    
    # 下载数据（使用默认日期范围）
    data = fetch_qqq_data(start_date="2023-01-01", end_date="2026-1-1", save_csv=True)
    
    if data is not None:
        print("\n数据下载完成。")
    else:
        print("\n数据下载失败，请根据错误信息排查。")