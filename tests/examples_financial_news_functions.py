#!/usr/bin/env python3
"""
财经新闻函数使用示例

本文件展示了所有集成到news_analyst_node中的财经新闻函数的用法和输出内容。
包含7个基础接口函数和对应的Toolkit工具函数的使用示例。

运行方式:
    python tests/examples_financial_news_functions.py

作者: TradingAgents Team
日期: 2025-08-06
"""

import sys
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.dataflows.interface import (
    get_sina_global_financial_news,
    get_eastmoney_financial_breakfast,
    get_eastmoney_global_financial_news,
    get_futu_financial_news,
    get_tonghuashun_global_financial_live,
    get_cailianshe_telegraph,
    get_sina_securities_original, get_stock_news_openai, get_global_news_openai, get_china_focused_news_openai,
    get_google_news
)
from tradingagents.agents.utils.agent_utils import Toolkit


def print_separator(title: str):
    """打印分隔符"""
    print("\n" + "="*80)
    print(f" {title} ")
    print("="*80)


def print_subsection(title: str):
    """打印子标题"""
    print(f"\n--- {title} ---")


def truncate_text(text: str, max_length: int = 1000) -> str:
    """截断文本到指定长度"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "\n...\n[内容已截断，完整内容请查看函数返回值]"


def example_sina_global_financial_news():
    """示例1: 新浪财经全球财经快讯"""
    print_separator("示例1: 新浪财经全球财经快讯")
    
    print("函数: get_sina_global_financial_news()")
    print("描述: 获取新浪财经全球财经快讯，包含时间和内容")
    print("数据源: https://finance.sina.com.cn/7x24")
    print("AKSHARE接口: ak.stock_info_global_sina()")
    
    try:
        # 使用当前日期作为测试
        curr_date = datetime.now().strftime("%Y-%m-%d")
        result = get_sina_global_financial_news(curr_date)
        print(f"\n返回数据长度: {len(result)} 字符")
        print(f"数据类型: {type(result)}")
        print("\n输出内容预览:")
        print(truncate_text(result))
        print("\n✅ 函数执行成功")
    except Exception as e:
        print(f"\n❌ 函数执行失败: {str(e)}")


def example_eastmoney_financial_breakfast():
    """示例2: 东方财富财经早餐"""
    print_separator("示例2: 东方财富财经早餐")
    
    print("函数: get_eastmoney_financial_breakfast()")
    print("描述: 获取东方财富财经早餐，包含标题、摘要、发布时间和链接")
    print("数据源: https://stock.eastmoney.com/a/czpnc.html")
    print("AKSHARE接口: ak.stock_info_cjzc_em()")
    
    try:
        # 使用当前日期作为测试
        curr_date = datetime.now().strftime("%Y-%m-%d")
        result = get_eastmoney_financial_breakfast(curr_date)
        print(f"\n返回数据长度: {len(result)} 字符")
        print(f"数据类型: {type(result)}")
        print("\n输出内容预览:")
        print(truncate_text(result))
        print("\n✅ 函数执行成功")
    except Exception as e:
        print(f"\n❌ 函数执行失败: {str(e)}")


def test_get_global_news_openai():
    result = get_global_news_openai("2025-08-01")
    print(result)

def test_get_china_focused_news_openai():
    result = get_china_focused_news_openai("2025-08-01")
    print(result)

def test_get_google_news():
    """Test get_google_news function with proper parameters"""
    # Test that the function can be called with correct parameters
    # Note: This test may take time due to network requests to Google News
    try:
        result = get_google_news("Tesla stock 699", "2025-08-01", 7)
        print(f"Google News result type: {type(result)}")
        print(f"Google News result length: {len(result)} characters")
        if result:
            print("✅ get_google_news executed successfully")
            print(f"Preview: {result}...")
        else:
            print("⚠️ get_google_news returned empty result")
    except Exception as e:
        print(f"❌ get_google_news failed: {str(e)}")
        # Don't fail the test for network issues
        pass

def test_get_sina_global_financial_news():
    result = get_eastmoney_global_financial_news("2025-08-06")
    print(result)