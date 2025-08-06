from typing import Annotated, Dict
from .yfin_utils import *
from .stockstats_utils import *
from .googlenews_utils import *
from .finnhub_utils import get_data_in_range
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import json
import os
import pandas as pd
from tqdm import tqdm
import yfinance as yf
import tushare as ts
from openai import OpenAI
import akshare as ak
try:
    from google import genai
    from google.genai import types
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    try:
        import google.generativeai as genai
        GOOGLE_GENAI_AVAILABLE = False  # Use fallback implementation
        print("Warning: google.genai not available, falling back to google.generativeai")
    except ImportError:
        GOOGLE_GENAI_AVAILABLE = False
        print("Warning: Google Gemini integration not available")
from .config import get_config, set_config, DATA_DIR

# AKShare imports for real social media data
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("Warning: AKShare not available, real social media data features will be limited")


def get_google_news(
    query: Annotated[str, "Query to search with"],
    curr_date: Annotated[str, "Curr date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    query = query.replace(" ", "+")

    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    news_results = getNewsData(query, before, curr_date)

    news_str = ""

    for news in news_results:
        news_str += (
            f"### {news['title']} (source: {news['source']}) \n\n{news['snippet']}\n\n"
        )

    if len(news_results) == 0:
        return ""

    return f"## {query} Google News, from {before} to {curr_date}:\n\n{news_str}"





def get_stock_stats_indicators_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    look_back_days: Annotated[int, "how many days to look back"],
    online: Annotated[bool, "to fetch data online or offline"],
) -> str:

    best_ind_params = {
        # Moving Averages
        "close_50_sma": (
            "50 SMA: A medium-term trend indicator. "
            "Usage: Identify trend direction and serve as dynamic support/resistance. "
            "Tips: It lags price; combine with faster indicators for timely signals."
        ),
        "close_200_sma": (
            "200 SMA: A long-term trend benchmark. "
            "Usage: Confirm overall market trend and identify golden/death cross setups. "
            "Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries."
        ),
        "close_10_ema": (
            "10 EMA: A responsive short-term average. "
            "Usage: Capture quick shifts in momentum and potential entry points. "
            "Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals."
        ),
        # MACD Related
        "macd": (
            "MACD: Computes momentum via differences of EMAs. "
            "Usage: Look for crossovers and divergence as signals of trend changes. "
            "Tips: Confirm with other indicators in low-volatility or sideways markets."
        ),
        "macds": (
            "MACD Signal: An EMA smoothing of the MACD line. "
            "Usage: Use crossovers with the MACD line to trigger trades. "
            "Tips: Should be part of a broader strategy to avoid false positives."
        ),
        "macdh": (
            "MACD Histogram: Shows the gap between the MACD line and its signal. "
            "Usage: Visualize momentum strength and spot divergence early. "
            "Tips: Can be volatile; complement with additional filters in fast-moving markets."
        ),
        # Momentum Indicators
        "rsi": (
            "RSI: Measures momentum to flag overbought/oversold conditions. "
            "Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. "
            "Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis."
        ),
        # Volatility Indicators
        "boll": (
            "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
            "Usage: Acts as a dynamic benchmark for price movement. "
            "Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals."
        ),
        "boll_ub": (
            "Bollinger Upper Band: Typically 2 standard deviations above the middle line. "
            "Usage: Signals potential overbought conditions and breakout zones. "
            "Tips: Confirm signals with other tools; prices may ride the band in strong trends."
        ),
        "boll_lb": (
            "Bollinger Lower Band: Typically 2 standard deviations below the middle line. "
            "Usage: Indicates potential oversold conditions. "
            "Tips: Use additional analysis to avoid false reversal signals."
        ),
        "atr": (
            "ATR: Averages true range to measure volatility. "
            "Usage: Set stop-loss levels and adjust position sizes based on current market volatility. "
            "Tips: It's a reactive measure, so use it as part of a broader risk management strategy."
        ),
        # Volume-Based Indicators
        "vwma": (
            "VWMA: A moving average weighted by volume. "
            "Usage: Confirm trends by integrating price action with volume data. "
            "Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."
        ),
        "mfi": (
            "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure. "
            "Usage: Identify overbought (>80) or oversold (<20) conditions and confirm the strength of trends or reversals. "
            "Tips: Use alongside RSI or MACD to confirm signals; divergence between price and MFI can indicate potential reversals."
        ),
    }

    if indicator not in best_ind_params:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(best_ind_params.keys())}"
        )

    end_date = curr_date
    curr_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date - relativedelta(days=look_back_days)

    if not online:
        # read from YFin data
        data = pd.read_csv(
            os.path.join(
                DATA_DIR,
                f"market_data/price_data/{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
            )
        )
        data["Date"] = pd.to_datetime(data["Date"], utc=True)
        dates_in_df = data["Date"].astype(str).str[:10]

        ind_string = ""
        while curr_date >= before:
            # only do the trading dates
            if curr_date.strftime("%Y-%m-%d") in dates_in_df.values:
                indicator_value = get_stockstats_indicator(
                    symbol, indicator, curr_date.strftime("%Y-%m-%d"), online
                )

                ind_string += f"{curr_date.strftime('%Y-%m-%d')}: {indicator_value}\n"

            curr_date = curr_date - relativedelta(days=1)
    else:
        # online gathering
        ind_string = ""
        while curr_date >= before:
            indicator_value = get_stockstats_indicator(
                symbol, indicator, curr_date.strftime("%Y-%m-%d"), online
            )

            ind_string += f"{curr_date.strftime('%Y-%m-%d')}: {indicator_value}\n"

            curr_date = curr_date - relativedelta(days=1)

    result_str = (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "No description available.")
    )

    return result_str


def get_stockstats_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    online: Annotated[bool, "to fetch data online or offline"],
) -> str:

    curr_date = datetime.strptime(curr_date, "%Y-%m-%d")
    curr_date = curr_date.strftime("%Y-%m-%d")

    try:
        indicator_value = StockstatsUtils.get_stock_stats(
            symbol,
            indicator,
            curr_date,
            os.path.join(DATA_DIR, "market_data", "price_data"),
            online=online,
        )
    except Exception as e:
        print(
            f"Error getting stockstats indicator data for indicator {indicator} on {curr_date}: {e}"
        )
        return ""

    return str(indicator_value)


def get_YFin_data_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    # calculate past days
    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=look_back_days)
    start_date = before.strftime("%Y-%m-%d")

    # read in data
    data = pd.read_csv(
        os.path.join(
            DATA_DIR,
            f"market_data/price_data/{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
        )
    )

    # Extract just the date part for comparison
    data["DateOnly"] = data["Date"].str[:10]

    # Filter data between the start and end dates (inclusive)
    filtered_data = data[
        (data["DateOnly"] >= start_date) & (data["DateOnly"] <= curr_date)
    ]

    # Drop the temporary column we created
    filtered_data = filtered_data.drop("DateOnly", axis=1)

    # Set pandas display options to show the full DataFrame
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", None
    ):
        df_string = filtered_data.to_string()

    return (
        f"## Raw Market Data for {symbol} from {start_date} to {curr_date}:\n\n"
        + df_string
    )


def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):
    from .tushare_utils import get_tushare_utils

    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    # Try Tushare first for Chinese stocks
    try:
        tushare_utils = get_tushare_utils()
        if tushare_utils.is_chinese_stock(symbol):
            data = tushare_utils.get_stock_data(symbol, start_date, end_date)

            if not data.empty:
                # Set Date as index for consistency with Yahoo Finance format
                data = data.set_index('Date')

                # Round numerical values to 2 decimal places for cleaner display
                numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
                for col in numeric_columns:
                    if col in data.columns:
                        data[col] = data[col].round(2)

                # Convert DataFrame to CSV string
                csv_string = data.to_csv()

                # Add header information
                header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date} (Tushare)\n"
                header += f"# Total records: {len(data)}\n"
                header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

                return header + csv_string
    except Exception as e:
        print(f"Tushare failed for {symbol}: {e}, falling back to Yahoo Finance")

    # Fallback to Yahoo Finance for non-Chinese stocks or if Tushare fails
    try:
        # Create ticker object
        ticker = yf.Ticker(symbol.upper())

        # Fetch historical data for the specified date range
        data = ticker.history(start=start_date, end=end_date)

        # Check if data is empty
        if data.empty:
            return (
                f"No data found for symbol '{symbol}' between {start_date} and {end_date}"
            )

        # Remove timezone info from index for cleaner output
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        # Round numerical values to 2 decimal places for cleaner display
        numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
        for col in numeric_columns:
            if col in data.columns:
                data[col] = data[col].round(2)

        # Convert DataFrame to CSV string
        csv_string = data.to_csv()

        # Add header information
        header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date} (Yahoo Finance)\n"
        header += f"# Total records: {len(data)}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + csv_string

    except Exception as e:
        return f"Error fetching data for symbol '{symbol}': {e}"


def get_tushare_data_online(
    symbol: Annotated[str, "Chinese stock ticker symbol (e.g., '000001' or '600000')"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):
    """
    Retrieve Chinese stock data using Tushare API
    """
    from .tushare_utils import get_tushare_utils

    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    try:
        tushare_utils = get_tushare_utils()
        data = tushare_utils.get_stock_data(symbol, start_date, end_date)

        if data.empty:
            return f"No data found for Chinese stock '{symbol}' between {start_date} and {end_date}"

        # Set Date as index for consistency
        data = data.set_index('Date')

        # Round numerical values to 2 decimal places for cleaner display
        numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
        for col in numeric_columns:
            if col in data.columns:
                data[col] = data[col].round(2)

        # Convert DataFrame to CSV string
        csv_string = data.to_csv()

        # Add header information
        header = f"# Chinese stock data for {symbol.upper()} from {start_date} to {end_date} (Tushare)\n"
        header += f"# Total records: {len(data)}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + csv_string

    except Exception as e:
        return f"Error fetching Chinese stock data for '{symbol}': {e}"


def get_YFin_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    # read in data
    data = pd.read_csv(
        os.path.join(
            DATA_DIR,
            f"market_data/price_data/{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
        )
    )

    if end_date > "2025-03-25":
        raise Exception(
            f"Get_YFin_Data: {end_date} is outside of the data range of 2015-01-01 to 2025-03-25"
        )

    # Extract just the date part for comparison
    data["DateOnly"] = data["Date"].str[:10]

    # Filter data between the start and end dates (inclusive)
    filtered_data = data[
        (data["DateOnly"] >= start_date) & (data["DateOnly"] <= end_date)
    ]

    # Drop the temporary column we created
    filtered_data = filtered_data.drop("DateOnly", axis=1)

    # remove the index from the dataframe
    filtered_data = filtered_data.reset_index(drop=True)

    return filtered_data


def get_stock_news_openai(
    ticker: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
    max_limit_per_day: Annotated[int, "Maximum number of news per day"]
):
    """
    获取股票新闻数据，使用AKSHARE的stock_news_em接口，并获取新闻链接的内容
    Get stock news data using AKSHARE's stock_news_em interface and fetch content from news links

    Args:
        ticker (str): 股票代码 / ticker symbol of the company
        start_date (str): 开始日期，格式为yyyy-mm-dd / Start date in yyyy-mm-dd format
        look_back_days (int): 向前查看的天数 / how many days to look back
        max_limit_per_day (int): 每天最大新闻数量 / Maximum number of news per day

    Returns:
        str: 格式化的新闻报告 / formatted news report
    """

    def fetch_news_content(url):
        """
        获取新闻链接的内容
        Fetch content from news URL
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            import time

            # 添加请求头，模拟浏览器访问
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }

            # 设置超时时间
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            response.encoding = response.apparent_encoding

            # 使用BeautifulSoup解析HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # 移除脚本和样式元素
            for script in soup(["script", "style"]):
                script.decompose()

            # 获取文本内容
            text = soup.get_text()

            # 清理文本
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            # 限制内容长度
            if len(text) > 1000:
                text = text[:1000] + "..."

            return text

        except Exception as e:
            return f"获取链接内容失败: {str(e)}"

    try:
        # 检查AKSHARE是否可用
        if not AKSHARE_AVAILABLE:
            return "AKShare 未安装，无法获取股票新闻数据"

        # 计算日期范围
        end_date = datetime.strptime(start_date, "%Y-%m-%d")
        start_date_obj = end_date - timedelta(days=look_back_days)

        # 使用AKSHARE获取股票新闻
        news_df = ak.stock_news_em(symbol=ticker)

        if news_df.empty:
            return f"未找到股票 {ticker} 的新闻数据"

        # 过滤日期范围内的新闻
        filtered_news = []

        # AKSHARE stock_news_em 返回的列名是固定的中文列名
        # ['关键词', '新闻标题', '新闻内容', '发布时间', '文章来源', '新闻链接']
        date_column = '发布时间'
        title_column = '新闻标题'
        content_column = '新闻内容'
        source_column = '文章来源'
        link_column = '新闻链接'

        # 检查必要的列是否存在
        if date_column not in news_df.columns:
            return f"数据格式错误：未找到'{date_column}'列。实际列名：{list(news_df.columns)}"
        if title_column not in news_df.columns:
            return f"数据格式错误：未找到'{title_column}'列。实际列名：{list(news_df.columns)}"

        # 处理新闻数据
        daily_news_count = {}

        for idx, row in news_df.iterrows():
            try:
                # 解析日期 - AKSHARE返回的格式是 "YYYY-MM-DD HH:MM:SS"
                news_date_str = str(row[date_column])

                # 解析日期
                news_date = datetime.strptime(news_date_str, "%Y-%m-%d %H:%M:%S")

                # 检查是否在指定日期范围内
                if start_date_obj <= news_date <= end_date:
                    date_key = news_date.strftime("%Y-%m-%d")

                    # 检查每日新闻数量限制
                    if date_key not in daily_news_count:
                        daily_news_count[date_key] = 0

                    if daily_news_count[date_key] < max_limit_per_day:
                        # 获取新闻链接内容
                        link_content = ""
                        news_link = ""
                        if link_column in news_df.columns and pd.notna(row[link_column]):
                            news_link = str(row[link_column])
                            if news_link and news_link != "":
                                print(f"正在获取新闻链接内容: {news_link}")
                                link_content = fetch_news_content(news_link)

                        news_item = {
                            'date': news_date.strftime("%Y-%m-%d %H:%M:%S"),
                            'title': str(row[title_column]),
                            'content': str(row[content_column]) if content_column in news_df.columns else "",
                            'source': str(row[source_column]) if source_column in news_df.columns else "",
                            'link': news_link,
                            'link_content': link_content
                        }
                        filtered_news.append(news_item)
                        daily_news_count[date_key] += 1

            except Exception as e:
                # 如果处理单条新闻出错，继续处理下一条
                print(f"处理新闻时出错: {e}, 新闻日期: {row[date_column]}")
                continue

        # 生成报告
        if not filtered_news:
            return f"在 {start_date_obj.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')} 期间未找到股票 {ticker} 的相关新闻"

        # 按日期排序
        filtered_news.sort(key=lambda x: x['date'], reverse=True)

        # 格式化输出
        report = f"## 股票 {ticker} 新闻报告\n"
        report += f"**时间范围**: {start_date_obj.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}\n"
        report += f"**查找天数**: {look_back_days} 天\n"
        report += f"**每日新闻限制**: {max_limit_per_day} 条\n"
        report += f"**总计新闻数量**: {len(filtered_news)} 条\n\n"

        # 按日期分组显示新闻
        current_date = None
        for news in filtered_news:
            news_date = news['date'][:10]  # 只取日期部分

            if current_date != news_date:
                current_date = news_date
                report += f"### {current_date}\n\n"

            report += f"**{news['date'][11:]}** - {news['title']}\n"
            if news['source']:
                report += f"*来源: {news['source']}*\n"

            # 显示原始新闻内容
            if news['content'] and news['content'] != "":
                # 限制内容长度
                content = news['content'][:200] + "..." if len(news['content']) > 200 else news['content']
                report += f"**摘要**: {content}\n\n"

            # 显示新闻链接
            if news['link']:
                report += f"**新闻链接**: {news['link']}\n\n"

            # 显示链接内容
            if news['link_content'] and news['link_content'] != "":
                if not news['link_content'].startswith("获取链接内容失败"):
                    report += f"**链接内容**:\n{news['link_content']}\n\n"
                else:
                    report += f"*{news['link_content']}*\n\n"

            report += "---\n\n"

        report += f"\n---\n*数据来源: AKShare stock_news_em*"

        return report

    except Exception as e:
        return f"获取股票新闻时发生错误: {str(e)}"

def get_global_news_openai(curr_date):
    config = get_config()

    # Check if using Google provider
    if config.get("llm_provider", "").lower() == "google":
        # Configure the Gemini client
        client = genai.Client()

        # Define the grounding tool for Google Search
        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )

        # Configure generation settings
        generation_config = types.GenerateContentConfig(
            tools=[grounding_tool],
            temperature=1,
            max_output_tokens=4096,
            top_p=1,
        )

        # Make the request
        response = client.models.generate_content(
            model=config["quick_think_llm"],
            contents=f"Can you search global or macroeconomics news from 7 days before {curr_date} to {curr_date} that would be informative for trading purposes? Please prioritize and include more Chinese economic news, Chinese market news, Chinese policy news, and China-related international trade news. Also include other major global economic news from US, Europe, and other regions. Make sure you only get the data posted during that period. Focus on: 1) Chinese economic indicators and policies (40% weight), 2) Chinese stock market and financial sector news (30% weight), 3) Global economic news affecting China (20% weight), 4) Other major global economic news (10% weight).",
            config=generation_config,
        )

        return response.text
    else:
        # Original OpenAI implementation
        client = OpenAI(base_url=config["backend_url"])

        response = client.responses.create(
            model=config["quick_think_llm"],
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Can you search global or macroeconomics news from 7 days before {curr_date} to {curr_date} that would be informative for trading purposes? Please prioritize and include more Chinese economic news, Chinese market news, Chinese policy news, and China-related international trade news. Also include other major global economic news from US, Europe, and other regions. Make sure you only get the data posted during that period. Focus on: 1) Chinese economic indicators and policies (40% weight), 2) Chinese stock market and financial sector news (30% weight), 3) Global economic news affecting China (20% weight), 4) Other major global economic news (10% weight).",
                        }
                    ],
                }
            ],
            text={"format": {"type": "text"}},
            reasoning={},
            tools=[
                {
                    "type": "web_search_preview",
                    "user_location": {"type": "approximate"},
                    "search_context_size": "medium",  # Increased context size for better coverage
                }
            ],
            temperature=1,
            max_output_tokens=4096,
            top_p=1,
            store=True,
        )

        return response.output[1].content[0].text

def get_china_focused_news_openai(curr_date):
    """
    Get China-focused economic and market news using OpenAI API or Google Gemini API
    """
    config = get_config()

    # Check if using Google provider
    if config.get("llm_provider", "").lower() == "google":
        # Configure the Gemini client
        client = genai.Client()

        # Define the grounding tool for Google Search
        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )

        # Configure generation settings
        generation_config = types.GenerateContentConfig(
            tools=[grounding_tool],
            temperature=1,
            max_output_tokens=4096,
            top_p=1,
        )

        # Make the request
        response = client.models.generate_content(
            model=config["quick_think_llm"],
            contents=f"Can you search for Chinese economic and financial market news from 7 days before {curr_date} to {curr_date} that would be informative for trading Chinese stocks? Please focus specifically on: 1) Chinese economic indicators (GDP, CPI, PMI, etc.), 2) Chinese monetary policy and central bank actions, 3) Chinese stock market news (A-shares, Hong Kong stocks), 4) Chinese regulatory changes affecting financial markets, 5) Chinese corporate earnings and major company news, 6) China-US trade relations and international economic relations, 7) Chinese real estate and property market news, 8) Chinese technology sector and policy changes. Make sure you only get the data posted during that period and prioritize Chinese sources and China-focused international coverage.",
            config=generation_config,
        )

        return response.text
    else:
        # Original OpenAI implementation
        client = OpenAI(base_url=config["backend_url"])

        response = client.responses.create(
            model=config["quick_think_llm"],
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Can you search for Chinese economic and financial market news from 7 days before {curr_date} to {curr_date} that would be informative for trading Chinese stocks? Please focus specifically on: 1) Chinese economic indicators (GDP, CPI, PMI, etc.), 2) Chinese monetary policy and central bank actions, 3) Chinese stock market news (A-shares, Hong Kong stocks), 4) Chinese regulatory changes affecting financial markets, 5) Chinese corporate earnings and major company news, 6) China-US trade relations and international economic relations, 7) Chinese real estate and property market news, 8) Chinese technology sector and policy changes. Make sure you only get the data posted during that period and prioritize Chinese sources and China-focused international coverage.",
                        }
                    ],
                }
            ],
            text={"format": {"type": "text"}},
            reasoning={},
            tools=[
                {
                    "type": "web_search_preview",
                    "user_location": {"type": "approximate"},
                    "search_context_size": "high",  # High context for comprehensive coverage
                }
            ],
            temperature=1,
            max_output_tokens=4096,
            top_p=1,
            store=True,
        )

        return response.output[1].content[0].text


def get_fundamentals_tushare(
    ticker: Annotated[str, "Stock ticker symbol (e.g., '000001' or '600000')"],
    curr_date: Annotated[str, "Current date in YYYY-MM-DD format"]
) -> str:
    """
    Get comprehensive fundamental data for a Chinese stock using Tushare API.

    This function retrieves data from multiple Tushare endpoints:
    - daily_basic: PE, PB, PS ratios, market cap, shares outstanding
    - fina_indicator: Financial indicators and ratios
    - top10_holders: Top 10 shareholders information
    - top10_floatholders: Top 10 floating shareholders information
    - income: Income statement data
    - balancesheet: Balance sheet data
    - cashflow: Cash flow statement data

    Args:
        ticker: Stock ticker symbol (Chinese stocks)
        curr_date: Current date for analysis

    Returns:
        str: Comprehensive fundamental analysis report
    """
    try:
        from .tushare_utils import get_tushare_utils
        tushare_utils = get_tushare_utils()

        if not tushare_utils.is_chinese_stock(ticker):
            return f"Error: {ticker} is not a Chinese stock. This function only supports Chinese stocks."

        # Convert to Tushare format
        ts_symbol = tushare_utils.convert_symbol_to_tushare(ticker)

        # Convert date format for Tushare (YYYYMMDD)
        from datetime import datetime, timedelta
        date_obj = datetime.strptime(curr_date, '%Y-%m-%d')
        ts_date = date_obj.strftime('%Y%m%d')

        # Initialize result dictionary
        result = {
            'symbol': ticker,
            'ts_symbol': ts_symbol,
            'analysis_date': curr_date,
            'data_sources': []
        }

        # 1. Get daily_basic data (PE, PB, PS ratios, market cap)
        try:
            # Try multiple recent dates to find available data
            dates_to_try = [
                ts_date,
                (date_obj - timedelta(days=1)).strftime('%Y%m%d'),
                (date_obj - timedelta(days=2)).strftime('%Y%m%d'),
                (date_obj - timedelta(days=3)).strftime('%Y%m%d'),
                (date_obj - timedelta(days=7)).strftime('%Y%m%d'),
            ]

            daily_basic_data = pd.DataFrame()
            for try_date in dates_to_try:
                try:
                    daily_basic_data = tushare_utils.pro.daily_basic(ts_code=ts_symbol, trade_date=try_date)
                    if not daily_basic_data.empty:
                        result['daily_basic_date'] = try_date
                        break
                except Exception as e:
                    continue

            if not daily_basic_data.empty:
                latest_basic = daily_basic_data.iloc[0]
                result['daily_basic'] = {
                    'pe_ratio': latest_basic.get('pe', 'N/A'),
                    'pb_ratio': latest_basic.get('pb', 'N/A'),
                    'ps_ratio': latest_basic.get('ps', 'N/A'),
                    'total_share': latest_basic.get('total_share', 'N/A'),
                    'float_share': latest_basic.get('float_share', 'N/A'),
                    'total_mv': latest_basic.get('total_mv', 'N/A'),
                    'circ_mv': latest_basic.get('circ_mv', 'N/A'),
                    'turnover_rate': latest_basic.get('turnover_rate', 'N/A'),
                    'volume_ratio': latest_basic.get('volume_ratio', 'N/A'),
                    'pe_ttm': latest_basic.get('pe_ttm', 'N/A'),
                    'pb_mrq': 'N/A',  # This field doesn't exist in daily_basic
                    'ps_ttm': latest_basic.get('ps_ttm', 'N/A'),
                }
                result['data_sources'].append('daily_basic')
        except Exception as e:
            result['daily_basic_error'] = str(e)

        # 2. Get fina_indicator data (financial indicators)
        try:
            # Get the most recent financial indicators
            fina_indicator_data = tushare_utils.pro.fina_indicator(ts_code=ts_symbol)
            if not fina_indicator_data.empty:
                # Sort by end_date to get the most recent data
                fina_indicator_data = fina_indicator_data.sort_values('end_date', ascending=False)
                latest_fina = fina_indicator_data.iloc[0]
                result['fina_indicator'] = {
                    'end_date': latest_fina.get('end_date', 'N/A'),
                    'eps': latest_fina.get('eps', 'N/A'),
                    'dt_eps': latest_fina.get('dt_eps', 'N/A'),
                    'total_revenue_ps': latest_fina.get('total_revenue_ps', 'N/A'),
                    'revenue_ps': latest_fina.get('revenue_ps', 'N/A'),
                    'capital_rese_ps': latest_fina.get('capital_rese_ps', 'N/A'),
                    'surplus_rese_ps': latest_fina.get('surplus_rese_ps', 'N/A'),
                    'undist_profit_ps': latest_fina.get('undist_profit_ps', 'N/A'),
                    'extra_item': latest_fina.get('extra_item', 'N/A'),
                    'profit_dedt': latest_fina.get('profit_dedt', 'N/A'),
                    'gross_margin': latest_fina.get('gross_margin', 'N/A'),
                    'current_ratio': latest_fina.get('current_ratio', 'N/A'),
                    'quick_ratio': latest_fina.get('quick_ratio', 'N/A'),
                    'cash_ratio': latest_fina.get('cash_ratio', 'N/A'),
                    'ar_turn': latest_fina.get('ar_turn', 'N/A'),
                    'ca_turn': latest_fina.get('ca_turn', 'N/A'),
                    'fa_turn': latest_fina.get('fa_turn', 'N/A'),
                    'assets_turn': latest_fina.get('assets_turn', 'N/A'),
                    'op_income': latest_fina.get('op_income', 'N/A'),
                    'valuechange_income': latest_fina.get('valuechange_income', 'N/A'),
                    'interst_income': latest_fina.get('interst_income', 'N/A'),
                    'daa': latest_fina.get('daa', 'N/A'),
                    'ebit': latest_fina.get('ebit', 'N/A'),
                    'ebitda': latest_fina.get('ebitda', 'N/A'),
                    'fcff': latest_fina.get('fcff', 'N/A'),
                    'fcfe': latest_fina.get('fcfe', 'N/A'),
                    'current_exint': latest_fina.get('current_exint', 'N/A'),
                    'noncurrent_exint': latest_fina.get('noncurrent_exint', 'N/A'),
                    'interestdebt': latest_fina.get('interestdebt', 'N/A'),
                    'netdebt': latest_fina.get('netdebt', 'N/A'),
                    'tangible_asset': latest_fina.get('tangible_asset', 'N/A'),
                    'working_capital': latest_fina.get('working_capital', 'N/A'),
                    'networking_capital': latest_fina.get('networking_capital', 'N/A'),
                    'invest_capital': latest_fina.get('invest_capital', 'N/A'),
                    'retained_earnings': latest_fina.get('retained_earnings', 'N/A'),
                    'diluted2_eps': latest_fina.get('diluted2_eps', 'N/A'),
                    'bps': latest_fina.get('bps', 'N/A'),
                    'ocfps': latest_fina.get('ocfps', 'N/A'),
                    'retainedps': latest_fina.get('retainedps', 'N/A'),
                    'cfps': latest_fina.get('cfps', 'N/A'),
                    'ebit_ps': latest_fina.get('ebit_ps', 'N/A'),
                    'fcff_ps': latest_fina.get('fcff_ps', 'N/A'),
                    'fcfe_ps': latest_fina.get('fcfe_ps', 'N/A'),
                    'netprofit_margin': latest_fina.get('netprofit_margin', 'N/A'),
                    'grossprofit_margin': latest_fina.get('grossprofit_margin', 'N/A'),
                    'cogs_of_sales': latest_fina.get('cogs_of_sales', 'N/A'),
                    'expense_of_sales': latest_fina.get('expense_of_sales', 'N/A'),
                    'profit_to_gr': latest_fina.get('profit_to_gr', 'N/A'),
                    'saleexp_to_gr': latest_fina.get('saleexp_to_gr', 'N/A'),
                    'adminexp_of_gr': latest_fina.get('adminexp_of_gr', 'N/A'),
                    'finaexp_of_gr': latest_fina.get('finaexp_of_gr', 'N/A'),
                    'impai_ttm': latest_fina.get('impai_ttm', 'N/A'),
                    'gc_of_gr': latest_fina.get('gc_of_gr', 'N/A'),
                    'op_of_gr': latest_fina.get('op_of_gr', 'N/A'),
                    'ebit_of_gr': latest_fina.get('ebit_of_gr', 'N/A'),
                    'roe': latest_fina.get('roe', 'N/A'),
                    'roe_waa': latest_fina.get('roe_waa', 'N/A'),
                    'roe_dt': latest_fina.get('roe_dt', 'N/A'),
                    'roa': latest_fina.get('roa', 'N/A'),
                    'npta': latest_fina.get('npta', 'N/A'),
                    'roic': latest_fina.get('roic', 'N/A'),
                    'roe_yearly': latest_fina.get('roe_yearly', 'N/A'),
                    'roa_yearly': latest_fina.get('roa_yearly', 'N/A'),
                    'roe_avg': latest_fina.get('roe_avg', 'N/A'),
                    'opincome_of_ebt': latest_fina.get('opincome_of_ebt', 'N/A'),
                    'investincome_of_ebt': latest_fina.get('investincome_of_ebt', 'N/A'),
                    'n_op_profit_of_ebt': latest_fina.get('n_op_profit_of_ebt', 'N/A'),
                    'tax_to_ebt': latest_fina.get('tax_to_ebt', 'N/A'),
                    'dtprofit_to_profit': latest_fina.get('dtprofit_to_profit', 'N/A'),
                    'salescash_to_or': latest_fina.get('salescash_to_or', 'N/A'),
                    'ocf_to_or': latest_fina.get('ocf_to_or', 'N/A'),
                    'ocf_to_opincome': latest_fina.get('ocf_to_opincome', 'N/A'),
                    'capitalized_to_da': latest_fina.get('capitalized_to_da', 'N/A'),
                    'debt_to_assets': latest_fina.get('debt_to_assets', 'N/A'),
                    'assets_to_eqt': latest_fina.get('assets_to_eqt', 'N/A'),
                    'dp_assets_to_eqt': latest_fina.get('dp_assets_to_eqt', 'N/A'),
                    'ca_to_assets': latest_fina.get('ca_to_assets', 'N/A'),
                    'nca_to_assets': latest_fina.get('nca_to_assets', 'N/A'),
                    'tbassets_to_totalassets': latest_fina.get('tbassets_to_totalassets', 'N/A'),
                    'int_to_talcap': latest_fina.get('int_to_talcap', 'N/A'),
                    'eqt_to_talcapital': latest_fina.get('eqt_to_talcapital', 'N/A'),
                    'currentdebt_to_debt': latest_fina.get('currentdebt_to_debt', 'N/A'),
                    'longdeb_to_debt': latest_fina.get('longdeb_to_debt', 'N/A'),
                    'ocf_to_shortdebt': latest_fina.get('ocf_to_shortdebt', 'N/A'),
                    'debt_to_eqt': latest_fina.get('debt_to_eqt', 'N/A'),
                    'eqt_to_debt': latest_fina.get('eqt_to_debt', 'N/A'),
                    'eqt_to_interestdebt': latest_fina.get('eqt_to_interestdebt', 'N/A'),
                    'tangibleasset_to_debt': latest_fina.get('tangibleasset_to_debt', 'N/A'),
                    'tangasset_to_intdebt': latest_fina.get('tangasset_to_intdebt', 'N/A'),
                    'tangibleasset_to_netdebt': latest_fina.get('tangibleasset_to_netdebt', 'N/A'),
                    'ocf_to_debt': latest_fina.get('ocf_to_debt', 'N/A'),
                    'turn_days': latest_fina.get('turn_days', 'N/A'),
                    'roa_yearly': latest_fina.get('roa_yearly', 'N/A'),
                    'roa_dp': latest_fina.get('roa_dp', 'N/A'),
                    'fixed_assets': latest_fina.get('fixed_assets', 'N/A'),
                    'profit_to_op': latest_fina.get('profit_to_op', 'N/A'),
                    'q_opincome': latest_fina.get('q_opincome', 'N/A'),
                    'q_investincome': latest_fina.get('q_investincome', 'N/A'),
                    'q_dtprofit': latest_fina.get('q_dtprofit', 'N/A'),
                    'q_eps': latest_fina.get('q_eps', 'N/A'),
                    'q_netprofit_margin': latest_fina.get('q_netprofit_margin', 'N/A'),
                    'q_gsprofit_margin': latest_fina.get('q_gsprofit_margin', 'N/A'),
                    'q_exp_to_sales': latest_fina.get('q_exp_to_sales', 'N/A'),
                    'q_profit_to_gr': latest_fina.get('q_profit_to_gr', 'N/A'),
                    'q_saleexp_to_gr': latest_fina.get('q_saleexp_to_gr', 'N/A'),
                    'q_adminexp_to_gr': latest_fina.get('q_adminexp_to_gr', 'N/A'),
                    'q_finaexp_to_gr': latest_fina.get('q_finaexp_to_gr', 'N/A'),
                    'q_impair_to_gr_ttm': latest_fina.get('q_impair_to_gr_ttm', 'N/A'),
                    'q_gc_to_gr': latest_fina.get('q_gc_to_gr', 'N/A'),
                    'q_op_to_gr': latest_fina.get('q_op_to_gr', 'N/A'),
                    'q_roe': latest_fina.get('q_roe', 'N/A'),
                    'q_dt_roe': latest_fina.get('q_dt_roe', 'N/A'),
                    'q_npta': latest_fina.get('q_npta', 'N/A'),
                    'q_opincome_to_ebt': latest_fina.get('q_opincome_to_ebt', 'N/A'),
                    'q_investincome_to_ebt': latest_fina.get('q_investincome_to_ebt', 'N/A'),
                    'q_dtprofit_to_profit': latest_fina.get('q_dtprofit_to_profit', 'N/A'),
                    'q_salescash_to_or': latest_fina.get('q_salescash_to_or', 'N/A'),
                    'q_ocf_to_sales': latest_fina.get('q_ocf_to_sales', 'N/A'),
                    'q_ocf_to_or': latest_fina.get('q_ocf_to_or', 'N/A'),
                    'basic_eps_yoy': latest_fina.get('basic_eps_yoy', 'N/A'),
                    'dt_eps_yoy': latest_fina.get('dt_eps_yoy', 'N/A'),
                    'cfps_yoy': latest_fina.get('cfps_yoy', 'N/A'),
                    'op_yoy': latest_fina.get('op_yoy', 'N/A'),
                    'ebt_yoy': latest_fina.get('ebt_yoy', 'N/A'),
                    'netprofit_yoy': latest_fina.get('netprofit_yoy', 'N/A'),
                    'dt_netprofit_yoy': latest_fina.get('dt_netprofit_yoy', 'N/A'),
                    'ocf_yoy': latest_fina.get('ocf_yoy', 'N/A'),
                    'roe_yoy': latest_fina.get('roe_yoy', 'N/A'),
                    'bps_yoy': latest_fina.get('bps_yoy', 'N/A'),
                    'assets_yoy': latest_fina.get('assets_yoy', 'N/A'),
                    'eqt_yoy': latest_fina.get('eqt_yoy', 'N/A'),
                    'tr_yoy': latest_fina.get('tr_yoy', 'N/A'),
                    'or_yoy': latest_fina.get('or_yoy', 'N/A'),
                    'q_gr_yoy': latest_fina.get('q_gr_yoy', 'N/A'),
                    'q_gr_qoq': latest_fina.get('q_gr_qoq', 'N/A'),
                    'q_sales_yoy': latest_fina.get('q_sales_yoy', 'N/A'),
                    'q_sales_qoq': latest_fina.get('q_sales_qoq', 'N/A'),
                    'q_op_yoy': latest_fina.get('q_op_yoy', 'N/A'),
                    'q_op_qoq': latest_fina.get('q_op_qoq', 'N/A'),
                    'q_profit_yoy': latest_fina.get('q_profit_yoy', 'N/A'),
                    'q_profit_qoq': latest_fina.get('q_profit_qoq', 'N/A'),
                    'q_netprofit_yoy': latest_fina.get('q_netprofit_yoy', 'N/A'),
                    'q_netprofit_qoq': latest_fina.get('q_netprofit_qoq', 'N/A'),
                    'equity_yoy': latest_fina.get('equity_yoy', 'N/A'),
                    'rd_exp': latest_fina.get('rd_exp', 'N/A'),
                }
                result['data_sources'].append('fina_indicator')
        except Exception as e:
            result['fina_indicator_error'] = str(e)

        # 3. Get top10_holders data (top 10 shareholders)
        try:
            # Get the most recent top 10 holders data
            top10_holders_data = tushare_utils.pro.top10_holders(ts_code=ts_symbol)
            if not top10_holders_data.empty:
                # Sort by end_date to get the most recent data
                top10_holders_data = top10_holders_data.sort_values('end_date', ascending=False)
                # Get unique end_date and take the most recent period
                latest_date = top10_holders_data['end_date'].iloc[0]
                latest_holders = top10_holders_data[top10_holders_data['end_date'] == latest_date]

                result['top10_holders'] = {
                    'end_date': latest_date,
                    'holders': []
                }

                for _, holder in latest_holders.iterrows():
                    result['top10_holders']['holders'].append({
                        'holder_name': holder.get('holder_name', 'N/A'),
                        'hold_amount': holder.get('hold_amount', 'N/A'),
                        'hold_ratio': holder.get('hold_ratio', 'N/A'),
                    })
                result['data_sources'].append('top10_holders')
        except Exception as e:
            result['top10_holders_error'] = str(e)

        # 4. Get top10_floatholders data (top 10 floating shareholders)
        try:
            # Get the most recent top 10 float holders data
            top10_floatholders_data = tushare_utils.pro.top10_floatholders(ts_code=ts_symbol)
            if not top10_floatholders_data.empty:
                # Sort by end_date to get the most recent data
                top10_floatholders_data = top10_floatholders_data.sort_values('end_date', ascending=False)
                # Get unique end_date and take the most recent period
                latest_date = top10_floatholders_data['end_date'].iloc[0]
                latest_float_holders = top10_floatholders_data[top10_floatholders_data['end_date'] == latest_date]

                result['top10_floatholders'] = {
                    'end_date': latest_date,
                    'holders': []
                }

                for _, holder in latest_float_holders.iterrows():
                    result['top10_floatholders']['holders'].append({
                        'holder_name': holder.get('holder_name', 'N/A'),
                        'hold_amount': holder.get('hold_amount', 'N/A'),
                        'hold_ratio': holder.get('hold_ratio', 'N/A'),
                    })
                result['data_sources'].append('top10_floatholders')
        except Exception as e:
            result['top10_floatholders_error'] = str(e)

        # 5. Get income statement data
        try:
            income_data = tushare_utils.pro.income(ts_code=ts_symbol)
            if not income_data.empty:
                # Sort by end_date to get the most recent data
                income_data = income_data.sort_values('end_date', ascending=False)
                latest_income = income_data.iloc[0]
                result['income'] = {
                    'end_date': latest_income.get('end_date', 'N/A'),
                    'revenue': latest_income.get('revenue', 'N/A'),
                    'total_revenue': latest_income.get('total_revenue', 'N/A'),
                    'int_income': latest_income.get('int_income', 'N/A'),
                    'prem_earned': latest_income.get('prem_earned', 'N/A'),
                    'comm_income': latest_income.get('comm_income', 'N/A'),
                    'n_commis_income': latest_income.get('n_commis_income', 'N/A'),
                    'n_oth_income': latest_income.get('n_oth_income', 'N/A'),  # Other income
                    'n_oth_b_income': latest_income.get('n_oth_b_income', 'N/A'),
                    'prem_income': latest_income.get('prem_income', 'N/A'),
                    'out_prem': latest_income.get('out_prem', 'N/A'),
                    'une_prem_reser': latest_income.get('une_prem_reser', 'N/A'),
                    'reins_income': latest_income.get('reins_income', 'N/A'),
                    'n_sec_tb_income': latest_income.get('n_sec_tb_income', 'N/A'),
                    'n_sec_uw_income': latest_income.get('n_sec_uw_income', 'N/A'),
                    'n_asset_mg_income': latest_income.get('n_asset_mg_income', 'N/A'),
                    'oth_b_income': latest_income.get('oth_b_income', 'N/A'),
                    'fv_value_chg_gain': latest_income.get('fv_value_chg_gain', 'N/A'),
                    'invest_income': latest_income.get('invest_income', 'N/A'),
                    'ass_invest_income': latest_income.get('ass_invest_income', 'N/A'),
                    'forex_gain': latest_income.get('forex_gain', 'N/A'),
                    'total_cogs': latest_income.get('total_cogs', 'N/A'),  # Total operating costs
                    'oper_cost': latest_income.get('oper_cost', 'N/A'),
                    'int_exp': latest_income.get('int_exp', 'N/A'),
                    'comm_exp': latest_income.get('comm_exp', 'N/A'),
                    'biz_tax_surchg': latest_income.get('biz_tax_surchg', 'N/A'),
                    'sell_exp': latest_income.get('sell_exp', 'N/A'),
                    'admin_exp': latest_income.get('admin_exp', 'N/A'),
                    'fin_exp': latest_income.get('fin_exp', 'N/A'),
                    'assets_impair_loss': latest_income.get('assets_impair_loss', 'N/A'),
                    'prem_refund': latest_income.get('prem_refund', 'N/A'),
                    'compens_payout': latest_income.get('compens_payout', 'N/A'),
                    'reser_insur_liab': latest_income.get('reser_insur_liab', 'N/A'),
                    'div_payt': latest_income.get('div_payt', 'N/A'),
                    'reins_exp': latest_income.get('reins_exp', 'N/A'),
                    'oper_exp': latest_income.get('oper_exp', 'N/A'),
                    'compens_payout_refu': latest_income.get('compens_payout_refu', 'N/A'),
                    'insur_reser_refu': latest_income.get('insur_reser_refu', 'N/A'),
                    'reins_cost_refund': latest_income.get('reins_cost_refund', 'N/A'),
                    'other_bus_cost': latest_income.get('other_bus_cost', 'N/A'),
                    'operate_profit': latest_income.get('operate_profit', 'N/A'),
                    'non_oper_income': latest_income.get('non_oper_income', 'N/A'),
                    'non_oper_exp': latest_income.get('non_oper_exp', 'N/A'),
                    'nca_disploss': latest_income.get('nca_disploss', 'N/A'),
                    'total_profit': latest_income.get('total_profit', 'N/A'),
                    'income_tax': latest_income.get('income_tax', 'N/A'),
                    'n_income': latest_income.get('n_income', 'N/A'),
                    'n_income_attr_p': latest_income.get('n_income_attr_p', 'N/A'),
                    'minority_gain': latest_income.get('minority_gain', 'N/A'),
                    'oth_compr_income': latest_income.get('oth_compr_income', 'N/A'),
                    't_compr_income': latest_income.get('t_compr_income', 'N/A'),
                    'compr_inc_attr_p': latest_income.get('compr_inc_attr_p', 'N/A'),
                    'compr_inc_attr_m_s': latest_income.get('compr_inc_attr_m_s', 'N/A'),
                    'ebit': latest_income.get('ebit', 'N/A'),
                    'ebitda': latest_income.get('ebitda', 'N/A'),
                    'insurance_exp': latest_income.get('insurance_exp', 'N/A'),
                    'undist_profit': latest_income.get('undist_profit', 'N/A'),
                    'distable_profit': latest_income.get('distable_profit', 'N/A'),
                    'rd_exp': latest_income.get('rd_exp', 'N/A'),
                    'fin_exp_int_exp': latest_income.get('fin_exp_int_exp', 'N/A'),
                    'fin_exp_int_inc': latest_income.get('fin_exp_int_inc', 'N/A'),
                    'transfer_surplus_rese': latest_income.get('transfer_surplus_rese', 'N/A'),
                    'transfer_housing_imprest': latest_income.get('transfer_housing_imprest', 'N/A'),
                    'transfer_oth': latest_income.get('transfer_oth', 'N/A'),
                    'adj_lossgain': latest_income.get('adj_lossgain', 'N/A'),
                    'withdra_legal_surplus': latest_income.get('withdra_legal_surplus', 'N/A'),
                    'withdra_legal_pubfund': latest_income.get('withdra_legal_pubfund', 'N/A'),
                    'withdra_biz_devfund': latest_income.get('withdra_biz_devfund', 'N/A'),
                    'withdra_rese_fund': latest_income.get('withdra_rese_fund', 'N/A'),
                    'withdra_oth_ersu': latest_income.get('withdra_oth_ersu', 'N/A'),
                    'workers_welfare': latest_income.get('workers_welfare', 'N/A'),
                    'distr_profit_shrhder': latest_income.get('distr_profit_shrhder', 'N/A'),
                    'prfshare_payable_dvd': latest_income.get('prfshare_payable_dvd', 'N/A'),
                    'comshare_payable_dvd': latest_income.get('comshare_payable_dvd', 'N/A'),
                    'capit_comstock_div': latest_income.get('capit_comstock_div', 'N/A'),
                    'continued_net_profit': latest_income.get('continued_net_profit', 'N/A'),
                    'end_net_profit': latest_income.get('end_net_profit', 'N/A'),
                    'credit_impa_loss': latest_income.get('credit_impa_loss', 'N/A'),
                    'net_expo_hedging_benefits': latest_income.get('net_expo_hedging_benefits', 'N/A'),
                    'oth_impair_loss_assets': latest_income.get('oth_impair_loss_assets', 'N/A'),
                    'total_opcost': latest_income.get('total_opcost', 'N/A'),
                    'amodcost_fin_assets': latest_income.get('amodcost_fin_assets', 'N/A'),
                    'oth_income': latest_income.get('oth_income', 'N/A'),
                    'asset_disp_income': latest_income.get('asset_disp_income', 'N/A'),
                    'continued_ebit': latest_income.get('continued_ebit', 'N/A'),
                    'non_current_liab_due': latest_income.get('non_current_liab_due', 'N/A'),
                    'update_flag': latest_income.get('update_flag', 'N/A'),
                }
                result['data_sources'].append('income')
        except Exception as e:
            result['income_error'] = str(e)

        # 6. Get balance sheet data
        try:
            balance_data = tushare_utils.pro.balancesheet(ts_code=ts_symbol)
            if not balance_data.empty:
                # Sort by end_date to get the most recent data
                balance_data = balance_data.sort_values('end_date', ascending=False)
                latest_balance = balance_data.iloc[0]
                result['balancesheet'] = {
                    'end_date': latest_balance.get('end_date', 'N/A'),
                    'total_share': latest_balance.get('total_share', 'N/A'),
                    'cap_rese': latest_balance.get('cap_rese', 'N/A'),
                    'undistr_porfit': latest_balance.get('undistr_porfit', 'N/A'),
                    'surplus_rese': latest_balance.get('surplus_rese', 'N/A'),
                    'special_rese': latest_balance.get('special_rese', 'N/A'),
                    'money_cap': latest_balance.get('money_cap', 'N/A'),
                    'trad_asset': latest_balance.get('trad_asset', 'N/A'),
                    'notes_receiv': latest_balance.get('notes_receiv', 'N/A'),
                    'accounts_receiv': latest_balance.get('accounts_receiv', 'N/A'),
                    'oth_receiv': latest_balance.get('oth_receiv', 'N/A'),
                    'prepayment': latest_balance.get('prepayment', 'N/A'),
                    'div_receiv': latest_balance.get('div_receiv', 'N/A'),
                    'int_receiv': latest_balance.get('int_receiv', 'N/A'),
                    'inventories': latest_balance.get('inventories', 'N/A'),
                    'amor_exp': latest_balance.get('amor_exp', 'N/A'),
                    'nca_within_1y': latest_balance.get('nca_within_1y', 'N/A'),
                    'sett_rsrv': latest_balance.get('sett_rsrv', 'N/A'),
                    'loanto_oth_bank_fi': latest_balance.get('loanto_oth_bank_fi', 'N/A'),
                    'premium_receiv': latest_balance.get('premium_receiv', 'N/A'),
                    'reinsur_receiv': latest_balance.get('reinsur_receiv', 'N/A'),
                    'reinsur_res_receiv': latest_balance.get('reinsur_res_receiv', 'N/A'),
                    'pur_resale_fa': latest_balance.get('pur_resale_fa', 'N/A'),
                    'oth_cur_assets': latest_balance.get('oth_cur_assets', 'N/A'),
                    'total_cur_assets': latest_balance.get('total_cur_assets', 'N/A'),
                    'fa_avail_for_sale': latest_balance.get('fa_avail_for_sale', 'N/A'),
                    'htm_invest': latest_balance.get('htm_invest', 'N/A'),
                    'lt_eqt_invest': latest_balance.get('lt_eqt_invest', 'N/A'),
                    'invest_real_estate': latest_balance.get('invest_real_estate', 'N/A'),
                    'time_deposits': latest_balance.get('time_deposits', 'N/A'),
                    'oth_assets': latest_balance.get('oth_assets', 'N/A'),
                    'lt_rec': latest_balance.get('lt_rec', 'N/A'),
                    'fix_assets': latest_balance.get('fix_assets', 'N/A'),
                    'cip': latest_balance.get('cip', 'N/A'),
                    'const_materials': latest_balance.get('const_materials', 'N/A'),
                    'fixed_assets_disp': latest_balance.get('fixed_assets_disp', 'N/A'),
                    'produc_bio_assets': latest_balance.get('produc_bio_assets', 'N/A'),
                    'oil_and_gas_assets': latest_balance.get('oil_and_gas_assets', 'N/A'),
                    'intan_assets': latest_balance.get('intan_assets', 'N/A'),
                    'r_and_d': latest_balance.get('r_and_d', 'N/A'),
                    'goodwill': latest_balance.get('goodwill', 'N/A'),
                    'lt_amor_exp': latest_balance.get('lt_amor_exp', 'N/A'),
                    'defer_tax_assets': latest_balance.get('defer_tax_assets', 'N/A'),
                    'decr_in_disbur': latest_balance.get('decr_in_disbur', 'N/A'),
                    'oth_nca': latest_balance.get('oth_nca', 'N/A'),
                    'total_nca': latest_balance.get('total_nca', 'N/A'),
                    'cash_reser_cb': latest_balance.get('cash_reser_cb', 'N/A'),
                    'depos_in_oth_bfi': latest_balance.get('depos_in_oth_bfi', 'N/A'),
                    'prec_metals': latest_balance.get('prec_metals', 'N/A'),
                    'deriv_assets': latest_balance.get('deriv_assets', 'N/A'),
                    'rr_reins_une_prem': latest_balance.get('rr_reins_une_prem', 'N/A'),
                    'rr_reins_outstd_cla': latest_balance.get('rr_reins_outstd_cla', 'N/A'),
                    'rr_reins_lins_liab': latest_balance.get('rr_reins_lins_liab', 'N/A'),
                    'rr_reins_lthins_liab': latest_balance.get('rr_reins_lthins_liab', 'N/A'),
                    'refund_depos': latest_balance.get('refund_depos', 'N/A'),
                    'ph_pledge_loans': latest_balance.get('ph_pledge_loans', 'N/A'),
                    'receiv_invest': latest_balance.get('receiv_invest', 'N/A'),
                    'receiv_sec_depos': latest_balance.get('receiv_sec_depos', 'N/A'),
                    'receiv_subrog_rec': latest_balance.get('receiv_subrog_rec', 'N/A'),
                    'receiv_cash_reinsur': latest_balance.get('receiv_cash_reinsur', 'N/A'),
                    'oth_agent_assets': latest_balance.get('oth_agent_assets', 'N/A'),
                    'oth_agent_assetss': latest_balance.get('oth_agent_assetss', 'N/A'),
                    'oth_cur_assetss': latest_balance.get('oth_cur_assetss', 'N/A'),
                    'total_assets': latest_balance.get('total_assets', 'N/A'),
                    'lt_borr': latest_balance.get('lt_borr', 'N/A'),
                    'st_borr': latest_balance.get('st_borr', 'N/A'),
                    'cb_borr': latest_balance.get('cb_borr', 'N/A'),
                    'depos_ib_deposits': latest_balance.get('depos_ib_deposits', 'N/A'),
                    'loan_oth_bank': latest_balance.get('loan_oth_bank', 'N/A'),
                    'trading_fl': latest_balance.get('trading_fl', 'N/A'),
                    'notes_payable': latest_balance.get('notes_payable', 'N/A'),
                    'acct_payable': latest_balance.get('acct_payable', 'N/A'),
                    'adv_receipts': latest_balance.get('adv_receipts', 'N/A'),
                    'sold_for_repur_fa': latest_balance.get('sold_for_repur_fa', 'N/A'),
                    'comm_payable': latest_balance.get('comm_payable', 'N/A'),
                    'payroll_payable': latest_balance.get('payroll_payable', 'N/A'),
                    'taxes_payable': latest_balance.get('taxes_payable', 'N/A'),
                    'int_payable': latest_balance.get('int_payable', 'N/A'),
                    'div_payable': latest_balance.get('div_payable', 'N/A'),
                    'oth_payable': latest_balance.get('oth_payable', 'N/A'),
                    'acc_exp': latest_balance.get('acc_exp', 'N/A'),
                    'deferred_inc': latest_balance.get('deferred_inc', 'N/A'),
                    'st_bonds_payable': latest_balance.get('st_bonds_payable', 'N/A'),
                    'payable_to_reinsurer': latest_balance.get('payable_to_reinsurer', 'N/A'),
                    'rsrv_insur_cont': latest_balance.get('rsrv_insur_cont', 'N/A'),
                    'acting_trading_sec': latest_balance.get('acting_trading_sec', 'N/A'),
                    'acting_uw_sec': latest_balance.get('acting_uw_sec', 'N/A'),
                    'non_cur_liab_due_1y': latest_balance.get('non_cur_liab_due_1y', 'N/A'),
                    'oth_cur_liab': latest_balance.get('oth_cur_liab', 'N/A'),
                    'total_cur_liab': latest_balance.get('total_cur_liab', 'N/A'),
                    'bond_payable': latest_balance.get('bond_payable', 'N/A'),
                    'lt_payable': latest_balance.get('lt_payable', 'N/A'),
                    'specific_payables': latest_balance.get('specific_payables', 'N/A'),
                    'estimated_liab': latest_balance.get('estimated_liab', 'N/A'),
                    'defer_tax_liab': latest_balance.get('defer_tax_liab', 'N/A'),
                    'defer_inc_non_cur_liab': latest_balance.get('defer_inc_non_cur_liab', 'N/A'),
                    'oth_ncl': latest_balance.get('oth_ncl', 'N/A'),
                    'total_ncl': latest_balance.get('total_ncl', 'N/A'),
                    'depos_oth_bfi': latest_balance.get('depos_oth_bfi', 'N/A'),
                    'deriv_liab': latest_balance.get('deriv_liab', 'N/A'),
                    'depos': latest_balance.get('depos', 'N/A'),
                    'agency_bus_liab': latest_balance.get('agency_bus_liab', 'N/A'),
                    'oth_liab': latest_balance.get('oth_liab', 'N/A'),
                    'prem_receiv_adva': latest_balance.get('prem_receiv_adva', 'N/A'),
                    'depos_received': latest_balance.get('depos_received', 'N/A'),
                    'ph_invest': latest_balance.get('ph_invest', 'N/A'),
                    'reser_une_prem': latest_balance.get('reser_une_prem', 'N/A'),
                    'reser_outstd_claims': latest_balance.get('reser_outstd_claims', 'N/A'),
                    'reser_lins_liab': latest_balance.get('reser_lins_liab', 'N/A'),
                    'reser_lthins_liab': latest_balance.get('reser_lthins_liab', 'N/A'),
                    'indept_acc_liab': latest_balance.get('indept_acc_liab', 'N/A'),
                    'pledge_borr': latest_balance.get('pledge_borr', 'N/A'),
                    'indem_payable': latest_balance.get('indem_payable', 'N/A'),
                    'policy_div_payable': latest_balance.get('policy_div_payable', 'N/A'),
                    'total_liab': latest_balance.get('total_liab', 'N/A'),
                    'treasury_share': latest_balance.get('treasury_share', 'N/A'),
                    'ordin_risk_reser': latest_balance.get('ordin_risk_reser', 'N/A'),
                    'forex_differ': latest_balance.get('forex_differ', 'N/A'),
                    'invest_loss_unconf': latest_balance.get('invest_loss_unconf', 'N/A'),
                    'minority_int': latest_balance.get('minority_int', 'N/A'),
                    'total_hldr_eqy_exc_min_int': latest_balance.get('total_hldr_eqy_exc_min_int', 'N/A'),
                    'total_hldr_eqy_inc_min_int': latest_balance.get('total_hldr_eqy_inc_min_int', 'N/A'),
                    'total_liab_hldr_eqy': latest_balance.get('total_liab_hldr_eqy', 'N/A'),
                    'lt_payroll_payable': latest_balance.get('lt_payroll_payable', 'N/A'),
                    'oth_comp_income': latest_balance.get('oth_comp_income', 'N/A'),
                    'oth_eqt_tools': latest_balance.get('oth_eqt_tools', 'N/A'),
                    'oth_eqt_tools_p_shr': latest_balance.get('oth_eqt_tools_p_shr', 'N/A'),
                    'lending_funds': latest_balance.get('lending_funds', 'N/A'),
                    'acc_receivable': latest_balance.get('acc_receivable', 'N/A'),
                    'st_fin_payable': latest_balance.get('st_fin_payable', 'N/A'),
                    'payables': latest_balance.get('payables', 'N/A'),
                    'hfs_assets': latest_balance.get('hfs_assets', 'N/A'),
                    'hfs_sales': latest_balance.get('hfs_sales', 'N/A'),
                    'cost_fin_assets': latest_balance.get('cost_fin_assets', 'N/A'),
                    'fair_value_fin_assets': latest_balance.get('fair_value_fin_assets', 'N/A'),
                    'cip_total': latest_balance.get('cip_total', 'N/A'),
                    'oth_pay_total': latest_balance.get('oth_pay_total', 'N/A'),
                    'long_pay_total': latest_balance.get('long_pay_total', 'N/A'),
                    'debt_invest': latest_balance.get('debt_invest', 'N/A'),
                    'oth_debt_invest': latest_balance.get('oth_debt_invest', 'N/A'),
                    'oth_eq_invest': latest_balance.get('oth_eq_invest', 'N/A'),
                    'oth_illiq_fin_assets': latest_balance.get('oth_illiq_fin_assets', 'N/A'),
                    'oth_eq_ppbond': latest_balance.get('oth_eq_ppbond', 'N/A'),
                    'receiv_financing': latest_balance.get('receiv_financing', 'N/A'),
                    'use_right_assets': latest_balance.get('use_right_assets', 'N/A'),
                    'lease_liab': latest_balance.get('lease_liab', 'N/A'),
                    'contract_assets': latest_balance.get('contract_assets', 'N/A'),
                    'contract_liab': latest_balance.get('contract_liab', 'N/A'),
                    'accounts_receiv_bill': latest_balance.get('accounts_receiv_bill', 'N/A'),
                    'accounts_pay': latest_balance.get('accounts_pay', 'N/A'),
                    'oth_rcv_total': latest_balance.get('oth_rcv_total', 'N/A'),
                    'fix_assets_total': latest_balance.get('fix_assets_total', 'N/A'),
                    'update_flag': latest_balance.get('update_flag', 'N/A'),
                }
                result['data_sources'].append('balancesheet')
        except Exception as e:
            result['balancesheet_error'] = str(e)

        # 7. Get cash flow data
        try:
            cashflow_data = tushare_utils.pro.cashflow(ts_code=ts_symbol)
            if not cashflow_data.empty:
                # Sort by end_date to get the most recent data
                cashflow_data = cashflow_data.sort_values('end_date', ascending=False)
                latest_cashflow = cashflow_data.iloc[0]
                result['cashflow'] = {
                    'end_date': latest_cashflow.get('end_date', 'N/A'),
                    'net_profit': latest_cashflow.get('net_profit', 'N/A'),
                    'finan_exp': latest_cashflow.get('finan_exp', 'N/A'),
                    'c_fr_sale_sg': latest_cashflow.get('c_fr_sale_sg', 'N/A'),
                    'recp_tax_rends': latest_cashflow.get('recp_tax_rends', 'N/A'),
                    'n_depos_incr_fi': latest_cashflow.get('n_depos_incr_fi', 'N/A'),
                    'n_incr_loans_cb': latest_cashflow.get('n_incr_loans_cb', 'N/A'),
                    'n_inc_borr_oth_fi': latest_cashflow.get('n_inc_borr_oth_fi', 'N/A'),
                    'prem_fr_orig_contr': latest_cashflow.get('prem_fr_orig_contr', 'N/A'),
                    'n_incr_insured_dep': latest_cashflow.get('n_incr_insured_dep', 'N/A'),
                    'n_reinsur_prem': latest_cashflow.get('n_reinsur_prem', 'N/A'),
                    'n_incr_disp_tfa': latest_cashflow.get('n_incr_disp_tfa', 'N/A'),
                    'ifc_cash_incr': latest_cashflow.get('ifc_cash_incr', 'N/A'),
                    'n_incr_disp_faas': latest_cashflow.get('n_incr_disp_faas', 'N/A'),
                    'n_incr_loans_oth_bank': latest_cashflow.get('n_incr_loans_oth_bank', 'N/A'),
                    'n_cap_incr_repur': latest_cashflow.get('n_cap_incr_repur', 'N/A'),
                    'c_fr_oth_operate_a': latest_cashflow.get('c_fr_oth_operate_a', 'N/A'),
                    'c_inf_fr_operate_a': latest_cashflow.get('c_inf_fr_operate_a', 'N/A'),
                    'c_paid_goods_s': latest_cashflow.get('c_paid_goods_s', 'N/A'),
                    'c_paid_to_for_empl': latest_cashflow.get('c_paid_to_for_empl', 'N/A'),
                    'c_paid_for_taxes': latest_cashflow.get('c_paid_for_taxes', 'N/A'),
                    'n_incr_clt_loan_adv': latest_cashflow.get('n_incr_clt_loan_adv', 'N/A'),
                    'n_incr_dep_cbob': latest_cashflow.get('n_incr_dep_cbob', 'N/A'),
                    'c_pay_claims_orig_inco': latest_cashflow.get('c_pay_claims_orig_inco', 'N/A'),
                    'pay_handling_chrg': latest_cashflow.get('pay_handling_chrg', 'N/A'),
                    'pay_comm_insur_plcy': latest_cashflow.get('pay_comm_insur_plcy', 'N/A'),
                    'oth_cash_pay_oper_act': latest_cashflow.get('oth_cash_pay_oper_act', 'N/A'),
                    'st_cash_out_act': latest_cashflow.get('st_cash_out_act', 'N/A'),
                    'n_cashflow_act': latest_cashflow.get('n_cashflow_act', 'N/A'),
                    'oth_recp_ral_inv_act': latest_cashflow.get('oth_recp_ral_inv_act', 'N/A'),
                    'c_disp_withdrwl_invest': latest_cashflow.get('c_disp_withdrwl_invest', 'N/A'),
                    'c_recp_return_invest': latest_cashflow.get('c_recp_return_invest', 'N/A'),
                    'n_recp_disp_fiolta': latest_cashflow.get('n_recp_disp_fiolta', 'N/A'),
                    'n_recp_disp_sobu': latest_cashflow.get('n_recp_disp_sobu', 'N/A'),
                    'stot_inflows_inv_act': latest_cashflow.get('stot_inflows_inv_act', 'N/A'),
                    'c_pay_acq_const_fiolta': latest_cashflow.get('c_pay_acq_const_fiolta', 'N/A'),
                    'c_paid_invest': latest_cashflow.get('c_paid_invest', 'N/A'),
                    'n_incr_pledge_loan': latest_cashflow.get('n_incr_pledge_loan', 'N/A'),
                    'n_pay_acq_sobu': latest_cashflow.get('n_pay_acq_sobu', 'N/A'),
                    'oth_pay_ral_inv_act': latest_cashflow.get('oth_pay_ral_inv_act', 'N/A'),
                    'n_incr_fiolta': latest_cashflow.get('n_incr_fiolta', 'N/A'),
                    'stot_out_inv_act': latest_cashflow.get('stot_out_inv_act', 'N/A'),
                    'n_cashflow_inv_act': latest_cashflow.get('n_cashflow_inv_act', 'N/A'),
                    'c_recp_borrow': latest_cashflow.get('c_recp_borrow', 'N/A'),
                    'proc_issue_bonds': latest_cashflow.get('proc_issue_bonds', 'N/A'),
                    'oth_cash_recp_ral_fnc_act': latest_cashflow.get('oth_cash_recp_ral_fnc_act', 'N/A'),
                    'stot_cash_in_fnc_act': latest_cashflow.get('stot_cash_in_fnc_act', 'N/A'),
                    'free_cashflow': latest_cashflow.get('free_cashflow', 'N/A'),
                    'c_prepay_amt_borr': latest_cashflow.get('c_prepay_amt_borr', 'N/A'),
                    'c_pay_dist_dpcp_int_exp': latest_cashflow.get('c_pay_dist_dpcp_int_exp', 'N/A'),
                    'incl_dvd_profit_paid_sc_ms': latest_cashflow.get('incl_dvd_profit_paid_sc_ms', 'N/A'),
                    'oth_cashpay_ral_fnc_act': latest_cashflow.get('oth_cashpay_ral_fnc_act', 'N/A'),
                    'stot_cashout_fnc_act': latest_cashflow.get('stot_cashout_fnc_act', 'N/A'),
                    'n_cash_flows_fnc_act': latest_cashflow.get('n_cash_flows_fnc_act', 'N/A'),
                    'eff_fx_flu_cash': latest_cashflow.get('eff_fx_flu_cash', 'N/A'),
                    'n_incr_cash_cash_equ': latest_cashflow.get('n_incr_cash_cash_equ', 'N/A'),
                    'c_cash_equ_beg_period': latest_cashflow.get('c_cash_equ_beg_period', 'N/A'),
                    'c_cash_equ_end_period': latest_cashflow.get('c_cash_equ_end_period', 'N/A'),
                    'c_recp_cap_contrib': latest_cashflow.get('c_recp_cap_contrib', 'N/A'),
                    'incl_cash_rec_saims': latest_cashflow.get('incl_cash_rec_saims', 'N/A'),
                    'uncon_invest_loss': latest_cashflow.get('uncon_invest_loss', 'N/A'),
                    'prov_depr_assets': latest_cashflow.get('prov_depr_assets', 'N/A'),
                    'depr_fa_coga_dpba': latest_cashflow.get('depr_fa_coga_dpba', 'N/A'),
                    'amort_intang_assets': latest_cashflow.get('amort_intang_assets', 'N/A'),
                    'lt_amort_deferred_exp': latest_cashflow.get('lt_amort_deferred_exp', 'N/A'),
                    'decr_deferred_exp': latest_cashflow.get('decr_deferred_exp', 'N/A'),
                    'incr_acc_exp': latest_cashflow.get('incr_acc_exp', 'N/A'),
                    'loss_disp_fiolta': latest_cashflow.get('loss_disp_fiolta', 'N/A'),
                    'loss_scr_fa': latest_cashflow.get('loss_scr_fa', 'N/A'),
                    'loss_fv_chg': latest_cashflow.get('loss_fv_chg', 'N/A'),
                    'invest_loss': latest_cashflow.get('invest_loss', 'N/A'),
                    'decr_def_inc_tax_assets': latest_cashflow.get('decr_def_inc_tax_assets', 'N/A'),
                    'incr_def_inc_tax_liab': latest_cashflow.get('incr_def_inc_tax_liab', 'N/A'),
                    'decr_inventories': latest_cashflow.get('decr_inventories', 'N/A'),
                    'decr_oper_payable': latest_cashflow.get('decr_oper_payable', 'N/A'),
                    'incr_oper_payable': latest_cashflow.get('incr_oper_payable', 'N/A'),
                    'others': latest_cashflow.get('others', 'N/A'),
                    'im_net_cashflow_oper_act': latest_cashflow.get('im_net_cashflow_oper_act', 'N/A'),
                    'conv_debt_into_cap': latest_cashflow.get('conv_debt_into_cap', 'N/A'),
                    'conv_copbonds_due_within_1y': latest_cashflow.get('conv_copbonds_due_within_1y', 'N/A'),
                    'fa_fnc_leases': latest_cashflow.get('fa_fnc_leases', 'N/A'),
                    'im_n_incr_cash_equ': latest_cashflow.get('im_n_incr_cash_equ', 'N/A'),
                    'net_dism_capital_add': latest_cashflow.get('net_dism_capital_add', 'N/A'),
                    'net_cash_rece_sec': latest_cashflow.get('net_cash_rece_sec', 'N/A'),
                    'credit_impa_loss': latest_cashflow.get('credit_impa_loss', 'N/A'),
                    'use_right_asset_dep': latest_cashflow.get('use_right_asset_dep', 'N/A'),
                    'oth_loss_asset': latest_cashflow.get('oth_loss_asset', 'N/A'),
                    'end_bal_cash': latest_cashflow.get('end_bal_cash', 'N/A'),
                    'beg_bal_cash': latest_cashflow.get('beg_bal_cash', 'N/A'),
                    'end_bal_cash_equ': latest_cashflow.get('end_bal_cash_equ', 'N/A'),
                    'beg_bal_cash_equ': latest_cashflow.get('beg_bal_cash_equ', 'N/A'),
                    'update_flag': latest_cashflow.get('update_flag', 'N/A'),
                }
                result['data_sources'].append('cashflow')
        except Exception as e:
            result['cashflow_error'] = str(e)

        return _format_comprehensive_fundamental_report(result)

    except Exception as e:
        return f"Error retrieving comprehensive fundamental data for {ticker}: {str(e)}"


def get_fundamentals_openai(ticker, curr_date):
    """
    Get fundamental data for a stock. For Chinese stocks, use Tushare API.
    For international stocks, use web search via OpenAI/Google.
    """
    config = get_config()

    # First, check if this is a Chinese stock and try Tushare
    try:
        from .tushare_utils import get_tushare_utils
        tushare_utils = get_tushare_utils()

        if tushare_utils.is_chinese_stock(ticker):
            print(f"Detected Chinese stock {ticker}, using Tushare for fundamental data...")
            fundamental_data = tushare_utils.get_fundamental_data(ticker, curr_date)
            return tushare_utils.format_fundamental_report(fundamental_data)
    except Exception as e:
        print(f"Tushare fundamental data failed for {ticker}: {e}, falling back to web search")

    # Fallback to web search for international stocks or if Tushare fails
    # Check if using Google provider and if Google Gemini is available
    if config.get("llm_provider", "").lower() == "google" and GOOGLE_GENAI_AVAILABLE:
        try:
            # Configure the Gemini client
            client = genai.Client()

            # Define the grounding tool for Google Search
            grounding_tool = types.Tool(
                google_search=types.GoogleSearch()
            )

            # Configure generation settings
            generation_config = types.GenerateContentConfig(
                tools=[grounding_tool],
                temperature=1,
                max_output_tokens=4096,
                top_p=1,
            )

            # Make the request
            response = client.models.generate_content(
                model=config["quick_think_llm"],
                contents=f"Can you search Fundamental for discussions on {ticker} during of the month before {curr_date} to the month of {curr_date}. Make sure you only get the data posted during that period. List as a table, with PE/PS/Cash flow/ etc",
                config=generation_config,
            )

            return response.text
        except Exception as e:
            print(f"Google Gemini API failed: {e}, falling back to OpenAI")
            # Fall through to OpenAI implementation
    else:
        # Original OpenAI implementation
        client = OpenAI(base_url=config["backend_url"])

        response = client.responses.create(
            model=config["quick_think_llm"],
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Can you search Fundamental for discussions on {ticker} during of the month before {curr_date} to the month of {curr_date}. Make sure you only get the data posted during that period. List as a table, with PE/PS/Cash flow/ etc",
                        }
                    ],
                }
            ],
            text={"format": {"type": "text"}},
            reasoning={},
            tools=[
                {
                    "type": "web_search_preview",
                    "user_location": {"type": "approximate"},
                    "search_context_size": "low",
                }
            ],
            temperature=1,
            max_output_tokens=4096,
            top_p=1,
            store=True,
        )

        return response.output[1].content[0].text


def _safe_format_value(value, value_type='number', unit=''):
    """
    Safely format values, handling None, nan, and other edge cases

    Args:
        value: The value to format
        value_type: Type of value ('number', 'percentage', 'currency', 'text')
        unit: Unit to append (e.g., '元', '%', '万股')

    Returns:
        str: Formatted value string
    """
    import pandas as pd
    import numpy as np

    # Handle None, nan, and empty values
    if value is None or pd.isna(value) or value == '' or str(value).lower() == 'none':
        return 'N/A'

    # Handle numpy nan specifically
    try:
        import numpy as np
        if np.isnan(float(value)):
            return 'N/A'
    except (ValueError, TypeError, ImportError):
        pass

    # Handle string values
    if isinstance(value, str):
        if value.lower() in ['n/a', 'na', 'null', '']:
            return 'N/A'
        return value

    # Handle numeric values
    try:
        if value_type == 'percentage':
            if abs(float(value)) < 0.01:
                return f"{float(value):.4f}%"
            else:
                return f"{float(value):.2f}%"
        elif value_type == 'currency':
            if abs(float(value)) >= 1e8:  # 亿
                return f"{float(value)/1e8:.2f}亿{unit}"
            elif abs(float(value)) >= 1e4:  # 万
                return f"{float(value)/1e4:.2f}万{unit}"
            else:
                return f"{float(value):.2f}{unit}"
        elif value_type == 'number':
            if abs(float(value)) >= 1e8:  # 亿
                return f"{float(value)/1e8:.2f}亿{unit}"
            elif abs(float(value)) >= 1e4:  # 万
                return f"{float(value)/1e4:.2f}万{unit}"
            else:
                return f"{float(value):.2f}{unit}"
        else:
            return f"{value}{unit}"
    except (ValueError, TypeError):
        return str(value) if value else 'N/A'


def _format_comprehensive_fundamental_report(result: dict) -> str:
    """
    Format comprehensive fundamental data into a readable report

    Args:
        result: Dictionary containing comprehensive fundamental data

    Returns:
        str: Formatted comprehensive report string
    """
    report = f"""
# Comprehensive Fundamental Analysis Report for {result.get('symbol', 'N/A')}

## Basic Information
- **Symbol**: {result.get('symbol', 'N/A')}
- **Tushare Symbol**: {result.get('ts_symbol', 'N/A')}
- **Analysis Date**: {result.get('analysis_date', 'N/A')}
- **Data Sources**: {', '.join(result.get('data_sources', []))}

"""

    # Daily Basic Data Section
    if 'daily_basic' in result:
        daily_basic = result['daily_basic']
        report += f"""## Market Valuation Metrics (Daily Basic)
- **Data Date**: {result.get('daily_basic_date', 'N/A')}
- **PE Ratio**: {_safe_format_value(daily_basic.get('pe_ratio'), 'number')}
- **PB Ratio**: {_safe_format_value(daily_basic.get('pb_ratio'), 'number')}
- **PS Ratio**: {_safe_format_value(daily_basic.get('ps_ratio'), 'number')}
- **PE TTM**: {_safe_format_value(daily_basic.get('pe_ttm'), 'number')}
- **PB MRQ**: {_safe_format_value(daily_basic.get('pb_mrq'), 'number')}
- **PS TTM**: {_safe_format_value(daily_basic.get('ps_ttm'), 'number')}

### Share Information
- **Total Shares**: {_safe_format_value(daily_basic.get('total_share'), 'number', '万股')}
- **Float Shares**: {_safe_format_value(daily_basic.get('float_share'), 'number', '万股')}
- **Total Market Value**: {_safe_format_value(daily_basic.get('total_mv'), 'currency', '元')}
- **Circulating Market Value**: {_safe_format_value(daily_basic.get('circ_mv'), 'currency', '元')}
- **Turnover Rate**: {_safe_format_value(daily_basic.get('turnover_rate'), 'percentage')}
- **Volume Ratio**: {_safe_format_value(daily_basic.get('volume_ratio'), 'number')}

"""
    elif 'daily_basic_error' in result:
        report += f"""## Market Valuation Metrics (Daily Basic)
**Error**: {result['daily_basic_error']}

"""

    # Financial Indicators Section
    if 'fina_indicator' in result:
        fina = result['fina_indicator']
        report += f"""## Financial Indicators
- **Report Period**: {fina.get('end_date', 'N/A')}

### Per Share Metrics
- **Earnings Per Share (EPS)**: {_safe_format_value(fina.get('eps'), 'number', '元')}
- **Diluted EPS**: {_safe_format_value(fina.get('dt_eps'), 'number', '元')}
- **Book Value Per Share (BPS)**: {_safe_format_value(fina.get('bps'), 'number', '元')}
- **Operating Cash Flow Per Share**: {_safe_format_value(fina.get('ocfps'), 'number', '元')}
- **Cash Flow Per Share**: {_safe_format_value(fina.get('cfps'), 'number', '元')}
- **Revenue Per Share**: {_safe_format_value(fina.get('revenue_ps'), 'number', '元')}

### Profitability Ratios
- **Return on Equity (ROE)**: {_safe_format_value(fina.get('roe'), 'percentage')}
- **Return on Assets (ROA)**: {_safe_format_value(fina.get('roa'), 'percentage')}
- **Return on Invested Capital (ROIC)**: {_safe_format_value(fina.get('roic'), 'percentage')}
- **Net Profit Margin**: {_safe_format_value(fina.get('netprofit_margin'), 'percentage')}
- **Gross Profit Margin**: {_safe_format_value(fina.get('grossprofit_margin'), 'percentage')}
- **EBIT**: {_safe_format_value(fina.get('ebit'), 'currency', '元')}
- **EBITDA**: {_safe_format_value(fina.get('ebitda'), 'currency', '元')}

### Liquidity Ratios
- **Current Ratio**: {_safe_format_value(fina.get('current_ratio'), 'number')}
- **Quick Ratio**: {_safe_format_value(fina.get('quick_ratio'), 'number')}
- **Cash Ratio**: {_safe_format_value(fina.get('cash_ratio'), 'number')}

### Efficiency Ratios
- **Asset Turnover**: {_safe_format_value(fina.get('assets_turn'), 'number')}
- **Accounts Receivable Turnover**: {_safe_format_value(fina.get('ar_turn'), 'number')}
- **Current Asset Turnover**: {_safe_format_value(fina.get('ca_turn'), 'number')}
- **Fixed Asset Turnover**: {_safe_format_value(fina.get('fa_turn'), 'number')}

### Leverage Ratios
- **Debt to Assets**: {_safe_format_value(fina.get('debt_to_assets'), 'percentage')}
- **Assets to Equity**: {_safe_format_value(fina.get('assets_to_eqt'), 'number')}
- **Debt to Equity**: {_safe_format_value(fina.get('debt_to_eqt'), 'number')}

### Growth Rates (YoY)
- **EPS Growth**: {_safe_format_value(fina.get('basic_eps_yoy'), 'percentage')}
- **Net Profit Growth**: {_safe_format_value(fina.get('netprofit_yoy'), 'percentage')}
- **Revenue Growth**: {_safe_format_value(fina.get('or_yoy'), 'percentage')}
- **ROE Growth**: {_safe_format_value(fina.get('roe_yoy'), 'percentage')}
- **Total Assets Growth**: {_safe_format_value(fina.get('assets_yoy'), 'percentage')}

"""
    elif 'fina_indicator_error' in result:
        report += f"""## Financial Indicators
**Error**: {result['fina_indicator_error']}

"""

    # Top 10 Holders Section
    if 'top10_holders' in result:
        holders = result['top10_holders']
        report += f"""## Top 10 Shareholders
- **Report Period**: {holders.get('end_date', 'N/A')}

| Rank | Shareholder Name | Holdings (万股) | Holding Ratio (%) |
|------|------------------|-----------------|-------------------|
"""
        for i, holder in enumerate(holders.get('holders', []), 1):
            report += f"| {i} | {holder.get('holder_name', 'N/A')} | {holder.get('hold_amount', 'N/A')} | {holder.get('hold_ratio', 'N/A')} |\n"
        report += "\n"
    elif 'top10_holders_error' in result:
        report += f"""## Top 10 Shareholders
**Error**: {result['top10_holders_error']}

"""

    # Top 10 Float Holders Section
    if 'top10_floatholders' in result:
        float_holders = result['top10_floatholders']
        report += f"""## Top 10 Floating Shareholders
- **Report Period**: {float_holders.get('end_date', 'N/A')}

| Rank | Shareholder Name | Holdings (万股) | Holding Ratio (%) |
|------|------------------|-----------------|-------------------|
"""
        for i, holder in enumerate(float_holders.get('holders', []), 1):
            report += f"| {i} | {holder.get('holder_name', 'N/A')} | {holder.get('hold_amount', 'N/A')} | {holder.get('hold_ratio', 'N/A')} |\n"
        report += "\n"
    elif 'top10_floatholders_error' in result:
        report += f"""## Top 10 Floating Shareholders
**Error**: {result['top10_floatholders_error']}

"""

    # Income Statement Section
    if 'income' in result:
        income = result['income']
        report += f"""## Income Statement
- **Report Period**: {income.get('end_date', 'N/A')}

### Revenue
- **Total Revenue**: {_safe_format_value(income.get('total_revenue'), 'currency', '元')}
- **Operating Revenue**: {_safe_format_value(income.get('revenue'), 'currency', '元')}
- **Investment Income**: {_safe_format_value(income.get('invest_income'), 'currency', '元')}
- **Other Income**: {_safe_format_value(income.get('n_oth_income'), 'currency', '元')}

### Costs and Expenses
- **Total Operating Costs**: {_safe_format_value(income.get('total_cogs'), 'currency', '元')}
- **Operating Costs**: {_safe_format_value(income.get('oper_cost'), 'currency', '元')}
- **Selling Expenses**: {_safe_format_value(income.get('sell_exp'), 'currency', '元')}
- **Administrative Expenses**: {_safe_format_value(income.get('admin_exp'), 'currency', '元')}
- **Financial Expenses**: {_safe_format_value(income.get('fin_exp'), 'currency', '元')}
- **R&D Expenses**: {_safe_format_value(income.get('rd_exp'), 'currency', '元')}
- **Asset Impairment Loss**: {_safe_format_value(income.get('assets_impair_loss'), 'currency', '元')}

### Profit
- **Operating Profit**: {_safe_format_value(income.get('operate_profit'), 'currency', '元')}
- **Total Profit**: {_safe_format_value(income.get('total_profit'), 'currency', '元')}
- **Net Income**: {_safe_format_value(income.get('n_income'), 'currency', '元')}
- **Net Income (Attributable to Parent)**: {_safe_format_value(income.get('n_income_attr_p'), 'currency', '元')}
- **EBIT**: {_safe_format_value(income.get('ebit'), 'currency', '元')}
- **EBITDA**: {_safe_format_value(income.get('ebitda'), 'currency', '元')}

### Tax
- **Income Tax**: {_safe_format_value(income.get('income_tax'), 'currency', '元')}

"""
    elif 'income_error' in result:
        report += f"""## Income Statement
**Error**: {result['income_error']}

"""

    # Balance Sheet Section
    if 'balancesheet' in result:
        balance = result['balancesheet']
        report += f"""## Balance Sheet
- **Report Period**: {balance.get('end_date', 'N/A')}

### Assets
- **Total Assets**: {_safe_format_value(balance.get('total_assets'), 'currency', '元')}
- **Total Current Assets**: {_safe_format_value(balance.get('total_cur_assets'), 'currency', '元')}
- **Total Non-Current Assets**: {_safe_format_value(balance.get('total_nca'), 'currency', '元')}

#### Current Assets
- **Cash and Cash Equivalents**: {_safe_format_value(balance.get('money_cap'), 'currency', '元')}
- **Accounts Receivable**: {_safe_format_value(balance.get('accounts_receiv'), 'currency', '元')}
- **Inventories**: {_safe_format_value(balance.get('inventories'), 'currency', '元')}
- **Other Current Assets**: {_safe_format_value(balance.get('oth_cur_assets'), 'currency', '元')}

#### Non-Current Assets
- **Fixed Assets**: {_safe_format_value(balance.get('fix_assets'), 'currency', '元')}
- **Intangible Assets**: {_safe_format_value(balance.get('intan_assets'), 'currency', '元')}
- **Goodwill**: {_safe_format_value(balance.get('goodwill'), 'currency', '元')}
- **Long-term Equity Investment**: {_safe_format_value(balance.get('lt_eqt_invest'), 'currency', '元')}

### Liabilities
- **Total Liabilities**: {_safe_format_value(balance.get('total_liab'), 'currency', '元')}
- **Total Current Liabilities**: {_safe_format_value(balance.get('total_cur_liab'), 'currency', '元')}
- **Total Non-Current Liabilities**: {_safe_format_value(balance.get('total_ncl'), 'currency', '元')}

#### Current Liabilities
- **Short-term Borrowings**: {_safe_format_value(balance.get('st_borr'), 'currency', '元')}
- **Accounts Payable**: {_safe_format_value(balance.get('acct_payable'), 'currency', '元')}
- **Other Current Liabilities**: {_safe_format_value(balance.get('oth_cur_liab'), 'currency', '元')}

#### Non-Current Liabilities
- **Long-term Borrowings**: {_safe_format_value(balance.get('lt_borr'), 'currency', '元')}
- **Bonds Payable**: {_safe_format_value(balance.get('bond_payable'), 'currency', '元')}

### Equity
- **Total Equity (Excluding Minority Interest)**: {_safe_format_value(balance.get('total_hldr_eqy_exc_min_int'), 'currency', '元')}
- **Total Equity (Including Minority Interest)**: {_safe_format_value(balance.get('total_hldr_eqy_inc_min_int'), 'currency', '元')}
- **Share Capital**: {_safe_format_value(balance.get('total_share'), 'currency', '元')}
- **Capital Reserve**: {_safe_format_value(balance.get('cap_rese'), 'currency', '元')}
- **Surplus Reserve**: {_safe_format_value(balance.get('surplus_rese'), 'currency', '元')}
- **Undistributed Profit**: {_safe_format_value(balance.get('undistr_porfit'), 'currency', '元')}

"""
    elif 'balancesheet_error' in result:
        report += f"""## Balance Sheet
**Error**: {result['balancesheet_error']}

"""

    # Cash Flow Section
    if 'cashflow' in result:
        cashflow = result['cashflow']
        report += f"""## Cash Flow Statement
- **Report Period**: {cashflow.get('end_date', 'N/A')}

### Operating Activities
- **Net Cash Flow from Operating Activities**: {_safe_format_value(cashflow.get('n_cashflow_act'), 'currency', '元')}
- **Cash Received from Sales**: {_safe_format_value(cashflow.get('c_fr_sale_sg'), 'currency', '元')}
- **Cash Paid for Goods and Services**: {_safe_format_value(cashflow.get('c_paid_goods_s'), 'currency', '元')}
- **Cash Paid to Employees**: {_safe_format_value(cashflow.get('c_paid_to_for_empl'), 'currency', '元')}
- **Cash Paid for Taxes**: {_safe_format_value(cashflow.get('c_paid_for_taxes'), 'currency', '元')}

### Investing Activities
- **Net Cash Flow from Investing Activities**: {_safe_format_value(cashflow.get('n_cashflow_inv_act'), 'currency', '元')}
- **Cash Paid for Fixed Assets**: {_safe_format_value(cashflow.get('c_pay_acq_const_fiolta'), 'currency', '元')}
- **Cash Paid for Investments**: {_safe_format_value(cashflow.get('c_paid_invest'), 'currency', '元')}
- **Cash Received from Disposal of Investments**: {_safe_format_value(cashflow.get('c_recp_return_invest'), 'currency', '元')}

### Financing Activities
- **Net Cash Flow from Financing Activities**: {_safe_format_value(cashflow.get('n_cash_flows_fnc_act'), 'currency', '元')}
- **Cash Received from Borrowings**: {_safe_format_value(cashflow.get('c_recp_borrow'), 'currency', '元')}
- **Cash Received from Capital Contributions**: {_safe_format_value(cashflow.get('c_recp_cap_contrib'), 'currency', '元')}
- **Cash Paid for Debt Repayment**: {_safe_format_value(cashflow.get('c_prepay_amt_borr'), 'currency', '元')}
- **Cash Paid for Dividends**: {_safe_format_value(cashflow.get('c_pay_dist_dpcp_int_exp'), 'currency', '元')}

### Cash Position
- **Net Increase in Cash**: {_safe_format_value(cashflow.get('n_incr_cash_cash_equ'), 'currency', '元')}
- **Cash at Beginning of Period**: {_safe_format_value(cashflow.get('c_cash_equ_beg_period'), 'currency', '元')}
- **Cash at End of Period**: {_safe_format_value(cashflow.get('c_cash_equ_end_period'), 'currency', '元')}
- **Free Cash Flow**: {_safe_format_value(cashflow.get('free_cashflow'), 'currency', '元')}

### Reconciliation Items
- **Net Profit**: {_safe_format_value(cashflow.get('net_profit'), 'currency', '元')}
- **Depreciation and Amortization**: {_safe_format_value(cashflow.get('depr_fa_coga_dpba'), 'currency', '元')}
- **Asset Impairment**: {_safe_format_value(cashflow.get('prov_depr_assets'), 'currency', '元')}

"""
    elif 'cashflow_error' in result:
        report += f"""## Cash Flow Statement
**Error**: {result['cashflow_error']}

"""

    # Add any errors at the end
    errors = []
    for key, value in result.items():
        if key.endswith('_error'):
            errors.append(f"- **{key.replace('_error', '').title()}**: {value}")

    if errors:
        report += f"""## Data Retrieval Errors
{chr(10).join(errors)}

"""

    report += """---
*Data source: Tushare API*
*Report generated by TradingAgents comprehensive fundamental analysis function*
"""

    return report


def get_sina_global_financial_news(curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"]) -> str:
    """
    获取新浪财经全球财经快讯

    Args:
        curr_date (str): current date you are trading at, yyyy-mm-dd

    Returns:
        str: 格式化的全球财经快讯字符串，包含时间和内容
    """
    try:
        # 调用akshare接口获取新浪财经全球财经快讯
        news_df = ak.stock_info_global_sina()

        if news_df.empty:
            return "暂无全球财经快讯数据"

        # 过滤当天的新闻
        curr_date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
        filtered_news = []

        for index, row in news_df.iterrows():
            time_str = row['时间']
            content = row['内容']

            try:
                # 解析时间字符串，支持多种格式
                news_time = None
                time_str_clean = str(time_str).strip()

                # 尝试不同的时间格式
                time_formats = [
                    "%Y-%m-%d %H:%M:%S",  # YYYY-MM-DD HH:MM:SS
                    "%Y-%m-%d %H:%M",     # YYYY-MM-DD HH:MM
                    "%Y-%m-%d",           # YYYY-MM-DD
                    "%m-%d %H:%M",        # MM-DD HH:MM
                ]

                for fmt in time_formats:
                    try:
                        if fmt == "%m-%d %H:%M":
                            # 对于 MM-DD HH:MM 格式，添加当前年份
                            time_with_year = f"{curr_date_obj.year}-{time_str_clean}"
                            news_time = datetime.strptime(time_with_year, "%Y-%m-%d %H:%M")
                        else:
                            news_time = datetime.strptime(time_str_clean, fmt)
                        break
                    except ValueError:
                        continue

                # 检查是否为当天新闻
                if news_time and news_time.date() == curr_date_obj.date():
                    filtered_news.append((time_str, content))
            except:
                # 如果时间解析失败，跳过该条新闻
                continue

        if not filtered_news:
            return f"暂无 {curr_date} 当天的新浪财经全球财经快讯数据"

        # 格式化输出
        news_str = f"## 新浪财经-全球财经快讯 ({curr_date})\n\n"

        for time_str, content in filtered_news:
            news_str += f"**{time_str}**\n{content}\n\n"

        return news_str

    except Exception as e:
        return f"获取新浪财经全球财经快讯时发生错误: {str(e)}"


def get_eastmoney_financial_breakfast(curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"]) -> str:
    """
    获取东方财富财经早餐

    Args:
        curr_date (str): current date you are trading at, yyyy-mm-dd

    Returns:
        str: 格式化的财经早餐字符串，包含标题、摘要、发布时间和链接
    """
    try:
        # 调用akshare接口获取东方财富财经早餐
        breakfast_df = ak.stock_info_cjzc_em()

        if breakfast_df.empty:
            return "暂无财经早餐数据"

        # 过滤当天的新闻
        curr_date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
        filtered_news = []

        for index, row in breakfast_df.iterrows():
            title = row['标题']
            summary = row['摘要']
            publish_time = row['发布时间']
            link = row['链接']

            try:
                # 解析发布时间，支持多种格式
                news_time = None
                publish_time_clean = str(publish_time).strip()

                # 尝试不同的时间格式
                time_formats = [
                    "%Y-%m-%d %H:%M:%S",  # YYYY-MM-DD HH:MM:SS
                    "%Y-%m-%d %H:%M",     # YYYY-MM-DD HH:MM
                    "%Y-%m-%d",           # YYYY-MM-DD
                ]

                for fmt in time_formats:
                    try:
                        news_time = datetime.strptime(publish_time_clean, fmt)
                        break
                    except ValueError:
                        continue

                # 检查是否为当天新闻
                if news_time and news_time.date() == curr_date_obj.date():
                    filtered_news.append((title, summary, publish_time, link))
            except:
                # 如果时间解析失败，跳过该条新闻
                continue

        if not filtered_news:
            return f"暂无 {curr_date} 当天的东方财富财经早餐数据"

        # 格式化输出
        breakfast_str = f"## 东方财富-财经早餐 ({curr_date})\n\n"

        for title, summary, publish_time, link in filtered_news:
            breakfast_str += f"### {title}\n"
            breakfast_str += f"**发布时间**: {publish_time}\n\n"
            breakfast_str += f"{summary}\n\n"
            breakfast_str += f"[查看详情]({link})\n\n"
            breakfast_str += "---\n\n"

        return breakfast_str

    except Exception as e:
        return f"获取东方财富财经早餐时发生错误: {str(e)}"


def get_eastmoney_global_financial_news(curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"]) -> str:
    """
    获取东方财富全球财经快讯

    Args:
        curr_date (str): current date you are trading at, yyyy-mm-dd

    Returns:
        str: 格式化的全球财经快讯字符串，包含标题、摘要、发布时间和链接
    """
    try:
        # 调用akshare接口获取东方财富全球财经快讯
        news_df = ak.stock_info_global_em()

        if news_df.empty:
            return "暂无东方财富全球财经快讯数据"

        # 过滤当天的新闻
        curr_date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
        filtered_news = []

        for index, row in news_df.iterrows():
            title = row['标题']
            summary = row['摘要']
            publish_time = row['发布时间']
            link = row['链接']

            try:
                # 解析发布时间，支持多种格式
                news_time = None
                publish_time_clean = str(publish_time).strip()

                # 尝试不同的时间格式
                time_formats = [
                    "%Y-%m-%d %H:%M:%S",  # YYYY-MM-DD HH:MM:SS
                    "%Y-%m-%d %H:%M",     # YYYY-MM-DD HH:MM
                    "%Y-%m-%d",           # YYYY-MM-DD
                ]

                for fmt in time_formats:
                    try:
                        news_time = datetime.strptime(publish_time_clean, fmt)
                        break
                    except ValueError:
                        continue

                # 检查是否为当天新闻
                if news_time and news_time.date() == curr_date_obj.date():
                    filtered_news.append((title, summary, publish_time, link))
            except:
                # 如果时间解析失败，跳过该条新闻
                continue

        if not filtered_news:
            return f"暂无 {curr_date} 当天的东方财富全球财经快讯数据"

        # 格式化输出
        news_str = f"## 东方财富-全球财经快讯 ({curr_date})\n\n"

        for title, summary, publish_time, link in filtered_news:
            news_str += f"### {title}\n"
            news_str += f"**发布时间**: {publish_time}\n\n"
            news_str += f"{summary}\n\n"
            news_str += f"[查看详情]({link})\n\n"
            news_str += "---\n\n"

        return news_str

    except Exception as e:
        return f"获取东方财富全球财经快讯时发生错误: {str(e)}"


def get_futu_financial_news(curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"]) -> str:
    """
    获取富途牛牛快讯

    Args:
        curr_date (str): current date you are trading at, yyyy-mm-dd

    Returns:
        str: 格式化的富途牛牛快讯字符串，包含标题、内容、发布时间和链接
    """
    try:
        # 调用akshare接口获取富途牛牛快讯
        news_df = ak.stock_info_global_futu()

        if news_df.empty:
            return "暂无富途牛牛快讯数据"

        # 过滤当天的新闻
        curr_date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
        filtered_news = []

        for index, row in news_df.iterrows():
            title = row['标题']
            content = row['内容']
            publish_time = row['发布时间']
            link = row['链接']

            try:
                # 解析发布时间，支持多种格式
                news_time = None
                publish_time_clean = str(publish_time).strip()

                # 尝试不同的时间格式
                time_formats = [
                    "%Y-%m-%d %H:%M:%S",  # YYYY-MM-DD HH:MM:SS
                    "%Y-%m-%d %H:%M",     # YYYY-MM-DD HH:MM
                    "%Y-%m-%d",           # YYYY-MM-DD
                ]

                for fmt in time_formats:
                    try:
                        news_time = datetime.strptime(publish_time_clean, fmt)
                        break
                    except ValueError:
                        continue

                # 检查是否为当天新闻
                if news_time and news_time.date() == curr_date_obj.date():
                    filtered_news.append((title, content, publish_time, link))
            except:
                # 如果时间解析失败，跳过该条新闻
                continue

        if not filtered_news:
            return f"暂无 {curr_date} 当天的富途牛牛快讯数据"

        # 格式化输出
        news_str = f"## 富途牛牛-快讯 ({curr_date})\n\n"

        for title, content, publish_time, link in filtered_news:
            news_str += f"### {title}\n"
            news_str += f"**发布时间**: {publish_time}\n\n"
            news_str += f"{content}\n\n"
            news_str += f"[查看详情]({link})\n\n"
            news_str += "---\n\n"

        return news_str

    except Exception as e:
        return f"获取富途牛牛快讯时发生错误: {str(e)}"


def get_tonghuashun_global_financial_live(curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"]) -> str:
    """
    获取同花顺全球财经直播

    Args:
        curr_date (str): current date you are trading at, yyyy-mm-dd

    Returns:
        str: 格式化的同花顺全球财经直播字符串，包含标题、内容、发布时间和链接
    """
    try:
        # 调用akshare接口获取同花顺全球财经直播
        news_df = ak.stock_info_global_ths()

        if news_df.empty:
            return "暂无同花顺全球财经直播数据"

        # 过滤当天的新闻
        curr_date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
        filtered_news = []

        for index, row in news_df.iterrows():
            title = row['标题']
            content = row['内容']
            publish_time = row['发布时间']
            link = row['链接']

            try:
                # 解析发布时间，支持多种格式
                news_time = None
                publish_time_clean = str(publish_time).strip()

                # 尝试不同的时间格式
                time_formats = [
                    "%Y-%m-%d %H:%M:%S",  # YYYY-MM-DD HH:MM:SS
                    "%Y-%m-%d %H:%M",     # YYYY-MM-DD HH:MM
                    "%Y-%m-%d",           # YYYY-MM-DD
                ]

                for fmt in time_formats:
                    try:
                        news_time = datetime.strptime(publish_time_clean, fmt)
                        break
                    except ValueError:
                        continue

                # 检查是否为当天新闻
                if news_time and news_time.date() == curr_date_obj.date():
                    filtered_news.append((title, content, publish_time, link))
            except:
                # 如果时间解析失败，跳过该条新闻
                continue

        if not filtered_news:
            return f"暂无 {curr_date} 当天的同花顺全球财经直播数据"

        # 格式化输出
        news_str = f"## 同花顺财经-全球财经直播 ({curr_date})\n\n"

        for title, content, publish_time, link in filtered_news:
            news_str += f"### {title}\n"
            news_str += f"**发布时间**: {publish_time}\n\n"
            news_str += f"{content}\n\n"
            news_str += f"[查看详情]({link})\n\n"
            news_str += "---\n\n"

        return news_str

    except Exception as e:
        return f"获取同花顺全球财经直播时发生错误: {str(e)}"


def get_cailianshe_telegraph(curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"]) -> str:
    """
    获取财联社电报

    Args:
        curr_date (str): current date you are trading at, yyyy-mm-dd

    Returns:
        str: 格式化的财联社电报字符串，包含标题、内容、发布日期和发布时间
    """
    try:
        # 调用akshare接口获取财联社电报（获取全部）
        news_df = ak.stock_info_global_cls(symbol='全部')

        if news_df.empty:
            return "暂无财联社电报数据"

        # 过滤当天的新闻
        curr_date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
        filtered_news = []

        for index, row in news_df.iterrows():
            title = row['标题']
            content = row['内容']
            publish_date = row['发布日期']
            publish_time = row['发布时间']

            try:
                # 解析发布日期，通常格式为 "YYYY-MM-DD"
                news_date = datetime.strptime(str(publish_date), "%Y-%m-%d")

                # 检查是否为当天新闻
                if news_date.date() == curr_date_obj.date():
                    filtered_news.append((title, content, publish_date, publish_time))
            except:
                # 如果日期解析失败，跳过该条新闻
                continue

        if not filtered_news:
            return f"暂无 {curr_date} 当天的财联社电报数据"

        # 格式化输出
        news_str = f"## 财联社-电报 ({curr_date})\n\n"

        for title, content, publish_date, publish_time in filtered_news:
            news_str += f"### {title}\n"
            news_str += f"**发布时间**: {publish_date} {publish_time}\n\n"
            news_str += f"{content}\n\n"
            news_str += "---\n\n"

        return news_str

    except Exception as e:
        return f"获取财联社电报时发生错误: {str(e)}"


def get_sina_securities_original(curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"]) -> str:
    """
    获取新浪财经证券原创

    Args:
        curr_date (str): current date you are trading at, yyyy-mm-dd

    Returns:
        str: 格式化的新浪财经证券原创字符串，包含时间、内容和链接
    """
    try:
        # 调用akshare接口获取新浪财经证券原创（第1页）
        news_df = ak.stock_info_broker_sina(page='1')

        if news_df.empty:
            return "暂无新浪财经证券原创数据"

        # 过滤当天的新闻
        curr_date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
        filtered_news = []

        for index, row in news_df.iterrows():
            time = row['时间']
            content = row['内容']
            link = row['链接']

            try:
                # 解析时间字符串，支持多种格式
                news_time = None
                time_clean = str(time).strip()

                # 尝试不同的时间格式
                time_formats = [
                    "%Y-%m-%d %H:%M:%S",  # YYYY-MM-DD HH:MM:SS
                    "%Y-%m-%d %H:%M",     # YYYY-MM-DD HH:MM
                    "%Y-%m-%d",           # YYYY-MM-DD
                    "%m-%d %H:%M",        # MM-DD HH:MM
                ]

                for fmt in time_formats:
                    try:
                        if fmt == "%m-%d %H:%M":
                            # 对于 MM-DD HH:MM 格式，添加当前年份
                            time_with_year = f"{curr_date_obj.year}-{time_clean}"
                            news_time = datetime.strptime(time_with_year, "%Y-%m-%d %H:%M")
                        else:
                            news_time = datetime.strptime(time_clean, fmt)
                        break
                    except ValueError:
                        continue

                # 检查是否为当天新闻
                if news_time and news_time.date() == curr_date_obj.date():
                    filtered_news.append((time, content, link))
            except:
                # 如果时间解析失败，跳过该条新闻
                continue

        if not filtered_news:
            return f"暂无 {curr_date} 当天的新浪财经证券原创数据"

        # 格式化输出
        news_str = f"## 新浪财经-证券原创 ({curr_date})\n\n"

        for time, content, link in filtered_news:
            news_str += f"### {content}\n"
            news_str += f"**发布时间**: {time}\n\n"
            news_str += f"[查看详情]({link})\n\n"
            news_str += "---\n\n"

        return news_str

    except Exception as e:
        return f"获取新浪财经证券原创时发生错误: {str(e)}"
