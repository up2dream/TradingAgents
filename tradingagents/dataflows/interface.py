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


def get_finnhub_company_insider_sentiment(
    ticker: Annotated[str, "ticker symbol for the company"],
    curr_date: Annotated[
        str,
        "current date of you are trading at, yyyy-mm-dd",
    ],
    look_back_days: Annotated[int, "number of days to look back"],
):
    """
    Retrieve insider sentiment about a company (retrieved from public SEC information) for the past 15 days
    Args:
        ticker (str): ticker symbol of the company
        curr_date (str): current date you are trading on, yyyy-mm-dd
    Returns:
        str: a report of the sentiment in the past 15 days starting at curr_date
    """

    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    data = get_data_in_range(ticker, before, curr_date, "insider_senti", DATA_DIR)

    if len(data) == 0:
        return ""

    result_str = ""
    seen_dicts = []
    for date, senti_list in data.items():
        for entry in senti_list:
            if entry not in seen_dicts:
                result_str += f"### {entry['year']}-{entry['month']}:\nChange: {entry['change']}\nMonthly Share Purchase Ratio: {entry['mspr']}\n\n"
                seen_dicts.append(entry)

    return (
        f"## {ticker} Insider Sentiment Data for {before} to {curr_date}:\n"
        + result_str
        + "The change field refers to the net buying/selling from all insiders' transactions. The mspr field refers to monthly share purchase ratio."
    )


def get_finnhub_company_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[
        str,
        "current date you are trading at, yyyy-mm-dd",
    ],
    look_back_days: Annotated[int, "how many days to look back"],
):
    """
    Retrieve insider transcaction information about a company (retrieved from public SEC information) for the past 15 days
    Args:
        ticker (str): ticker symbol of the company
        curr_date (str): current date you are trading at, yyyy-mm-dd
    Returns:
        str: a report of the company's insider transaction/trading informtaion in the past 15 days
    """

    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    data = get_data_in_range(ticker, before, curr_date, "insider_trans", DATA_DIR)

    if len(data) == 0:
        return ""

    result_str = ""

    seen_dicts = []
    for date, senti_list in data.items():
        for entry in senti_list:
            if entry not in seen_dicts:
                result_str += f"### Filing Date: {entry['filingDate']}, {entry['name']}:\nChange:{entry['change']}\nShares: {entry['share']}\nTransaction Price: {entry['transactionPrice']}\nTransaction Code: {entry['transactionCode']}\n\n"
                seen_dicts.append(entry)

    return (
        f"## {ticker} insider transactions from {before} to {curr_date}:\n"
        + result_str
        + "The change field reflects the variation in share count—here a negative number indicates a reduction in holdings—while share specifies the total number of shares involved. The transactionPrice denotes the per-share price at which the trade was executed, and transactionDate marks when the transaction occurred. The name field identifies the insider making the trade, and transactionCode (e.g., S for sale) clarifies the nature of the transaction. FilingDate records when the transaction was officially reported, and the unique id links to the specific SEC filing, as indicated by the source. Additionally, the symbol ties the transaction to a particular company, isDerivative flags whether the trade involves derivative securities, and currency notes the currency context of the transaction."
    )


def get_simfin_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
        str,
        "reporting frequency of the company's financial history: annual / quarterly",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
):
    data_path = os.path.join(
        DATA_DIR,
        "fundamental_data",
        "simfin_data_all",
        "balance_sheet",
        "companies",
        "us",
        f"us-balance-{freq}.csv",
    )
    df = pd.read_csv(data_path, sep=";")

    # Convert date strings to datetime objects and remove any time components
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()

    # Convert the current date to datetime and normalize
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()

    # Filter the DataFrame for the given ticker and for reports that were published on or before the current date
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    # Check if there are any available reports; if not, return a notification
    if filtered_df.empty:
        print("No balance sheet available before the given current date.")
        return ""

    # Get the most recent balance sheet by selecting the row with the latest Publish Date
    latest_balance_sheet = filtered_df.loc[filtered_df["Publish Date"].idxmax()]

    # drop the SimFinID column
    latest_balance_sheet = latest_balance_sheet.drop("SimFinId")

    return (
        f"## {freq} balance sheet for {ticker} released on {str(latest_balance_sheet['Publish Date'])[0:10]}: \n"
        + str(latest_balance_sheet)
        + "\n\nThis includes metadata like reporting dates and currency, share details, and a breakdown of assets, liabilities, and equity. Assets are grouped as current (liquid items like cash and receivables) and noncurrent (long-term investments and property). Liabilities are split between short-term obligations and long-term debts, while equity reflects shareholder funds such as paid-in capital and retained earnings. Together, these components ensure that total assets equal the sum of liabilities and equity."
    )


def get_simfin_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
        str,
        "reporting frequency of the company's financial history: annual / quarterly",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
):
    data_path = os.path.join(
        DATA_DIR,
        "fundamental_data",
        "simfin_data_all",
        "cash_flow",
        "companies",
        "us",
        f"us-cashflow-{freq}.csv",
    )
    df = pd.read_csv(data_path, sep=";")

    # Convert date strings to datetime objects and remove any time components
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()

    # Convert the current date to datetime and normalize
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()

    # Filter the DataFrame for the given ticker and for reports that were published on or before the current date
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    # Check if there are any available reports; if not, return a notification
    if filtered_df.empty:
        print("No cash flow statement available before the given current date.")
        return ""

    # Get the most recent cash flow statement by selecting the row with the latest Publish Date
    latest_cash_flow = filtered_df.loc[filtered_df["Publish Date"].idxmax()]

    # drop the SimFinID column
    latest_cash_flow = latest_cash_flow.drop("SimFinId")

    return (
        f"## {freq} cash flow statement for {ticker} released on {str(latest_cash_flow['Publish Date'])[0:10]}: \n"
        + str(latest_cash_flow)
        + "\n\nThis includes metadata like reporting dates and currency, share details, and a breakdown of cash movements. Operating activities show cash generated from core business operations, including net income adjustments for non-cash items and working capital changes. Investing activities cover asset acquisitions/disposals and investments. Financing activities include debt transactions, equity issuances/repurchases, and dividend payments. The net change in cash represents the overall increase or decrease in the company's cash position during the reporting period."
    )


def get_simfin_income_statements(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
        str,
        "reporting frequency of the company's financial history: annual / quarterly",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
):
    data_path = os.path.join(
        DATA_DIR,
        "fundamental_data",
        "simfin_data_all",
        "income_statements",
        "companies",
        "us",
        f"us-income-{freq}.csv",
    )
    df = pd.read_csv(data_path, sep=";")

    # Convert date strings to datetime objects and remove any time components
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()

    # Convert the current date to datetime and normalize
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()

    # Filter the DataFrame for the given ticker and for reports that were published on or before the current date
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    # Check if there are any available reports; if not, return a notification
    if filtered_df.empty:
        print("No income statement available before the given current date.")
        return ""

    # Get the most recent income statement by selecting the row with the latest Publish Date
    latest_income = filtered_df.loc[filtered_df["Publish Date"].idxmax()]

    # drop the SimFinID column
    latest_income = latest_income.drop("SimFinId")

    return (
        f"## {freq} income statement for {ticker} released on {str(latest_income['Publish Date'])[0:10]}: \n"
        + str(latest_income)
        + "\n\nThis includes metadata like reporting dates and currency, share details, and a comprehensive breakdown of the company's financial performance. Starting with Revenue, it shows Cost of Revenue and resulting Gross Profit. Operating Expenses are detailed, including SG&A, R&D, and Depreciation. The statement then shows Operating Income, followed by non-operating items and Interest Expense, leading to Pretax Income. After accounting for Income Tax and any Extraordinary items, it concludes with Net Income, representing the company's bottom-line profit or loss for the period."
    )


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
