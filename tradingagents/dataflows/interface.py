from typing import Annotated, Dict
from .yfin_utils import *
from .stockstats_utils import *
from .googlenews_utils import *
from .finnhub_utils import get_data_in_range
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
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


def get_stock_news_openai(ticker, curr_date):
    config = get_config()

    # Check if this is a Chinese stock
    from .tushare_utils import get_tushare_utils
    try:
        tushare_utils = get_tushare_utils()
        is_chinese_stock = tushare_utils.is_chinese_stock(ticker)
    except:
        is_chinese_stock = False

    # Check if using Google provider
    if config.get("llm_provider", "").lower() == "google":
        try:
            if GOOGLE_GENAI_AVAILABLE:
                # Use new google.genai API
                client = genai.Client()
                grounding_tool = types.Tool(google_search=types.GoogleSearch())
                generation_config = types.GenerateContentConfig(
                    tools=[grounding_tool],
                    temperature=1,
                    max_output_tokens=4096,
                    top_p=1,
                )
            else:
                # Use fallback google.generativeai API
                import google.generativeai as genai_fallback
                genai_fallback.configure(api_key=os.getenv('GOOGLE_API_KEY'))
                model = genai_fallback.GenerativeModel('gemini-1.5-flash')

            # Create enhanced search query for Chinese stocks
            if is_chinese_stock:
                # Get company name if possible
                try:
                    company_info = tushare_utils.get_stock_basic_info(ticker)
                    company_name = company_info.get('name', ticker)
                except:
                    company_name = ticker

                search_query = f"""请搜索关于股票代码 {ticker} ({company_name}) 在 {curr_date} 前7天到 {curr_date} 期间的中国社交媒体讨论。

请重点搜索以下平台的相关内容：
1. 微博 (weibo.com) - 搜索 "{ticker}" "{company_name}" 相关讨论
2. 雪球 (xueqiu.com) - 搜索股票讨论和分析
3. 知乎 (zhihu.com) - 搜索相关问答和分析
4. 东方财富股吧 (guba.eastmoney.com) - 搜索股票讨论
5. 同花顺 (10jqka.com.cn) - 搜索股票资讯和讨论

请确保只获取指定时间段内发布的内容，并总结投资者情绪和主要观点。"""
            else:
                search_query = f"""Search for social media discussions about stock ticker {ticker} from 7 days before {curr_date} to {curr_date}.

Focus on these platforms:
1. Twitter/X - search for "${ticker}" discussions
2. Reddit - search r/investing, r/stocks, r/SecurityAnalysis for {ticker}
3. StockTwits - search for {ticker} sentiment
4. Yahoo Finance comments
5. Seeking Alpha comments

Make sure you only get data posted during the specified time period and summarize investor sentiment and key points."""

            # Make the request
            if GOOGLE_GENAI_AVAILABLE:
                response = client.models.generate_content(
                    model=config["quick_think_llm"],
                    contents=search_query,
                    config=generation_config,
                )
                return response.text
            else:
                # Use fallback API without grounding tools
                response = model.generate_content(search_query)
                return response.text

        except Exception as e:
            print(f"Google API failed: {e}, falling back to OpenAI")
            # Fall through to OpenAI implementation

    # OpenAI implementation (either as fallback or primary)
    if config.get("llm_provider", "").lower() != "google" or True:  # Always execute as fallback
        # Original OpenAI implementation with enhanced query
        client = OpenAI(base_url=config["backend_url"])

        if is_chinese_stock:
            # Get company name if possible
            try:
                company_info = tushare_utils.get_stock_basic_info(ticker)
                company_name = company_info.get('name', ticker)
            except:
                company_name = ticker

            search_text = f"""请搜索关于股票代码 {ticker} ({company_name}) 在 {curr_date} 前7天到 {curr_date} 期间的中国社交媒体讨论。

请重点搜索以下平台：微博、雪球、知乎、东方财富股吧、同花顺等平台的相关讨论。
请确保只获取指定时间段内发布的内容，并总结投资者情绪和主要观点。"""
        else:
            search_text = f"Search social media (Twitter, Reddit, StockTwits) for {ticker} from 7 days before {curr_date} to {curr_date}. Focus on investor sentiment and key discussions."

        response = client.responses.create(
            model=config["quick_think_llm"],
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": search_text,
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
                    "search_context_size": "medium",  # Increased for better coverage
                }
            ],
            temperature=1,
            max_output_tokens=4096,
            top_p=1,
            store=True,
        )

        return response.output[1].content[0].text


def get_china_social_media_openai(ticker, curr_date):
    """
    专门搜索中国社交媒体平台上关于特定股票的讨论
    Specifically search Chinese social media platforms for stock discussions
    """
    config = get_config()

    # Get company information for better search
    from .tushare_utils import get_tushare_utils
    try:
        tushare_utils = get_tushare_utils()
        company_info = tushare_utils.get_stock_basic_info(ticker)
        company_name = company_info.get('name', ticker)
        industry = company_info.get('industry', '')
    except:
        company_name = ticker
        industry = ''

    # Check if using Google provider
    if config.get("llm_provider", "").lower() == "google":
        try:
            if GOOGLE_GENAI_AVAILABLE:
                # Use new google.genai API
                client = genai.Client()
                grounding_tool = types.Tool(google_search=types.GoogleSearch())
                generation_config = types.GenerateContentConfig(
                    tools=[grounding_tool],
                    temperature=1,
                    max_output_tokens=4096,
                    top_p=1,
                )
            else:
                # Use fallback google.generativeai API
                import google.generativeai as genai_fallback
                genai_fallback.configure(api_key=os.getenv('GOOGLE_API_KEY'))
                model = genai_fallback.GenerativeModel('gemini-1.5-flash')

            # Create comprehensive Chinese social media search query
            search_query = f"""请深度搜索中国社交媒体平台关于股票 {ticker} ({company_name}) 在 {curr_date} 前7天到 {curr_date} 期间的所有相关讨论和分析。

具体搜索策略：

1. 微博 (weibo.com)：
   - 搜索关键词："{ticker}" "{company_name}" "股票" "投资"
   - 查找财经博主、分析师、投资者的相关微博
   - 关注转发量和评论量高的内容

2. 雪球 (xueqiu.com)：
   - 搜索股票代码 {ticker} 的专门讨论页面
   - 查找用户发布的分析报告和观点
   - 关注热门讨论和专业投资者观点

3. 知乎 (zhihu.com)：
   - 搜索关于 {company_name} 或 {ticker} 的问答
   - 查找行业分析和投资建议
   - 关注专业人士的回答

4. 东方财富股吧 (guba.eastmoney.com)：
   - 搜索 {ticker} 专门的股吧讨论
   - 查找散户投资者的情绪和观点
   - 关注热门帖子和讨论

5. 同花顺 (10jqka.com.cn)：
   - 搜索股票资讯和用户讨论
   - 查找技术分析和基本面分析

6. 其他平台：
   - 财联社、证券时报等财经媒体的评论区
   - 各大财经APP的用户讨论

请分析并总结：
- 整体投资者情绪（看多/看空/中性）
- 主要讨论话题和关注点
- 重要的利好或利空消息
- 技术分析观点
- 基本面分析观点
- 风险提示和担忧
- 目标价位和投资建议

请确保只获取指定时间段 ({curr_date} 前7天到 {curr_date}) 内发布的内容。"""

            # Make the request
            if GOOGLE_GENAI_AVAILABLE:
                response = client.models.generate_content(
                    model=config["quick_think_llm"],
                    contents=search_query,
                    config=generation_config,
                )
                return response.text
            else:
                # Use fallback API without grounding tools
                response = model.generate_content(search_query)
                return response.text

        except Exception as e:
            print(f"Google API failed: {e}, falling back to OpenAI")

    # OpenAI implementation (either as fallback or primary)
    if config.get("llm_provider", "").lower() != "google" or True:  # Always execute as fallback
        # OpenAI implementation
        client = OpenAI(base_url=config["backend_url"])

        search_text = f"""请深度搜索中国社交媒体平台关于股票 {ticker} ({company_name}) 在 {curr_date} 前7天到 {curr_date} 期间的讨论。

重点搜索：微博、雪球、知乎、东方财富股吧、同花顺等平台。
分析投资者情绪、主要观点、利好利空消息、技术和基本面分析。
请确保只获取指定时间段内的内容。"""

        response = client.responses.create(
            model=config["quick_think_llm"],
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": search_text,
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
                    "search_context_size": "high",  # High context for comprehensive search
                }
            ],
            temperature=1,
            max_output_tokens=4096,
            top_p=1,
            store=True,
        )

        return response.output[1].content[0].text


def get_china_social_media_real_data(ticker, curr_date):
    """
    获取真实的中国社交媒体数据 - 使用 AKShare 等本地库
    Get real Chinese social media data using AKShare and other local libraries
    """
    import pandas as pd
    from datetime import datetime, timedelta

    results = {}

    try:
        # 1. 东方财富热度数据 - Eastmoney Hot Data (这个最可靠)
        try:
            if AKSHARE_AVAILABLE:
                # 获取东方财富人气排行榜
                hot_rank_df = ak.stock_hot_rank_em()

                # 查找目标股票
                target_hot = hot_rank_df[hot_rank_df['代码'].str.contains(ticker, na=False)] if not hot_rank_df.empty else pd.DataFrame()

                # 获取东方财富飙升榜
                try:
                    hot_up_df = ak.stock_hot_up_em()
                    target_up = hot_up_df[hot_up_df['代码'].str.contains(ticker, na=False)] if not hot_up_df.empty else pd.DataFrame()
                except:
                    target_up = pd.DataFrame()

                results['东方财富数据'] = {
                    '人气排行': target_hot.to_dict('records') if not target_hot.empty else [],
                    '飙升排行': target_up.to_dict('records') if not target_up.empty else []
                }
            else:
                results['东方财富数据'] = "AKShare 未安装，无法获取东方财富数据"

        except Exception as e:
            results['东方财富数据'] = f"获取失败: {e}"

        # 2. 个股详细热度数据 - Individual Stock Heat Data
        try:
            if AKSHARE_AVAILABLE:
                # 获取整体热度排行榜，然后查找目标股票
                hot_rank_df = ak.stock_hot_rank_em()
                target_hot_rank = hot_rank_df[hot_rank_df['代码'].str.contains(ticker, na=False)] if not hot_rank_df.empty else pd.DataFrame()

                # 尝试获取实时热度数据
                try:
                    hot_realtime_df = ak.stock_hot_rank_detail_realtime_em()
                    target_realtime = hot_realtime_df[hot_realtime_df['代码'].str.contains(ticker, na=False)] if not hot_realtime_df.empty else pd.DataFrame()
                except:
                    target_realtime = pd.DataFrame()

                results['个股热度详情'] = {
                    '热度排行': target_hot_rank.to_dict('records') if not target_hot_rank.empty else [],
                    '实时热度': target_realtime.to_dict('records') if not target_realtime.empty else []
                }
            else:
                results['个股热度详情'] = "AKShare 未安装，无法获取个股热度数据"

        except Exception as e:
            results['个股热度详情'] = f"获取失败: {e}"

        # 3. 互动平台数据 - Interactive Platform Data
        try:
            if AKSHARE_AVAILABLE:
                # 尝试获取互动易数据 - 某些股票可能不支持
                interact_df = pd.DataFrame()

                # 尝试多种股票代码格式
                symbols_to_try = [ticker]

                # 如果是6位数字，尝试添加交易所后缀
                if ticker.isdigit() and len(ticker) == 6:
                    if ticker.startswith(('000', '002', '300')):
                        symbols_to_try.append(f"{ticker}.SZ")
                    elif ticker.startswith(('600', '601', '603', '688')):
                        symbols_to_try.append(f"{ticker}.SH")

                # 尝试不同的股票代码格式
                for symbol_format in symbols_to_try:
                    try:
                        interact_df = ak.stock_irm_cninfo(symbol=symbol_format)
                        if not interact_df.empty:
                            break  # 成功获取数据，跳出循环
                    except Exception as format_error:
                        print(f"尝试格式 {symbol_format} 失败: {format_error}")
                        continue

                # 处理获取到的数据
                if not interact_df.empty:
                    # 检查是否有时间列用于过滤
                    time_columns = ['问题时间', '提问时间', '更新时间']
                    time_col = None
                    for col in time_columns:
                        if col in interact_df.columns:
                            time_col = col
                            break

                    if time_col:
                        try:
                            # 转换日期格式并过滤最近7天的数据
                            interact_df[time_col] = pd.to_datetime(interact_df[time_col])
                            end_date = datetime.strptime(curr_date, "%Y-%m-%d")
                            start_date = end_date - timedelta(days=7)
                            recent_data = interact_df[interact_df[time_col] >= start_date]
                            results['互动平台数据'] = recent_data.head(10).to_dict('records') if not recent_data.empty else interact_df.head(5).to_dict('records')
                        except:
                            # 如果日期过滤失败，返回最新的几条数据
                            results['互动平台数据'] = interact_df.head(5).to_dict('records')
                    else:
                        # 没有时间列，返回最新的几条数据
                        results['互动平台数据'] = interact_df.head(5).to_dict('records')
                else:
                    # 如果互动易数据获取失败，尝试获取其他相关数据作为替代
                    try:
                        # 尝试获取公司公告数据作为替代
                        from .tushare_utils import get_tushare_utils
                        tushare_utils = get_tushare_utils()

                        if tushare_utils.is_chinese_stock(ticker):
                            ts_symbol = tushare_utils.convert_symbol_to_tushare(ticker)
                            # 获取最近的公告数据
                            end_date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
                            start_date_obj = end_date_obj - timedelta(days=7)

                            announcements = tushare_utils.pro.anns(
                                ts_code=ts_symbol,
                                start_date=start_date_obj.strftime('%Y%m%d'),
                                end_date=end_date_obj.strftime('%Y%m%d')
                            )

                            if not announcements.empty:
                                results['互动平台数据'] = announcements.head(5).to_dict('records')
                            else:
                                results['互动平台数据'] = []
                        else:
                            results['互动平台数据'] = []
                    except:
                        results['互动平台数据'] = []
            else:
                results['互动平台数据'] = "AKShare 未安装，无法获取互动平台数据"

        except Exception as e:
            # 如果所有尝试都失败，提供更友好的错误信息
            results['互动平台数据'] = f"该股票暂无互动平台数据或数据源不支持 (错误: {str(e)[:50]}...)"

        # 4. 股票新闻数据 - Stock News Data
        try:
            if AKSHARE_AVAILABLE:
                # 获取个股新闻
                news_df = ak.stock_news_em(symbol=ticker)
                results['股票新闻'] = news_df.head(10).to_dict('records') if not news_df.empty else []
            else:
                results['股票新闻'] = "AKShare 未安装，无法获取股票新闻数据"

        except Exception as e:
            results['股票新闻'] = f"获取失败: {e}"

        # 5. 龙虎榜数据 - Dragon Tiger List Data (反映机构和游资关注度)
        try:
            if AKSHARE_AVAILABLE:
                # 获取龙虎榜数据 - 按日期获取所有股票的龙虎榜数据
                lhb_df = ak.stock_lhb_detail_em(start_date=curr_date.replace('-', ''), end_date=curr_date.replace('-', ''))
                # 查找目标股票
                if not lhb_df.empty and '代码' in lhb_df.columns:
                    target_lhb = lhb_df[lhb_df['代码'].str.contains(ticker, na=False)]
                    results['龙虎榜数据'] = target_lhb.to_dict('records') if not target_lhb.empty else []
                else:
                    results['龙虎榜数据'] = []
            else:
                results['龙虎榜数据'] = "AKShare 未安装，无法获取龙虎榜数据"

        except Exception as e:
            results['龙虎榜数据'] = f"获取失败: {e}"

        # 6. Tushare 社交媒体相关数据 - Tushare Social Media Related Data
        try:
            from .tushare_utils import get_tushare_utils
            tushare_utils = get_tushare_utils()

            if tushare_utils.is_chinese_stock(ticker):
                # 获取概念题材数据
                try:
                    ts_symbol = tushare_utils.convert_symbol_to_tushare(ticker)
                    concept_detail = tushare_utils.pro.concept_detail(ts_code=ts_symbol)
                    results['概念题材'] = concept_detail.to_dict('records') if not concept_detail.empty else []
                except Exception as e:
                    results['概念题材'] = f"获取失败: {e}"

                # 获取限售股解禁数据（反映市场关注度）
                try:
                    share_float = tushare_utils.pro.share_float(ts_code=ts_symbol, start_date=curr_date.replace('-', ''), end_date=curr_date.replace('-', ''))
                    results['限售解禁'] = share_float.to_dict('records') if not share_float.empty else []
                except Exception as e:
                    results['限售解禁'] = f"获取失败: {e}"

                # 获取股东人数变化（反映散户关注度）
                try:
                    # 获取最近的股东人数数据
                    end_date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
                    # 查找最近的季度末日期
                    year = end_date_obj.year
                    month = end_date_obj.month
                    if month >= 10:
                        quarter_end = f"{year}0930"
                    elif month >= 7:
                        quarter_end = f"{year}0630"
                    elif month >= 4:
                        quarter_end = f"{year}0331"
                    else:
                        quarter_end = f"{year-1}1231"

                    stk_holdernumber = tushare_utils.pro.stk_holdernumber(ts_code=ts_symbol, end_date=quarter_end)
                    results['股东人数'] = stk_holdernumber.to_dict('records') if not stk_holdernumber.empty else []
                except Exception as e:
                    results['股东人数'] = f"获取失败: {e}"
            else:
                results['概念题材'] = "非中国股票，无法获取 Tushare 数据"
                results['限售解禁'] = "非中国股票，无法获取 Tushare 数据"
                results['股东人数'] = "非中国股票，无法获取 Tushare 数据"

        except Exception as e:
            results['概念题材'] = f"Tushare 获取失败: {e}"
            results['限售解禁'] = f"Tushare 获取失败: {e}"
            results['股东人数'] = f"Tushare 获取失败: {e}"

        return format_social_media_report(ticker, curr_date, results)

    except Exception as e:
        return f"获取中国社交媒体数据时发生错误: {e}"


def get_china_comprehensive_social_media_data(ticker, curr_date):
    """
    获取综合的中国社交媒体数据 - 整合所有数据源
    Get comprehensive Chinese social media data - integrating all data sources
    """
    try:
        # 获取基础社交媒体数据
        basic_data = get_china_social_media_real_data(ticker, curr_date)

        # 获取主流论坛数据
        forum_data = get_china_forum_data(ticker, curr_date)

        # 整合报告
        comprehensive_report = f"""# {ticker} 综合中国社交媒体数据报告
## 数据日期: {curr_date}

# 第一部分：基础社交媒体数据
{basic_data}

---

# 第二部分：主流投资论坛数据
{forum_data}

---

## 综合分析总结
基于以上多个数据源的综合分析：

### 数据覆盖范围
- ✅ **AKShare数据**: 东方财富热度、股票新闻、龙虎榜
- ✅ **Tushare数据**: 概念题材、股东结构、解禁信息
- ✅ **论坛数据**: 东方财富股吧、雪球、同花顺
- ✅ **互动数据**: 投资者问答、公司公告

### 投资者关注度评估
通过多维度数据分析，该股票的社交媒体表现和投资者关注度可以从新闻热度、论坛讨论、概念题材、股东变化等多个角度进行评估。

---
*综合数据来源: AKShare + Tushare + 主流投资论坛*
"""

        return comprehensive_report

    except Exception as e:
        return f"获取综合中国社交媒体数据时发生错误: {e}"


def get_china_forum_data(ticker, curr_date):
    """
    获取中国主流投资论坛数据 - 增强版，包含更有价值的内容
    Get data from mainstream Chinese investment forums - Enhanced version with more valuable content
    """
    import pandas as pd
    from datetime import datetime, timedelta

    results = {}

    try:
        # 1. 东方财富股吧数据 - Eastmoney Forum Data (Enhanced)
        try:
            if AKSHARE_AVAILABLE:
                # 获取东方财富股票评论总览
                comment_overview = ak.stock_comment_em()

                # 查找目标股票
                if not comment_overview.empty and '代码' in comment_overview.columns:
                    target_comment = comment_overview[comment_overview['代码'].str.contains(ticker, na=False)]
                    results['东方财富股吧总览'] = target_comment.to_dict('records') if not target_comment.empty else []
                else:
                    results['东方财富股吧总览'] = []

                # 获取投资者情绪数据 - 每日愿望评论
                try:
                    desire_daily = ak.stock_comment_detail_scrd_desire_daily_em(symbol=ticker)
                    results['投资者情绪变化'] = desire_daily.to_dict('records') if not desire_daily.empty else []
                except:
                    results['投资者情绪变化'] = []

                # 获取综合评价历史评分
                try:
                    rating_history = ak.stock_comment_detail_zhpj_lspf_em(symbol=ticker)
                    results['综合评价评分'] = rating_history.head(10).to_dict('records') if not rating_history.empty else []
                except:
                    results['综合评价评分'] = []

                # 获取机构参与度数据
                try:
                    institution_data = ak.stock_comment_detail_zlkp_jgcyd_em(symbol=ticker)
                    results['机构参与度变化'] = institution_data.head(10).to_dict('records') if not institution_data.empty else []
                except:
                    results['机构参与度变化'] = []

            else:
                results['东方财富股吧总览'] = "AKShare 未安装，无法获取东方财富股吧数据"
                results['投资者情绪变化'] = "AKShare 未安装，无法获取投资者情绪数据"
                results['综合评价评分'] = "AKShare 未安装，无法获取综合评价数据"
                results['机构参与度变化'] = "AKShare 未安装，无法获取机构参与度数据"

        except Exception as e:
            results['东方财富股吧总览'] = f"获取失败: {e}"
            results['投资者情绪变化'] = f"获取失败: {e}"
            results['综合评价评分'] = f"获取失败: {e}"
            results['机构参与度变化'] = f"获取失败: {e}"

        # 2. 雪球论坛数据 - Xueqiu Forum Data
        try:
            if AKSHARE_AVAILABLE:
                # 尝试不同的雪球股票代码格式
                xq_formats = [ticker]
                if ticker.isdigit() and len(ticker) == 6:
                    if ticker.startswith(('000', '002', '300')):
                        xq_formats.extend([f"SZ{ticker}", f"{ticker}.SZ"])
                    elif ticker.startswith(('600', '601', '603', '688')):
                        xq_formats.extend([f"SH{ticker}", f"{ticker}.SH"])

                # 尝试获取雪球热门推文
                xq_tweets = pd.DataFrame()
                for xq_format in xq_formats:
                    try:
                        xq_tweets = ak.stock_hot_tweet_xq(symbol=xq_format)
                        if not xq_tweets.empty:
                            break
                    except:
                        continue

                results['雪球热门推文'] = xq_tweets.head(10).to_dict('records') if not xq_tweets.empty else []

                # 尝试获取雪球关注数据
                xq_follow = pd.DataFrame()
                for xq_format in xq_formats:
                    try:
                        xq_follow = ak.stock_hot_follow_xq(symbol=xq_format)
                        if not xq_follow.empty:
                            break
                    except:
                        continue

                results['雪球关注数据'] = xq_follow.head(10).to_dict('records') if not xq_follow.empty else []

            else:
                results['雪球热门推文'] = "AKShare 未安装，无法获取雪球数据"
                results['雪球关注数据'] = "AKShare 未安装，无法获取雪球数据"

        except Exception as e:
            results['雪球热门推文'] = f"获取失败: {e}"
            results['雪球关注数据'] = f"获取失败: {e}"

        # 3. 机构研报和分析师观点 - Institution Research and Analyst Views
        try:
            if AKSHARE_AVAILABLE:
                # 获取机构推荐详情
                try:
                    institute_recommend = ak.stock_institute_recommend_detail(symbol=ticker)
                    # 过滤最近的推荐
                    if not institute_recommend.empty:
                        # 按评级日期排序，获取最新的推荐
                        institute_recommend['评级日期'] = pd.to_datetime(institute_recommend['评级日期'])
                        recent_recommend = institute_recommend.sort_values('评级日期', ascending=False).head(10)
                        results['机构推荐观点'] = recent_recommend.to_dict('records')
                    else:
                        results['机构推荐观点'] = []
                except Exception as e:
                    results['机构推荐观点'] = f"获取失败: {e}"

                # 获取研究报告
                try:
                    research_reports = ak.stock_research_report_em(symbol=ticker)
                    if not research_reports.empty:
                        # 获取最新的研究报告
                        recent_reports = research_reports.head(5)
                        results['研究报告'] = recent_reports.to_dict('records')
                    else:
                        results['研究报告'] = []
                except Exception as e:
                    results['研究报告'] = f"获取失败: {e}"

            else:
                results['机构推荐观点'] = "AKShare 未安装，无法获取机构推荐数据"
                results['研究报告'] = "AKShare 未安装，无法获取研究报告数据"

        except Exception as e:
            results['机构推荐观点'] = f"获取失败: {e}"
            results['研究报告'] = f"获取失败: {e}"

        # 4. 同花顺论坛数据 - Tonghuashun Forum Data (简化版)
        try:
            if AKSHARE_AVAILABLE:
                # 获取同花顺概念板块信息 (简化，只作为补充)
                try:
                    ths_concept = ak.stock_board_concept_name_ths()
                    if not ths_concept.empty:
                        results['概念板块参考'] = ths_concept.head(5).to_dict('records')
                    else:
                        results['概念板块参考'] = []
                except:
                    results['概念板块参考'] = []

            else:
                results['概念板块参考'] = "AKShare 未安装，无法获取概念板块数据"

        except Exception as e:
            results['概念板块参考'] = f"获取失败: {e}"

        return format_forum_report(ticker, curr_date, results)

    except Exception as e:
        return f"获取中国主流论坛数据时发生错误: {e}"


def format_forum_report(ticker, curr_date, data):
    """
    格式化主流论坛报告
    Format mainstream forum report
    """

    def format_forum_section(title, data_dict):
        if isinstance(data_dict, str):
            return f"### {title}\n{data_dict}\n"

        result = f"### {title}\n"
        for category, data_list in data_dict.items():
            if data_list:
                result += f"**{category}**:\n"
                for item in data_list[:3]:  # 只显示前3条
                    result += f"- {item}\n"
            else:
                result += f"**{category}**: 暂无数据\n"
        return result + "\n"

    def analyze_forum_sentiment(data):
        """基于论坛数据分析投资者情绪 - 增强版"""
        sentiment_indicators = []
        detailed_analysis = []

        # 分析投资者情绪变化
        sentiment_data = data.get('投资者情绪变化', [])
        if sentiment_data and not isinstance(sentiment_data, str):
            try:
                latest_sentiment = sentiment_data[0] if sentiment_data else {}
                if '当日意愿上升' in latest_sentiment:
                    sentiment_value = latest_sentiment['当日意愿上升']
                    if sentiment_value > 10:
                        sentiment_indicators.append("投资者情绪较为乐观")
                        detailed_analysis.append(f"当日意愿上升 {sentiment_value}%，显示积极情绪")
                    elif sentiment_value < -10:
                        sentiment_indicators.append("投资者情绪较为悲观")
                        detailed_analysis.append(f"当日意愿下降 {abs(sentiment_value)}%，显示消极情绪")
                    else:
                        sentiment_indicators.append("投资者情绪相对平稳")
            except:
                pass

        # 分析综合评价评分
        rating_data = data.get('综合评价评分', [])
        if rating_data and not isinstance(rating_data, str):
            try:
                latest_rating = rating_data[0] if rating_data else {}
                if '评分' in latest_rating:
                    rating_value = latest_rating['评分']
                    if rating_value > 70:
                        sentiment_indicators.append("综合评价较高")
                        detailed_analysis.append(f"最新综合评分 {rating_value:.1f}，属于较高水平")
                    elif rating_value < 50:
                        sentiment_indicators.append("综合评价偏低")
                        detailed_analysis.append(f"最新综合评分 {rating_value:.1f}，需要关注")
            except:
                pass

        # 分析机构推荐观点
        recommend_data = data.get('机构推荐观点', [])
        if recommend_data and not isinstance(recommend_data, str):
            try:
                buy_count = sum(1 for item in recommend_data if '买入' in str(item.get('最新评级', '')))
                hold_count = sum(1 for item in recommend_data if '持有' in str(item.get('最新评级', '')))
                total_count = len(recommend_data)

                if buy_count > total_count * 0.6:
                    sentiment_indicators.append("机构普遍看好")
                    detailed_analysis.append(f"近期 {total_count} 家机构中 {buy_count} 家给出买入评级")
                elif buy_count > 0:
                    sentiment_indicators.append("机构观点偏积极")
                    detailed_analysis.append(f"近期 {total_count} 家机构中 {buy_count} 家给出买入评级")
            except:
                pass

        # 分析研究报告
        report_data = data.get('研究报告', [])
        if report_data and not isinstance(report_data, str):
            try:
                report_count = len(report_data)
                if report_count > 0:
                    sentiment_indicators.append("机构研究关注度较高")
                    detailed_analysis.append(f"近期有 {report_count} 份研究报告发布")
            except:
                pass

        # 分析机构参与度
        institution_data = data.get('机构参与度变化', [])
        if institution_data and not isinstance(institution_data, str):
            try:
                if len(institution_data) >= 2:
                    latest = institution_data[0]['机构参与度']
                    previous = institution_data[1]['机构参与度']
                    if latest > previous:
                        sentiment_indicators.append("机构参与度上升")
                        detailed_analysis.append(f"机构参与度从 {previous:.1f}% 上升至 {latest:.1f}%")
                    elif latest < previous:
                        sentiment_indicators.append("机构参与度下降")
                        detailed_analysis.append(f"机构参与度从 {previous:.1f}% 下降至 {latest:.1f}%")
            except:
                pass

        # 生成分析报告
        if sentiment_indicators:
            analysis = "基于主流论坛和机构数据分析，该股票具有以下特征：\n"
            analysis += "\n".join([f"- {indicator}" for indicator in sentiment_indicators])

            if detailed_analysis:
                analysis += "\n\n详细分析：\n"
                analysis += "\n".join([f"• {detail}" for detail in detailed_analysis])

            return analysis
        else:
            return "基于当前数据，该股票在主流论坛和机构关注度相对较低。"

    report = f"""# {ticker} 中国主流投资论坛数据报告 (增强版)
## 数据日期: {curr_date}

## 1. 东方财富投资者情绪数据
{format_forum_section("投资者情绪分析", {
    '股吧总览': data.get('东方财富股吧总览', []),
    '情绪变化': data.get('投资者情绪变化', []),
    '综合评分': data.get('综合评价评分', []),
    '机构参与度': data.get('机构参与度变化', [])
})}

## 2. 机构研报和分析师观点
{format_forum_section("专业机构观点", {
    '机构推荐': data.get('机构推荐观点', []),
    '研究报告': data.get('研究报告', [])
})}

## 3. 雪球论坛数据
{format_forum_section("雪球论坛", {
    '热门推文': data.get('雪球热门推文', []),
    '关注数据': data.get('雪球关注数据', [])
})}

## 4. 市场概念参考
{format_forum_section("概念板块", {
    '相关概念': data.get('概念板块参考', [])
})}

## 5. 综合情绪分析
{analyze_forum_sentiment(data)}

---
*数据来源: 东方财富(情绪+机构) + 雪球 + 研报分析 - 增强版投资者情绪分析*
"""
    return report


def format_social_media_report(ticker, curr_date, data):
    """
    格式化社交媒体报告
    Format social media report
    """

    def format_data_section(title, data_dict):
        if isinstance(data_dict, str):
            return f"### {title}\n{data_dict}\n"

        result = f"### {title}\n"
        for category, data_list in data_dict.items():
            if data_list:
                result += f"**{category}**:\n"
                for item in data_list[:3]:  # 只显示前3条
                    result += f"- {item}\n"
            else:
                result += f"**{category}**: 暂无数据\n"
        return result + "\n"

    def analyze_sentiment(data):
        """基于获取的数据分析投资者情绪"""
        sentiment_indicators = []

        # 分析东方财富数据
        em_data = data.get('东方财富数据', {})
        if isinstance(em_data, dict):
            if em_data.get('人气排行'):
                sentiment_indicators.append("东方财富平台人气较高")
            if em_data.get('飙升排行'):
                sentiment_indicators.append("东方财富平台热度飙升")

        # 分析个股热度数据
        heat_data = data.get('个股热度详情', {})
        if isinstance(heat_data, dict):
            if heat_data.get('热度排行') or heat_data.get('实时热度'):
                sentiment_indicators.append("个股热度数据显示关注度较高")
        elif heat_data and not isinstance(heat_data, str):
            sentiment_indicators.append("个股热度数据显示关注度较高")

        # 分析互动数据
        interact_data = data.get('互动平台数据', [])
        if interact_data and not isinstance(interact_data, str):
            sentiment_indicators.append("投资者互动较为活跃")

        # 分析新闻数据
        news_data = data.get('股票新闻', [])
        if news_data and not isinstance(news_data, str):
            sentiment_indicators.append("近期新闻关注度较高")

        # 分析龙虎榜数据
        lhb_data = data.get('龙虎榜数据', [])
        if lhb_data and not isinstance(lhb_data, str):
            sentiment_indicators.append("机构和游资关注度较高")

        # 分析概念题材数据
        concept_data = data.get('概念题材', [])
        if concept_data and not isinstance(concept_data, str):
            sentiment_indicators.append("属于热门概念题材")

        # 分析限售解禁数据
        float_data = data.get('限售解禁', [])
        if float_data and not isinstance(float_data, str):
            sentiment_indicators.append("近期有限售股解禁，需关注流通盘变化")

        # 分析股东人数数据
        holder_data = data.get('股东人数', [])
        if holder_data and not isinstance(holder_data, str):
            sentiment_indicators.append("股东人数数据可用，反映散户参与度")

        if sentiment_indicators:
            return "基于真实数据分析，该股票具有以下特征：\n" + "\n".join([f"- {indicator}" for indicator in sentiment_indicators])
        else:
            return "基于当前数据，该股票在金融平台的活跃度相对较低。"

    report = f"""# {ticker} 中国社交媒体数据报告 (真实数据)
## 数据日期: {curr_date}

## 1. 东方财富平台数据
{format_data_section("东方财富平台", data.get('东方财富数据', {}))}

## 2. 个股热度详情
{format_data_section("个股热度", data.get('个股热度详情', {}) if isinstance(data.get('个股热度详情', {}), dict) else {'热度数据': data.get('个股热度详情', [])})}

## 3. 互动平台数据
{format_data_section("互动平台", {'投资者互动': data.get('互动平台数据', [])})}

## 4. 股票新闻
{format_data_section("股票新闻", {'最新新闻': data.get('股票新闻', [])})}

## 5. 龙虎榜数据
{format_data_section("龙虎榜", {'机构游资': data.get('龙虎榜数据', [])})}

## 6. Tushare 社交媒体相关数据
{format_data_section("概念题材", {'概念分类': data.get('概念题材', [])})}
{format_data_section("限售解禁", {'解禁情况': data.get('限售解禁', [])})}
{format_data_section("股东人数", {'股东变化': data.get('股东人数', [])})}

## 7. 投资者情绪分析
{analyze_sentiment(data)}

---
*数据来源: AKShare + Tushare - 真实的中国金融平台数据*
"""
    return report


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


def get_sina_global_financial_news() -> str:
    """
    获取新浪财经全球财经快讯

    Returns:
        str: 格式化的全球财经快讯字符串，包含时间和内容
    """
    try:
        # 调用akshare接口获取新浪财经全球财经快讯
        news_df = ak.stock_info_global_sina()

        if news_df.empty:
            return "暂无全球财经快讯数据"

        # 格式化输出
        news_str = "## 新浪财经-全球财经快讯\n\n"

        for index, row in news_df.iterrows():
            time_str = row['时间']
            content = row['内容']
            news_str += f"**{time_str}**\n{content}\n\n"

        return news_str

    except Exception as e:
        return f"获取新浪财经全球财经快讯时发生错误: {str(e)}"


def get_eastmoney_financial_breakfast() -> str:
    """
    获取东方财富财经早餐

    Returns:
        str: 格式化的财经早餐字符串，包含标题、摘要、发布时间和链接
    """
    try:
        # 调用akshare接口获取东方财富财经早餐
        breakfast_df = ak.stock_info_cjzc_em()

        if breakfast_df.empty:
            return "暂无财经早餐数据"

        # 格式化输出，只显示最近10条
        breakfast_str = "## 东方财富-财经早餐\n\n"

        # 取最新的10条数据
        recent_data = breakfast_df.head(10)

        for index, row in recent_data.iterrows():
            title = row['标题']
            summary = row['摘要']
            publish_time = row['发布时间']
            link = row['链接']

            breakfast_str += f"### {title}\n"
            breakfast_str += f"**发布时间**: {publish_time}\n\n"
            breakfast_str += f"{summary}\n\n"
            breakfast_str += f"[查看详情]({link})\n\n"
            breakfast_str += "---\n\n"

        return breakfast_str

    except Exception as e:
        return f"获取东方财富财经早餐时发生错误: {str(e)}"


def get_eastmoney_global_financial_news() -> str:
    """
    获取东方财富全球财经快讯

    Returns:
        str: 格式化的全球财经快讯字符串，包含标题、摘要、发布时间和链接
    """
    try:
        # 调用akshare接口获取东方财富全球财经快讯
        news_df = ak.stock_info_global_em()

        if news_df.empty:
            return "暂无东方财富全球财经快讯数据"

        # 格式化输出，显示最新15条
        news_str = "## 东方财富-全球财经快讯\n\n"

        # 取最新的15条数据
        recent_data = news_df.head(15)

        for index, row in recent_data.iterrows():
            title = row['标题']
            summary = row['摘要']
            publish_time = row['发布时间']
            link = row['链接']

            news_str += f"### {title}\n"
            news_str += f"**发布时间**: {publish_time}\n\n"
            news_str += f"{summary}\n\n"
            news_str += f"[查看详情]({link})\n\n"
            news_str += "---\n\n"

        return news_str

    except Exception as e:
        return f"获取东方财富全球财经快讯时发生错误: {str(e)}"


def get_futu_financial_news() -> str:
    """
    获取富途牛牛快讯

    Returns:
        str: 格式化的富途牛牛快讯字符串，包含标题、内容、发布时间和链接
    """
    try:
        # 调用akshare接口获取富途牛牛快讯
        news_df = ak.stock_info_global_futu()

        if news_df.empty:
            return "暂无富途牛牛快讯数据"

        # 格式化输出，显示最新15条
        news_str = "## 富途牛牛-快讯\n\n"

        # 取最新的15条数据
        recent_data = news_df.head(15)

        for index, row in recent_data.iterrows():
            title = row['标题']
            content = row['内容']
            publish_time = row['发布时间']
            link = row['链接']

            news_str += f"### {title}\n"
            news_str += f"**发布时间**: {publish_time}\n\n"
            news_str += f"{content}\n\n"
            news_str += f"[查看详情]({link})\n\n"
            news_str += "---\n\n"

        return news_str

    except Exception as e:
        return f"获取富途牛牛快讯时发生错误: {str(e)}"


def get_tonghuashun_global_financial_live() -> str:
    """
    获取同花顺全球财经直播

    Returns:
        str: 格式化的同花顺全球财经直播字符串，包含标题、内容、发布时间和链接
    """
    try:
        # 调用akshare接口获取同花顺全球财经直播
        news_df = ak.stock_info_global_ths()

        if news_df.empty:
            return "暂无同花顺全球财经直播数据"

        # 格式化输出，显示所有数据（通常20条）
        news_str = "## 同花顺财经-全球财经直播\n\n"

        for index, row in news_df.iterrows():
            title = row['标题']
            content = row['内容']
            publish_time = row['发布时间']
            link = row['链接']

            news_str += f"### {title}\n"
            news_str += f"**发布时间**: {publish_time}\n\n"
            news_str += f"{content}\n\n"
            news_str += f"[查看详情]({link})\n\n"
            news_str += "---\n\n"

        return news_str

    except Exception as e:
        return f"获取同花顺全球财经直播时发生错误: {str(e)}"


def get_cailianshe_telegraph() -> str:
    """
    获取财联社电报

    Returns:
        str: 格式化的财联社电报字符串，包含标题、内容、发布日期和发布时间
    """
    try:
        # 调用akshare接口获取财联社电报（获取全部）
        news_df = ak.stock_info_global_cls(symbol='全部')

        if news_df.empty:
            return "暂无财联社电报数据"

        # 格式化输出，显示所有数据（通常20条）
        news_str = "## 财联社-电报\n\n"

        for index, row in news_df.iterrows():
            title = row['标题']
            content = row['内容']
            publish_date = row['发布日期']
            publish_time = row['发布时间']

            news_str += f"### {title}\n"
            news_str += f"**发布时间**: {publish_date} {publish_time}\n\n"
            news_str += f"{content}\n\n"
            news_str += "---\n\n"

        return news_str

    except Exception as e:
        return f"获取财联社电报时发生错误: {str(e)}"


def get_sina_securities_original() -> str:
    """
    获取新浪财经证券原创

    Returns:
        str: 格式化的新浪财经证券原创字符串，包含时间、内容和链接
    """
    try:
        # 调用akshare接口获取新浪财经证券原创（第1页）
        news_df = ak.stock_info_broker_sina(page='1')

        if news_df.empty:
            return "暂无新浪财经证券原创数据"

        # 格式化输出，显示所有数据
        news_str = "## 新浪财经-证券原创\n\n"

        for index, row in news_df.iterrows():
            time = row['时间']
            content = row['内容']
            link = row['链接']

            news_str += f"### {content}\n"
            news_str += f"**发布时间**: {time}\n\n"
            news_str += f"[查看详情]({link})\n\n"
            news_str += "---\n\n"

        return news_str

    except Exception as e:
        return f"获取新浪财经证券原创时发生错误: {str(e)}"
