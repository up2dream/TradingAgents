from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from typing import List
from typing import Annotated
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import RemoveMessage
from langchain_core.tools import tool
from datetime import date, timedelta, datetime
import functools
import pandas as pd
import os
from dateutil.relativedelta import relativedelta
from langchain_openai import ChatOpenAI
import tradingagents.dataflows.interface as interface
from tradingagents.default_config import DEFAULT_CONFIG
from langchain_core.messages import HumanMessage


def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]
        
        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]
        
        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")
        
        return {"messages": removal_operations + [placeholder]}
    
    return delete_messages


class Toolkit:
    _config = DEFAULT_CONFIG.copy()

    @classmethod
    def update_config(cls, config):
        """Update the class-level configuration."""
        cls._config.update(config)

    @property
    def config(self):
        """Access the configuration."""
        return self._config

    def __init__(self, config=None):
        if config:
            self.update_config(config)



    @staticmethod
    @tool
    def get_YFin_data(
        symbol: Annotated[str, "ticker symbol of the company"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    ) -> str:
        """
        Retrieve the stock price data for a given ticker symbol from Yahoo Finance.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
        Returns:
            str: A formatted dataframe containing the stock price data for the specified ticker symbol in the specified date range.
        """

        result_data = interface.get_YFin_data(symbol, start_date, end_date)

        return result_data

    @staticmethod
    @tool
    def get_YFin_data_online(
        symbol: Annotated[str, "ticker symbol of the company"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    ) -> str:
        """
        Retrieve the stock price data for a given ticker symbol from Yahoo Finance.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
        Returns:
            str: A formatted dataframe containing the stock price data for the specified ticker symbol in the specified date range.
        """

        result_data = interface.get_YFin_data_online(symbol, start_date, end_date)

        return result_data

    @staticmethod
    @tool
    def get_stockstats_indicators_report(
        symbol: Annotated[str, "ticker symbol of the company"],
        indicator: Annotated[
            str, "technical indicator to get the analysis and report of"
        ],
        curr_date: Annotated[
            str, "The current trading date you are trading on, YYYY-mm-dd"
        ],
        look_back_days: Annotated[int, "how many days to look back"] = 30,
    ) -> str:
        """
        Retrieve stock stats indicators for a given ticker symbol and indicator.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            indicator (str): Technical indicator to get the analysis and report of
            curr_date (str): The current trading date you are trading on, YYYY-mm-dd
            look_back_days (int): How many days to look back, default is 30
        Returns:
            str: A formatted dataframe containing the stock stats indicators for the specified ticker symbol and indicator.
        """

        result_stockstats = interface.get_stock_stats_indicators_window(
            symbol, indicator, curr_date, look_back_days, False
        )

        return result_stockstats

    @staticmethod
    @tool
    def get_stockstats_indicators_report_online(
        symbol: Annotated[str, "ticker symbol of the company"],
        indicator: Annotated[
            str, "technical indicator to get the analysis and report of"
        ],
        curr_date: Annotated[
            str, "The current trading date you are trading on, YYYY-mm-dd"
        ],
        look_back_days: Annotated[int, "how many days to look back"] = 30,
    ) -> str:
        """
        Retrieve stock stats indicators for a given ticker symbol and indicator.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            indicator (str): Technical indicator to get the analysis and report of
            curr_date (str): The current trading date you are trading on, YYYY-mm-dd
            look_back_days (int): How many days to look back, default is 30
        Returns:
            str: A formatted dataframe containing the stock stats indicators for the specified ticker symbol and indicator.
        """

        result_stockstats = interface.get_stock_stats_indicators_window(
            symbol, indicator, curr_date, look_back_days, True
        )

        return result_stockstats

    @staticmethod
    @tool
    def get_finnhub_company_insider_sentiment(
        ticker: Annotated[str, "ticker symbol for the company"],
        curr_date: Annotated[
            str,
            "current date of you are trading at, yyyy-mm-dd",
        ],
    ):
        """
        Retrieve insider sentiment information about a company (retrieved from public SEC information) for the past 30 days
        Args:
            ticker (str): ticker symbol of the company
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
            str: a report of the sentiment in the past 30 days starting at curr_date
        """

        data_sentiment = interface.get_finnhub_company_insider_sentiment(
            ticker, curr_date, 30
        )

        return data_sentiment

    @staticmethod
    @tool
    def get_finnhub_company_insider_transactions(
        ticker: Annotated[str, "ticker symbol"],
        curr_date: Annotated[
            str,
            "current date you are trading at, yyyy-mm-dd",
        ],
    ):
        """
        Retrieve insider transaction information about a company (retrieved from public SEC information) for the past 30 days
        Args:
            ticker (str): ticker symbol of the company
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
            str: a report of the company's insider transactions/trading information in the past 30 days
        """

        data_trans = interface.get_finnhub_company_insider_transactions(
            ticker, curr_date, 30
        )

        return data_trans

    @staticmethod
    @tool
    def get_simfin_balance_sheet(
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[
            str,
            "reporting frequency of the company's financial history: annual/quarterly",
        ],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    ):
        """
        Retrieve the most recent balance sheet of a company
        Args:
            ticker (str): ticker symbol of the company
            freq (str): reporting frequency of the company's financial history: annual / quarterly
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
            str: a report of the company's most recent balance sheet
        """

        data_balance_sheet = interface.get_simfin_balance_sheet(ticker, freq, curr_date)

        return data_balance_sheet

    @staticmethod
    @tool
    def get_simfin_cashflow(
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[
            str,
            "reporting frequency of the company's financial history: annual/quarterly",
        ],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    ):
        """
        Retrieve the most recent cash flow statement of a company
        Args:
            ticker (str): ticker symbol of the company
            freq (str): reporting frequency of the company's financial history: annual / quarterly
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
                str: a report of the company's most recent cash flow statement
        """

        data_cashflow = interface.get_simfin_cashflow(ticker, freq, curr_date)

        return data_cashflow

    @staticmethod
    @tool
    def get_simfin_income_stmt(
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[
            str,
            "reporting frequency of the company's financial history: annual/quarterly",
        ],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    ):
        """
        Retrieve the most recent income statement of a company
        Args:
            ticker (str): ticker symbol of the company
            freq (str): reporting frequency of the company's financial history: annual / quarterly
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
                str: a report of the company's most recent income statement
        """

        data_income_stmt = interface.get_simfin_income_statements(
            ticker, freq, curr_date
        )

        return data_income_stmt

    @staticmethod
    @tool
    def get_google_news(
        query: Annotated[str, "Query to search with"],
        curr_date: Annotated[str, "Curr date in yyyy-mm-dd format"],
    ):
        """
        Retrieve the latest news from Google News based on a query and date range.
        Args:
            query (str): Query to search with
            curr_date (str): Current date in yyyy-mm-dd format
            look_back_days (int): How many days to look back
        Returns:
            str: A formatted string containing the latest news from Google News based on the query and date range.
        """

        google_news_results = interface.get_google_news(query, curr_date, 7)

        return google_news_results

    @staticmethod
    @tool
    def get_stock_news_openai(
        ticker: Annotated[str, "the company's ticker"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        look_back_days: Annotated[int, "how many days to look back"] = 7,
        max_limit_per_day: Annotated[int, "Maximum number of news per day"] = 3,
    ):
        """
        Retrieve the latest news about a given stock using AKSHARE's stock_news_em interface.
        Args:
            ticker (str): Ticker of a company. e.g. AAPL, TSM, 000001
            start_date (str): Start date in yyyy-mm-dd format
            look_back_days (int): how many days to look back (default: 7)
            max_limit_per_day (int): Maximum number of news per day (default: 3)
        Returns:
            str: A formatted string containing the latest news about the company.
        """

        openai_news_results = interface.get_stock_news_openai(ticker, start_date, look_back_days, max_limit_per_day)

        return openai_news_results

    @staticmethod
    @tool
    def get_global_news_openai(
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    ):
        """
        Retrieve the latest macroeconomics news on a given date using OpenAI's macroeconomics news API.
        Args:
            curr_date (str): Current date in yyyy-mm-dd format
        Returns:
            str: A formatted string containing the latest macroeconomic news on the given date.
        """

        openai_news_results = interface.get_global_news_openai(curr_date)

        return openai_news_results

    @staticmethod
    @tool
    def get_china_focused_news_openai(
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    ):
        """
        Retrieve the latest China-focused economic and market news using OpenAI's news API.
        This function specifically focuses on Chinese economic indicators, market news, policy changes,
        and China-related international economic news that would be relevant for trading Chinese stocks.
        Args:
            curr_date (str): Current date in yyyy-mm-dd format
        Returns:
            str: A formatted string containing the latest China-focused economic and market news.
        """

        china_news_results = interface.get_china_focused_news_openai(curr_date)

        return china_news_results

    @staticmethod
    @tool
    def get_fundamentals_openai(
        ticker: Annotated[str, "the company's ticker"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    ):
        """
        Retrieve the latest fundamental information about a given stock on a given date by using OpenAI's news API.
        Args:
            ticker (str): Ticker of a company. e.g. AAPL, TSM
            curr_date (str): Current date in yyyy-mm-dd format
        Returns:
            str: A formatted string containing the latest fundamental information about the company on the given date.
        """

        openai_fundamentals_results = interface.get_fundamentals_openai(
            ticker, curr_date
        )

        return openai_fundamentals_results

    @staticmethod
    @tool
    def get_fundamentals_tushare(
        ticker: Annotated[str, "Stock ticker symbol (e.g., '000001' or '600000')"],
        curr_date: Annotated[str, "Current date in YYYY-MM-DD format"]
    ):
        """
        Get comprehensive fundamental data for a Chinese stock using Tushare API.

        This function retrieves data from multiple Tushare endpoints including:
        - daily_basic: PE, PB, PS ratios, market cap, shares outstanding
        - fina_indicator: Financial indicators and ratios
        - top10_holders: Top 10 shareholders information
        - income: Income statement data
        - balancesheet: Balance sheet data
        - cashflow: Cash flow statement data

        Args:
            ticker (str): Stock ticker symbol (Chinese stocks)
            curr_date (str): Current date for analysis
        Returns:
            str: Comprehensive fundamental analysis report
        """

        tushare_fundamentals_results = interface.get_fundamentals_tushare(
            ticker, curr_date
        )

        return tushare_fundamentals_results

    @staticmethod
    @tool
    def get_sina_global_financial_news(
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"]
    ):
        """
        获取新浪财经全球财经快讯数据

        Args:
            curr_date (str): current date you are trading at, yyyy-mm-dd

        Returns:
            str: 格式化的全球财经快讯字符串，包含最新的财经新闻和时间信息
        """

        result = interface.get_sina_global_financial_news(curr_date)

        return result

    @staticmethod
    @tool
    def get_eastmoney_financial_breakfast(
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"]
    ):
        """
        获取东方财富财经早餐数据

        Args:
            curr_date (str): current date you are trading at, yyyy-mm-dd

        Returns:
            str: 格式化的财经早餐字符串，包含最新的财经早餐内容、标题、摘要、发布时间和链接
        """

        result = interface.get_eastmoney_financial_breakfast(curr_date)

        return result

    @staticmethod
    @tool
    def get_eastmoney_global_financial_news(
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"]
    ):
        """
        获取东方财富全球财经快讯数据

        Args:
            curr_date (str): current date you are trading at, yyyy-mm-dd

        Returns:
            str: 格式化的全球财经快讯字符串，包含最新的财经新闻标题、摘要、发布时间和链接
        """

        result = interface.get_eastmoney_global_financial_news(curr_date)

        return result

    @staticmethod
    @tool
    def get_futu_financial_news(
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"]
    ):
        """
        获取富途牛牛快讯数据

        Args:
            curr_date (str): current date you are trading at, yyyy-mm-dd

        Returns:
            str: 格式化的富途牛牛快讯字符串，包含最新的财经快讯标题、内容、发布时间和链接
        """

        result = interface.get_futu_financial_news(curr_date)

        return result

    @staticmethod
    @tool
    def get_tonghuashun_global_financial_live(
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"]
    ):
        """
        获取同花顺全球财经直播数据

        Args:
            curr_date (str): current date you are trading at, yyyy-mm-dd

        Returns:
            str: 格式化的同花顺全球财经直播字符串，包含最新的财经直播标题、内容、发布时间和链接
        """

        result = interface.get_tonghuashun_global_financial_live(curr_date)

        return result

    @staticmethod
    @tool
    def get_cailianshe_telegraph(
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"]
    ):
        """
        获取财联社电报数据

        Args:
            curr_date (str): current date you are trading at, yyyy-mm-dd

        Returns:
            str: 格式化的财联社电报字符串，包含最新的财经电报标题、内容、发布日期和时间
        """

        result = interface.get_cailianshe_telegraph(curr_date)

        return result

    @staticmethod
    @tool
    def get_sina_securities_original(
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"]
    ):
        """
        获取新浪财经证券原创数据

        Args:
            curr_date (str): current date you are trading at, yyyy-mm-dd

        Returns:
            str: 格式化的新浪财经证券原创字符串，包含最新的证券原创文章时间、内容和链接
        """

        result = interface.get_sina_securities_original(curr_date)

        return result
