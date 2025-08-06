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
    def get_reddit_stock_info(
        ticker: Annotated[
            str,
            "Ticker of a company. e.g. AAPL, TSM",
        ],
        curr_date: Annotated[str, "Current date you want to get news for"],
    ) -> str:
        """
        Retrieve the latest news about a given stock from Reddit, given the current date.
        Args:
            ticker (str): Ticker of a company. e.g. AAPL, TSM
            curr_date (str): current date in yyyy-mm-dd format to get news for
        Returns:
            str: A formatted dataframe containing the latest news about the company on the given date
        """

        stock_news_results = interface.get_reddit_company_news(ticker, curr_date, 7, 5)

        return stock_news_results

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
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    ):
        """
        Retrieve the latest news about a given stock by using OpenAI's news API.
        Args:
            ticker (str): Ticker of a company. e.g. AAPL, TSM
            curr_date (str): Current date in yyyy-mm-dd format
        Returns:
            str: A formatted string containing the latest news about the company on the given date.
        """

        openai_news_results = interface.get_stock_news_openai(ticker, curr_date)

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
    def get_china_social_media_openai(
        ticker: Annotated[str, "Stock ticker symbol"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    ):
        """
        专门搜索中国社交媒体平台上关于特定股票的讨论和分析。
        Specifically search Chinese social media platforms (Weibo, Xueqiu, Zhihu, etc.) for stock discussions.

        This function focuses on Chinese social media platforms including:
        - Weibo (微博) - for general social media sentiment
        - Xueqiu (雪球) - for professional investment discussions
        - Zhihu (知乎) - for in-depth analysis and Q&A
        - Eastmoney Guba (东方财富股吧) - for retail investor sentiment
        - Tonghuashun (同花顺) - for technical analysis discussions

        Args:
            ticker (str): Stock ticker symbol (e.g., '000001', '600000')
            curr_date (str): Current date in yyyy-mm-dd format
        Returns:
            str: Comprehensive analysis of Chinese social media sentiment and discussions about the stock.
        """

        china_social_results = interface.get_china_social_media_openai(ticker, curr_date)

        return china_social_results

    @staticmethod
    @tool
    def get_china_social_media_real_data(
        ticker: Annotated[str, "Stock ticker symbol"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    ):
        """
        获取真实的中国社交媒体数据，使用 AKShare 等本地库直接获取数据。
        Get real Chinese social media data using AKShare and other local libraries for direct data access.

        This function provides REAL data from Chinese social media platforms including:
        - Xueqiu (雪球) - Real attention, discussion, and trading rankings
        - Eastmoney (东方财富) - Real popularity and trending rankings
        - Individual stock heat trends - Real heat trend data
        - Interactive platforms - Real investor interaction data

        Unlike the LLM-based search, this function returns actual structured data from APIs.

        Args:
            ticker (str): Stock ticker symbol (e.g., '000001', '600000')
            curr_date (str): Current date in yyyy-mm-dd format
        Returns:
            str: Comprehensive report with real Chinese social media data and analysis.
        """

        real_social_results = interface.get_china_social_media_real_data(ticker, curr_date)

        return real_social_results

    @staticmethod
    @tool
    def get_china_forum_data(
        ticker: Annotated[str, "Stock ticker symbol"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    ):
        """
        获取中国主流投资论坛数据，包括东方财富股吧、雪球、同花顺等平台。
        Get data from mainstream Chinese investment forums including Eastmoney Guba, Xueqiu, Tonghuashun, etc.

        This function provides data from major Chinese investment forums:
        - Eastmoney Guba (东方财富股吧) - Popular retail investor discussions
        - Xueqiu (雪球) - Professional investment community
        - Tonghuashun (同花顺) - Concept and industry discussions

        Args:
            ticker (str): Stock ticker symbol (e.g., '000001', '688111')
            curr_date (str): Current date in yyyy-mm-dd format
        Returns:
            str: Comprehensive report with mainstream forum data and sentiment analysis.
        """

        forum_results = interface.get_china_forum_data(ticker, curr_date)

        return forum_results

    @staticmethod
    @tool
    def get_china_comprehensive_social_media_data(
        ticker: Annotated[str, "Stock ticker symbol"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    ):
        """
        获取综合的中国社交媒体数据，整合所有数据源包括基础数据和主流论坛。
        Get comprehensive Chinese social media data integrating all sources including basic data and mainstream forums.

        This function provides the most complete Chinese social media analysis by combining:
        - Basic social media data (AKShare + Tushare)
        - Mainstream forum data (Eastmoney, Xueqiu, Tonghuashun)
        - Interactive platform data
        - News and sentiment analysis

        Args:
            ticker (str): Stock ticker symbol (e.g., '000001', '688111')
            curr_date (str): Current date in yyyy-mm-dd format
        Returns:
            str: Comprehensive integrated report with all Chinese social media data sources.
        """

        comprehensive_results = interface.get_china_comprehensive_social_media_data(ticker, curr_date)

        return comprehensive_results

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
    def get_sina_global_financial_news():
        """
        获取新浪财经全球财经快讯数据

        Returns:
            str: 格式化的全球财经快讯字符串，包含最新的财经新闻和时间信息
        """

        result = interface.get_sina_global_financial_news()

        return result

    @staticmethod
    @tool
    def get_eastmoney_financial_breakfast():
        """
        获取东方财富财经早餐数据

        Returns:
            str: 格式化的财经早餐字符串，包含最新的财经早餐内容、标题、摘要、发布时间和链接
        """

        result = interface.get_eastmoney_financial_breakfast()

        return result

    @staticmethod
    @tool
    def get_eastmoney_global_financial_news():
        """
        获取东方财富全球财经快讯数据

        Returns:
            str: 格式化的全球财经快讯字符串，包含最新的财经新闻标题、摘要、发布时间和链接
        """

        result = interface.get_eastmoney_global_financial_news()

        return result

    @staticmethod
    @tool
    def get_futu_financial_news():
        """
        获取富途牛牛快讯数据

        Returns:
            str: 格式化的富途牛牛快讯字符串，包含最新的财经快讯标题、内容、发布时间和链接
        """

        result = interface.get_futu_financial_news()

        return result

    @staticmethod
    @tool
    def get_tonghuashun_global_financial_live():
        """
        获取同花顺全球财经直播数据

        Returns:
            str: 格式化的同花顺全球财经直播字符串，包含最新的财经直播标题、内容、发布时间和链接
        """

        result = interface.get_tonghuashun_global_financial_live()

        return result

    @staticmethod
    @tool
    def get_cailianshe_telegraph():
        """
        获取财联社电报数据

        Returns:
            str: 格式化的财联社电报字符串，包含最新的财经电报标题、内容、发布日期和时间
        """

        result = interface.get_cailianshe_telegraph()

        return result

    @staticmethod
    @tool
    def get_sina_securities_original():
        """
        获取新浪财经证券原创数据

        Returns:
            str: 格式化的新浪财经证券原创字符串，包含最新的证券原创文章时间、内容和链接
        """

        result = interface.get_sina_securities_original()

        return result
