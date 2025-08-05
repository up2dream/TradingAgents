import pandas as pd
import tushare as ts
from typing import Annotated
import os
from datetime import datetime, timedelta
from .config import get_config


class TushareUtils:
    """Tushare utility class for Chinese stock market data"""
    
    def __init__(self):
        """Initialize Tushare with token from environment"""
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()

        config = get_config()
        token = os.getenv('TUSHARE_TOKEN')
        if not token:
            raise ValueError("TUSHARE_TOKEN not found in environment variables")
        ts.set_token(token)
        self.pro = ts.pro_api()
    
    @staticmethod
    def convert_symbol_to_tushare(symbol: str) -> str:
        """
        Convert symbol to Tushare format
        Examples: 
        - '000001' -> '000001.SZ' (Shenzhen)
        - '600000' -> '600000.SH' (Shanghai)
        - 'AAPL' -> 'AAPL' (keep as is for US stocks, though Tushare mainly supports Chinese stocks)
        """
        if symbol.isdigit() and len(symbol) == 6:
            # Chinese stock codes
            if symbol.startswith(('000', '002', '300')):
                return f"{symbol}.SZ"  # Shenzhen
            elif symbol.startswith(('600', '601', '603', '688')):
                return f"{symbol}.SH"  # Shanghai
            else:
                return f"{symbol}.SZ"  # Default to Shenzhen for other codes
        elif '.' in symbol:
            return symbol  # Already in correct format
        else:
            return symbol  # Keep as is for other formats
    
    def get_stock_data(
        self,
        symbol: Annotated[str, "ticker symbol"],
        start_date: Annotated[str, "start date for retrieving stock price data, YYYY-mm-dd"],
        end_date: Annotated[str, "end date for retrieving stock price data, YYYY-mm-dd"],
    ) -> pd.DataFrame:
        """
        Retrieve stock price data for designated ticker symbol using Tushare
        """
        ts_symbol = self.convert_symbol_to_tushare(symbol)
        
        try:
            # Get daily stock data from Tushare
            data = self.pro.daily(
                ts_code=ts_symbol,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', '')
            )
            
            if data.empty:
                return pd.DataFrame()
            
            # Convert to Yahoo Finance format for compatibility
            data = data.sort_values('trade_date')
            data['Date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
            
            # Rename columns to match Yahoo Finance format
            data = data.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'vol': 'Volume',
                'pre_close': 'Adj Close'  # Tushare's pre_close corresponds to Yahoo's Adj Close
            })

            # If pre_close is not available, use close as fallback
            if 'Adj Close' not in data.columns:
                data['Adj Close'] = data['Close']
            
            # Select and reorder columns to match Yahoo Finance format
            data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
            data = data.reset_index(drop=True)
            
            return data
            
        except Exception as e:
            print(f"Error fetching data from Tushare for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_stock_basic_info(self, symbol: str) -> dict:
        """Get basic stock information"""
        ts_symbol = self.convert_symbol_to_tushare(symbol)
        
        try:
            # Get stock basic info
            basic_info = self.pro.stock_basic(ts_code=ts_symbol)
            if not basic_info.empty:
                info = basic_info.iloc[0]
                return {
                    'symbol': info.get('symbol', ''),
                    'name': info.get('name', ''),
                    'area': info.get('area', ''),
                    'industry': info.get('industry', ''),
                    'market': info.get('market', ''),
                    'list_date': info.get('list_date', '')
                }
        except Exception as e:
            print(f"Error fetching basic info from Tushare for {symbol}: {e}")
        
        return {}
    
    def is_chinese_stock(self, symbol: str) -> bool:
        """Check if the symbol is a Chinese stock"""
        # Chinese stock codes are typically 6 digits
        if symbol.isdigit() and len(symbol) == 6:
            return True
        # Or already in Tushare format
        if '.' in symbol and symbol.split('.')[1] in ['SH', 'SZ']:
            return True
        return False

    def get_fundamental_data(self, symbol: str, curr_date: str) -> dict:
        """
        Get comprehensive fundamental data for a Chinese stock

        Args:
            symbol: Stock symbol (e.g., '688111' or '688111.SH')
            curr_date: Current date in YYYY-MM-DD format

        Returns:
            dict: Fundamental data including PE, PS, PB, ROE, etc.
        """
        ts_symbol = self.convert_symbol_to_tushare(symbol)

        try:
            # Convert date format for Tushare (YYYYMMDD)
            date_obj = datetime.strptime(curr_date, '%Y-%m-%d')
            ts_date = date_obj.strftime('%Y%m%d')

            # Get basic financial indicators (try multiple recent dates)
            basic_data = pd.DataFrame()
            dates_to_try = [
                ts_date,
                (date_obj - timedelta(days=1)).strftime('%Y%m%d'),
                (date_obj - timedelta(days=2)).strftime('%Y%m%d'),
                (date_obj - timedelta(days=3)).strftime('%Y%m%d'),
                (date_obj - timedelta(days=7)).strftime('%Y%m%d'),
            ]

            for try_date in dates_to_try:
                try:
                    basic_data = self.pro.daily_basic(ts_code=ts_symbol, trade_date=try_date)
                    if not basic_data.empty:
                        break
                except Exception as e:
                    print(f"Failed to get daily_basic for {ts_symbol} on {try_date}: {e}")
                    continue

            # Get company basic info
            company_info = self.get_stock_basic_info(symbol)

            # Get latest financial statements (get all data and select most recent)
            income_data = pd.DataFrame()
            balance_data = pd.DataFrame()
            cashflow_data = pd.DataFrame()

            try:
                # Get all income statement data
                income_data = self.pro.income(ts_code=ts_symbol)
                if not income_data.empty:
                    # Sort by end_date to get the most recent data
                    income_data = income_data.sort_values('end_date', ascending=False)
                    print(f"Retrieved {len(income_data)} income statement records")
            except Exception as e:
                print(f"Failed to get income data: {e}")

            try:
                # Get all balance sheet data
                balance_data = self.pro.balancesheet(ts_code=ts_symbol)
                if not balance_data.empty:
                    # Sort by end_date to get the most recent data
                    balance_data = balance_data.sort_values('end_date', ascending=False)
                    print(f"Retrieved {len(balance_data)} balance sheet records")
            except Exception as e:
                print(f"Failed to get balance sheet data: {e}")

            try:
                # Get all cash flow data
                cashflow_data = self.pro.cashflow(ts_code=ts_symbol)
                if not cashflow_data.empty:
                    # Sort by end_date to get the most recent data
                    cashflow_data = cashflow_data.sort_values('end_date', ascending=False)
                    print(f"Retrieved {len(cashflow_data)} cash flow records")
            except Exception as e:
                print(f"Failed to get cash flow data: {e}")

            result = {
                'symbol': symbol,
                'company_name': company_info.get('name', 'N/A'),
                'industry': company_info.get('industry', 'N/A'),
                'market': company_info.get('market', 'N/A'),
                'date': curr_date
            }

            # Add basic financial ratios if available
            if not basic_data.empty:
                latest_basic = basic_data.iloc[0]
                result.update({
                    'pe_ratio': latest_basic.get('pe', 'N/A'),
                    'pb_ratio': latest_basic.get('pb', 'N/A'),
                    'ps_ratio': latest_basic.get('ps', 'N/A'),
                    'total_share': latest_basic.get('total_share', 'N/A'),
                    'float_share': latest_basic.get('float_share', 'N/A'),
                    'total_mv': latest_basic.get('total_mv', 'N/A'),  # Total market value
                    'circ_mv': latest_basic.get('circ_mv', 'N/A'),   # Circulating market value
                })

            # Add income statement data if available
            if not income_data.empty:
                latest_income = income_data.iloc[0]
                result.update({
                    'revenue': latest_income.get('revenue', 'N/A'),
                    'total_revenue': latest_income.get('total_revenue', 'N/A'),
                    'operate_profit': latest_income.get('operate_profit', 'N/A'),
                    'total_profit': latest_income.get('total_profit', 'N/A'),
                    'n_income': latest_income.get('n_income', 'N/A'),  # Net income
                    'n_income_attr_p': latest_income.get('n_income_attr_p', 'N/A'),  # Net income attributable to parent
                    'eps': latest_income.get('basic_eps', 'N/A'),     # Earnings per share
                    'diluted_eps': latest_income.get('diluted_eps', 'N/A'),  # Diluted EPS
                    'end_date': latest_income.get('end_date', 'N/A'),  # Report period
                })

            # Add balance sheet data if available
            if not balance_data.empty:
                latest_balance = balance_data.iloc[0]
                result.update({
                    'total_assets': latest_balance.get('total_assets', 'N/A'),
                    'total_liab': latest_balance.get('total_liab', 'N/A'),
                    'total_equity': latest_balance.get('total_hldr_eqy_exc_min_int', 'N/A'),
                    'monetary_cap': latest_balance.get('money_cap', 'N/A'),  # Cash and equivalents
                    'total_cur_assets': latest_balance.get('total_cur_assets', 'N/A'),  # Total current assets
                    'total_nca': latest_balance.get('total_nca', 'N/A'),  # Total non-current assets
                    'balance_end_date': latest_balance.get('end_date', 'N/A'),  # Balance sheet period
                })

            # Add cash flow data if available
            if not cashflow_data.empty:
                latest_cashflow = cashflow_data.iloc[0]

                # Try to find a record with valid net_profit data
                net_profit_cf = latest_cashflow.get('net_profit', 'N/A')
                if pd.isna(net_profit_cf) or net_profit_cf == 'N/A':
                    # Look for records with valid net_profit
                    valid_net_profit_records = cashflow_data[cashflow_data['net_profit'].notna()]
                    if not valid_net_profit_records.empty:
                        net_profit_cf = valid_net_profit_records.iloc[0].get('net_profit', 'N/A')
                        print(f"Found net_profit from {valid_net_profit_records.iloc[0].get('end_date', 'N/A')}: {net_profit_cf}")
                    else:
                        net_profit_cf = 'N/A'

                result.update({
                    'operating_cashflow': latest_cashflow.get('n_cashflow_act', 'N/A'),
                    'investing_cashflow': latest_cashflow.get('n_cashflow_inv_act', 'N/A'),
                    'financing_cashflow': latest_cashflow.get('n_cash_flows_fnc_act', 'N/A'),
                    'net_profit_cf': net_profit_cf,  # Net profit from cash flow statement
                    'cashflow_end_date': latest_cashflow.get('end_date', 'N/A'),  # Cash flow period
                })

            return result

        except Exception as e:
            print(f"Error fetching fundamental data from Tushare for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'symbol': symbol,
                'error': str(e),
                'date': curr_date,
                'company_name': 'N/A',
                'industry': 'N/A',
                'market': 'N/A'
            }

    def format_fundamental_report(self, fundamental_data: dict) -> str:
        """
        Format fundamental data into a readable report

        Args:
            fundamental_data: Dictionary containing fundamental data

        Returns:
            str: Formatted report string
        """
        if 'error' in fundamental_data:
            return f"Error retrieving fundamental data for {fundamental_data['symbol']}: {fundamental_data['error']}"

        report = f"""
# Fundamental Analysis Report for {fundamental_data.get('symbol', 'N/A')}

## Company Information
- **Company Name**: {fundamental_data.get('company_name', 'N/A')}
- **Industry**: {fundamental_data.get('industry', 'N/A')}
- **Market**: {fundamental_data.get('market', 'N/A')}
- **Analysis Date**: {fundamental_data.get('date', 'N/A')}

## Valuation Ratios
- **PE Ratio**: {fundamental_data.get('pe_ratio', 'N/A')}
- **PB Ratio**: {fundamental_data.get('pb_ratio', 'N/A')}
- **PS Ratio**: {fundamental_data.get('ps_ratio', 'N/A')}

## Market Capitalization
- **Total Market Value**: {fundamental_data.get('total_mv', 'N/A')} (万元)
- **Circulating Market Value**: {fundamental_data.get('circ_mv', 'N/A')} (万元)
- **Total Shares**: {fundamental_data.get('total_share', 'N/A')} (万股)
- **Float Shares**: {fundamental_data.get('float_share', 'N/A')} (万股)

## Financial Performance
- **Total Revenue**: {fundamental_data.get('total_revenue', 'N/A')} (元)
- **Revenue**: {fundamental_data.get('revenue', 'N/A')} (元)
- **Operating Profit**: {fundamental_data.get('operate_profit', 'N/A')} (元)
- **Total Profit**: {fundamental_data.get('total_profit', 'N/A')} (元)
- **Net Income**: {fundamental_data.get('n_income', 'N/A')} (元)
- **Net Income (Attributable to Parent)**: {fundamental_data.get('n_income_attr_p', 'N/A')} (元)
- **Basic EPS**: {fundamental_data.get('eps', 'N/A')} (元)
- **Diluted EPS**: {fundamental_data.get('diluted_eps', 'N/A')} (元)
- **Report Period**: {fundamental_data.get('end_date', 'N/A')}

## Balance Sheet
- **Total Assets**: {fundamental_data.get('total_assets', 'N/A')} (元)
- **Total Current Assets**: {fundamental_data.get('total_cur_assets', 'N/A')} (元)
- **Total Non-Current Assets**: {fundamental_data.get('total_nca', 'N/A')} (元)
- **Total Liabilities**: {fundamental_data.get('total_liab', 'N/A')} (元)
- **Total Equity**: {fundamental_data.get('total_equity', 'N/A')} (元)
- **Cash and Equivalents**: {fundamental_data.get('monetary_cap', 'N/A')} (元)
- **Balance Sheet Period**: {fundamental_data.get('balance_end_date', 'N/A')}

## Cash Flow
- **Operating Cash Flow**: {fundamental_data.get('operating_cashflow', 'N/A')} (元)
- **Investing Cash Flow**: {fundamental_data.get('investing_cashflow', 'N/A')} (元)
- **Financing Cash Flow**: {fundamental_data.get('financing_cashflow', 'N/A')} (元)
- **Net Profit (Cash Flow Statement)**: {fundamental_data.get('net_profit_cf', 'N/A')} (元)
- **Cash Flow Period**: {fundamental_data.get('cashflow_end_date', 'N/A')}

---
*Data source: Tushare API*
"""
        return report


# Global instance
_tushare_utils = None

def get_tushare_utils():
    """Get global Tushare utils instance"""
    global _tushare_utils
    if _tushare_utils is None:
        _tushare_utils = TushareUtils()
    return _tushare_utils
