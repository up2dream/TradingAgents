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


# Global instance
_tushare_utils = None

def get_tushare_utils():
    """Get global Tushare utils instance"""
    global _tushare_utils
    if _tushare_utils is None:
        _tushare_utils = TushareUtils()
    return _tushare_utils
