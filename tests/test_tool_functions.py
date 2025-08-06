import unittest
import sys
import os
from datetime import datetime
import unittest.mock
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables from .env file
dotenv_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))).joinpath('.env')
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)

from tradingagents.dataflows.interface import get_china_focused_news_openai, get_global_news_openai
from tradingagents.dataflows.config import get_config, set_config

class TestChinaFocusedNews(unittest.TestCase):
    """Test cases for the get_china_focused_news_openai function."""
    
    def setUp(self):
        """Set up test environment."""
        # Store original environment variables
        self.original_openai_key = os.environ.get('OPENAI_API_KEY')
        self.original_google_key = os.environ.get('GOOGLE_API_KEY')
        
        # Store original config
        self.original_config = get_config()
    
    def tearDown(self):
        """Restore environment after tests."""
        # Restore original environment variables
        if self.original_openai_key:
            os.environ['OPENAI_API_KEY'] = self.original_openai_key
        elif 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
            
        if self.original_google_key:
            os.environ['GOOGLE_API_KEY'] = self.original_google_key
        elif 'GOOGLE_API_KEY' in os.environ:
            del os.environ['GOOGLE_API_KEY']
        
        # Restore original config
        set_config(self.original_config)

    def test_get_global_news_openai(self):
        """Test the OpenAI path of the get_global_news_openai function."""
        custom_config = self.original_config.copy()
        set_config(custom_config)
        
        try:
            # Call the function with actual API
            test_date = "2025-08-06"
            result = get_global_news_openai(test_date)
            
            # Verify the result is not empty
            self.assertIsNotNone(result)
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
            
            # Print the result for display
            print(f"\nOpenAI Path Result: {result}...")
        except Exception as e:
            # Skip test if API key is invalid
            if "invalid_api_key" in str(e) or "AuthenticationError" in str(e) or "API key not valid" in str(e):
                print(f"\nSkipping OpenAI test: Invalid API key")
                self.skipTest(f"Invalid OpenAI API key: {str(e)}")
            else:
                self.fail(f"OpenAI test failed with error: {str(e)}")

    def test_get_china_focused_news_google(self):
        """Test the Google Gemini path of the get_china_focused_news_openai function."""
        custom_config = self.original_config.copy()
        set_config(custom_config)
        
        try:
            # Call the function with actual API
            test_date = "2025-08-06"
            result = get_china_focused_news_openai(test_date)
            
            # Verify the result is not empty
            self.assertIsNotNone(result)
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
            
            # Print the result for display
            print(f"\nGoogle Gemini Path Result: {result}...")
        except Exception as e:
            # Skip test if API key is invalid
            if "invalid_api_key" in str(e) or "AuthenticationError" in str(e) or "API key not valid" in str(e):
                print(f"\nSkipping Google Gemini test: Invalid API key")
                self.skipTest(f"Invalid Google API key: {str(e)}")
            else:
                self.fail(f"Google Gemini test failed with error: {str(e)}")

if __name__ == "__main__":
    unittest.main()