#!/usr/bin/env python3
"""
Batch processing script for TradingAgents analysis.

Commands:
- uv run batch.py run: Run analysis for stocks from yaml config
- uv run batch.py continue: Continue unfinished analysis
- uv run batch.py clear: Clear all analysis results
- uv run batch.py generate_stock_list --code 000688.SH [--append]: Generate stock list from index
"""

import argparse
import yaml
import os
import sys
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Any, Optional
import shutil
from tqdm import tqdm

# Default configuration file
CONFIG_FILE = "batch_config.yaml"
PROGRESS_FILE = "batch_progress.json"
SUMMARY_FILE = "batch_summary.json"

class BatchProcessor:
    def __init__(self, config_file: str = CONFIG_FILE):
        self.config_file = config_file
        self.progress_file = PROGRESS_FILE
        self.summary_file = SUMMARY_FILE
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from yaml file."""
        if not os.path.exists(self.config_file):
            self.create_default_config()
        
        with open(self.config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def create_default_config(self):
        """Create default configuration file."""
        default_config = {
            'stocks': [
                '688111.SH',  # 金山办公
                '000001.SZ',  # 平安银行
                'AAPL',       # Apple
                'NVDA'        # NVIDIA
            ],
            'analysis_date': str(date.today()),
            'results_base_dir': './results',
            'batch_results_dir': './batch_results',
            'max_concurrent': 1,  # Number of concurrent analyses
            'timeout_minutes': 30,  # Timeout for each analysis
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ Created default configuration file: {self.config_file}")
        print("Please edit the configuration file to add your stock list.")
    
    def save_config(self):
        """Save current configuration to file."""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    def load_progress(self) -> Dict[str, Any]:
        """Load progress from file."""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'completed': [],
            'failed': [],
            'remaining': self.config['stocks'].copy(),
            'start_time': None,
            'last_update': None
        }
    
    def save_progress(self, progress: Dict[str, Any]):
        """Save progress to file."""
        progress['last_update'] = datetime.now().isoformat()
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)
    
    def run_analysis_for_stock(self, stock: str, analysis_date: str) -> bool:
        """Run analysis for a single stock in a new console window."""
        print(f"\n🔄 Starting analysis for {stock}...")

        # Check if analysis already exists
        results_dir = Path(self.config['results_base_dir']) / stock / analysis_date
        final_decision_file = results_dir / "reports" / "final_trade_decision.md"

        if final_decision_file.exists():
            print(f"✅ Analysis already exists for {stock}, skipping...")
            return True

        # Create a temporary main.py with the specific stock
        temp_main = f"temp_main_{stock.replace('.', '_').replace('-', '_')}.py"

        try:
            # Read the original main.py and modify it
            with open('main.py', 'r', encoding='utf-8') as f:
                main_content = f.read()

            # Replace the stock symbol and date in main.py
            modified_content = main_content.replace(
                'ta.propagate("688111", "2025-08-03")',
                f'ta.propagate("{stock}", "{analysis_date}")'
            )

            # Write temporary main file
            with open(temp_main, 'w', encoding='utf-8') as f:
                f.write(modified_content)

            # Run the analysis directly in the current process for better control
            print(f"🚀 Running analysis for {stock}...")

            # Use subprocess to run the analysis
            cmd = ['uv', 'run', temp_main]

            # Start the process with timeout
            timeout = self.config.get('timeout_minutes', 30) * 60

            try:
                result = subprocess.run(
                    cmd,
                    timeout=timeout,
                    capture_output=True,
                    text=True,
                    cwd=os.getcwd()
                )

                if result.returncode == 0:
                    # Check if results were generated
                    if final_decision_file.exists():
                        print(f"✅ Analysis completed successfully for {stock}")
                        return True
                    else:
                        print(f"⚠️  Analysis process completed but no results found for {stock}")
                        print(f"stdout: {result.stdout[-500:]}")  # Last 500 chars
                        print(f"stderr: {result.stderr[-500:]}")  # Last 500 chars
                        return False
                else:
                    print(f"❌ Analysis failed for {stock} (return code: {result.returncode})")
                    print(f"stderr: {result.stderr[-500:]}")  # Last 500 chars
                    return False

            except subprocess.TimeoutExpired:
                print(f"⏰ Timeout reached for {stock} after {timeout} seconds")
                return False

        except Exception as e:
            print(f"❌ Error running analysis for {stock}: {e}")
            return False
        finally:
            # Clean up temporary file
            if os.path.exists(temp_main):
                os.remove(temp_main)
    
    def collect_results(self, stocks: List[str], analysis_date: str) -> Dict[str, Any]:
        """Collect analysis results from all completed stocks."""
        summary = {
            'analysis_date': analysis_date,
            'total_stocks': len(stocks),
            'completed_stocks': [],
            'failed_stocks': [],
            'recommendations': {}
        }
        
        for stock in stocks:
            results_dir = Path(self.config['results_base_dir']) / stock / analysis_date
            final_decision_file = results_dir / "reports" / "final_trade_decision.md"
            
            if final_decision_file.exists():
                try:
                    with open(final_decision_file, 'r', encoding='utf-8') as f:
                        decision_content = f.read()
                    
                    summary['completed_stocks'].append(stock)
                    summary['recommendations'][stock] = {
                        'decision_file': str(final_decision_file),
                        'decision_content': decision_content,
                        'reports_dir': str(results_dir / "reports")
                    }
                except Exception as e:
                    print(f"⚠️ Error reading results for {stock}: {e}")
                    summary['failed_stocks'].append(stock)
            else:
                summary['failed_stocks'].append(stock)
        
        return summary
    
    def generate_summary_report(self, summary: Dict[str, Any]):
        """Generate a summary report with all recommendations."""
        batch_results_dir = Path(self.config['batch_results_dir'])
        batch_results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = batch_results_dir / f"batch_summary_{timestamp}.md"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"# 批量股票分析汇总报告\n\n")
            f.write(f"**分析日期**: {summary['analysis_date']}\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**总股票数**: {summary['total_stocks']}\n")
            f.write(f"**成功分析**: {len(summary['completed_stocks'])}\n")
            f.write(f"**失败分析**: {len(summary['failed_stocks'])}\n\n")
            
            if summary['failed_stocks']:
                f.write("## ❌ 分析失败的股票\n\n")
                for stock in summary['failed_stocks']:
                    f.write(f"- {stock}\n")
                f.write("\n")
            
            f.write("## 📊 投资建议汇总\n\n")
            
            for stock, data in summary['recommendations'].items():
                f.write(f"### {stock}\n\n")
                f.write(f"**报告路径**: {data['reports_dir']}\n\n")
                f.write("**投资决策**:\n")
                f.write("```\n")
                f.write(data['decision_content'])
                f.write("\n```\n\n")
                f.write("---\n\n")
        
        print(f"📄 Summary report generated: {summary_file}")
        
        # Also save as JSON
        json_file = batch_results_dir / f"batch_summary_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return summary_file

    def run_batch_analysis(self):
        """Run batch analysis for all stocks in configuration."""
        print("🚀 Starting batch analysis...")

        # Load or create progress
        progress = self.load_progress()

        if not progress['start_time']:
            progress['start_time'] = datetime.now().isoformat()
            progress['remaining'] = self.config['stocks'].copy()

        analysis_date = self.config['analysis_date']
        total_stocks = len(self.config['stocks'])

        print(f"📅 Analysis date: {analysis_date}")
        print(f"📊 Total stocks: {total_stocks}")
        print(f"✅ Completed: {len(progress['completed'])}")
        print(f"❌ Failed: {len(progress['failed'])}")
        print(f"⏳ Remaining: {len(progress['remaining'])}")

        # Process remaining stocks
        with tqdm(total=len(progress['remaining']), desc="Analyzing stocks") as pbar:
            for stock in progress['remaining'].copy():
                pbar.set_description(f"Analyzing {stock}")

                success = self.run_analysis_for_stock(stock, analysis_date)

                if success:
                    progress['completed'].append(stock)
                    print(f"✅ {stock} completed successfully")
                else:
                    progress['failed'].append(stock)
                    print(f"❌ {stock} failed")

                progress['remaining'].remove(stock)
                self.save_progress(progress)
                pbar.update(1)

        # Generate summary report
        print("\n📊 Generating summary report...")
        summary = self.collect_results(self.config['stocks'], analysis_date)
        summary_file = self.generate_summary_report(summary)

        # Save final summary
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n🎉 Batch analysis completed!")
        print(f"📄 Summary report: {summary_file}")

        # Clean up progress file
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)

    def continue_analysis(self):
        """Continue unfinished analysis."""
        if not os.path.exists(self.progress_file):
            print("❌ No unfinished analysis found. Use 'run' command to start a new analysis.")
            return

        progress = self.load_progress()

        if not progress['remaining']:
            print("✅ All analyses are already completed!")
            return

        print(f"🔄 Continuing analysis...")
        print(f"⏳ Remaining stocks: {len(progress['remaining'])}")
        print(f"📝 Stocks to analyze: {', '.join(progress['remaining'])}")

        self.run_batch_analysis()

    def clear_results(self):
        """Clear all analysis results."""
        confirm = input("⚠️  Are you sure you want to clear all results? This cannot be undone. (y/N): ")

        if confirm.lower() != 'y':
            print("❌ Operation cancelled.")
            return

        # Clear results directory
        results_dir = Path(self.config['results_base_dir'])
        if results_dir.exists():
            shutil.rmtree(results_dir)
            print(f"🗑️  Cleared results directory: {results_dir}")

        # Clear batch results directory
        batch_results_dir = Path(self.config['batch_results_dir'])
        if batch_results_dir.exists():
            shutil.rmtree(batch_results_dir)
            print(f"🗑️  Cleared batch results directory: {batch_results_dir}")

        # Clear progress files
        for file in [self.progress_file, self.summary_file]:
            if os.path.exists(file):
                os.remove(file)
                print(f"🗑️  Cleared progress file: {file}")

        print("✅ All results cleared successfully!")

    def generate_stock_list(self, index_code: str, append: bool = False):
        """Generate stock list from index code."""
        print(f"📈 Generating stock list for index: {index_code}")

        try:
            # Try to get real index constituents using akshare
            new_stocks = self._fetch_index_constituents(index_code)

            if not new_stocks:
                # Fallback to sample data
                sample_stocks = {
                    '000688.SH': ['600519.SH', '000858.SZ', '002415.SZ', '600036.SH', '000002.SZ'],  # 科创50样例
                    '000300.SH': ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '600519.SH'],  # 沪深300样例
                    '000905.SH': ['002415.SZ', '300059.SZ', '002304.SZ', '000725.SZ', '002142.SZ'],  # 中证500样例
                }

                if index_code in sample_stocks:
                    new_stocks = sample_stocks[index_code]
                    print(f"⚠️  Using sample data for {index_code}")
                else:
                    print(f"⚠️  Index {index_code} not found. Adding as single stock.")
                    new_stocks = [index_code]

            if append:
                # Append to existing stocks
                existing_stocks = set(self.config['stocks'])
                for stock in new_stocks:
                    if stock not in existing_stocks:
                        self.config['stocks'].append(stock)
                print(f"➕ Appended {len(new_stocks)} stocks to configuration")
            else:
                # Replace existing stocks
                self.config['stocks'] = new_stocks
                print(f"🔄 Replaced stock list with {len(new_stocks)} stocks")

            # Save updated configuration
            self.save_config()

            print(f"📝 Updated stock list: {', '.join(self.config['stocks'])}")
            print(f"💾 Configuration saved to: {self.config_file}")

        except Exception as e:
            print(f"❌ Error generating stock list: {e}")

    def _fetch_index_constituents(self, index_code: str) -> List[str]:
        """Fetch index constituents using akshare."""
        try:
            import akshare as ak

            # Map index codes to akshare function calls
            index_mapping = {
                '000300.SH': lambda: ak.index_stock_cons(symbol="000300"),  # 沪深300
                '000905.SH': lambda: ak.index_stock_cons(symbol="000905"),  # 中证500
                '000688.SH': lambda: ak.index_stock_cons(symbol="000688"),  # 科创50
                '399006.SZ': lambda: ak.index_stock_cons(symbol="399006"),  # 创业板指
                '000001.SH': lambda: ak.index_stock_cons(symbol="000001"),  # 上证指数
            }

            if index_code in index_mapping:
                print(f"🔍 Fetching constituents for {index_code}...")
                df = index_mapping[index_code]()

                if df is not None and not df.empty:
                    # Extract stock codes and add appropriate suffix
                    stocks = []
                    for _, row in df.iterrows():
                        code = str(row['品种代码']) if '品种代码' in row else str(row['代码'])

                        # Add appropriate suffix based on exchange
                        if code.startswith('6'):
                            stock_code = f"{code}.SH"
                        elif code.startswith(('0', '3')):
                            stock_code = f"{code}.SZ"
                        else:
                            stock_code = code

                        stocks.append(stock_code)

                    print(f"✅ Found {len(stocks)} constituents for {index_code}")
                    return stocks[:20]  # Limit to first 20 stocks to avoid too many

            print(f"⚠️  Could not fetch constituents for {index_code}")
            return []

        except ImportError:
            print("⚠️  akshare not available, using sample data")
            return []
        except Exception as e:
            print(f"⚠️  Error fetching constituents: {e}")
            return []


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch processing script for TradingAgents analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run batch.py run                                    # Run analysis for all stocks
  uv run batch.py continue                               # Continue unfinished analysis
  uv run batch.py clear                                  # Clear all results
  uv run batch.py generate_stock_list --code 000300.SH  # Generate from index (replace)
  uv run batch.py generate_stock_list --code 000300.SH --append  # Generate from index (append)
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run batch analysis for all stocks')

    # Continue command
    continue_parser = subparsers.add_parser('continue', help='Continue unfinished analysis')

    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear all analysis results')

    # Generate stock list command
    generate_parser = subparsers.add_parser('generate_stock_list', help='Generate stock list from index')
    generate_parser.add_argument('--code', required=True, help='Index code (e.g., 000300.SH)')
    generate_parser.add_argument('--append', action='store_true', help='Append to existing list instead of replacing')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize batch processor
    processor = BatchProcessor()

    # Execute command
    if args.command == 'run':
        processor.run_batch_analysis()
    elif args.command == 'continue':
        processor.continue_analysis()
    elif args.command == 'clear':
        processor.clear_results()
    elif args.command == 'generate_stock_list':
        processor.generate_stock_list(args.code, args.append)


if __name__ == "__main__":
    main()
