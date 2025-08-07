#!/usr/bin/env python3
"""
Batch processing script for TradingAgents analysis.

Commands:
- uv run batch.py run [--trade-date YYYY-MM-DD]: Run analysis for stocks from yaml config
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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import LLM classes and config for confidence scoring
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from tradingagents.default_config import DEFAULT_CONFIG

# Default configuration file
CONFIG_FILE = "batch_config.yaml"
PROGRESS_FILE = "batch_progress.json"
SUMMARY_FILE = "batch_summary.json"

class BatchProcessor:
    def __init__(self, config_file: str = CONFIG_FILE, trade_date: Optional[str] = None):
        self.config_file = config_file
        self.progress_file = PROGRESS_FILE
        self.summary_file = SUMMARY_FILE
        self.config = self.load_config()
        self.trade_date = self._validate_and_set_trade_date(trade_date)

        # Initialize LLM for confidence scoring (using same config as main.py)
        self.llm_config = self._get_llm_config()
        try:
            self.confidence_llm = self._initialize_llm()
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize LLM for confidence scoring: {e}\n"
                f"Please ensure the required API keys are properly configured:\n"
                f"- Create a .env file in the project root directory\n"
                f"- Add the appropriate API key:\n"
                f"  * For OpenAI: OPENAI_API_KEY=your-api-key\n"
                f"  * For Anthropic: ANTHROPIC_API_KEY=your-api-key\n"
                f"  * For Google: GOOGLE_API_KEY=your-api-key or GEMINI_API_KEY=your-api-key\n"
                f"- Or set the environment variable directly in your shell"
            )

    def _validate_and_set_trade_date(self, trade_date: Optional[str]) -> str:
        """Validate and set the trade date."""
        if trade_date is None:
            return str(date.today())

        try:
            # Validate date format
            datetime.strptime(trade_date, '%Y-%m-%d')
            return trade_date
        except ValueError:
            raise ValueError(f"Invalid date format: {trade_date}. Please use YYYY-MM-DD format.")

    def _get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration matching main.py settings."""
        # Create a custom config matching main.py
        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = "openai"
        config["deep_think_llm"] = "o4-mini"
        config["quick_think_llm"] = "gpt-4o-mini"
        config["backend_url"] = "https://api.gptsapi.net/v1"
        return config

    def _initialize_llm(self):
        """Initialize LLM for confidence scoring based on provider."""
        if self.llm_config["llm_provider"].lower() == "openai":
            return ChatOpenAI(
                model=self.llm_config["quick_think_llm"],
                base_url=self.llm_config["backend_url"]
            )
        elif self.llm_config["llm_provider"].lower() == "anthropic":
            return ChatAnthropic(
                model=self.llm_config["quick_think_llm"],
                base_url=self.llm_config["backend_url"]
            )
        elif self.llm_config["llm_provider"].lower() == "google":
            return ChatGoogleGenerativeAI(model=self.llm_config["quick_think_llm"])
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_config['llm_provider']}")
        
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
        """Run analysis for a single stock in the current console."""
        print(f"\n{'='*60}")
        print(f"🔄 Starting analysis for {stock} on {analysis_date}")
        print(f"{'='*60}")

        # Check if analysis already exists
        results_dir = Path(self.config['results_base_dir']) / stock / analysis_date
        final_decision_file = results_dir / "reports" / "final_trade_decision.md"

        if final_decision_file.exists():
            print(f"✅ Analysis already exists for {stock}, skipping...")
            return True

        try:
            # Ensure results directory exists
            results_base_dir = Path(self.config['results_base_dir'])
            results_base_dir.mkdir(exist_ok=True)

            # Prepare command to run main.py with arguments
            cmd = ['uv', 'run', 'main.py', '--stock', stock, '--date', analysis_date]

            # Start the process with timeout
            timeout = self.config.get('timeout_minutes', 30) * 60

            print(f"🚀 Running analysis for {stock}...")
            print(f"⏰ Timeout: {timeout//60} minutes")
            print(f"📝 Command: {' '.join(cmd)}")
            print()

            try:
                # Run the analysis process
                result = subprocess.run(
                    cmd,
                    timeout=timeout,
                    text=True,
                    cwd=os.getcwd()
                )

                if result.returncode == 0:
                    # Check if results were generated
                    if final_decision_file.exists():
                        print(f"\n✅ Analysis completed successfully for {stock}")
                        return True
                    else:
                        print(f"\n⚠️  Analysis process completed but no results found for {stock}")
                        return False
                else:
                    print(f"\n❌ Analysis failed for {stock} (return code: {result.returncode})")
                    return False

            except subprocess.TimeoutExpired:
                print(f"\n⏰ Timeout reached for {stock} after {timeout//60} minutes")
                return False

        except Exception as e:
            print(f"\n❌ Error running analysis for {stock}: {e}")
            return False
    
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

            # Prepare data for sorting
            stock_data = []
            total_stocks = len(summary['recommendations'])

            print(f"🤖 Analyzing {total_stocks} stock reports with AI model...")

            # Use tqdm for progress tracking during confidence scoring
            with tqdm(total=total_stocks, desc="AI Confidence Scoring", unit="stock") as pbar:
                for stock, data in summary['recommendations'].items():
                    pbar.set_description(f"Analyzing {stock}")

                    # Extract the recommendation from decision content
                    decision_content = data['decision_content']
                    recommendation = "UNKNOWN"
                    confidence_score = 0

                    # Try to extract the recommendation more accurately
                    content_upper = decision_content.upper()

                    # Check for explicit recommendation patterns first
                    if "RECOMMENDATION: BUY" in content_upper or "FINAL TRANSACTION PROPOSAL: BUY" in content_upper:
                        recommendation = "🟢 BUY"
                    elif "RECOMMENDATION: SELL" in content_upper or "FINAL TRANSACTION PROPOSAL: SELL" in content_upper:
                        recommendation = "🔴 SELL"
                    elif "RECOMMENDATION: HOLD" in content_upper or "FINAL TRANSACTION PROPOSAL: HOLD" in content_upper:
                        recommendation = "🟡 HOLD"
                    # Fallback to general keyword search
                    elif "BUY" in content_upper and "SELL" not in content_upper:
                        recommendation = "🟢 BUY"
                    elif "SELL" in content_upper and "BUY" not in content_upper:
                        recommendation = "🔴 SELL"
                    elif "HOLD" in content_upper:
                        recommendation = "🟡 HOLD"

                    # Calculate confidence score with error handling
                    try:
                        confidence_score = self._calculate_confidence_score(decision_content)
                        confidence_display = f"{confidence_score}%"
                    except Exception as e:
                        print(f"\n⚠️ Failed to calculate confidence score for {stock}: {e}")
                        confidence_score = 50  # Default fallback score
                        confidence_display = "50% (fallback)"

                    # Create clickable link to final decision report (absolute path)
                    import os
                    current_dir = os.getcwd()
                    final_report_path = f"file://{current_dir}/{data['reports_dir']}/final_trade_decision.md"

                    stock_data.append({
                        'stock': stock,
                        'recommendation': recommendation,
                        'confidence_score': confidence_score,
                        'confidence_display': confidence_display,
                        'report_path': final_report_path
                    })

                    pbar.update(1)

            # Sort by confidence score (high to low)
            stock_data.sort(key=lambda x: x['confidence_score'], reverse=True)

            # Create table with confidence column
            f.write("| 股票代码 | 投资建议 | 建议准确程度 | 详细报告 |\n")
            f.write("|---------|---------|-------------|----------|\n")

            for item in stock_data:
                f.write(f"| {item['stock']} | {item['recommendation']} | {item['confidence_display']} | [查看详细分析]({item['report_path']}) |\n")

            f.write("\n---\n\n")
            f.write("💡 **说明**: \n")
            f.write("- 建议准确程度由AI大模型分析投资决策报告质量生成，评分范围30-95分\n")
            f.write("- 评分考虑因素：建议明确性、技术分析深度、风险管理、推理逻辑、多角度分析等\n")
            f.write("- 点击\"查看详细分析\"链接可查看完整的投资决策依据和分析过程\n")
            f.write("- 表格已按建议准确程度从高到低排序\n\n")
        
        print(f"📄 Summary report generated: {summary_file}")
        
        # Also save as JSON with recommendations added
        json_summary = summary.copy()

        # Add recommendation field to existing recommendations
        for stock, data in json_summary['recommendations'].items():
            # Extract the recommendation from decision content
            decision_content = data['decision_content']
            recommendation = "UNKNOWN"

            # Try to extract the recommendation more accurately
            content_upper = decision_content.upper()

            # Check for explicit recommendation patterns first
            if "RECOMMENDATION: BUY" in content_upper or "FINAL TRANSACTION PROPOSAL: BUY" in content_upper:
                recommendation = "BUY"
            elif "RECOMMENDATION: SELL" in content_upper or "FINAL TRANSACTION PROPOSAL: SELL" in content_upper:
                recommendation = "SELL"
            elif "RECOMMENDATION: HOLD" in content_upper or "FINAL TRANSACTION PROPOSAL: HOLD" in content_upper:
                recommendation = "HOLD"
            # Fallback to general keyword search
            elif "BUY" in content_upper and "SELL" not in content_upper:
                recommendation = "BUY"
            elif "SELL" in content_upper and "BUY" not in content_upper:
                recommendation = "SELL"
            elif "HOLD" in content_upper:
                recommendation = "HOLD"

            # Add recommendation field to existing data
            data['recommendation'] = recommendation

        json_file = batch_results_dir / f"batch_summary_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_summary, f, indent=2, ensure_ascii=False)
        
        return summary_file

    def _calculate_confidence_score(self, decision_content: str) -> int:
        """Calculate confidence score using LLM analysis."""
        if self.confidence_llm is None:
            raise RuntimeError("LLM not initialized. Please ensure API keys are properly configured.")

        return self._calculate_confidence_score_with_llm(decision_content)

    def _calculate_confidence_score_with_llm(self, decision_content: str) -> int:
        """Calculate confidence score using LLM analysis with timeout and retry."""
        # Truncate very long content to avoid token limits and reduce latency
        max_content_length = 3000
        if len(decision_content) > max_content_length:
            decision_content = decision_content[:max_content_length] + "\n...(内容已截断)"

        prompt = f"""
请分析以下股票投资决策报告的质量和可信度，并给出一个30-95之间的准确程度评分。

评分标准：
- 30-50分：分析简单，缺乏深度，建议不明确
- 51-65分：分析基本完整，有一定依据，但缺乏细节
- 66-80分：分析较为全面，有技术指标支持，风险考虑充分
- 81-95分：分析非常详细，多角度论证，风险管理完善，建议明确

请重点考虑以下因素：
1. 投资建议的明确性和具体性
2. 技术分析的深度（技术指标使用情况）
3. 风险管理和止损策略的完整性
4. 推理逻辑的严密性和论证的充分性
5. 多角度分析（牛熊观点、不同视角）
6. 具体的执行计划和目标设定
7. 分析内容的详细程度和专业性

投资决策报告内容：
{decision_content}

请只返回一个30-95之间的整数评分，不要包含其他文字。
"""

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                # Set a reasonable timeout for the API call
                response = self.confidence_llm.invoke(prompt)
                # Extract the score from response
                score_text = response.content.strip()

                # Try to extract number from response
                import re
                numbers = re.findall(r'\d+', score_text)
                if numbers:
                    score = int(numbers[0])
                    # Ensure score is within bounds
                    return min(max(score, 30), 95)
                else:
                    raise ValueError("No valid score found in LLM response")

            except Exception as e:
                if attempt < max_retries:
                    print(f"   Retry {attempt + 1}/{max_retries} for confidence scoring...")
                    time.sleep(1)  # Brief delay before retry
                    continue
                else:
                    raise Exception(f"LLM scoring failed after {max_retries + 1} attempts: {e}")



    def run_batch_analysis(self):
        """Run batch analysis for all stocks in configuration."""
        print("🚀 Starting batch analysis...")

        # Load or create progress
        progress = self.load_progress()

        if not progress['start_time']:
            progress['start_time'] = datetime.now().isoformat()
            progress['remaining'] = self.config['stocks'].copy()

        # Use trade_date parameter if provided, otherwise fall back to config
        analysis_date = self.trade_date
        total_stocks = len(self.config['stocks'])

        print(f"📅 Analysis date: {analysis_date}")
        print(f"📊 Total stocks: {total_stocks}")
        print(f"✅ Completed: {len(progress['completed'])}")
        print(f"❌ Failed: {len(progress['failed'])}")
        print(f"⏳ Remaining: {len(progress['remaining'])}")

        # Show analysis plan
        if progress['remaining']:
            print("\n" + "="*60)
            print("📋 ANALYSIS PLAN")
            print("="*60)
            print(f"📊 Stocks to analyze: {len(progress['remaining'])}")
            print(f"⏰ Estimated time: {len(progress['remaining']) * 15} minutes")
            print(f"🔄 Each analysis will run sequentially in this console")
            print("="*60)
            print("\n🚀 Starting analysis automatically...")

        # Process remaining stocks
        with tqdm(total=len(progress['remaining']), desc="Analyzing stocks") as pbar:
            for stock in progress['remaining'].copy():
                pbar.set_description(f"Analyzing {stock}")

                max_retries = 2  # Allow up to 2 retries for each stock
                retry_count = 0
                success = False

                while retry_count <= max_retries and not success:
                    if retry_count > 0:
                        print(f"🔄 Retry {retry_count}/{max_retries} for {stock}")

                    success = self.run_analysis_for_stock(stock, analysis_date)

                    if not success and retry_count < max_retries:
                        print(f"⚠️  Analysis failed for {stock}. Retrying...")
                        retry_count += 1
                    else:
                        break

                if success:
                    progress['completed'].append(stock)
                    print(f"✅ {stock} completed successfully")
                else:
                    progress['failed'].append(stock)
                    print(f"❌ {stock} failed after {retry_count} retries")

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

    def generate_stock_list(self, index_code: str, append: bool = False, limit: int = None):
        """Generate stock list from index code."""
        print(f"📈 Generating stock list for index: {index_code}")

        try:
            # Try to get real index constituents using akshare
            new_stocks = self._fetch_index_constituents(index_code)

            # Apply limit if specified
            if limit and len(new_stocks) > limit:
                print(f"📊 Limiting to first {limit} stocks (out of {len(new_stocks)} total)")
                new_stocks = new_stocks[:limit]

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
                    # Return all stocks by default, let user decide how many to use
                    return stocks

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
  uv run batch.py run                                    # Run analysis for all stocks (today's date)
  uv run batch.py run --trade-date 2025-08-04           # Run analysis for specific date
  uv run batch.py continue                               # Continue unfinished analysis
  uv run batch.py clear                                  # Clear all results
  uv run batch.py generate_stock_list --code 000300.SH  # Generate from index (replace, all stocks)
  uv run batch.py generate_stock_list --code 000300.SH --limit 20  # Generate first 20 stocks
  uv run batch.py generate_stock_list --code 000688.SH --append    # Append to existing list
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run batch analysis for all stocks')
    run_parser.add_argument('--trade-date', type=str, help='Trade date for analysis (YYYY-MM-DD format, default: today)')

    # Continue command
    continue_parser = subparsers.add_parser('continue', help='Continue unfinished analysis')

    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear all analysis results')

    # Generate stock list command
    generate_parser = subparsers.add_parser('generate_stock_list', help='Generate stock list from index')
    generate_parser.add_argument('--code', required=True, help='Index code (e.g., 000300.SH)')
    generate_parser.add_argument('--append', action='store_true', help='Append to existing list instead of replacing')
    generate_parser.add_argument('--limit', type=int, help='Limit number of stocks to fetch (default: all)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize batch processor
    if args.command == 'run':
        processor = BatchProcessor(trade_date=args.trade_date)
    else:
        processor = BatchProcessor()

    # Execute command
    if args.command == 'run':
        processor.run_batch_analysis()
    elif args.command == 'continue':
        processor.continue_analysis()
    elif args.command == 'clear':
        processor.clear_results()
    elif args.command == 'generate_stock_list':
        processor.generate_stock_list(args.code, args.append, args.limit)


if __name__ == "__main__":
    main()
