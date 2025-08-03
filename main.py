from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta

load_dotenv()


def print_config_info(config):
    """Print configuration information in a formatted way."""
    print("=" * 60)
    print("TRADING AGENTS CONFIGURATION")
    print("=" * 60)

    # LLM Configuration
    print("\n🤖 LLM Configuration:")
    print(f"  Provider: {config.get('llm_provider', 'N/A')}")
    print(f"  Deep Think Model: {config.get('deep_think_llm', 'N/A')}")
    print(f"  Quick Think Model: {config.get('quick_think_llm', 'N/A')}")
    print(f"  Backend URL: {config.get('backend_url', 'N/A')}")

    # Debate Configuration
    print("\n💬 Debate Configuration:")
    print(f"  Max Debate Rounds: {config.get('max_debate_rounds', 'N/A')}")
    print(f"  Max Risk Discussion Rounds: {config.get('max_risk_discuss_rounds', 'N/A')}")
    print(f"  Max Recursion Limit: {config.get('max_recur_limit', 'N/A')}")

    # Tool Configuration
    print("\n🔧 Tool Configuration:")
    print(f"  Online Tools Enabled: {config.get('online_tools', 'N/A')}")

    # Directory Configuration
    print("\n📁 Directory Configuration:")
    print(f"  Project Directory: {config.get('project_dir', 'N/A')}")
    print(f"  Results Directory: {config.get('results_dir', 'N/A')}")
    print(f"  Data Directory: {config.get('data_dir', 'N/A')}")
    print(f"  Data Cache Directory: {config.get('data_cache_dir', 'N/A')}")

    print("=" * 60)
    print()


def format_duration(seconds):
    """Format duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.2f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h {remaining_minutes}m {remaining_seconds:.2f}s"


def print_execution_summary(start_time, end_time):
    """Print execution time summary."""
    duration = end_time - start_time

    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    print(f"⏰ Start Time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🏁 End Time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total Duration: {format_duration(duration)}")
    print("=" * 60)


# Record start time
start_time = time.time()

# Create a custom config
config = DEFAULT_CONFIG.copy()
# config["llm_provider"] = "google"  # Use a different model
# config["backend_url"] = "https://generativelanguage.googleapis.com/v1"  # Use a different backend
# config["deep_think_llm"] = "gemini-2.0-flash"  # Use a different model
# config["quick_think_llm"] = "gemini-2.0-flash"  # Use a different model
# config["max_debate_rounds"] = 1  # Increase debate rounds
# config["online_tools"] = True  # Increase debate rounds

# Print configuration information
print_config_info(config)

print(f"🚀 Starting trading analysis at {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
print()

try:
    # Initialize with custom config
    ta = TradingAgentsGraph(debug=True, config=config)

    # forward propagate
    _, decision = ta.propagate("688111", "2025-08-03")
    print(decision)

    # Memorize mistakes and reflect
    # ta.reflect_and_remember(1000) # parameter is the position returns

except Exception as e:
    print(f"❌ Error occurred during execution: {e}")
    raise
finally:
    # Record end time and print summary
    end_time = time.time()
    print_execution_summary(start_time, end_time)
