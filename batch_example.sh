#!/bin/bash

# TradingAgents 批量分析示例脚本
# Example script for TradingAgents batch analysis

echo "🚀 TradingAgents 批量分析示例"
echo "================================"

echo ""
echo "1. 生成沪深300成分股列表 (前20只)"
uv run batch.py generate_stock_list --code 000300.SH

echo ""
echo "2. 查看当前配置"
echo "当前配置文件内容:"
cat batch_config.yaml

echo ""
echo "3. 开始批量分析 (注意：这将需要较长时间)"
read -p "是否继续批量分析? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    uv run batch.py run
else
    echo "跳过批量分析"
fi

echo ""
echo "4. 查看结果目录结构"
if [ -d "batch_results" ]; then
    echo "批量分析结果:"
    ls -la batch_results/
    
    echo ""
    echo "最新汇总报告预览:"
    latest_report=$(ls -t batch_results/batch_summary_*.md 2>/dev/null | head -1)
    if [ -n "$latest_report" ]; then
        head -20 "$latest_report"
        echo "..."
        echo "(查看完整报告: $latest_report)"
    fi
else
    echo "暂无批量分析结果"
fi

echo ""
echo "5. 其他可用命令:"
echo "   uv run batch.py continue                               # 继续未完成的分析"
echo "   uv run batch.py clear                                  # 清除所有结果"
echo "   uv run batch.py generate_stock_list --code 000688.SH --append  # 追加科创50成分股"

echo ""
echo "✅ 示例完成!"
