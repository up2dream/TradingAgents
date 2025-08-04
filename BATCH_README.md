# TradingAgents 批量分析工具

这是一个用于批量运行TradingAgents股票分析的工具，支持多股票自动化分析、进度跟踪和结果汇总。

## 功能特性

- 📊 **批量分析**: 从配置文件读取股票列表，自动分析每支股票
- 🔄 **断点续传**: 支持继续未完成的分析任务
- 📈 **指数成分股**: 自动获取指数成分股并生成股票列表
- 📋 **进度跟踪**: 实时显示分析进度和状态
- 📄 **汇总报告**: 自动生成包含所有股票买卖建议的汇总报告
- 🗑️ **结果清理**: 一键清除所有分析结果

## 安装依赖

确保已安装所需的Python包：

```bash
# 项目使用uv管理依赖
uv sync

# 或者手动安装额外依赖
uv add pyyaml tqdm
```

## 使用方法

### 1. 运行批量分析

```bash
# 运行配置文件中所有股票的分析
uv run batch.py run
```

这将：
- 读取 `batch_config.yaml` 中的股票列表
- 为每支股票在新的控制台窗口中运行分析
- 显示实时进度
- 收集所有分析结果
- 生成汇总报告

### 2. 继续未完成的分析

```bash
# 继续上次中断的分析
uv run batch.py continue
```

如果之前的分析被中断，使用此命令可以从中断处继续，无需重新分析已完成的股票。

### 3. 生成股票列表

```bash
# 从指数生成股票列表（替换现有列表，获取所有成分股）
uv run batch.py generate_stock_list --code 000300.SH

# 限制获取前20只股票
uv run batch.py generate_stock_list --code 000300.SH --limit 20

# 追加到现有列表
uv run batch.py generate_stock_list --code 000688.SH --append

# 追加并限制数量
uv run batch.py generate_stock_list --code 000688.SH --append --limit 10
```

支持的指数代码：
- `000300.SH`: 沪深300
- `000905.SH`: 中证500  
- `000688.SH`: 科创50
- `399006.SZ`: 创业板指
- `000001.SH`: 上证指数

### 4. 清除分析结果

```bash
# 清除所有分析结果和进度文件
uv run batch.py clear
```

⚠️ **注意**: 此操作不可逆，会删除所有分析结果。

## 配置文件

### batch_config.yaml

```yaml
# 需要分析的股票列表
stocks:
  - '688111.SH'  # 金山办公
  - '000001.SZ'  # 平安银行
  - 'AAPL'       # Apple
  - 'NVDA'       # NVIDIA

# 分析日期
analysis_date: '2025-08-03'

# 目录配置
results_base_dir: './results'        # 单个分析结果
batch_results_dir: './batch_results' # 批量汇总结果

# 性能设置
max_concurrent: 1      # 并发数量（建议保持为1）
timeout_minutes: 30    # 单个分析超时时间
```

## 输出结果

### 目录结构

```
├── results/                    # 单个股票分析结果
│   ├── 688111.SH/
│   │   └── 2025-08-03/
│   │       ├── reports/
│   │       │   ├── final_trade_decision.md
│   │       │   ├── market_report.md
│   │       │   └── ...
│   │       └── message_tool.log
│   └── AAPL/
│       └── 2025-08-03/
│           └── ...
├── batch_results/              # 批量分析汇总
│   ├── batch_summary_20250803_143022.md
│   └── batch_summary_20250803_143022.json
├── batch_config.yaml           # 配置文件
├── batch_progress.json         # 进度文件（临时）
└── batch_summary.json          # 最终汇总（临时）
```

### 汇总报告示例

生成的汇总报告包含：
- 分析概览（成功/失败数量）
- 每支股票的投资建议
- 详细报告路径链接

## 注意事项

1. **并发限制**: 建议 `max_concurrent` 保持为1，避免多个分析进程冲突
2. **资源消耗**: 每个分析会消耗大量API调用，请注意成本控制
3. **网络依赖**: 需要稳定的网络连接获取实时数据
4. **时间消耗**: 每支股票分析可能需要5-15分钟，请合理安排时间
5. **存储空间**: 分析结果会占用较多磁盘空间

## 故障排除

### 常见问题

1. **分析卡住不动**
   - 检查网络连接
   - 查看控制台窗口是否有错误信息
   - 使用 `continue` 命令重新开始

2. **配置文件错误**
   - 检查YAML格式是否正确
   - 确认股票代码格式正确
   - 验证日期格式为 YYYY-MM-DD

3. **依赖包缺失**
   ```bash
   uv add pyyaml tqdm akshare
   ```

4. **权限问题**
   - 确保有写入当前目录的权限
   - 检查防火墙设置

### 日志查看

- 单个股票分析日志: `results/{stock}/{date}/message_tool.log`
- 批量处理进度: `batch_progress.json`
- 最终汇总结果: `batch_results/batch_summary_*.json`

## 扩展功能

可以根据需要扩展以下功能：
- 支持更多指数和交易所
- 添加邮件通知功能
- 集成钉钉/企业微信通知
- 支持定时任务
- 添加风险控制参数

## 技术支持

如有问题，请检查：
1. TradingAgents主程序是否正常工作
2. 相关依赖包是否正确安装
3. 配置文件格式是否正确
4. 网络连接是否稳定
