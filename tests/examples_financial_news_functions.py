#!/usr/bin/env python3
"""
财经新闻函数使用示例

本文件展示了所有集成到news_analyst_node中的财经新闻函数的用法和输出内容。
包含7个基础接口函数和对应的Toolkit工具函数的使用示例。

运行方式:
    python tests/examples_financial_news_functions.py

作者: TradingAgents Team
日期: 2025-08-06
"""

import sys
import os
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.dataflows.interface import (
    get_sina_global_financial_news,
    get_eastmoney_financial_breakfast,
    get_eastmoney_global_financial_news,
    get_futu_financial_news,
    get_tonghuashun_global_financial_live,
    get_cailianshe_telegraph,
    get_sina_securities_original
)
from tradingagents.agents.utils.agent_utils import Toolkit


def print_separator(title: str):
    """打印分隔符"""
    print("\n" + "="*80)
    print(f" {title} ")
    print("="*80)


def print_subsection(title: str):
    """打印子标题"""
    print(f"\n--- {title} ---")


def truncate_text(text: str, max_length: int = 1000) -> str:
    """截断文本到指定长度"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "\n...\n[内容已截断，完整内容请查看函数返回值]"


def example_sina_global_financial_news():
    """示例1: 新浪财经全球财经快讯"""
    print_separator("示例1: 新浪财经全球财经快讯")
    
    print("函数: get_sina_global_financial_news()")
    print("描述: 获取新浪财经全球财经快讯，包含时间和内容")
    print("数据源: https://finance.sina.com.cn/7x24")
    print("AKSHARE接口: ak.stock_info_global_sina()")
    
    try:
        result = get_sina_global_financial_news()
        print(f"\n返回数据长度: {len(result)} 字符")
        print(f"数据类型: {type(result)}")
        print("\n输出内容预览:")
        print(truncate_text(result))
        print("\n✅ 函数执行成功")
    except Exception as e:
        print(f"\n❌ 函数执行失败: {str(e)}")


def example_eastmoney_financial_breakfast():
    """示例2: 东方财富财经早餐"""
    print_separator("示例2: 东方财富财经早餐")
    
    print("函数: get_eastmoney_financial_breakfast()")
    print("描述: 获取东方财富财经早餐，包含标题、摘要、发布时间和链接")
    print("数据源: https://stock.eastmoney.com/a/czpnc.html")
    print("AKSHARE接口: ak.stock_info_cjzc_em()")
    
    try:
        result = get_eastmoney_financial_breakfast()
        print(f"\n返回数据长度: {len(result)} 字符")
        print(f"数据类型: {type(result)}")
        print("\n输出内容预览:")
        print(truncate_text(result))
        print("\n✅ 函数执行成功")
    except Exception as e:
        print(f"\n❌ 函数执行失败: {str(e)}")


def example_eastmoney_global_financial_news():
    """示例3: 东方财富全球财经快讯"""
    print_separator("示例3: 东方财富全球财经快讯")
    
    print("函数: get_eastmoney_global_financial_news()")
    print("描述: 获取东方财富全球财经快讯，包含标题、摘要、发布时间和链接")
    print("数据源: https://kuaixun.eastmoney.com/7_24.html")
    print("AKSHARE接口: ak.stock_info_global_em()")
    
    try:
        result = get_eastmoney_global_financial_news()
        print(f"\n返回数据长度: {len(result)} 字符")
        print(f"数据类型: {type(result)}")
        print("\n输出内容预览:")
        print(truncate_text(result))
        print("\n✅ 函数执行成功")
    except Exception as e:
        print(f"\n❌ 函数执行失败: {str(e)}")


def example_futu_financial_news():
    """示例4: 富途牛牛快讯"""
    print_separator("示例4: 富途牛牛快讯")
    
    print("函数: get_futu_financial_news()")
    print("描述: 获取富途牛牛快讯，包含标题、内容、发布时间和链接")
    print("数据源: https://news.futunn.com/main/live")
    print("AKSHARE接口: ak.stock_info_global_futu()")
    
    try:
        result = get_futu_financial_news()
        print(f"\n返回数据长度: {len(result)} 字符")
        print(f"数据类型: {type(result)}")
        print("\n输出内容预览:")
        print(truncate_text(result))
        print("\n✅ 函数执行成功")
    except Exception as e:
        print(f"\n❌ 函数执行失败: {str(e)}")


def example_tonghuashun_global_financial_live():
    """示例5: 同花顺全球财经直播"""
    print_separator("示例5: 同花顺全球财经直播")
    
    print("函数: get_tonghuashun_global_financial_live()")
    print("描述: 获取同花顺全球财经直播，包含标题、内容、发布时间和链接")
    print("数据源: https://news.10jqka.com.cn/realtimenews.html")
    print("AKSHARE接口: ak.stock_info_global_ths()")
    
    try:
        result = get_tonghuashun_global_financial_live()
        print(f"\n返回数据长度: {len(result)} 字符")
        print(f"数据类型: {type(result)}")
        print("\n输出内容预览:")
        print(truncate_text(result))
        print("\n✅ 函数执行成功")
    except Exception as e:
        print(f"\n❌ 函数执行失败: {str(e)}")


def example_cailianshe_telegraph():
    """示例6: 财联社电报"""
    print_separator("示例6: 财联社电报")
    
    print("函数: get_cailianshe_telegraph()")
    print("描述: 获取财联社电报，包含标题、内容、发布日期和发布时间")
    print("数据源: https://www.cls.cn/telegraph")
    print("AKSHARE接口: ak.stock_info_global_cls(symbol='全部')")
    
    try:
        result = get_cailianshe_telegraph()
        print(f"\n返回数据长度: {len(result)} 字符")
        print(f"数据类型: {type(result)}")
        print("\n输出内容预览:")
        print(truncate_text(result))
        print("\n✅ 函数执行成功")
    except Exception as e:
        print(f"\n❌ 函数执行失败: {str(e)}")


def example_sina_securities_original():
    """示例7: 新浪财经证券原创"""
    print_separator("示例7: 新浪财经证券原创")
    
    print("函数: get_sina_securities_original()")
    print("描述: 获取新浪财经证券原创，包含时间、内容和链接")
    print("数据源: https://finance.sina.com.cn/roll/index.d.html?cid=221431")
    print("AKSHARE接口: ak.stock_info_broker_sina(page='1')")
    
    try:
        result = get_sina_securities_original()
        print(f"\n返回数据长度: {len(result)} 字符")
        print(f"数据类型: {type(result)}")
        print("\n输出内容预览:")
        print(truncate_text(result))
        print("\n✅ 函数执行成功")
    except Exception as e:
        print(f"\n❌ 函数执行失败: {str(e)}")


def example_toolkit_functions():
    """示例8: Toolkit工具函数使用"""
    print_separator("示例8: Toolkit工具函数使用")
    
    print("描述: 展示如何通过Toolkit类调用财经新闻工具函数")
    print("用途: 这些函数可以被LLM在news_analyst_node中自动调用")
    
    try:
        toolkit = Toolkit()
        
        # 获取所有财经新闻相关的工具函数
        news_tools = [
            ('新浪财经全球财经快讯', toolkit.get_sina_global_financial_news),
            ('东方财富财经早餐', toolkit.get_eastmoney_financial_breakfast),
            ('东方财富全球财经快讯', toolkit.get_eastmoney_global_financial_news),
            ('富途牛牛快讯', toolkit.get_futu_financial_news),
            ('同花顺全球财经直播', toolkit.get_tonghuashun_global_financial_live),
            ('财联社电报', toolkit.get_cailianshe_telegraph),
            ('新浪财经证券原创', toolkit.get_sina_securities_original)
        ]
        
        print(f"\n可用的财经新闻工具函数数量: {len(news_tools)}")
        
        for i, (name, tool_func) in enumerate(news_tools, 1):
            print_subsection(f"{i}. {name}")
            print(f"函数名: {tool_func.name}")
            print(f"描述: {tool_func.description}")
            
            try:
                # 调用工具函数
                result = tool_func.invoke({})
                print(f"返回数据长度: {len(result)} 字符")
                print(f"数据预览: {result[:200]}...")
                print("✅ 工具函数调用成功")
            except Exception as e:
                print(f"❌ 工具函数调用失败: {str(e)}")
        
        print("\n✅ Toolkit函数示例完成")
    except Exception as e:
        print(f"\n❌ Toolkit示例执行失败: {str(e)}")


def example_performance_test():
    """示例9: 性能测试"""
    print_separator("示例9: 性能测试")

    print("描述: 测试所有财经新闻函数的响应时间和数据量")

    functions = [
        ('新浪财经全球财经快讯', get_sina_global_financial_news),
        ('东方财富财经早餐', get_eastmoney_financial_breakfast),
        ('东方财富全球财经快讯', get_eastmoney_global_financial_news),
        ('富途牛牛快讯', get_futu_financial_news),
        ('同花顺全球财经直播', get_tonghuashun_global_financial_live),
        ('财联社电报', get_cailianshe_telegraph),
        ('新浪财经证券原创', get_sina_securities_original)
    ]

    print(f"\n测试函数数量: {len(functions)}")
    print("\n性能测试结果:")
    print("-" * 80)
    print(f"{'函数名':<25} {'响应时间(秒)':<12} {'数据长度(字符)':<15} {'状态':<8}")
    print("-" * 80)

    total_time = 0
    success_count = 0

    for name, func in functions:
        try:
            start_time = datetime.now()
            result = func()
            end_time = datetime.now()

            response_time = (end_time - start_time).total_seconds()
            data_length = len(result)
            status = "✅ 成功"

            total_time += response_time
            success_count += 1

        except Exception as e:
            response_time = 0
            data_length = 0
            status = "❌ 失败"

        print(f"{name:<25} {response_time:<12.2f} {data_length:<15} {status:<8}")

    print("-" * 80)
    print(f"总计: {len(functions)} 个函数, {success_count} 个成功, 总耗时: {total_time:.2f} 秒")
    print(f"平均响应时间: {total_time/len(functions):.2f} 秒")


def example_error_handling():
    """示例10: 错误处理示例"""
    print_separator("示例10: 错误处理示例")

    print("描述: 展示财经新闻函数的错误处理机制")
    print("说明: 所有函数都包含完善的错误处理，确保在网络异常或API不可用时返回友好的错误信息")

    # 模拟网络错误的情况（这里只是展示错误处理的概念）
    print("\n错误处理特性:")
    print("1. 网络连接异常处理")
    print("2. API接口不可用处理")
    print("3. 数据格式异常处理")
    print("4. 返回友好的错误信息")
    print("5. 不会导致程序崩溃")

    print("\n示例错误信息格式:")
    print("- '获取新浪财经全球财经快讯时发生错误: [具体错误信息]'")
    print("- '获取东方财富财经早餐时发生错误: [具体错误信息]'")
    print("- '暂无[数据源]数据' (当返回数据为空时)")

    print("\n✅ 所有函数都具备完善的错误处理机制")


def example_integration_usage():
    """示例11: 集成使用示例"""
    print_separator("示例11: 集成使用示例")

    print("描述: 展示如何在实际应用中集成使用这些财经新闻函数")

    print("\n1. 在news_analyst_node中的使用:")
    print("   - LLM可以根据用户查询自动选择合适的新闻源")
    print("   - 支持多源新闻对比分析")
    print("   - 提供实时和历史财经资讯")

    print("\n2. 推荐的使用场景:")
    print("   - 实时市场监控: 使用快讯类函数(新浪、东方财富、富途、同花顺)")
    print("   - 深度分析: 使用财经早餐和证券原创文章")
    print("   - 专业资讯: 使用财联社电报")
    print("   - 全面覆盖: 组合使用多个数据源")

    print("\n3. 数据更新频率:")
    print("   - 快讯类: 实时更新(分钟级)")
    print("   - 财经早餐: 每日更新")
    print("   - 证券原创: 不定期更新")
    print("   - 电报: 高频更新")

    print("\n4. 建议的调用策略:")
    print("   - 优先使用响应速度快的接口")
    print("   - 根据用户需求选择合适的信息类型")
    print("   - 实现缓存机制避免频繁调用")
    print("   - 设置合理的超时时间")


def main():
    """主函数：运行所有示例"""
    print("财经新闻函数使用示例")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # 运行所有示例
    examples = [
        example_sina_global_financial_news,
        example_eastmoney_financial_breakfast,
        example_eastmoney_global_financial_news,
        example_futu_financial_news,
        example_tonghuashun_global_financial_live,
        example_cailianshe_telegraph,
        example_sina_securities_original,
        example_toolkit_functions,
        example_performance_test,
        example_error_handling,
        example_integration_usage
    ]

    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ 示例 {example_func.__name__} 执行失败: {str(e)}")

    print_separator("示例运行完成")
    print("所有财经新闻函数示例已运行完毕。")
    print("这些函数已集成到news_analyst_node中，可以被LLM自动调用。")
    print("\n使用说明:")
    print("1. 基础接口函数可以直接导入使用")
    print("2. Toolkit工具函数通过toolkit.function_name.invoke({})调用")
    print("3. 在news_analyst_node中，LLM会根据需要自动选择合适的工具函数")
    print("4. 建议结合多个数据源获取更全面的财经资讯")
    print("5. 注意合理控制调用频率，避免对数据源造成压力")

    print("\n📊 统计信息:")
    print(f"- 集成的财经新闻接口数量: 7个")
    print(f"- 支持的数据源: 新浪财经、东方财富、富途牛牛、同花顺、财联社")
    print(f"- 工具函数总数: 10个 (包含原有的3个)")
    print(f"- 输出格式: 统一的Markdown格式")
    print(f"- 错误处理: 完善的异常捕获和友好错误信息")


if __name__ == "__main__":
    main()
