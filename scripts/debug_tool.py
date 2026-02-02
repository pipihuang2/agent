"""MCP 工具调试脚本

用法:
    uv run python scripts/debug_tool.py

可以在这里测试任何工具，修改下面的代码来调试不同的工具。
"""

import json
import sys
sys.path.insert(0, "src")

from industrial_agent.mcp_server.tools.data_analysis import get_analyzer
from industrial_agent.mcp_server.tools.chart_generator import get_chart_generator
from industrial_agent.mcp_server.tools.data_query import get_query_manager
from industrial_agent.mcp_server.tools.mysql_query import get_mysql_manager


def print_result(name: str, result: dict):
    """格式化打印结果"""
    print(f"\n{'='*60}")
    print(f"工具: {name}")
    print('='*60)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


def test_data_query():
    """测试数据查询工具"""
    manager = get_query_manager()

    # 获取数据摘要
    result = manager.get_data_summary()
    print_result("get_data_summary", result)

    # 按时间查询
    result = manager.query_by_time_range(
        start_time="2024-01-01T00:00:00",
        end_time="2024-01-01T05:00:00",
        columns=["timestamp", "product_id", "measurement_1"]
    )
    print_result("query_by_time_range", result)

    # 按产品查询
    result = manager.query_by_product(product_id="P001")
    print_result("query_by_product", result)


def test_data_analysis():
    """测试数据分析工具"""
    analyzer = get_analyzer()

    # 测试数据
    data = [10.1, 10.2, 10.3, 9.8, 10.5, 10.0, 10.4, 9.9, 10.1, 10.2,
            10.0, 10.3, 10.1, 10.2, 9.7, 10.4, 10.2, 10.1, 10.3, 10.0]

    # 统计分析
    result = analyzer.statistical_analysis(
        data=data,
        include_capability=True,
        usl=11.0,
        lsl=9.0
    )
    print_result("statistical_analysis", result)

    # 异常检测
    data_with_outlier = data + [15.0, 5.0]  # 添加异常值
    result = analyzer.anomaly_detection(
        data=data_with_outlier,
        method="zscore",
        threshold=2.0
    )
    print_result("anomaly_detection (zscore)", result)

    # 趋势分析
    trend_data = [10.0 + i * 0.1 for i in range(20)]  # 上升趋势
    result = analyzer.trend_analysis(data=trend_data)
    print_result("trend_analysis", result)

    # 控制图分析
    result = analyzer.control_chart_analysis(
        data=data,
        chart_type="individuals"
    )
    print_result("control_chart_analysis", result)


def test_chart_generator():
    """测试图表生成工具"""
    generator = get_chart_generator()

    # 折线图
    result = generator.line_chart(
        data=[10.1, 10.2, 10.3, 9.8, 10.5, 10.0],
        title="测试折线图",
        xlabel="样本",
        ylabel="测量值",
        show_trend=True
    )
    print_result("generate_line_chart", result)

    # 柱状图
    result = generator.bar_chart(
        data=[85, 90, 78, 92, 88],
        labels=["产品A", "产品B", "产品C", "产品D", "产品E"],
        title="产品合格率对比"
    )
    print_result("generate_bar_chart", result)

    # 直方图
    import numpy as np
    data = np.random.normal(10, 0.5, 100).tolist()
    result = generator.histogram(
        data=data,
        bins=15,
        title="测量值分布",
        show_normal=True,
        usl=11.0,
        lsl=9.0
    )
    print_result("generate_histogram", result)

    # 控制图
    result = generator.control_chart(
        data=[10.1, 10.2, 10.3, 9.8, 10.5, 10.0, 10.8, 10.1],
        ucl=10.6,
        lcl=9.6,
        cl=10.1,
        title="I-MR 控制图"
    )
    print_result("generate_control_chart", result)


def test_mysql():
    """测试 MySQL 工具（需要配置数据库连接）"""
    manager = get_mysql_manager()

    # 测试连接
    result = manager.test_connection()
    print_result("mysql_test_connection", result)

    if result.get("success"):
        # 列出表
        result = manager.list_tables()
        print_result("mysql_list_tables", result)


def main():
    """主函数 - 选择要测试的工具"""
    print("=" * 60)
    print("MCP 工具调试脚本")
    print("=" * 60)
    print("\n选择要测试的工具类别:")
    print("1. 数据查询工具 (data_query)")
    print("2. 数据分析工具 (data_analysis)")
    print("3. 图表生成工具 (chart_generator)")
    print("4. MySQL 数据库工具 (mysql)")
    print("5. 全部测试")
    print("0. 退出")

    choice = input("\n请输入选项 (0-5): ").strip()

    if choice == "1":
        test_data_query()
    elif choice == "2":
        test_data_analysis()
    elif choice == "3":
        test_chart_generator()
    elif choice == "4":
        test_mysql()
    elif choice == "5":
        test_data_query()
        test_data_analysis()
        test_chart_generator()
        test_mysql()
    elif choice == "0":
        print("退出")
    else:
        print("无效选项")


if __name__ == "__main__":
    main()
