"""单个工具快速测试

用法: uv run python scripts/test_single_tool.py

直接修改下面的代码来测试你想调试的工具
"""

import json
import sys
sys.path.insert(0, "src")


def main():
    # ============================================================
    # 在这里修改你要测试的工具
    # ============================================================

    # --- 示例 1: 统计分析 ---
    from industrial_agent.mcp_server.tools.data_analysis import get_analyzer
    analyzer = get_analyzer()

    result = analyzer.statistical_analysis(
        data=[10.1, 10.2, 10.3, 9.8, 10.5, 10.0, 10.4, 9.9],
        include_capability=True,
        usl=11.0,
        lsl=9.0
    )

    # --- 示例 2: 异常检测 ---
    # result = analyzer.anomaly_detection(
    #     data=[10, 10, 10, 10, 50, 10, 10],  # 50 是异常值
    #     method="zscore",
    #     threshold=2.0
    # )

    # --- 示例 3: 生成图表 ---
    # from industrial_agent.mcp_server.tools.chart_generator import get_chart_generator
    # generator = get_chart_generator()
    # result = generator.line_chart(
    #     data=[1, 2, 3, 4, 5, 4, 3],
    #     title="我的测试图表",
    #     show_trend=True
    # )

    # --- 示例 4: 数据查询 ---
    # from industrial_agent.mcp_server.tools.data_query import get_query_manager
    # manager = get_query_manager()
    # result = manager.get_data_summary()

    # --- 示例 5: MySQL 查询 ---
    # from industrial_agent.mcp_server.tools.mysql_query import get_mysql_manager
    # manager = get_mysql_manager()
    # result = manager.test_connection()

    # ============================================================
    # 打印结果
    # ============================================================
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
