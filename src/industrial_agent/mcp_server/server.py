"""MCP Server implementation for industrial quality inspection tools.

This module provides the main MCP server that exposes data query,
analysis, and chart generation capabilities.
"""

import asyncio
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .tools import register_analysis_tools, register_chart_tools, register_query_tools


def create_server() -> Server:
    """Create and configure the MCP server with all tools."""
    server = Server("industrial-quality-agent")

    # Store all tools from different modules
    all_tools: list[Tool] = []
    tool_handlers: dict[str, Any] = {}

    # We need to manually aggregate tools since MCP doesn't support multiple handlers
    # Let's create a unified approach

    return server


class IndustrialMCPServer:
    """Industrial quality inspection MCP server."""

    def __init__(self):
        """Initialize the MCP server."""
        self.server = Server("industrial-quality-agent")
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Set up all tool handlers."""
        from .tools.chart_generator import ChartGenerator, get_chart_generator
        from .tools.data_analysis import DataAnalyzer, get_analyzer
        from .tools.data_query import DataQueryManager, get_query_manager
        from .tools.mysql_query import MySQLQueryManager, get_mysql_manager

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List all available tools."""
            return [
                # Data Query Tools
                Tool(
                    name="query_by_time_range",
                    description="Query quality inspection data within a specified time range",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "start_time": {
                                "type": "string",
                                "description": "Start time in ISO format (e.g., 2024-01-01T00:00:00)",
                            },
                            "end_time": {
                                "type": "string",
                                "description": "End time in ISO format",
                            },
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of columns to return",
                            },
                        },
                    },
                ),
                Tool(
                    name="query_by_product",
                    description="Query quality inspection data for a specific product",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "product_id": {
                                "type": "string",
                                "description": "The product ID to query",
                            },
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of columns to return",
                            },
                        },
                        "required": ["product_id"],
                    },
                ),
                Tool(
                    name="query_by_batch",
                    description="Query quality inspection data for a specific batch",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "batch_id": {
                                "type": "string",
                                "description": "The batch ID to query",
                            },
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of columns to return",
                            },
                        },
                        "required": ["batch_id"],
                    },
                ),
                Tool(
                    name="get_data_summary",
                    description="Get a summary of available quality inspection data",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                Tool(
                    name="get_available_columns",
                    description="Get list of available data columns",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                # Data Analysis Tools
                Tool(
                    name="statistical_analysis",
                    description="Perform statistical analysis on quality data including mean, std, percentiles, and process capability",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "List of numerical values to analyze",
                            },
                            "include_capability": {
                                "type": "boolean",
                                "description": "Whether to include process capability metrics",
                                "default": False,
                            },
                            "usl": {
                                "type": "number",
                                "description": "Upper specification limit",
                            },
                            "lsl": {
                                "type": "number",
                                "description": "Lower specification limit",
                            },
                        },
                        "required": ["data"],
                    },
                ),
                Tool(
                    name="anomaly_detection",
                    description="Detect anomalies in quality data using z-score or IQR method",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "List of numerical values",
                            },
                            "method": {
                                "type": "string",
                                "enum": ["zscore", "iqr"],
                                "description": "Detection method",
                                "default": "zscore",
                            },
                            "threshold": {
                                "type": "number",
                                "description": "Threshold for anomaly detection",
                                "default": 3.0,
                            },
                        },
                        "required": ["data"],
                    },
                ),
                Tool(
                    name="trend_analysis",
                    description="Analyze trends in time series quality data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "List of numerical values in time order",
                            },
                            "timestamps": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional list of timestamps",
                            },
                        },
                        "required": ["data"],
                    },
                ),
                Tool(
                    name="control_chart_analysis",
                    description="Perform control chart analysis (I-MR or X-bar)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "List of numerical values",
                            },
                            "chart_type": {
                                "type": "string",
                                "enum": ["individuals", "xbar"],
                                "description": "Type of control chart",
                                "default": "individuals",
                            },
                            "subgroup_size": {
                                "type": "integer",
                                "description": "Size of subgroups for X-bar chart",
                                "default": 5,
                            },
                        },
                        "required": ["data"],
                    },
                ),
                # Chart Generation Tools
                Tool(
                    name="generate_line_chart",
                    description="Generate a line chart for time series or sequential data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "List of values to plot",
                            },
                            "labels": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "X-axis labels",
                            },
                            "title": {"type": "string", "description": "Chart title"},
                            "xlabel": {"type": "string", "description": "X-axis label"},
                            "ylabel": {"type": "string", "description": "Y-axis label"},
                            "show_trend": {
                                "type": "boolean",
                                "description": "Show trend line",
                                "default": False,
                            },
                        },
                        "required": ["data"],
                    },
                ),
                Tool(
                    name="generate_bar_chart",
                    description="Generate a bar chart for categorical data comparison",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "List of values",
                            },
                            "labels": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Category labels",
                            },
                            "title": {"type": "string", "description": "Chart title"},
                            "xlabel": {"type": "string", "description": "X-axis label"},
                            "ylabel": {"type": "string", "description": "Y-axis label"},
                            "horizontal": {
                                "type": "boolean",
                                "description": "Create horizontal bars",
                                "default": False,
                            },
                        },
                        "required": ["data", "labels"],
                    },
                ),
                Tool(
                    name="generate_scatter_plot",
                    description="Generate a scatter plot for correlation analysis",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "x_data": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "X-axis values",
                            },
                            "y_data": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Y-axis values",
                            },
                            "title": {"type": "string", "description": "Chart title"},
                            "xlabel": {"type": "string", "description": "X-axis label"},
                            "ylabel": {"type": "string", "description": "Y-axis label"},
                            "show_regression": {
                                "type": "boolean",
                                "description": "Show regression line",
                                "default": False,
                            },
                        },
                        "required": ["x_data", "y_data"],
                    },
                ),
                Tool(
                    name="generate_heatmap",
                    description="Generate a heatmap for matrix data visualization",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                },
                                "description": "2D array of values",
                            },
                            "x_labels": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Column labels",
                            },
                            "y_labels": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Row labels",
                            },
                            "title": {"type": "string", "description": "Chart title"},
                            "colorbar_label": {
                                "type": "string",
                                "description": "Colorbar label",
                            },
                        },
                        "required": ["data"],
                    },
                ),
                Tool(
                    name="generate_control_chart",
                    description="Generate a control chart with UCL, LCL, and center line",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "List of values to plot",
                            },
                            "ucl": {
                                "type": "number",
                                "description": "Upper control limit",
                            },
                            "lcl": {
                                "type": "number",
                                "description": "Lower control limit",
                            },
                            "cl": {"type": "number", "description": "Center line"},
                            "title": {"type": "string", "description": "Chart title"},
                            "ylabel": {"type": "string", "description": "Y-axis label"},
                            "labels": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "X-axis labels",
                            },
                        },
                        "required": ["data", "ucl", "lcl", "cl"],
                    },
                ),
                Tool(
                    name="generate_histogram",
                    description="Generate a histogram for data distribution analysis",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "List of values",
                            },
                            "bins": {
                                "type": "integer",
                                "description": "Number of bins",
                                "default": 20,
                            },
                            "title": {"type": "string", "description": "Chart title"},
                            "xlabel": {"type": "string", "description": "X-axis label"},
                            "ylabel": {"type": "string", "description": "Y-axis label"},
                            "show_normal": {
                                "type": "boolean",
                                "description": "Show normal distribution overlay",
                                "default": False,
                            },
                            "usl": {
                                "type": "number",
                                "description": "Upper specification limit",
                            },
                            "lsl": {
                                "type": "number",
                                "description": "Lower specification limit",
                            },
                        },
                        "required": ["data"],
                    },
                ),
                # Defect Heatmap Tool
                Tool(
                    name="generate_defect_heatmap",
                    description="Generate a defect heatmap overlay on a product image. Takes NG (defect) pixel coordinates and creates a heatmap showing defect density - redder areas indicate higher defect frequency.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image_path": {
                                "type": "string",
                                "description": "Path to the product image file (supports PNG, JPG, etc.)",
                            },
                            "ng_coordinates": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "x": {"type": "integer", "description": "X pixel coordinate"},
                                        "y": {"type": "integer", "description": "Y pixel coordinate"},
                                    },
                                    "required": ["x", "y"],
                                },
                                "description": "List of NG (defect) pixel coordinates. Example: [{\"x\": 100, \"y\": 200}, {\"x\": 150, \"y\": 250}]",
                            },
                            "title": {
                                "type": "string",
                                "description": "Chart title",
                                "default": "Defect Heatmap",
                            },
                            "sigma": {
                                "type": "number",
                                "description": "Gaussian blur sigma for smoothing. Higher values create smoother heat regions.",
                                "default": 20.0,
                            },
                            "alpha": {
                                "type": "number",
                                "description": "Heatmap transparency (0.0-1.0). Lower values make the heatmap more transparent.",
                                "default": 0.6,
                            },
                            "colorbar_label": {
                                "type": "string",
                                "description": "Label for the colorbar",
                                "default": "Defect Density",
                            },
                        },
                        "required": ["image_path", "ng_coordinates"],
                    },
                ),
                # MySQL Database Tools
                Tool(
                    name="mysql_test_connection",
                    description="Test the MySQL database connection",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                Tool(
                    name="mysql_list_tables",
                    description="List all tables in the MySQL database",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                Tool(
                    name="mysql_describe_table",
                    description="Get the schema/structure of a MySQL table (column names, types, keys, etc.)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "Name of the table to describe",
                            },
                        },
                        "required": ["table_name"],
                    },
                ),
                Tool(
                    name="mysql_execute_query",
                    description="Execute a SELECT SQL query on the MySQL database. Only SELECT queries are allowed for security.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "SQL SELECT query to execute",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of rows to return",
                                "default": 1000,
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="mysql_query_table",
                    description="Query a MySQL table with optional filtering and sorting",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "Name of the table to query",
                            },
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of columns to select (omit for all columns)",
                            },
                            "where": {
                                "type": "string",
                                "description": "WHERE clause conditions (without 'WHERE' keyword). Example: \"status = 'FAIL' AND temperature > 30\"",
                            },
                            "order_by": {
                                "type": "string",
                                "description": "ORDER BY clause (without 'ORDER BY' keyword). Example: \"timestamp DESC\"",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of rows to return",
                                "default": 1000,
                            },
                        },
                        "required": ["table_name"],
                    },
                ),
                Tool(
                    name="mysql_get_table_stats",
                    description="Get statistical summary (count, mean, min, max, std) for a numeric column in a MySQL table",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "Name of the table",
                            },
                            "column": {
                                "type": "string",
                                "description": "Name of the numeric column to analyze",
                            },
                        },
                        "required": ["table_name", "column"],
                    },
                ),
                Tool(
                    name="mysql_get_column_values",
                    description="Get all values from a specific column (useful for feeding into analysis tools)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "Name of the table",
                            },
                            "column": {
                                "type": "string",
                                "description": "Name of the column",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of values to return",
                                "default": 10000,
                            },
                        },
                        "required": ["table_name", "column"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Handle tool calls."""
            query_manager = get_query_manager()
            analyzer = get_analyzer()
            chart_generator = get_chart_generator()

            result: dict[str, Any] = {}

            # Data Query Tools
            if name == "query_by_time_range":
                result = query_manager.query_by_time_range(
                    start_time=arguments.get("start_time"),
                    end_time=arguments.get("end_time"),
                    columns=arguments.get("columns"),
                )
            elif name == "query_by_product":
                result = query_manager.query_by_product(
                    product_id=arguments["product_id"],
                    columns=arguments.get("columns"),
                )
            elif name == "query_by_batch":
                result = query_manager.query_by_batch(
                    batch_id=arguments["batch_id"],
                    columns=arguments.get("columns"),
                )
            elif name == "get_data_summary":
                result = query_manager.get_data_summary()
            elif name == "get_available_columns":
                result = {"columns": query_manager.get_available_columns()}

            # Data Analysis Tools
            elif name == "statistical_analysis":
                result = analyzer.statistical_analysis(
                    data=arguments["data"],
                    include_capability=arguments.get("include_capability", False),
                    usl=arguments.get("usl"),
                    lsl=arguments.get("lsl"),
                )
            elif name == "anomaly_detection":
                result = analyzer.anomaly_detection(
                    data=arguments["data"],
                    method=arguments.get("method", "zscore"),
                    threshold=arguments.get("threshold", 3.0),
                )
            elif name == "trend_analysis":
                result = analyzer.trend_analysis(
                    data=arguments["data"],
                    timestamps=arguments.get("timestamps"),
                )
            elif name == "control_chart_analysis":
                result = analyzer.control_chart_analysis(
                    data=arguments["data"],
                    chart_type=arguments.get("chart_type", "individuals"),
                    subgroup_size=arguments.get("subgroup_size", 5),
                )

            # Chart Generation Tools
            elif name == "generate_line_chart":
                result = chart_generator.line_chart(
                    data=arguments["data"],
                    labels=arguments.get("labels"),
                    title=arguments.get("title", "Line Chart"),
                    xlabel=arguments.get("xlabel", "Index"),
                    ylabel=arguments.get("ylabel", "Value"),
                    show_trend=arguments.get("show_trend", False),
                )
            elif name == "generate_bar_chart":
                result = chart_generator.bar_chart(
                    data=arguments["data"],
                    labels=arguments["labels"],
                    title=arguments.get("title", "Bar Chart"),
                    xlabel=arguments.get("xlabel", "Category"),
                    ylabel=arguments.get("ylabel", "Value"),
                    horizontal=arguments.get("horizontal", False),
                )
            elif name == "generate_scatter_plot":
                result = chart_generator.scatter_plot(
                    x_data=arguments["x_data"],
                    y_data=arguments["y_data"],
                    title=arguments.get("title", "Scatter Plot"),
                    xlabel=arguments.get("xlabel", "X"),
                    ylabel=arguments.get("ylabel", "Y"),
                    show_regression=arguments.get("show_regression", False),
                )
            elif name == "generate_heatmap":
                result = chart_generator.heatmap(
                    data=arguments["data"],
                    x_labels=arguments.get("x_labels"),
                    y_labels=arguments.get("y_labels"),
                    title=arguments.get("title", "Heatmap"),
                    colorbar_label=arguments.get("colorbar_label", "Value"),
                )
            elif name == "generate_control_chart":
                result = chart_generator.control_chart(
                    data=arguments["data"],
                    ucl=arguments["ucl"],
                    lcl=arguments["lcl"],
                    cl=arguments["cl"],
                    title=arguments.get("title", "Control Chart"),
                    ylabel=arguments.get("ylabel", "Value"),
                    labels=arguments.get("labels"),
                )
            elif name == "generate_histogram":
                result = chart_generator.histogram(
                    data=arguments["data"],
                    bins=arguments.get("bins", 20),
                    title=arguments.get("title", "Histogram"),
                    xlabel=arguments.get("xlabel", "Value"),
                    ylabel=arguments.get("ylabel", "Frequency"),
                    show_normal=arguments.get("show_normal", False),
                    usl=arguments.get("usl"),
                    lsl=arguments.get("lsl"),
                )
            elif name == "generate_defect_heatmap":
                result = chart_generator.defect_heatmap(
                    image_path=arguments["image_path"],
                    ng_coordinates=arguments["ng_coordinates"],
                    title=arguments.get("title", "Defect Heatmap"),
                    sigma=arguments.get("sigma", 20.0),
                    alpha=arguments.get("alpha", 0.6),
                    colorbar_label=arguments.get("colorbar_label", "Defect Density"),
                )

            # MySQL Database Tools
            elif name == "mysql_test_connection":
                mysql_manager = get_mysql_manager()
                result = mysql_manager.test_connection()
            elif name == "mysql_list_tables":
                mysql_manager = get_mysql_manager()
                result = mysql_manager.list_tables()
            elif name == "mysql_describe_table":
                mysql_manager = get_mysql_manager()
                result = mysql_manager.describe_table(
                    table_name=arguments["table_name"],
                )
            elif name == "mysql_execute_query":
                mysql_manager = get_mysql_manager()
                result = mysql_manager.execute_query(
                    query=arguments["query"],
                    limit=arguments.get("limit", 1000),
                )
            elif name == "mysql_query_table":
                mysql_manager = get_mysql_manager()
                result = mysql_manager.query_table(
                    table_name=arguments["table_name"],
                    columns=arguments.get("columns"),
                    where=arguments.get("where"),
                    order_by=arguments.get("order_by"),
                    limit=arguments.get("limit", 1000),
                )
            elif name == "mysql_get_table_stats":
                mysql_manager = get_mysql_manager()
                result = mysql_manager.get_table_stats(
                    table_name=arguments["table_name"],
                    column=arguments["column"],
                )
            elif name == "mysql_get_column_values":
                mysql_manager = get_mysql_manager()
                result = mysql_manager.get_column_values(
                    table_name=arguments["table_name"],
                    column=arguments["column"],
                    limit=arguments.get("limit", 10000),
                )
            else:
                result = {"error": f"Unknown tool: {name}"}

            return [TextContent(type="text", text=json.dumps(result, default=str, indent=2))]

    async def run(self) -> None:
        """Run the MCP server using stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


async def main() -> None:
    """Main entry point for the MCP server."""
    server = IndustrialMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
