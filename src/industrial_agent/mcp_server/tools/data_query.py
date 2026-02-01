"""Data query tools for MCP server.

This module provides a framework for data querying. Users can extend this
to connect to their specific data sources (databases, APIs, files, etc.).
"""

from datetime import datetime
from typing import Any, Optional

import pandas as pd
from mcp.server import Server
from mcp.types import TextContent, Tool

# Sample data for demonstration purposes
# Users should replace this with their actual data source
SAMPLE_DATA = pd.DataFrame(
    {
        "timestamp": pd.date_range(start="2024-01-01", periods=100, freq="H"),
        "product_id": [f"P{i % 10:03d}" for i in range(100)],
        "measurement_1": [10.0 + (i % 5) * 0.1 + (i % 3) * 0.05 for i in range(100)],
        "measurement_2": [5.0 + (i % 4) * 0.2 - (i % 2) * 0.1 for i in range(100)],
        "temperature": [25.0 + (i % 10) * 0.5 for i in range(100)],
        "humidity": [60.0 + (i % 8) * 1.0 for i in range(100)],
        "pass_fail": ["PASS" if i % 10 != 7 else "FAIL" for i in range(100)],
        "batch_id": [f"B{i // 10:03d}" for i in range(100)],
    }
)


class DataQueryManager:
    """Manager for data query operations.

    This class provides a framework for querying quality inspection data.
    Users can extend or modify this to connect to their specific data sources.
    """

    def __init__(self, data_source: Optional[pd.DataFrame] = None):
        """
        Initialize the data query manager.

        Args:
            data_source: Optional DataFrame to use as the data source.
                        Defaults to sample data for demonstration.
        """
        self._data = data_source if data_source is not None else SAMPLE_DATA.copy()

    def query_by_time_range(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        columns: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Query data within a time range.

        Args:
            start_time: Start time in ISO format (optional).
            end_time: End time in ISO format (optional).
            columns: List of columns to return (optional, returns all if not specified).

        Returns:
            Dictionary containing query results.
        """
        df = self._data.copy()

        if start_time:
            start = pd.to_datetime(start_time)
            df = df[df["timestamp"] >= start]

        if end_time:
            end = pd.to_datetime(end_time)
            df = df[df["timestamp"] <= end]

        if columns:
            available_cols = [c for c in columns if c in df.columns]
            if "timestamp" not in available_cols:
                available_cols.insert(0, "timestamp")
            df = df[available_cols]

        return {
            "row_count": len(df),
            "columns": list(df.columns),
            "data": df.to_dict(orient="records"),
        }

    def query_by_product(
        self,
        product_id: str,
        columns: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Query data for a specific product.

        Args:
            product_id: The product ID to filter by.
            columns: List of columns to return (optional).

        Returns:
            Dictionary containing query results.
        """
        df = self._data[self._data["product_id"] == product_id].copy()

        if columns:
            available_cols = [c for c in columns if c in df.columns]
            df = df[available_cols]

        return {
            "product_id": product_id,
            "row_count": len(df),
            "columns": list(df.columns),
            "data": df.to_dict(orient="records"),
        }

    def query_by_batch(
        self,
        batch_id: str,
        columns: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Query data for a specific batch.

        Args:
            batch_id: The batch ID to filter by.
            columns: List of columns to return (optional).

        Returns:
            Dictionary containing query results.
        """
        df = self._data[self._data["batch_id"] == batch_id].copy()

        if columns:
            available_cols = [c for c in columns if c in df.columns]
            df = df[available_cols]

        return {
            "batch_id": batch_id,
            "row_count": len(df),
            "columns": list(df.columns),
            "data": df.to_dict(orient="records"),
        }

    def get_available_columns(self) -> list[str]:
        """Get list of available data columns."""
        return list(self._data.columns)

    def get_data_summary(self) -> dict[str, Any]:
        """Get a summary of the available data."""
        return {
            "total_rows": len(self._data),
            "columns": list(self._data.columns),
            "time_range": {
                "start": str(self._data["timestamp"].min()),
                "end": str(self._data["timestamp"].max()),
            },
            "unique_products": self._data["product_id"].nunique(),
            "unique_batches": self._data["batch_id"].nunique(),
        }


# Global query manager instance
_query_manager: Optional[DataQueryManager] = None


def get_query_manager() -> DataQueryManager:
    """Get or create the global query manager instance."""
    global _query_manager
    if _query_manager is None:
        _query_manager = DataQueryManager()
    return _query_manager


def register_query_tools(server: Server) -> None:
    """Register data query tools with the MCP server."""

    @server.list_tools()
    async def list_query_tools() -> list[Tool]:
        """List available query tools."""
        return [
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
        ]

    @server.call_tool()
    async def handle_query_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle query tool calls."""
        manager = get_query_manager()

        if name == "query_by_time_range":
            result = manager.query_by_time_range(
                start_time=arguments.get("start_time"),
                end_time=arguments.get("end_time"),
                columns=arguments.get("columns"),
            )
        elif name == "query_by_product":
            result = manager.query_by_product(
                product_id=arguments["product_id"],
                columns=arguments.get("columns"),
            )
        elif name == "query_by_batch":
            result = manager.query_by_batch(
                batch_id=arguments["batch_id"],
                columns=arguments.get("columns"),
            )
        elif name == "get_data_summary":
            result = manager.get_data_summary()
        elif name == "get_available_columns":
            result = {"columns": manager.get_available_columns()}
        else:
            result = {"error": f"Unknown tool: {name}"}

        import json

        return [TextContent(type="text", text=json.dumps(result, default=str, indent=2))]
