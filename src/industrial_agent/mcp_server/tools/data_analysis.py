"""Data analysis tools for MCP server.

This module provides statistical analysis, anomaly detection, and trend analysis
tools for industrial quality inspection data.
"""

import json
from typing import Any, Optional

import numpy as np
import pandas as pd
from mcp.server import Server
from mcp.types import TextContent, Tool


class DataAnalyzer:
    """Analyzer for quality inspection data."""

    def statistical_analysis(
        self,
        data: list[float],
        include_capability: bool = False,
        usl: Optional[float] = None,
        lsl: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Perform statistical analysis on data.

        Args:
            data: List of numerical values to analyze.
            include_capability: Whether to include process capability metrics.
            usl: Upper specification limit (required for capability analysis).
            lsl: Lower specification limit (required for capability analysis).

        Returns:
            Dictionary containing statistical metrics.
        """
        arr = np.array(data)
        n = len(arr)

        result = {
            "count": n,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if n > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "q1": float(np.percentile(arr, 25)),
            "q3": float(np.percentile(arr, 75)),
            "range": float(np.max(arr) - np.min(arr)),
            "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        }

        # Add skewness and kurtosis if enough data
        if n >= 3:
            mean = np.mean(arr)
            std = np.std(arr, ddof=1)
            if std > 0:
                result["skewness"] = float(
                    np.mean(((arr - mean) / std) ** 3) * n / ((n - 1) * (n - 2))
                    if n > 2
                    else 0
                )
                result["kurtosis"] = float(
                    np.mean(((arr - mean) / std) ** 4) - 3 if n > 3 else 0
                )

        # Process capability analysis
        if include_capability and usl is not None and lsl is not None:
            mean = np.mean(arr)
            std = np.std(arr, ddof=1)
            if std > 0:
                cp = (usl - lsl) / (6 * std)
                cpu = (usl - mean) / (3 * std)
                cpl = (mean - lsl) / (3 * std)
                cpk = min(cpu, cpl)

                result["capability"] = {
                    "cp": float(cp),
                    "cpk": float(cpk),
                    "cpu": float(cpu),
                    "cpl": float(cpl),
                    "within_spec_percentage": float(
                        np.sum((arr >= lsl) & (arr <= usl)) / n * 100
                    ),
                }

        return result

    def anomaly_detection(
        self,
        data: list[float],
        method: str = "zscore",
        threshold: float = 3.0,
    ) -> dict[str, Any]:
        """
        Detect anomalies in data.

        Args:
            data: List of numerical values.
            method: Detection method ('zscore' or 'iqr').
            threshold: Threshold for anomaly detection.

        Returns:
            Dictionary containing anomaly information.
        """
        arr = np.array(data)
        n = len(arr)

        if method == "zscore":
            mean = np.mean(arr)
            std = np.std(arr, ddof=1)
            if std > 0:
                z_scores = np.abs((arr - mean) / std)
                anomaly_mask = z_scores > threshold
            else:
                z_scores = np.zeros_like(arr)
                anomaly_mask = np.zeros(n, dtype=bool)

            anomalies = [
                {
                    "index": int(i),
                    "value": float(arr[i]),
                    "z_score": float(z_scores[i]),
                }
                for i in np.where(anomaly_mask)[0]
            ]

            return {
                "method": "z-score",
                "threshold": threshold,
                "total_points": n,
                "anomaly_count": int(np.sum(anomaly_mask)),
                "anomaly_percentage": float(np.sum(anomaly_mask) / n * 100),
                "anomalies": anomalies,
            }

        elif method == "iqr":
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            anomaly_mask = (arr < lower_bound) | (arr > upper_bound)

            anomalies = [
                {
                    "index": int(i),
                    "value": float(arr[i]),
                    "deviation": "below" if arr[i] < lower_bound else "above",
                }
                for i in np.where(anomaly_mask)[0]
            ]

            return {
                "method": "IQR",
                "threshold_multiplier": threshold,
                "bounds": {
                    "lower": float(lower_bound),
                    "upper": float(upper_bound),
                },
                "total_points": n,
                "anomaly_count": int(np.sum(anomaly_mask)),
                "anomaly_percentage": float(np.sum(anomaly_mask) / n * 100),
                "anomalies": anomalies,
            }

        else:
            return {"error": f"Unknown method: {method}"}

    def trend_analysis(
        self,
        data: list[float],
        timestamps: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Analyze trends in time series data.

        Args:
            data: List of numerical values.
            timestamps: Optional list of timestamps.

        Returns:
            Dictionary containing trend information.
        """
        arr = np.array(data)
        n = len(arr)
        x = np.arange(n)

        # Linear regression for trend
        if n > 1:
            slope, intercept = np.polyfit(x, arr, 1)
            predicted = slope * x + intercept
            residuals = arr - predicted
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((arr - np.mean(arr)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            slope = 0.0
            intercept = float(arr[0]) if n > 0 else 0.0
            r_squared = 0.0

        # Calculate moving average
        window_size = min(5, n)
        if n >= window_size:
            moving_avg = np.convolve(arr, np.ones(window_size) / window_size, mode="valid")
            moving_avg_list = [float(v) for v in moving_avg]
        else:
            moving_avg_list = []

        # Detect trend direction
        if n > 1:
            first_half_mean = np.mean(arr[: n // 2])
            second_half_mean = np.mean(arr[n // 2 :])
            if second_half_mean > first_half_mean * 1.05:
                trend_direction = "increasing"
            elif second_half_mean < first_half_mean * 0.95:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "insufficient_data"

        return {
            "data_points": n,
            "trend_direction": trend_direction,
            "linear_trend": {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_squared),
            },
            "moving_average": {
                "window_size": window_size,
                "values": moving_avg_list,
            },
            "period_comparison": {
                "first_half_mean": float(np.mean(arr[: n // 2])) if n > 1 else None,
                "second_half_mean": float(np.mean(arr[n // 2 :])) if n > 1 else None,
            },
        }

    def control_chart_analysis(
        self,
        data: list[float],
        chart_type: str = "xbar",
        subgroup_size: int = 5,
    ) -> dict[str, Any]:
        """
        Perform control chart analysis.

        Args:
            data: List of numerical values.
            chart_type: Type of control chart ('xbar', 'individuals', 'range').
            subgroup_size: Size of subgroups for X-bar charts.

        Returns:
            Dictionary containing control chart metrics.
        """
        arr = np.array(data)
        n = len(arr)

        if chart_type == "individuals":
            # Individual X chart (I-MR)
            mean = np.mean(arr)
            if n > 1:
                mr = np.abs(np.diff(arr))
                mr_bar = np.mean(mr)
                sigma_est = mr_bar / 1.128  # d2 for n=2
            else:
                mr_bar = 0
                sigma_est = 0

            ucl = mean + 3 * sigma_est
            lcl = mean - 3 * sigma_est

            out_of_control = [
                {"index": int(i), "value": float(arr[i]), "type": "out_of_limits"}
                for i in range(n)
                if arr[i] > ucl or arr[i] < lcl
            ]

            return {
                "chart_type": "Individuals (I-MR)",
                "center_line": float(mean),
                "ucl": float(ucl),
                "lcl": float(lcl),
                "mr_bar": float(mr_bar),
                "sigma_estimate": float(sigma_est),
                "out_of_control_points": out_of_control,
                "in_control": len(out_of_control) == 0,
            }

        elif chart_type == "xbar":
            # X-bar chart
            if n < subgroup_size:
                return {"error": "Insufficient data for subgroup analysis"}

            num_subgroups = n // subgroup_size
            subgroups = arr[: num_subgroups * subgroup_size].reshape(
                num_subgroups, subgroup_size
            )
            subgroup_means = np.mean(subgroups, axis=1)
            subgroup_ranges = np.ptp(subgroups, axis=1)

            xbar = np.mean(subgroup_means)
            rbar = np.mean(subgroup_ranges)

            # A2 factors for different subgroup sizes
            a2_factors = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483}
            a2 = a2_factors.get(subgroup_size, 0.577)

            ucl = xbar + a2 * rbar
            lcl = xbar - a2 * rbar

            out_of_control = [
                {
                    "subgroup": int(i),
                    "mean": float(subgroup_means[i]),
                    "type": "out_of_limits",
                }
                for i in range(num_subgroups)
                if subgroup_means[i] > ucl or subgroup_means[i] < lcl
            ]

            return {
                "chart_type": "X-bar",
                "subgroup_size": subgroup_size,
                "num_subgroups": num_subgroups,
                "center_line": float(xbar),
                "ucl": float(ucl),
                "lcl": float(lcl),
                "r_bar": float(rbar),
                "subgroup_means": [float(m) for m in subgroup_means],
                "out_of_control_points": out_of_control,
                "in_control": len(out_of_control) == 0,
            }

        else:
            return {"error": f"Unknown chart type: {chart_type}"}


# Global analyzer instance
_analyzer: Optional[DataAnalyzer] = None


def get_analyzer() -> DataAnalyzer:
    """Get or create the global analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = DataAnalyzer()
    return _analyzer


def register_analysis_tools(server: Server) -> None:
    """Register data analysis tools with the MCP server."""

    @server.list_tools()
    async def list_analysis_tools() -> list[Tool]:
        """List available analysis tools."""
        return [
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
                            "description": "Threshold for anomaly detection (3.0 for z-score, 1.5 for IQR)",
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
        ]

    @server.call_tool()
    async def handle_analysis_tool(
        name: str, arguments: dict[str, Any]
    ) -> list[TextContent]:
        """Handle analysis tool calls."""
        analyzer = get_analyzer()

        if name == "statistical_analysis":
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
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]
