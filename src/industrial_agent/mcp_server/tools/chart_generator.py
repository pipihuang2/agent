"""Chart generation tools for MCP server.

This module provides visualization tools for quality inspection data,
including line charts, bar charts, scatter plots, heatmaps, and control charts.
"""

import base64
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy.ndimage import gaussian_filter
from mcp.server import Server
from mcp.types import TextContent, Tool

from industrial_agent.config import get_settings


class ChartGenerator:
    """Generator for quality inspection visualization charts."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the chart generator.

        Args:
            output_dir: Directory to save generated charts.
        """
        if output_dir is None:
            settings = get_settings()
            output_dir = settings.chart_output_dir

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set default style
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["font.family"] = ["DejaVu Sans", "SimHei", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False

    def _save_chart(
        self,
        fig: plt.Figure,
        name: str,
        return_base64: bool = False,
    ) -> dict[str, Any]:
        """Save chart to file and optionally return as base64."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.png"
        filepath = self.output_dir / filename

        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")

        result = {
            "success": True,
            "filepath": str(filepath),
            "filename": filename,
        }

        if return_base64:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
            buf.seek(0)
            result["base64"] = base64.b64encode(buf.read()).decode("utf-8")

        plt.close(fig)
        return result

    def line_chart(
        self,
        data: list[float],
        labels: Optional[list[str]] = None,
        title: str = "Line Chart",
        xlabel: str = "Index",
        ylabel: str = "Value",
        show_trend: bool = False,
        return_base64: bool = False,
    ) -> dict[str, Any]:
        """
        Generate a line chart.

        Args:
            data: List of values to plot.
            labels: Optional x-axis labels.
            title: Chart title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            show_trend: Whether to show trend line.
            return_base64: Whether to return chart as base64.

        Returns:
            Dictionary with file path and optional base64 data.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        x = range(len(data))
        ax.plot(x, data, marker="o", markersize=4, linewidth=1.5, color="#2196F3")

        if show_trend and len(data) > 1:
            z = np.polyfit(list(x), data, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "--", color="#FF5722", linewidth=1.5, label="Trend")
            ax.legend()

        if labels:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        return self._save_chart(fig, "line_chart", return_base64)

    def bar_chart(
        self,
        data: list[float],
        labels: list[str],
        title: str = "Bar Chart",
        xlabel: str = "Category",
        ylabel: str = "Value",
        horizontal: bool = False,
        return_base64: bool = False,
    ) -> dict[str, Any]:
        """
        Generate a bar chart.

        Args:
            data: List of values.
            labels: Category labels.
            title: Chart title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            horizontal: Whether to create horizontal bars.
            return_base64: Whether to return chart as base64.

        Returns:
            Dictionary with file path and optional base64 data.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(data)))

        if horizontal:
            ax.barh(labels, data, color=colors)
            ax.set_xlabel(ylabel)
            ax.set_ylabel(xlabel)
        else:
            ax.bar(labels, data, color=colors)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            plt.xticks(rotation=45, ha="right")

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y" if not horizontal else "x")

        return self._save_chart(fig, "bar_chart", return_base64)

    def scatter_plot(
        self,
        x_data: list[float],
        y_data: list[float],
        title: str = "Scatter Plot",
        xlabel: str = "X",
        ylabel: str = "Y",
        show_regression: bool = False,
        return_base64: bool = False,
    ) -> dict[str, Any]:
        """
        Generate a scatter plot.

        Args:
            x_data: X-axis values.
            y_data: Y-axis values.
            title: Chart title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            show_regression: Whether to show regression line.
            return_base64: Whether to return chart as base64.

        Returns:
            Dictionary with file path and optional base64 data.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(x_data, y_data, alpha=0.6, c="#2196F3", edgecolors="white", s=50)

        if show_regression and len(x_data) > 1:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(x_data), max(x_data), 100)
            ax.plot(x_line, p(x_line), "--", color="#FF5722", linewidth=2, label="Regression")

            # Calculate R²
            y_pred = p(x_data)
            ss_res = np.sum((np.array(y_data) - y_pred) ** 2)
            ss_tot = np.sum((np.array(y_data) - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            ax.legend(title=f"R² = {r_squared:.4f}")

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        return self._save_chart(fig, "scatter_plot", return_base64)

    def heatmap(
        self,
        data: list[list[float]],
        x_labels: Optional[list[str]] = None,
        y_labels: Optional[list[str]] = None,
        title: str = "Heatmap",
        colorbar_label: str = "Value",
        return_base64: bool = False,
    ) -> dict[str, Any]:
        """
        Generate a heatmap.

        Args:
            data: 2D list of values.
            x_labels: Column labels.
            y_labels: Row labels.
            title: Chart title.
            colorbar_label: Colorbar label.
            return_base64: Whether to return chart as base64.

        Returns:
            Dictionary with file path and optional base64 data.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        data_array = np.array(data)
        im = ax.imshow(data_array, cmap="RdYlGn_r", aspect="auto")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(colorbar_label)

        if x_labels:
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha="right")

        if y_labels:
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels)

        # Add text annotations
        for i in range(data_array.shape[0]):
            for j in range(data_array.shape[1]):
                value = data_array[i, j]
                color = "white" if value > (data_array.max() + data_array.min()) / 2 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)

        ax.set_title(title, fontsize=14, fontweight="bold")

        return self._save_chart(fig, "heatmap", return_base64)

    def control_chart(
        self,
        data: list[float],
        ucl: float,
        lcl: float,
        cl: float,
        title: str = "Control Chart",
        ylabel: str = "Value",
        labels: Optional[list[str]] = None,
        return_base64: bool = False,
    ) -> dict[str, Any]:
        """
        Generate a control chart.

        Args:
            data: List of values to plot.
            ucl: Upper control limit.
            lcl: Lower control limit.
            cl: Center line.
            title: Chart title.
            ylabel: Y-axis label.
            labels: Optional x-axis labels.
            return_base64: Whether to return chart as base64.

        Returns:
            Dictionary with file path and optional base64 data.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        x = range(len(data))

        # Plot control limits
        ax.axhline(y=ucl, color="#F44336", linestyle="--", linewidth=1.5, label=f"UCL = {ucl:.3f}")
        ax.axhline(y=cl, color="#4CAF50", linestyle="-", linewidth=1.5, label=f"CL = {cl:.3f}")
        ax.axhline(y=lcl, color="#F44336", linestyle="--", linewidth=1.5, label=f"LCL = {lcl:.3f}")

        # Plot data points
        colors = ["#F44336" if v > ucl or v < lcl else "#2196F3" for v in data]
        ax.scatter(x, data, c=colors, s=50, zorder=5, edgecolors="white")
        ax.plot(x, data, color="#2196F3", linewidth=1, alpha=0.5)

        # Mark out-of-control points
        out_of_control = [(i, v) for i, v in enumerate(data) if v > ucl or v < lcl]
        if out_of_control:
            for idx, val in out_of_control:
                ax.annotate(
                    f"{val:.2f}",
                    (idx, val),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    color="#F44336",
                )

        if labels:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Sample")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        return self._save_chart(fig, "control_chart", return_base64)

    def histogram(
        self,
        data: list[float],
        bins: int = 20,
        title: str = "Histogram",
        xlabel: str = "Value",
        ylabel: str = "Frequency",
        show_normal: bool = False,
        usl: Optional[float] = None,
        lsl: Optional[float] = None,
        return_base64: bool = False,
    ) -> dict[str, Any]:
        """
        Generate a histogram.

        Args:
            data: List of values.
            bins: Number of bins.
            title: Chart title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            show_normal: Whether to show normal distribution overlay.
            usl: Upper specification limit.
            lsl: Lower specification limit.
            return_base64: Whether to return chart as base64.

        Returns:
            Dictionary with file path and optional base64 data.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        data_array = np.array(data)
        n, bins_arr, patches = ax.hist(
            data_array, bins=bins, color="#2196F3", alpha=0.7, edgecolor="white"
        )

        if show_normal:
            mean = np.mean(data_array)
            std = np.std(data_array)
            x = np.linspace(data_array.min(), data_array.max(), 100)
            normal_dist = (
                1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std) ** 2)
            )
            # Scale to match histogram
            scale = len(data_array) * (bins_arr[1] - bins_arr[0])
            ax.plot(x, normal_dist * scale, "r-", linewidth=2, label="Normal Fit")
            ax.legend()

        if usl is not None:
            ax.axvline(x=usl, color="#F44336", linestyle="--", linewidth=2, label=f"USL = {usl}")
        if lsl is not None:
            ax.axvline(x=lsl, color="#F44336", linestyle="--", linewidth=2, label=f"LSL = {lsl}")

        if usl is not None or lsl is not None:
            ax.legend()

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis="y")

        return self._save_chart(fig, "histogram", return_base64)

    def defect_heatmap(
        self,
        image_path: str,
        ng_coordinates: list[dict[str, int]],
        title: str = "Defect Heatmap",
        sigma: float = 20.0,
        alpha: float = 0.6,
        colorbar_label: str = "Defect Density",
        return_base64: bool = False,
    ) -> dict[str, Any]:
        """
        Generate a defect heatmap overlay on a product image.

        This method takes an image and a list of NG (defect) pixel coordinates,
        then generates a heatmap showing defect density. Redder areas indicate
        higher defect frequency.

        Args:
            image_path: Path to the product image file.
            ng_coordinates: List of defect coordinates, each with 'x' and 'y' keys.
                           Example: [{"x": 100, "y": 200}, {"x": 150, "y": 250}]
            title: Chart title.
            sigma: Gaussian blur sigma for smoothing the heatmap. Higher values
                   create smoother, more spread out heat regions.
            alpha: Transparency of the heatmap overlay (0.0-1.0).
            colorbar_label: Label for the colorbar.
            return_base64: Whether to return chart as base64.

        Returns:
            Dictionary with file path, statistics, and optional base64 data.
        """
        # Load the image
        try:
            img = Image.open(image_path)
            img_array = np.array(img)
        except Exception as e:
            return {"success": False, "error": f"Failed to load image: {str(e)}"}

        height, width = img_array.shape[:2]

        # Create a 2D array to accumulate defect counts
        heatmap_data = np.zeros((height, width), dtype=np.float64)

        # Mark each NG coordinate
        valid_points = 0
        for coord in ng_coordinates:
            x = coord.get("x", 0)
            y = coord.get("y", 0)
            # Ensure coordinates are within image bounds
            if 0 <= x < width and 0 <= y < height:
                heatmap_data[y, x] += 1
                valid_points += 1

        if valid_points == 0:
            return {
                "success": False,
                "error": "No valid NG coordinates within image bounds",
            }

        # Apply Gaussian blur to create smooth heatmap
        heatmap_smoothed = gaussian_filter(heatmap_data, sigma=sigma)

        # Normalize to 0-1 range
        if heatmap_smoothed.max() > 0:
            heatmap_normalized = heatmap_smoothed / heatmap_smoothed.max()
        else:
            heatmap_normalized = heatmap_smoothed

        # Create custom colormap: transparent -> yellow -> red
        colors = [
            (0, 0, 0, 0),       # Transparent for low values
            (1, 1, 0, 0.3),     # Yellow with low alpha
            (1, 0.5, 0, 0.6),   # Orange
            (1, 0, 0, 0.9),     # Red for high values
        ]
        cmap = LinearSegmentedColormap.from_list("defect_heatmap", colors, N=256)

        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Display the original image
        ax.imshow(img_array)

        # Overlay the heatmap
        heatmap_plot = ax.imshow(
            heatmap_normalized,
            cmap=cmap,
            alpha=alpha,
            interpolation="bilinear",
        )

        # Add colorbar
        cbar = plt.colorbar(heatmap_plot, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label, fontsize=10)

        # Set title
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")  # Hide axes for cleaner look

        # Calculate statistics
        stats = {
            "total_ng_points": len(ng_coordinates),
            "valid_ng_points": valid_points,
            "image_size": {"width": width, "height": height},
            "max_density_location": {
                "y": int(np.unravel_index(heatmap_smoothed.argmax(), heatmap_smoothed.shape)[0]),
                "x": int(np.unravel_index(heatmap_smoothed.argmax(), heatmap_smoothed.shape)[1]),
            },
        }

        result = self._save_chart(fig, "defect_heatmap", return_base64)
        result["statistics"] = stats

        return result


# Global chart generator instance
_chart_generator: Optional[ChartGenerator] = None


def get_chart_generator() -> ChartGenerator:
    """Get or create the global chart generator instance."""
    global _chart_generator
    if _chart_generator is None:
        _chart_generator = ChartGenerator()
    return _chart_generator


def register_chart_tools(server: Server) -> None:
    """Register chart generation tools with the MCP server."""

    @server.list_tools()
    async def list_chart_tools() -> list[Tool]:
        """List available chart tools."""
        return [
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
                        "return_base64": {
                            "type": "boolean",
                            "description": "Return chart as base64",
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
                        "return_base64": {
                            "type": "boolean",
                            "description": "Return chart as base64",
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
                        "return_base64": {
                            "type": "boolean",
                            "description": "Return chart as base64",
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
                        "return_base64": {
                            "type": "boolean",
                            "description": "Return chart as base64",
                            "default": False,
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
                        "return_base64": {
                            "type": "boolean",
                            "description": "Return chart as base64",
                            "default": False,
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
                        "return_base64": {
                            "type": "boolean",
                            "description": "Return chart as base64",
                            "default": False,
                        },
                    },
                    "required": ["data"],
                },
            ),
        ]

    @server.call_tool()
    async def handle_chart_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle chart tool calls."""
        generator = get_chart_generator()

        if name == "generate_line_chart":
            result = generator.line_chart(
                data=arguments["data"],
                labels=arguments.get("labels"),
                title=arguments.get("title", "Line Chart"),
                xlabel=arguments.get("xlabel", "Index"),
                ylabel=arguments.get("ylabel", "Value"),
                show_trend=arguments.get("show_trend", False),
                return_base64=arguments.get("return_base64", False),
            )
        elif name == "generate_bar_chart":
            result = generator.bar_chart(
                data=arguments["data"],
                labels=arguments["labels"],
                title=arguments.get("title", "Bar Chart"),
                xlabel=arguments.get("xlabel", "Category"),
                ylabel=arguments.get("ylabel", "Value"),
                horizontal=arguments.get("horizontal", False),
                return_base64=arguments.get("return_base64", False),
            )
        elif name == "generate_scatter_plot":
            result = generator.scatter_plot(
                x_data=arguments["x_data"],
                y_data=arguments["y_data"],
                title=arguments.get("title", "Scatter Plot"),
                xlabel=arguments.get("xlabel", "X"),
                ylabel=arguments.get("ylabel", "Y"),
                show_regression=arguments.get("show_regression", False),
                return_base64=arguments.get("return_base64", False),
            )
        elif name == "generate_heatmap":
            result = generator.heatmap(
                data=arguments["data"],
                x_labels=arguments.get("x_labels"),
                y_labels=arguments.get("y_labels"),
                title=arguments.get("title", "Heatmap"),
                colorbar_label=arguments.get("colorbar_label", "Value"),
                return_base64=arguments.get("return_base64", False),
            )
        elif name == "generate_control_chart":
            result = generator.control_chart(
                data=arguments["data"],
                ucl=arguments["ucl"],
                lcl=arguments["lcl"],
                cl=arguments["cl"],
                title=arguments.get("title", "Control Chart"),
                ylabel=arguments.get("ylabel", "Value"),
                labels=arguments.get("labels"),
                return_base64=arguments.get("return_base64", False),
            )
        elif name == "generate_histogram":
            result = generator.histogram(
                data=arguments["data"],
                bins=arguments.get("bins", 20),
                title=arguments.get("title", "Histogram"),
                xlabel=arguments.get("xlabel", "Value"),
                ylabel=arguments.get("ylabel", "Frequency"),
                show_normal=arguments.get("show_normal", False),
                usl=arguments.get("usl"),
                lsl=arguments.get("lsl"),
                return_base64=arguments.get("return_base64", False),
            )
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]
