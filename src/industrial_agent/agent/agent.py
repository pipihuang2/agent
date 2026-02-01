"""Industrial quality inspection agent using pydantic-ai."""

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

from pydantic_ai import Agent

from industrial_agent.config import Settings, get_settings

from .models import get_model
from .prompts import SYSTEM_PROMPT


@dataclass
class AgentDependencies:
    """Dependencies injected into the agent."""

    settings: Settings
    # Add MCP client or other dependencies here as needed


class IndustrialAgent:
    """Industrial quality inspection agent wrapper."""

    def __init__(self, settings: Settings | None = None):
        """
        Initialize the industrial agent.

        Args:
            settings: Optional settings instance.
        """
        self.settings = settings or get_settings()
        self.model = get_model(self.settings)
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent[AgentDependencies, str]:
        """Create the pydantic-ai agent with tools."""
        agent = Agent(
            model=self.model,
            system_prompt=SYSTEM_PROMPT,
            deps_type=AgentDependencies,
            output_type=str,
        )

        # Register tools
        self._register_tools(agent)

        return agent

    def _register_tools(self, agent: Agent[AgentDependencies, str]) -> None:
        """Register agent tools."""

        @agent.tool_plain
        async def analyze_data(
            data_description: str,
            analysis_type: str = "statistical",
        ) -> str:
            """
            Analyze quality inspection data.

            Args:
                data_description: Description of the data to analyze.
                analysis_type: Type of analysis (statistical, anomaly, trend).

            Returns:
                Analysis results as a string.
            """
            # This is a placeholder - actual implementation will use MCP tools
            return f"Analysis of {data_description} using {analysis_type} method."

        @agent.tool_plain
        async def generate_chart(
            chart_type: str,
            data_description: str,
        ) -> str:
            """
            Generate a chart for visualization.

            Args:
                chart_type: Type of chart (line, bar, scatter, heatmap, control).
                data_description: Description of the data to visualize.

            Returns:
                Path to the generated chart or status message.
            """
            # This is a placeholder - actual implementation will use MCP tools
            return f"Generated {chart_type} chart for {data_description}."

    async def run(self, user_input: str) -> str:
        """
        Run the agent with user input.

        Args:
            user_input: The user's query or request.

        Returns:
            The agent's response.
        """
        deps = AgentDependencies(settings=self.settings)
        result = await self.agent.run(user_input, deps=deps)
        return result.output

    async def run_stream(self, user_input: str):
        """
        Run the agent with streaming output.

        Args:
            user_input: The user's query or request.

        Yields:
            Streamed response chunks.
        """
        deps = AgentDependencies(settings=self.settings)
        async with self.agent.run_stream(user_input, deps=deps) as response:
            async for chunk in response.stream():
                yield chunk


def create_agent(settings: Settings | None = None) -> IndustrialAgent:
    """
    Factory function to create an industrial agent.

    Args:
        settings: Optional settings instance.

    Returns:
        An initialized IndustrialAgent instance.
    """
    return IndustrialAgent(settings)
