"""Main entry point for the industrial quality inspection agent."""

import argparse
import asyncio
import sys

from industrial_agent.agent import create_agent
from industrial_agent.config import get_settings


async def run_agent_interactive() -> None:
    """Run the agent in interactive mode."""
    settings = get_settings()

    print("=" * 60)
    print("工业质量检测数据分析 Agent")
    print("=" * 60)
    print(f"模型提供商: {settings.model_provider.value}")
    print(f"模型名称: {settings.model_name}")
    print("-" * 60)
    print("输入您的问题，输入 'quit' 或 'exit' 退出")
    print("-" * 60)

    try:
        agent = create_agent(settings)
    except ValueError as e:
        print(f"错误: {e}")
        print("请检查您的配置文件 (.env)")
        sys.exit(1)

    while True:
        try:
            user_input = input("\n您: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                print("再见！")
                break

            print("\n助手: ", end="", flush=True)

            # Use streaming for better UX
            response_text = ""
            async for chunk in agent.run_stream(user_input):
                print(chunk, end="", flush=True)
                response_text += chunk

            print()  # New line after response

        except KeyboardInterrupt:
            print("\n\n中断。再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")


async def run_agent_single(query: str) -> None:
    """Run the agent with a single query."""
    settings = get_settings()

    try:
        agent = create_agent(settings)
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)

    try:
        response = await agent.run(query)
        print(response)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


async def run_mcp_server() -> None:
    """Run the MCP server."""
    from industrial_agent.mcp_server.server import IndustrialMCPServer

    print("启动 MCP 服务器...", file=sys.stderr)
    server = IndustrialMCPServer()
    await server.run()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="工业质量检测数据分析 Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互模式
  python -m industrial_agent.main

  # 单次查询
  python -m industrial_agent.main --query "分析最近的质量数据"

  # 启动 MCP 服务器
  python -m industrial_agent.main --mcp-server
        """,
    )

    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="单次查询模式，直接执行指定的查询",
    )

    parser.add_argument(
        "--mcp-server",
        action="store_true",
        help="启动 MCP 服务器模式",
    )

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version="%(prog)s 0.1.0",
    )

    args = parser.parse_args()

    if args.mcp_server:
        asyncio.run(run_mcp_server())
    elif args.query:
        asyncio.run(run_agent_single(args.query))
    else:
        asyncio.run(run_agent_interactive())


if __name__ == "__main__":
    main()
