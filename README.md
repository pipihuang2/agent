# Industrial Quality Inspection Agent

基于 pydantic-ai 的工业质量检测数据分析 Agent，通过 MCP 协议提供数据查询、分析和图表生成功能。

## 功能特性

- **数据查询**: 按时间范围、产品ID、批次ID查询质量检测数据
- **统计分析**: 描述性统计、过程能力分析(Cp/Cpk)、异常检测、趋势分析
- **控制图分析**: I-MR控制图、X-bar控制图
- **图表生成**: 折线图、柱状图、散点图、热力图、直方图、控制图

## 安装

```bash
uv sync
```

## 配置

复制环境变量示例文件并配置:

```bash
cp .env.example .env
```

编辑 `.env` 文件设置:
- `MODEL_PROVIDER`: 模型提供商 (deepseek/ollama)
- `MODEL_NAME`: 模型名称
- `DEEPSEEK_API_KEY`: DeepSeek API密钥 (使用DeepSeek时必需)
- `OLLAMA_BASE_URL`: Ollama服务地址 (使用Ollama时)

## 使用

### 交互模式

```bash
uv run python -m industrial_agent.main
```

### 单次查询

```bash
uv run python -m industrial_agent.main --query "分析最近的质量数据"
```

### 启动MCP服务器

```bash
uv run python -m industrial_agent.main --mcp-server
```

## 技术栈

- **Agent框架**: pydantic-ai
- **MCP**: mcp (官方Python SDK)
- **图表生成**: matplotlib
- **数据分析**: pandas + numpy
- **配置管理**: pydantic-settings
