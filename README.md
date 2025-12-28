# Question Extraction Agent (image2test)

一个基于 LangChain/LangGraph 的“题目提取”Agent：从图片中识别选择题/判断题，并输出为 JSON 与 Word（`.docx`）。

## 依赖与环境

- Python：`>= 3.12`（见 `pyproject.toml`）
- 包管理：`uv`（推荐）
- 主要依赖：`langchain` / `langgraph`、OpenAI-compatible Chat API（默认使用豆包 Ark 兼容接口）、`python-docx`、`pydantic`、`click`、`rich`

## 初始化（首次运行）

1. 安装依赖（会创建/更新项目虚拟环境并安装本项目）：
   ```bash
   uv sync
   ```
2. 配置环境变量：复制示例文件并填写 Key
   - PowerShell：
     ```powershell
     Copy-Item .env.example .env
     ```
   - macOS/Linux：
     ```bash
     cp .env.example .env
     ```
3. 最小必配（用于图片识别）：在 `.env` 中填写 `DOUBAO_API_KEY=...`

默认输出目录为 `./output/`（可通过 `DEFAULT_OUTPUT_DIR` 修改）。

## 使用方式（CLI）

推荐通过 `uv run` 调用，避免 PATH/安装方式差异：

```bash
# 交互模式（对话式提取/保存/校验）
uv run question-agent interactive

# 单图提取
uv run question-agent extract .\\images\\1.jpg

# 批量处理目录
uv run question-agent batch .\\images\\
```

## 开发与测试

- 运行测试：`uv run pytest`
- 代码检查（Ruff）：`uv run ruff check .`

## （可选）LangSmith / LangGraph Studio

1. 在 `.env` 中配置：
   ```dotenv
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_langsmith_api_key_here
   LANGCHAIN_PROJECT=image2test
   ```
2. 启动：`uv run langgraph dev`（会在浏览器中打开 Studio）

## 项目结构

- `src/agent/`: Agent 逻辑与 prompts
- `src/tools/`: 图片分析、JSON/Word 生成工具
- `src/models/`: Pydantic 数据模型与配置（`.env` 读取）
- `src/utils/`: 通用工具函数
- `tests/`: `pytest` 测试
- `images/`: 示例图片
