需求:基于excel上传的数据,llm_agent进行规划,生成todo,调用工具:
1. pandas进行数据预处理,基于sklearn进行数据划分
2. 基于xgboost、sklearn、Prophet、ARIMA进行预测
3. 基于pyecharts进行可视化趋势预测
4. 附上llm大模型对预测结果的分析

llm大模型:
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = "<your_api_key>"

python版本:
python==3.12

AI框架: langChain + ReAct 


## 开发指南

### 目标
基于 Excel 上传数据，完成数据预处理、训练多种时序模型（XGBoost、sklearn、Prophet、ARIMA），生成预测结果、可视化与 LLM 中文分析，并提供 LangChain ReAct 风格的流水线/代理。

### 目录结构
```
src/forecast_agent/
  config.py          # 配置与环境变量
  data_loader.py     # Excel 数据加载
  preprocess.py      # 清洗、拆分、特征工程
  models.py          # 多模型训练与预测
  evaluation.py      # MAE/RMSE/MAPE 指标
  visualization.py   # pyecharts 预测对比图
  llm_agent.py       # LangChain + DeepSeek 分析
  pipeline.py        # 端到端流水线
main.py              # CLI 入口
scripts/generate_sample_data.py # 生成示例 Excel
data/sample_data.xlsx # 示例数据（运行脚本生成）
outputs/             # 图表与报告输出
```

### 环境准备
1. Python 3.12+。
2. 创建虚拟环境并安装依赖：
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. 配置环境变量（建议 `.env`）：
   ```
   DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions
   DEEPSEEK_API_KEY=你的密钥
   ```
   **安全提醒**：请将真实密钥移出本文件，存放在 `.env`，避免泄露。

### 快速开始
1. 生成示例数据：
   ```bash
   python scripts/generate_sample_data.py
   ```
2. 运行完整流水线（默认读取 `data/sample_data.xlsx`，结果写入 `outputs`）：
   ```bash
   python main.py
   ```
   自定义参数：
   ```bash
   python main.py --data-path your.xlsx --date-column 日期列 --target-column 数值列 --output-dir outputs
   ```
3. 查看输出：
   - 预测对比图：`outputs/forecast_chart.html`
   - 终端打印指标与（如配置了密钥）LLM 中文分析。

### 设计思路
- **数据处理**：`pandas` 清洗、插值；`train_test_split_ts` 按时间划分；`create_lag_features` 生成滞后特征供树模型/XGBoost 使用。
- **建模**：
  - `XGBoost`、`RandomForest` 通过滞后特征 + 滚动预测避免窥视未来。
  - `Prophet` 使用 `ds/y` 规范输入，生成未来时间节点预测。
  - `ARIMA` 由 `pmdarima.auto_arima` 自动寻参。
- **评估/可视化**：MAE/RMSE/MAPE；`pyecharts` 生成交互式 HTML 对比图。
- **LLM 分析**：`langchain-openai` 适配 DeepSeek OpenAI 兼容接口，生成中文总结与改进建议。
- **代理/自动化**：`pipeline.py` 串联加载、清洗、训练、评估、可视化与 LLM 分析，可嵌入 LangChain ReAct 工具链。

### 后续优化方向
- 增加交叉验证与超参搜索。
- 支持多目标/多维特征与异常检测。
- 引入 MLflow/Weights & Biases 做实验追踪。
- 增加单元测试与 CI。

