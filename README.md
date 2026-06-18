# DataViz_StreamlitApp — 生成式AI与学生学业表现交互式分析平台

> 24211870140 黄贵崇 | 数据可视化课程大作业 | Streamlit Cloud 部署版

## 应用简介

本应用基于 50,000 条学生样本数据，提供从原始数据清洗到可视化分析、风险识别、建模解释与学习行为画像的一体化交互式仪表盘。

### 核心功能

| 页面模块 | 功能描述 |
|---|---|
| ① 数据导入与清洗 | 上传CSV、字段标准化、缺失值审计、异常值审计、派生变量生成 |
| ② 总体画像 | 样本量、GPA变化、AI使用时长、技能保留、倦怠比例等关键指标卡片 |
| ③ 学业成绩 | GPA前后变化、AI使用时长/占比与GPA变化关系分析 |
| ④ AI使用行为 | 使用目的、工具多样性、付费订阅对学习结果的影响 |
| ⑤ 群体差异 | 专业、年级、传统学习×AI使用交互分析 |
| ⑥ AI素养与政策 | 提示词能力、学校AI政策、交叉分析热力图 |
| ⑦ 心理风险 | 依赖度、焦虑水平与倦怠风险的关系矩阵 |
| ⑧ 建模分析 | 相关性矩阵、线性回归、随机森林、K-Means聚类、PCA投影 |
| ⑨ 结论与导出 | 自动化结论生成、数据下载 |

### 侧边栏筛选器

支持按专业类别、年级、AI主要用途、提示词能力、学校AI政策、倦怠风险等级、AI使用时长范围、传统学习时长范围和GPA变化范围进行全局筛选，所有图表联动更新。

## 本地运行

```bash
# 1. 克隆仓库
git clone https://github.com/YOUR_USERNAME/DataViz_StreamlitApp.git
cd DataViz_StreamlitApp

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动应用
streamlit run StreamlitApp.py

# 4. 浏览器访问 http://localhost:8501
```

## Streamlit Cloud 部署

本应用已配置 `.streamlit/config.toml`，可直接部署至 [Streamlit Cloud](https://share.streamlit.io)：

1. 登录 [share.streamlit.io](https://share.streamlit.io)
2. 点击 "New app"
3. 选择本仓库，分支 main，主文件路径 `StreamlitApp.py`
4. 点击 "Deploy!"

## 数据说明

- **数据来源**：Kaggle - AI Usage & Student Academic Performance Analysis
- **样本量**：50,000 条学生记录
- **字段数**：16 个原始字段 + 12 个派生变量
- **应用内置**：默认数据集位于 `data/ai_student_impact_dataset.csv`
- **支持上传**：用户可通过侧边栏上传自定义 CSV 文件

## 技术栈

- **框架**：Streamlit ≥ 1.39
- **可视化**：Plotly ≥ 5.20
- **数据处理**：pandas ≥ 2.0, numpy ≥ 1.24
- **机器学习**：scikit-learn ≥ 1.3

## 论文引用

本应用对应的课程论文《生成式人工智能如何重塑学习：学生学业表现与行为的多维影响研究》包含 29 张静态图表及完整分析，附录中包含图表字典、程序代码、字段说明和模型参数。
