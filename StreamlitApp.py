"""
生成式人工智能应用与学生学业成绩分析 - Streamlit 交互式仪表盘

运行方式：
1. 将 ai_student_impact_dataset.csv 放在 data/ 目录下，或在页面左侧上传原始 CSV。
2. 安装依赖：pip install -r requirements.txt
3. 启动应用：streamlit run StreamlitApp.py

说明：
- 支持原始数据导入、字段标准化、缺失值处理、异常值审计、派生变量生成。
- 覆盖论文中的全部核心分析维度：GPA 变化、AI 使用时长、AI 占比、使用目的、提示词能力、专业、年级、制度政策、付费订阅、工具多样性、传统学习投入、心理风险、相关性、回归、随机森林、聚类画像。
"""

from __future__ import annotations

import io
import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="生成式AI与学生学业表现分析平台",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# 全局配置
# -----------------------------------------------------------------------------

APP_TITLE = "生成式AI应用与学生学业成绩分析平台"
APP_SUBTITLE = "从原始数据清洗到可视化分析、风险识别、建模解释与学习行为画像的一体化展示"
DEFAULT_DATA_PATHS = [
    "ai_student_impact_dataset.csv",
    "data/ai_student_impact_dataset.csv",
    "ai_student_impact_dataset.csv",
    "EnhancedProcessedAiStudentImpactData.csv",
    "ProcessedData/EnhancedProcessedAiStudentImpactData.csv",
    "StepFourEnhancedOutputs/ProcessedData/EnhancedProcessedAiStudentImpactData.csv",
]

CHINESE_LABELS = {
    "StudentId": "学生编号",
    "MajorCategory": "专业类别",
    "YearOfStudy": "年级",
    "PreSemesterGpa": "AI使用前GPA",
    "PostSemesterGpa": "AI使用后GPA",
    "GpaChange": "GPA变化",
    "WeeklyGenAiHours": "每周AI使用时长",
    "TraditionalStudyHours": "每周传统学习时长",
    "TotalStudyHours": "每周总学习时长",
    "GenAiStudyShare": "AI学习时间占比",
    "PrimaryUseCase": "AI主要用途",
    "PromptEngineeringSkill": "提示词能力",
    "ToolDiversity": "AI工具多样性",
    "PaidSubscription": "是否付费订阅",
    "PerceivedAiDependency": "AI依赖度",
    "InstitutionalPolicy": "学校AI政策",
    "AnxietyLevelDuringExams": "考试焦虑水平",
    "SkillRetentionScore": "技能保留分数",
    "BurnoutRiskLevel": "倦怠风险等级",
    "HighBurnoutFlag": "高倦怠风险",
    "MediumOrHighBurnoutFlag": "中高倦怠风险",
    "GenAiHoursBand": "AI使用时长分组",
    "GenAiShareBand": "AI学习占比分组",
    "TraditionalHoursBand": "传统学习时长分组",
    "DependencyBand": "AI依赖分组",
    "LearningCluster": "学习行为聚类",
    "LearningClusterName": "学习行为类型",
}

MAJOR_LABELS = {
    "STEM": "理工科",
    "Business": "商科",
    "Humanities": "人文学科",
    "Medical": "医学类",
    "Arts": "艺术类",
    "Social Sciences": "社会科学",
    "Education": "教育学",
    "Law": "法学",
    "Unknown": "未知",
}

YEAR_ORDER = ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"]
YEAR_LABELS = {
    "Freshman": "大一",
    "Sophomore": "大二",
    "Junior": "大三",
    "Senior": "大四",
    "Graduate": "研究生",
    "Unknown": "未知",
}

USE_CASE_LABELS = {
    "Debugging/Troubleshooting": "调试与问题排查",
    "Direct_Answer_Generation": "直接生成答案",
    "Direct Answer Generation": "直接生成答案",
    "Copywriting/Drafting": "写作与草稿生成",
    "Summarizing_Reading": "阅读总结",
    "Summarizing Reading": "阅读总结",
    "Ideation": "创意构思",
    "Unknown": "未知",
}

PROMPT_SKILL_ORDER = ["Beginner", "Intermediate", "Advanced"]
PROMPT_SKILL_LABELS = {"Beginner": "初级", "Intermediate": "中级", "Advanced": "高级", "Unknown": "未知"}

POLICY_LABELS = {
    "Allowed": "允许使用",
    "Allowed_With_Citation": "允许但需引用",
    "Allowed With Citation": "允许但需引用",
    "Restricted": "有限制使用",
    "Strict_Ban": "严格禁止",
    "Strict Ban": "严格禁止",
    "No_Policy": "无明确政策",
    "No Policy": "无明确政策",
    "Unknown": "未知",
}

BURNOUT_ORDER = ["Low", "Medium", "High"]
BURNOUT_LABELS = {"Low": "低风险", "Medium": "中风险", "High": "高风险", "Unknown": "未知"}

BOOLEAN_LABELS = {True: "是", False: "否", 1: "是", 0: "否", "True": "是", "False": "否"}

NUMERIC_COLUMNS = [
    "PreSemesterGpa",
    "PostSemesterGpa",
    "GpaChange",
    "WeeklyGenAiHours",
    "TraditionalStudyHours",
    "TotalStudyHours",
    "GenAiStudyShare",
    "ToolDiversity",
    "PerceivedAiDependency",
    "AnxietyLevelDuringExams",
    "SkillRetentionScore",
]

MODEL_FEATURE_COLUMNS = [
    "PreSemesterGpa",
    "WeeklyGenAiHours",
    "TraditionalStudyHours",
    "GenAiStudyShare",
    "ToolDiversity",
    "PerceivedAiDependency",
    "AnxietyLevelDuringExams",
    "SkillRetentionScore",
    "MajorCategory",
    "YearOfStudy",
    "PrimaryUseCase",
    "PromptEngineeringSkill",
    "PaidSubscription",
    "InstitutionalPolicy",
]

PLOT_TEMPLATE = "plotly_white"


@dataclass
class DataLoadResult:
    frame: pd.DataFrame
    sourceName: str
    sourceMode: str


# -----------------------------------------------------------------------------
# 页面样式
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>
    :root {
        --main-blue: #1f4e79;
        --deep-blue: #12324f;
        --soft-blue: #eaf2fb;
        --soft-gray: #f7f9fc;
        --card-border: #dfe7f3;
    }
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }
    .hero-card {
        padding: 1.4rem 1.6rem;
        border-radius: 22px;
        background: linear-gradient(120deg, #12324f 0%, #1f4e79 52%, #4d8bc7 100%);
        color: white;
        box-shadow: 0 14px 36px rgba(18, 50, 79, 0.22);
        margin-bottom: 1rem;
    }
    .hero-card h1 {
        margin: 0;
        font-size: 2.05rem;
        font-weight: 800;
        letter-spacing: 0.02em;
    }
    .hero-card p {
        margin: 0.55rem 0 0 0;
        font-size: 1rem;
        color: rgba(255,255,255,0.88);
    }
    .section-note {
        padding: 0.85rem 1rem;
        background: #f7f9fc;
        border-left: 5px solid #1f4e79;
        border-radius: 12px;
        margin: 0.6rem 0 1rem 0;
        color: #26384d;
        line-height: 1.65;
    }
    .insight-card {
        padding: 0.95rem 1rem;
        background: #ffffff;
        border: 1px solid #dfe7f3;
        border-radius: 16px;
        box-shadow: 0 6px 18px rgba(18, 50, 79, 0.06);
        min-height: 118px;
    }
    .insight-card h4 {
        margin: 0 0 0.45rem 0;
        color: #12324f;
        font-size: 1.02rem;
    }
    .insight-card p {
        margin: 0;
        color: #4a5568;
        line-height: 1.55;
        font-size: 0.93rem;
    }
    .small-caption {
        color: #637083;
        font-size: 0.88rem;
        margin-top: -0.2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.55rem;
        color: #12324f;
    }
    div[data-testid="stMetricLabel"] {
        color: #44556b;
        font-weight: 650;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 0.45rem 0.85rem;
        background-color: #f0f5fb;
        color: #12324f;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f4e79 !important;
        color: white !important;
    }
    .download-card {
        background: #f7f9fc;
        border: 1px dashed #9bb7d6;
        border-radius: 14px;
        padding: 1rem;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 数据读取、清洗与转换
# -----------------------------------------------------------------------------


def findDefaultDataPath() -> Optional[str]:
    for pathItem in DEFAULT_DATA_PATHS:
        if os.path.exists(pathItem):
            return pathItem
    return None


@st.cache_data(show_spinner=False)
def loadDefaultFrame(defaultPath: str) -> pd.DataFrame:
    return pd.read_csv(defaultPath)


def readUploadedFrame(uploadedFile) -> pd.DataFrame:
    return pd.read_csv(uploadedFile)


def normalizeColumnName(columnName: str) -> str:
    columnName = str(columnName).strip()
    columnMap = {
        "Student_ID": "StudentId",
        "StudentId": "StudentId",
        "Major_Category": "MajorCategory",
        "MajorCategory": "MajorCategory",
        "Year_of_Study": "YearOfStudy",
        "YearOfStudy": "YearOfStudy",
        "Pre_Semester_GPA": "PreSemesterGpa",
        "PreSemesterGpa": "PreSemesterGpa",
        "Weekly_GenAI_Hours": "WeeklyGenAiHours",
        "WeeklyGenAIHours": "WeeklyGenAiHours",
        "WeeklyGenAiHours": "WeeklyGenAiHours",
        "Primary_Use_Case": "PrimaryUseCase",
        "PrimaryUseCase": "PrimaryUseCase",
        "Prompt_Engineering_Skill": "PromptEngineeringSkill",
        "PromptEngineeringSkill": "PromptEngineeringSkill",
        "Tool_Diversity": "ToolDiversity",
        "ToolDiversity": "ToolDiversity",
        "Paid_Subscription": "PaidSubscription",
        "PaidSubscription": "PaidSubscription",
        "Traditional_Study_Hours": "TraditionalStudyHours",
        "TraditionalStudyHours": "TraditionalStudyHours",
        "Perceived_AI_Dependency": "PerceivedAiDependency",
        "PerceivedAiDependency": "PerceivedAiDependency",
        "Institutional_Policy": "InstitutionalPolicy",
        "InstitutionalPolicy": "InstitutionalPolicy",
        "Anxiety_Level_During_Exams": "AnxietyLevelDuringExams",
        "AnxietyLevelDuringExams": "AnxietyLevelDuringExams",
        "Post_Semester_GPA": "PostSemesterGpa",
        "PostSemesterGpa": "PostSemesterGpa",
        "Skill_Retention_Score": "SkillRetentionScore",
        "SkillRetentionScore": "SkillRetentionScore",
        "Burnout_Risk_Level": "BurnoutRiskLevel",
        "BurnoutRiskLevel": "BurnoutRiskLevel",
        "GpaChange": "GpaChange",
        "TotalStudyHours": "TotalStudyHours",
        "GenAiStudyShare": "GenAiStudyShare",
        "GenAiHoursBand": "GenAiHoursBand",
        "GenAiShareBand": "GenAiShareBand",
        "TraditionalHoursBand": "TraditionalHoursBand",
        "HighBurnoutFlag": "HighBurnoutFlag",
        "MediumOrHighBurnoutFlag": "MediumOrHighBurnoutFlag",
        "HighGenAiUseFlag": "HighGenAiUseFlag",
        "LowRetentionFlag": "LowRetentionFlag",
        "GpaImprovedFlag": "GpaImprovedFlag",
        "DependencyBand": "DependencyBand",
        "LearningClusterName": "LearningClusterName",
        "LearningCluster": "LearningCluster",
    }
    return columnMap.get(columnName, columnName.replace(" ", ""))


def convertBooleanSeries(seriesObject: pd.Series) -> pd.Series:
    if seriesObject.dtype == bool:
        return seriesObject
    trueValues = {"true", "1", "yes", "y", "是", "paid", "付费"}
    falseValues = {"false", "0", "no", "n", "否", "free", "免费"}

    def convertValue(valueItem):
        if pd.isna(valueItem):
            return np.nan
        valueText = str(valueItem).strip().lower()
        if valueText in trueValues:
            return True
        if valueText in falseValues:
            return False
        return bool(valueItem) if isinstance(valueItem, (int, float, np.integer, np.floating)) else np.nan

    return seriesObject.map(convertValue)


def cleanCategoryValue(valueItem, defaultValue="Unknown"):
    if pd.isna(valueItem):
        return defaultValue
    valueText = str(valueItem).strip()
    if valueText == "" or valueText.lower() in {"nan", "none", "null"}:
        return defaultValue
    return valueText


def makeBandSeries(values: pd.Series, bins: List[float], labels: List[str]) -> pd.Series:
    return pd.cut(values, bins=bins, labels=labels, include_lowest=True, right=False).astype(str)


@st.cache_data(show_spinner=False)
def prepareData(rawFrame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    cleanFrame = rawFrame.copy()
    cleanFrame.columns = [normalizeColumnName(columnName) for columnName in cleanFrame.columns]

    requiredColumns = [
        "StudentId",
        "MajorCategory",
        "YearOfStudy",
        "PreSemesterGpa",
        "WeeklyGenAiHours",
        "PrimaryUseCase",
        "PromptEngineeringSkill",
        "ToolDiversity",
        "PaidSubscription",
        "TraditionalStudyHours",
        "PerceivedAiDependency",
        "InstitutionalPolicy",
        "AnxietyLevelDuringExams",
        "PostSemesterGpa",
        "SkillRetentionScore",
        "BurnoutRiskLevel",
    ]
    missingRequiredColumns = [columnItem for columnItem in requiredColumns if columnItem not in cleanFrame.columns]
    if missingRequiredColumns:
        raise ValueError("数据缺少必要字段：" + ", ".join(missingRequiredColumns))

    originalMissingAudit = cleanFrame.isna().sum().reset_index()
    originalMissingAudit.columns = ["Field", "MissingCount"]
    originalMissingAudit["MissingRate"] = originalMissingAudit["MissingCount"] / max(len(cleanFrame), 1)

    cleaningLog = []
    initialRows = len(cleanFrame)
    cleanFrame = cleanFrame.drop_duplicates(subset=["StudentId"]).copy()
    cleaningLog.append(f"按 StudentId 去重：删除 {initialRows - len(cleanFrame):,} 条重复记录。")

    numericBaseColumns = [
        "PreSemesterGpa",
        "PostSemesterGpa",
        "WeeklyGenAiHours",
        "TraditionalStudyHours",
        "ToolDiversity",
        "PerceivedAiDependency",
        "AnxietyLevelDuringExams",
        "SkillRetentionScore",
    ]
    for columnItem in numericBaseColumns:
        cleanFrame[columnItem] = pd.to_numeric(cleanFrame[columnItem], errors="coerce")
        missingCount = int(cleanFrame[columnItem].isna().sum())
        if missingCount > 0:
            fillValue = cleanFrame[columnItem].median()
            cleanFrame[columnItem] = cleanFrame[columnItem].fillna(fillValue)
            cleaningLog.append(f"数值字段 {columnItem} 存在 {missingCount:,} 个缺失值，已使用中位数 {fillValue:.3f} 填补。")

    categoryColumns = [
        "MajorCategory",
        "YearOfStudy",
        "PrimaryUseCase",
        "PromptEngineeringSkill",
        "InstitutionalPolicy",
        "BurnoutRiskLevel",
    ]
    for columnItem in categoryColumns:
        missingCount = int(cleanFrame[columnItem].isna().sum())
        cleanFrame[columnItem] = cleanFrame[columnItem].map(cleanCategoryValue)
        if missingCount > 0:
            cleaningLog.append(f"分类字段 {columnItem} 存在 {missingCount:,} 个缺失值，已填充为 Unknown。")

    paidMissingCount = int(cleanFrame["PaidSubscription"].isna().sum())
    cleanFrame["PaidSubscription"] = convertBooleanSeries(cleanFrame["PaidSubscription"])
    if cleanFrame["PaidSubscription"].isna().sum() > 0:
        cleanFrame["PaidSubscription"] = cleanFrame["PaidSubscription"].fillna(False)
        cleaningLog.append(f"PaidSubscription 存在 {paidMissingCount:,} 个缺失或不可识别值，已默认处理为 False。")

    beforeClipFrame = cleanFrame.copy()
    cleanFrame["PreSemesterGpa"] = cleanFrame["PreSemesterGpa"].clip(0, 4)
    cleanFrame["PostSemesterGpa"] = cleanFrame["PostSemesterGpa"].clip(0, 4)
    cleanFrame["WeeklyGenAiHours"] = cleanFrame["WeeklyGenAiHours"].clip(lower=0)
    cleanFrame["TraditionalStudyHours"] = cleanFrame["TraditionalStudyHours"].clip(lower=0)
    cleanFrame["ToolDiversity"] = cleanFrame["ToolDiversity"].clip(lower=0)
    cleanFrame["PerceivedAiDependency"] = cleanFrame["PerceivedAiDependency"].clip(1, 10)
    cleanFrame["AnxietyLevelDuringExams"] = cleanFrame["AnxietyLevelDuringExams"].clip(1, 10)
    cleanFrame["SkillRetentionScore"] = cleanFrame["SkillRetentionScore"].clip(0, 100)
    changedClipCount = int((beforeClipFrame[numericBaseColumns] != cleanFrame[numericBaseColumns]).sum().sum())
    cleaningLog.append(f"业务范围裁剪：共调整 {changedClipCount:,} 个超出合理范围的数值单元。")

    cleanFrame["GpaChange"] = cleanFrame["PostSemesterGpa"] - cleanFrame["PreSemesterGpa"]
    cleanFrame["TotalStudyHours"] = cleanFrame["WeeklyGenAiHours"] + cleanFrame["TraditionalStudyHours"]
    cleanFrame["GenAiStudyShare"] = np.where(cleanFrame["TotalStudyHours"] > 0, cleanFrame["WeeklyGenAiHours"] / cleanFrame["TotalStudyHours"], 0)
    cleanFrame["GenAiStudyShare"] = cleanFrame["GenAiStudyShare"].clip(0, 1)
    cleanFrame["HighBurnoutFlag"] = (cleanFrame["BurnoutRiskLevel"] == "High").astype(int)
    cleanFrame["MediumOrHighBurnoutFlag"] = cleanFrame["BurnoutRiskLevel"].isin(["Medium", "High"]).astype(int)
    cleanFrame["HighGenAiUseFlag"] = (cleanFrame["WeeklyGenAiHours"] >= 20).astype(int)
    cleanFrame["LowRetentionFlag"] = (cleanFrame["SkillRetentionScore"] < 60).astype(int)
    cleanFrame["GpaImprovedFlag"] = (cleanFrame["GpaChange"] > 0).astype(int)

    cleanFrame["GenAiHoursBand"] = makeBandSeries(
        cleanFrame["WeeklyGenAiHours"],
        bins=[0, 2, 5, 10, 20, np.inf],
        labels=["0-2小时", "2-5小时", "5-10小时", "10-20小时", "20小时以上"],
    )
    cleanFrame["GenAiShareBand"] = makeBandSeries(
        cleanFrame["GenAiStudyShare"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.01],
        labels=["0%-20%", "20%-40%", "40%-60%", "60%-80%", "80%-100%"],
    )
    cleanFrame["TraditionalHoursBand"] = makeBandSeries(
        cleanFrame["TraditionalStudyHours"],
        bins=[0, 5, 10, 15, 20, np.inf],
        labels=["0-5小时", "5-10小时", "10-15小时", "15-20小时", "20小时以上"],
    )
    cleanFrame["DependencyBand"] = makeBandSeries(
        cleanFrame["PerceivedAiDependency"],
        bins=[1, 3, 6, 8, 11],
        labels=["低依赖", "中低依赖", "中高依赖", "高依赖"],
    )

    for columnItem in ["MajorCategory", "YearOfStudy", "PrimaryUseCase", "PromptEngineeringSkill", "InstitutionalPolicy", "BurnoutRiskLevel"]:
        cleanFrame[columnItem] = cleanFrame[columnItem].astype(str)

    cleanFrame["MajorCategoryCn"] = cleanFrame["MajorCategory"].map(MAJOR_LABELS).fillna(cleanFrame["MajorCategory"])
    cleanFrame["YearOfStudyCn"] = cleanFrame["YearOfStudy"].map(YEAR_LABELS).fillna(cleanFrame["YearOfStudy"])
    cleanFrame["PrimaryUseCaseCn"] = cleanFrame["PrimaryUseCase"].map(USE_CASE_LABELS).fillna(cleanFrame["PrimaryUseCase"])
    cleanFrame["PromptEngineeringSkillCn"] = cleanFrame["PromptEngineeringSkill"].map(PROMPT_SKILL_LABELS).fillna(cleanFrame["PromptEngineeringSkill"])
    cleanFrame["InstitutionalPolicyCn"] = cleanFrame["InstitutionalPolicy"].map(POLICY_LABELS).fillna(cleanFrame["InstitutionalPolicy"])
    cleanFrame["BurnoutRiskLevelCn"] = cleanFrame["BurnoutRiskLevel"].map(BURNOUT_LABELS).fillna(cleanFrame["BurnoutRiskLevel"])
    cleanFrame["PaidSubscriptionCn"] = cleanFrame["PaidSubscription"].map(BOOLEAN_LABELS).fillna(cleanFrame["PaidSubscription"].astype(str))

    outlierRecords = []
    for columnItem in [
        "PreSemesterGpa",
        "PostSemesterGpa",
        "GpaChange",
        "WeeklyGenAiHours",
        "TraditionalStudyHours",
        "TotalStudyHours",
        "GenAiStudyShare",
        "ToolDiversity",
        "PerceivedAiDependency",
        "AnxietyLevelDuringExams",
        "SkillRetentionScore",
    ]:
        quartileOne = cleanFrame[columnItem].quantile(0.25)
        quartileThree = cleanFrame[columnItem].quantile(0.75)
        interQuartileRange = quartileThree - quartileOne
        lowerBound = quartileOne - 1.5 * interQuartileRange
        upperBound = quartileThree + 1.5 * interQuartileRange
        flagColumn = f"{columnItem}OutlierFlag"
        cleanFrame[flagColumn] = ((cleanFrame[columnItem] < lowerBound) | (cleanFrame[columnItem] > upperBound)).astype(int)
        outlierRecords.append(
            {
                "Field": columnItem,
                "ChineseField": CHINESE_LABELS.get(columnItem, columnItem),
                "LowerBound": lowerBound,
                "UpperBound": upperBound,
                "OutlierCount": int(cleanFrame[flagColumn].sum()),
                "OutlierRate": float(cleanFrame[flagColumn].mean()),
            }
        )
    outlierAudit = pd.DataFrame(outlierRecords)
    cleaningLog.append("已基于 IQR 方法生成数值变量异常值审计结果；异常值默认保留，用于识别高使用、高依赖、低保留等研究群体。")
    cleaningLog.append("已生成 GpaChange、TotalStudyHours、GenAiStudyShare、GenAiHoursBand、GenAiShareBand、TraditionalHoursBand 等派生变量。")

    return cleanFrame, originalMissingAudit, outlierAudit, cleaningLog


# -----------------------------------------------------------------------------
# 通用图表与表格工具
# -----------------------------------------------------------------------------


def toChineseColumnName(columnName: str) -> str:
    return CHINESE_LABELS.get(columnName, columnName)


def formatPercent(valueItem: float) -> str:
    if pd.isna(valueItem):
        return "--"
    return f"{valueItem:.1%}"


def formatNumber(valueItem: float, digits: int = 3) -> str:
    if pd.isna(valueItem):
        return "--"
    return f"{valueItem:.{digits}f}"


def makeMetricDelta(currentValue: float, compareValue: float, digits: int = 3) -> str:
    if pd.isna(currentValue) or pd.isna(compareValue):
        return None
    deltaValue = currentValue - compareValue
    return f"{deltaValue:+.{digits}f} vs 全样本"


def makeCiSummary(frame: pd.DataFrame, groupColumn: str, valueColumn: str) -> pd.DataFrame:
    summaryFrame = frame.groupby(groupColumn, observed=False)[valueColumn].agg(["mean", "std", "count"]).reset_index()
    summaryFrame["StandardError"] = summaryFrame["std"] / np.sqrt(summaryFrame["count"].clip(lower=1))
    summaryFrame["Ci95"] = 1.96 * summaryFrame["StandardError"].fillna(0)
    return summaryFrame


def sortByKnownOrder(frame: pd.DataFrame, columnName: str, orderList: List[str]) -> pd.DataFrame:
    frame = frame.copy()
    frame[columnName] = pd.Categorical(frame[columnName], categories=orderList, ordered=True)
    return frame.sort_values(columnName)


def figureLayout(figObject: go.Figure, titleText: str, heightValue: int = 430) -> go.Figure:
    figObject.update_layout(
        title={"text": titleText, "x": 0.02, "xanchor": "left", "font": {"size": 20, "color": "#12324f"}},
        template=PLOT_TEMPLATE,
        height=heightValue,
        margin=dict(l=35, r=30, t=65, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Microsoft YaHei, SimHei, Arial", size=13),
    )
    return figObject


def plotlyChart(figObject: go.Figure):
    st.plotly_chart(figObject, use_container_width=True, config={"displayModeBar": True, "responsive": True})


def createDownloadBytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def renderInsightCard(titleText: str, bodyText: str):
    st.markdown(f"""
    <div class="insight-card">
        <h4>{titleText}</h4>
        <p>{bodyText}</p>
    </div>
    """, unsafe_allow_html=True)


def renderSectionNote(noteText: str):
    st.markdown(f"<div class='section-note'>{noteText}</div>", unsafe_allow_html=True)


def createCategoricalCountChart(frame: pd.DataFrame, columnName: str, titleText: str) -> go.Figure:
    countFrame = frame[columnName].value_counts(dropna=False).reset_index()
    countFrame.columns = [columnName, "Count"]
    figObject = px.bar(
        countFrame,
        x=columnName,
        y="Count",
        text="Count",
        labels={columnName: toChineseColumnName(columnName), "Count": "样本数"},
    )
    figObject.update_traces(texttemplate="%{text:,}", textposition="outside")
    return figureLayout(figObject, titleText)


def createDistributionChart(frame: pd.DataFrame, columnName: str, titleText: str) -> go.Figure:
    figObject = px.histogram(
        frame,
        x=columnName,
        nbins=45,
        marginal="box",
        labels={columnName: toChineseColumnName(columnName)},
    )
    figObject.update_layout(yaxis_title="样本数", xaxis_title=toChineseColumnName(columnName))
    return figureLayout(figObject, titleText)


def createBoxChart(frame: pd.DataFrame, xColumn: str, yColumn: str, titleText: str) -> go.Figure:
    figObject = px.box(
        frame,
        x=xColumn,
        y=yColumn,
        points=False,
        labels={xColumn: toChineseColumnName(xColumn), yColumn: toChineseColumnName(yColumn)},
    )
    return figureLayout(figObject, titleText)


def createViolinChart(frame: pd.DataFrame, xColumn: str, yColumn: str, titleText: str) -> go.Figure:
    figObject = px.violin(
        frame,
        x=xColumn,
        y=yColumn,
        box=True,
        points=False,
        labels={xColumn: toChineseColumnName(xColumn), yColumn: toChineseColumnName(yColumn)},
    )
    return figureLayout(figObject, titleText)


def createGroupedMeanChart(frame: pd.DataFrame, groupColumn: str, valueColumn: str, titleText: str, orderList: Optional[List[str]] = None) -> go.Figure:
    summaryFrame = makeCiSummary(frame, groupColumn, valueColumn)
    if orderList is not None:
        summaryFrame = sortByKnownOrder(summaryFrame, groupColumn, orderList)
    figObject = px.bar(
        summaryFrame,
        x=groupColumn,
        y="mean",
        error_y="Ci95",
        text=summaryFrame["mean"].map(lambda valueItem: f"{valueItem:.3f}"),
        labels={groupColumn: toChineseColumnName(groupColumn), "mean": toChineseColumnName(valueColumn), "Ci95": "95%置信区间"},
    )
    figObject.update_traces(textposition="outside")
    return figureLayout(figObject, titleText)


def createHeatmap(matrixFrame: pd.DataFrame, titleText: str, zFormat: str = ".2f", colorTitle: str = "数值") -> go.Figure:
    figObject = go.Figure(
        data=go.Heatmap(
            z=matrixFrame.values,
            x=matrixFrame.columns.astype(str),
            y=matrixFrame.index.astype(str),
            colorscale="Blues",
            colorbar=dict(title=colorTitle),
            text=np.round(matrixFrame.values.astype(float), 3),
            hovertemplate="横轴=%{x}<br>纵轴=%{y}<br>数值=%{z:" + zFormat + "}<extra></extra>",
        )
    )
    return figureLayout(figObject, titleText, heightValue=max(430, 35 * len(matrixFrame.index) + 120))


def createCorrelationHeatmap(frame: pd.DataFrame, numericColumns: List[str]) -> go.Figure:
    corrFrame = frame[numericColumns].corr(numeric_only=True)
    corrFrame.index = [toChineseColumnName(columnItem) for columnItem in corrFrame.index]
    corrFrame.columns = [toChineseColumnName(columnItem) for columnItem in corrFrame.columns]
    figObject = go.Figure(
        data=go.Heatmap(
            z=corrFrame.values,
            x=corrFrame.columns,
            y=corrFrame.index,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            colorbar=dict(title="相关系数"),
            text=np.round(corrFrame.values, 2),
            texttemplate="%{text}",
            hovertemplate="%{y} 与 %{x}<br>相关系数=%{z:.3f}<extra></extra>",
        )
    )
    return figureLayout(figObject, "核心数值指标相关性热力图", heightValue=720)


def createPivotMean(frame: pd.DataFrame, indexColumn: str, columnColumn: str, valueColumn: str, aggFunc="mean") -> pd.DataFrame:
    pivotFrame = frame.pivot_table(index=indexColumn, columns=columnColumn, values=valueColumn, aggfunc=aggFunc, observed=False)
    return pivotFrame.fillna(0)


def makeSampleFrame(frame: pd.DataFrame, sampleSize: int, randomState: int = 42) -> pd.DataFrame:
    if len(frame) <= sampleSize:
        return frame.copy()
    return frame.sample(sampleSize, random_state=randomState)


# -----------------------------------------------------------------------------
# 建模函数
# -----------------------------------------------------------------------------


def getEncoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@st.cache_data(show_spinner=False)
def runRegressionModel(frame: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    featureColumns = [columnItem for columnItem in MODEL_FEATURE_COLUMNS if columnItem in frame.columns and columnItem != "SkillRetentionScore"]
    targetColumn = "GpaChange"
    modelFrame = frame[featureColumns + [targetColumn]].dropna().copy()
    numericFeatureColumns = [columnItem for columnItem in featureColumns if pd.api.types.is_numeric_dtype(modelFrame[columnItem])]
    categoricalFeatureColumns = [columnItem for columnItem in featureColumns if columnItem not in numericFeatureColumns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numericFeatureColumns),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", getEncoder())]), categoricalFeatureColumns),
        ]
    )
    modelPipeline = Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())])
    xFrame = modelFrame[featureColumns]
    ySeries = modelFrame[targetColumn]
    modelPipeline.fit(xFrame, ySeries)
    predictionValues = modelPipeline.predict(xFrame)
    scoreValue = float(r2_score(ySeries, predictionValues))

    featureNameList = []
    if numericFeatureColumns:
        featureNameList.extend([toChineseColumnName(columnItem) for columnItem in numericFeatureColumns])
    if categoricalFeatureColumns:
        encoderStep = modelPipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
        encodedNames = encoderStep.get_feature_names_out(categoricalFeatureColumns)
        featureNameList.extend([nameItem.replace("_", "=") for nameItem in encodedNames])

    coefficientFrame = pd.DataFrame({"Feature": featureNameList, "Coefficient": modelPipeline.named_steps["model"].coef_})
    coefficientFrame["AbsCoefficient"] = coefficientFrame["Coefficient"].abs()
    coefficientFrame = coefficientFrame.sort_values("AbsCoefficient", ascending=False).head(18)
    return coefficientFrame, scoreValue


@st.cache_data(show_spinner=False)
def runRandomForestModels(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    featureColumns = [columnItem for columnItem in MODEL_FEATURE_COLUMNS if columnItem in frame.columns]
    targetGpa = "GpaChange"
    targetBurnout = "HighBurnoutFlag"
    modelFrame = frame[featureColumns + [targetGpa, targetBurnout]].dropna().copy()
    numericFeatureColumns = [columnItem for columnItem in featureColumns if pd.api.types.is_numeric_dtype(modelFrame[columnItem])]
    categoricalFeatureColumns = [columnItem for columnItem in featureColumns if columnItem not in numericFeatureColumns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numericFeatureColumns),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", getEncoder())]), categoricalFeatureColumns),
        ]
    )

    xFrame = modelFrame[featureColumns]
    yGpa = modelFrame[targetGpa]
    yBurnout = modelFrame[targetBurnout]
    xTrain, xTest, yGpaTrain, yGpaTest, yBurnoutTrain, yBurnoutTest = train_test_split(
        xFrame,
        yGpa,
        yBurnout,
        test_size=0.25,
        random_state=42,
        stratify=yBurnout if yBurnout.nunique() > 1 else None,
    )

    regressorPipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=160, random_state=42, min_samples_leaf=25, n_jobs=-1)),
        ]
    )
    classifierPipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=160, random_state=42, min_samples_leaf=25, n_jobs=-1, class_weight="balanced")),
        ]
    )
    regressorPipeline.fit(xTrain, yGpaTrain)
    classifierPipeline.fit(xTrain, yBurnoutTrain)

    transformedPreprocessor = regressorPipeline.named_steps["preprocessor"]
    featureNameList = []
    if numericFeatureColumns:
        featureNameList.extend([toChineseColumnName(columnItem) for columnItem in numericFeatureColumns])
    if categoricalFeatureColumns:
        encoderStep = transformedPreprocessor.named_transformers_["cat"].named_steps["encoder"]
        encodedNames = encoderStep.get_feature_names_out(categoricalFeatureColumns)
        featureNameList.extend([nameItem.replace("_", "=") for nameItem in encodedNames])

    gpaImportanceFrame = pd.DataFrame(
        {"Feature": featureNameList, "Importance": regressorPipeline.named_steps["model"].feature_importances_}
    ).sort_values("Importance", ascending=False).head(18)
    burnoutImportanceFrame = pd.DataFrame(
        {"Feature": featureNameList, "Importance": classifierPipeline.named_steps["model"].feature_importances_}
    ).sort_values("Importance", ascending=False).head(18)

    metricDict = {
        "GpaR2": float(r2_score(yGpaTest, regressorPipeline.predict(xTest))),
        "BurnoutAccuracy": float(accuracy_score(yBurnoutTest, classifierPipeline.predict(xTest))),
    }
    return gpaImportanceFrame, burnoutImportanceFrame, metricDict


@st.cache_data(show_spinner=False)
def runClusterModel(frame: pd.DataFrame, clusterCount: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    clusterColumns = [
        "WeeklyGenAiHours",
        "TraditionalStudyHours",
        "GenAiStudyShare",
        "ToolDiversity",
        "PerceivedAiDependency",
        "AnxietyLevelDuringExams",
        "SkillRetentionScore",
        "GpaChange",
        "HighBurnoutFlag",
    ]
    clusterFrame = frame[clusterColumns].dropna().copy()
    scalerObject = StandardScaler()
    scaledValues = scalerObject.fit_transform(clusterFrame)
    kmeansObject = KMeans(n_clusters=clusterCount, random_state=42, n_init=20)
    clusterLabels = kmeansObject.fit_predict(scaledValues)
    silhouetteValue = float(silhouette_score(scaledValues, clusterLabels)) if clusterCount > 1 and len(set(clusterLabels)) > 1 else np.nan

    pcaObject = PCA(n_components=2, random_state=42)
    pcaValues = pcaObject.fit_transform(scaledValues)
    projectionFrame = clusterFrame.copy()
    projectionFrame["Cluster"] = clusterLabels
    projectionFrame["PcaOne"] = pcaValues[:, 0]
    projectionFrame["PcaTwo"] = pcaValues[:, 1]

    profileFrame = projectionFrame.groupby("Cluster")[clusterColumns].mean().reset_index()
    profileFrame["ClusterSize"] = projectionFrame.groupby("Cluster").size().values
    profileFrame["ClusterShare"] = profileFrame["ClusterSize"] / len(projectionFrame)
    profileFrame["ClusterName"] = profileFrame.apply(assignClusterName, axis=1)
    nameMap = dict(zip(profileFrame["Cluster"], profileFrame["ClusterName"]))
    projectionFrame["ClusterName"] = projectionFrame["Cluster"].map(nameMap)
    return projectionFrame, profileFrame, pd.DataFrame({"Feature": clusterColumns, "PcaOneLoading": pcaObject.components_[0], "PcaTwoLoading": pcaObject.components_[1]}), silhouetteValue


def assignClusterName(rowItem: pd.Series) -> str:
    if rowItem["WeeklyGenAiHours"] >= 15 and rowItem["PerceivedAiDependency"] >= 5:
        return "高依赖风险型"
    if rowItem["TraditionalStudyHours"] >= 12 and rowItem["WeeklyGenAiHours"] < 6:
        return "传统主导型"
    if rowItem["WeeklyGenAiHours"] >= 5 and rowItem["WeeklyGenAiHours"] < 12 and rowItem["SkillRetentionScore"] >= 75:
        return "适度辅助型"
    if rowItem["ToolDiversity"] >= 3 and rowItem["GenAiStudyShare"] >= 0.4:
        return "工具探索型"
    return "低投入观察型"


# -----------------------------------------------------------------------------
# 图表函数
# -----------------------------------------------------------------------------


def renderDataImportTab(frame: pd.DataFrame, missingAudit: pd.DataFrame, outlierAudit: pd.DataFrame, cleaningLog: List[str]):
    st.subheader("1. 原始数据导入、数据清洗与预处理")
    renderSectionNote(
        "本页面完整展示从原始 CSV 到分析数据集的处理流程：字段标准化、类型转换、缺失值审计、重复记录处理、业务范围校验、异常值审计和派生变量生成。异常值默认保留，因为高 AI 使用、高依赖和低技能保留样本正是本研究的关键观察对象。"
    )

    columnA, columnB, columnC, columnD = st.columns(4)
    columnA.metric("清洗后样本数", f"{len(frame):,}")
    columnB.metric("字段数量", f"{frame.shape[1]:,}")
    columnC.metric("缺失单元格", f"{int(frame.isna().sum().sum()):,}")
    columnD.metric("学生ID唯一数", f"{frame['StudentId'].nunique():,}")

    leftColumn, rightColumn = st.columns([1, 1])
    with leftColumn:
        figObject = px.bar(
            missingAudit.sort_values("MissingCount", ascending=False),
            x="Field",
            y="MissingCount",
            text="MissingCount",
            labels={"Field": "字段", "MissingCount": "缺失值数量"},
        )
        figObject.update_traces(textposition="outside")
        plotlyChart(figureLayout(figObject, "原始数据缺失值审计", heightValue=460))
    with rightColumn:
        outlierDisplay = outlierAudit.sort_values("OutlierCount", ascending=False)
        figObject = px.bar(
            outlierDisplay,
            x="ChineseField",
            y="OutlierCount",
            text="OutlierCount",
            labels={"ChineseField": "字段", "OutlierCount": "IQR异常值数量"},
        )
        figObject.update_traces(textposition="outside")
        plotlyChart(figureLayout(figObject, "数值变量 IQR 异常值审计", heightValue=460))

    st.markdown("#### 数据清洗日志")
    for logItem in cleaningLog:
        st.write("- " + logItem)

    st.markdown("#### 字段字典与派生变量")
    dictionaryRecords = []
    for columnItem in [
        "StudentId", "MajorCategory", "YearOfStudy", "PreSemesterGpa", "PostSemesterGpa", "GpaChange",
        "WeeklyGenAiHours", "TraditionalStudyHours", "TotalStudyHours", "GenAiStudyShare", "PrimaryUseCase",
        "PromptEngineeringSkill", "ToolDiversity", "PaidSubscription", "PerceivedAiDependency", "InstitutionalPolicy",
        "AnxietyLevelDuringExams", "SkillRetentionScore", "BurnoutRiskLevel", "HighBurnoutFlag", "GenAiHoursBand", "GenAiShareBand",
    ]:
        if columnItem in frame.columns:
            dictionaryRecords.append({"标准字段名": columnItem, "中文含义": CHINESE_LABELS.get(columnItem, columnItem), "数据类型": str(frame[columnItem].dtype)})
    st.dataframe(pd.DataFrame(dictionaryRecords), use_container_width=True, hide_index=True)

    st.markdown("#### 清洗后数据预览")
    displayColumns = [columnItem for columnItem in [
        "StudentId", "MajorCategoryCn", "YearOfStudyCn", "PreSemesterGpa", "PostSemesterGpa", "GpaChange",
        "WeeklyGenAiHours", "TraditionalStudyHours", "GenAiStudyShare", "PrimaryUseCaseCn", "PromptEngineeringSkillCn",
        "PerceivedAiDependency", "AnxietyLevelDuringExams", "SkillRetentionScore", "BurnoutRiskLevelCn",
    ] if columnItem in frame.columns]
    st.dataframe(frame[displayColumns].head(1000), use_container_width=True, hide_index=True)

    st.download_button(
        label="下载清洗后的分析数据 CSV",
        data=createDownloadBytes(frame),
        file_name="CleanedAiStudentImpactData.csv",
        mime="text/csv",
    )


def renderOverviewTab(frame: pd.DataFrame, overallFrame: pd.DataFrame):
    st.subheader("2. 项目总览与总体画像")
    renderSectionNote(
        "本页用于回答：样本总体学业变化如何？AI 使用强度处在什么水平？学生的依赖、焦虑、倦怠风险和技能保留总体状况如何？"
    )

    metricColumns = st.columns(6)
    metricColumns[0].metric("样本数", f"{len(frame):,}", makeMetricDelta(len(frame), len(overallFrame), 0))
    metricColumns[1].metric("平均GPA变化", formatNumber(frame["GpaChange"].mean(), 3), makeMetricDelta(frame["GpaChange"].mean(), overallFrame["GpaChange"].mean(), 3))
    metricColumns[2].metric("AI时长均值", formatNumber(frame["WeeklyGenAiHours"].mean(), 2), "小时/周")
    metricColumns[3].metric("传统学习均值", formatNumber(frame["TraditionalStudyHours"].mean(), 2), "小时/周")
    metricColumns[4].metric("技能保留均值", formatNumber(frame["SkillRetentionScore"].mean(), 2), makeMetricDelta(frame["SkillRetentionScore"].mean(), overallFrame["SkillRetentionScore"].mean(), 2))
    metricColumns[5].metric("高倦怠比例", formatPercent(frame["HighBurnoutFlag"].mean()), makeMetricDelta(frame["HighBurnoutFlag"].mean(), overallFrame["HighBurnoutFlag"].mean(), 3))

    insightColumns = st.columns(4)
    with insightColumns[0]:
        renderInsightCard("学业表现", f"筛选样本的 GPA 平均变化为 {frame['GpaChange'].mean():.3f}，中位数为 {frame['GpaChange'].median():.3f}。")
    with insightColumns[1]:
        renderInsightCard("AI使用强度", f"每周 AI 使用时长均值为 {frame['WeeklyGenAiHours'].mean():.2f} 小时，AI 学习占比均值为 {frame['GenAiStudyShare'].mean():.1%}。")
    with insightColumns[2]:
        renderInsightCard("学习保留", f"技能保留分数均值为 {frame['SkillRetentionScore'].mean():.2f}，低保留样本比例为 {frame['LowRetentionFlag'].mean():.1%}。")
    with insightColumns[3]:
        renderInsightCard("心理风险", f"高倦怠风险比例为 {frame['HighBurnoutFlag'].mean():.1%}，考试焦虑均值为 {frame['AnxietyLevelDuringExams'].mean():.2f}。")

    leftColumn, rightColumn = st.columns([1, 1])
    with leftColumn:
        figObject = go.Figure()
        figObject.add_trace(go.Histogram(x=frame["PreSemesterGpa"], nbinsx=40, name="AI使用前GPA", opacity=0.65))
        figObject.add_trace(go.Histogram(x=frame["PostSemesterGpa"], nbinsx=40, name="AI使用后GPA", opacity=0.65))
        figObject.update_layout(barmode="overlay", xaxis_title="GPA", yaxis_title="样本数")
        plotlyChart(figureLayout(figObject, "AI使用前后 GPA 分布对比", heightValue=480))
    with rightColumn:
        figObject = createDistributionChart(frame, "GpaChange", "GPA变化分布：成绩提升与下降样本的整体形态")
        figObject.add_vline(x=0, line_dash="dash", annotation_text="无变化", annotation_position="top")
        plotlyChart(figObject)

    chartColumns = st.columns([1, 1])
    with chartColumns[0]:
        plotlyChart(createDistributionChart(frame, "WeeklyGenAiHours", "每周生成式AI使用时长分布"))
    with chartColumns[1]:
        plotlyChart(createDistributionChart(frame, "GenAiStudyShare", "AI学习时间占比分布"))

    summaryColumns = [columnItem for columnItem in NUMERIC_COLUMNS if columnItem in frame.columns]
    summaryFrame = frame[summaryColumns].describe().T.reset_index()
    summaryFrame["index"] = summaryFrame["index"].map(toChineseColumnName)
    summaryFrame = summaryFrame.rename(columns={"index": "指标", "count": "样本数", "mean": "均值", "std": "标准差", "min": "最小值", "25%": "下四分位", "50%": "中位数", "75%": "上四分位", "max": "最大值"})
    st.markdown("#### 描述性统计表")
    st.dataframe(summaryFrame.round(3), use_container_width=True, hide_index=True)


def renderAcademicOutcomeTab(frame: pd.DataFrame):
    st.subheader("3. 学业成绩变化分析")
    renderSectionNote(
        "本页集中展示 GPA 前后变化、AI 使用时长与 GPA 变化、AI 学习占比与 GPA 变化、传统学习投入与成绩变化之间的关系，用于支撑论文中‘适度辅助有益、过度依赖有风险’的核心判断。"
    )

    leftColumn, rightColumn = st.columns([1, 1])
    with leftColumn:
        hourOrder = ["0-2小时", "2-5小时", "5-10小时", "10-20小时", "20小时以上"]
        summaryFrame = makeCiSummary(frame, "GenAiHoursBand", "GpaChange")
        summaryFrame = sortByKnownOrder(summaryFrame, "GenAiHoursBand", hourOrder)
        figObject = go.Figure()
        figObject.add_trace(
            go.Scatter(
                x=summaryFrame["GenAiHoursBand"].astype(str),
                y=summaryFrame["mean"],
                mode="lines+markers",
                error_y=dict(type="data", array=summaryFrame["Ci95"], visible=True),
                name="平均GPA变化",
                hovertemplate="AI时长分组=%{x}<br>平均GPA变化=%{y:.3f}<extra></extra>",
            )
        )
        figObject.update_layout(xaxis_title="AI使用时长分组", yaxis_title="平均GPA变化")
        plotlyChart(figureLayout(figObject, "不同AI使用时长组的GPA变化趋势", heightValue=470))
    with rightColumn:
        shareOrder = ["0%-20%", "20%-40%", "40%-60%", "60%-80%", "80%-100%"]
        plotlyChart(createGroupedMeanChart(frame, "GenAiShareBand", "GpaChange", "不同AI学习时间占比下的GPA变化", shareOrder))

    sampleFrame = makeSampleFrame(frame, 5000)
    leftColumn, rightColumn = st.columns([1, 1])
    with leftColumn:
        figObject = px.scatter(
            sampleFrame,
            x="PreSemesterGpa",
            y="PostSemesterGpa",
            color="GenAiHoursBand",
            opacity=0.55,
            trendline="ols",
            labels={"PreSemesterGpa": "AI使用前GPA", "PostSemesterGpa": "AI使用后GPA", "GenAiHoursBand": "AI使用时长分组"},
        )
        figObject.add_trace(go.Scatter(x=[0, 4], y=[0, 4], mode="lines", name="前后相等参考线", line=dict(dash="dash")))
        plotlyChart(figureLayout(figObject, "GPA前后对比散点图（抽样显示）", heightValue=500))
    with rightColumn:
        figObject = px.scatter(
            sampleFrame,
            x="WeeklyGenAiHours",
            y="GpaChange",
            color="TraditionalStudyHours",
            opacity=0.55,
            trendline="ols",
            labels={"WeeklyGenAiHours": "每周AI使用时长", "GpaChange": "GPA变化", "TraditionalStudyHours": "传统学习时长"},
        )
        figObject.add_hline(y=0, line_dash="dash")
        plotlyChart(figureLayout(figObject, "AI使用时长与GPA变化散点图（颜色表示传统学习投入）", heightValue=500))

    gpaByHour = frame.groupby("GenAiHoursBand", observed=False).agg(
        样本数=("StudentId", "count"),
        平均GPA变化=("GpaChange", "mean"),
        GPA提升比例=("GpaImprovedFlag", "mean"),
        技能保留均值=("SkillRetentionScore", "mean"),
        高倦怠比例=("HighBurnoutFlag", "mean"),
    ).reset_index()
    gpaByHour = sortByKnownOrder(gpaByHour, "GenAiHoursBand", hourOrder)
    st.markdown("#### AI使用时长分组统计")
    st.dataframe(gpaByHour.round(4), use_container_width=True, hide_index=True)


def renderAiBehaviorTab(frame: pd.DataFrame):
    st.subheader("4. AI使用行为与学习方式分析")
    renderSectionNote(
        "本页关注学生如何使用 AI：每周使用强度、AI 在总学习时间中的占比、主要用途、工具多样性和付费订阅情况。该部分对应论文中关于‘AI 使用方式比使用频率更重要’的分析。"
    )

    chartColumns = st.columns([1, 1])
    with chartColumns[0]:
        useCaseSummary = frame.groupby("PrimaryUseCaseCn", observed=False).agg(
            样本数=("StudentId", "count"),
            平均GPA变化=("GpaChange", "mean"),
            技能保留均值=("SkillRetentionScore", "mean"),
            高倦怠比例=("HighBurnoutFlag", "mean"),
        ).reset_index().sort_values("平均GPA变化", ascending=False)
        figObject = make_subplots(specs=[[{"secondary_y": True}]])
        figObject.add_trace(go.Bar(x=useCaseSummary["PrimaryUseCaseCn"], y=useCaseSummary["平均GPA变化"], name="平均GPA变化"), secondary_y=False)
        figObject.add_trace(go.Scatter(x=useCaseSummary["PrimaryUseCaseCn"], y=useCaseSummary["技能保留均值"], name="技能保留均值", mode="lines+markers"), secondary_y=True)
        figObject.update_yaxes(title_text="平均GPA变化", secondary_y=False)
        figObject.update_yaxes(title_text="技能保留均值", secondary_y=True)
        figObject.update_layout(xaxis_title="AI主要用途")
        plotlyChart(figureLayout(figObject, "不同AI使用目的下的GPA变化与技能保留", heightValue=500))
    with chartColumns[1]:
        figObject = px.box(
            frame,
            x="PrimaryUseCaseCn",
            y="WeeklyGenAiHours",
            color="PrimaryUseCaseCn",
            labels={"PrimaryUseCaseCn": "AI主要用途", "WeeklyGenAiHours": "每周AI使用时长"},
        )
        figObject.update_layout(showlegend=False)
        plotlyChart(figureLayout(figObject, "不同AI使用目的下的AI使用时长分布", heightValue=500))

    chartColumns = st.columns([1, 1])
    with chartColumns[0]:
        toolSummary = frame.groupby("ToolDiversity", observed=False).agg(
            平均GPA变化=("GpaChange", "mean"),
            技能保留均值=("SkillRetentionScore", "mean"),
            平均AI依赖度=("PerceivedAiDependency", "mean"),
            样本数=("StudentId", "count"),
        ).reset_index()
        figObject = px.line(
            toolSummary,
            x="ToolDiversity",
            y=["平均GPA变化", "平均AI依赖度"],
            markers=True,
            labels={"ToolDiversity": "AI工具多样性", "value": "指标值", "variable": "指标"},
        )
        plotlyChart(figureLayout(figObject, "工具多样性与GPA变化、AI依赖度的关系", heightValue=470))
    with chartColumns[1]:
        paidSummary = frame.groupby("PaidSubscriptionCn", observed=False).agg(
            平均GPA变化=("GpaChange", "mean"),
            技能保留均值=("SkillRetentionScore", "mean"),
            AI时长均值=("WeeklyGenAiHours", "mean"),
            高倦怠比例=("HighBurnoutFlag", "mean"),
        ).reset_index()
        figObject = px.bar(
            paidSummary.melt(id_vars="PaidSubscriptionCn", var_name="指标", value_name="数值"),
            x="PaidSubscriptionCn",
            y="数值",
            color="指标",
            barmode="group",
            labels={"PaidSubscriptionCn": "是否付费订阅"},
        )
        plotlyChart(figureLayout(figObject, "付费订阅与学习结果、AI使用强度对比", heightValue=470))

    st.markdown("#### AI使用目的统计表")
    st.dataframe(useCaseSummary.round(4), use_container_width=True, hide_index=True)


def renderGroupDifferenceTab(frame: pd.DataFrame):
    st.subheader("5. 专业、年级与学习群体差异")
    renderSectionNote(
        "本页展示不同专业、不同年级学生在 AI 使用强度、GPA 变化、技能保留、AI 依赖度和倦怠风险方面的差异，用于支持论文中的群体结构分析。"
    )

    profileColumns = ["WeeklyGenAiHours", "GenAiStudyShare", "GpaChange", "SkillRetentionScore", "PerceivedAiDependency", "HighBurnoutFlag"]
    majorProfile = frame.groupby("MajorCategoryCn", observed=False)[profileColumns].mean()
    majorProfile.columns = [toChineseColumnName(columnItem) for columnItem in profileColumns]
    standardizedMajorProfile = (majorProfile - majorProfile.mean()) / majorProfile.std(ddof=0).replace(0, 1)
    plotlyChart(createHeatmap(standardizedMajorProfile.round(2), "不同专业类别学生AI使用与学习结果标准化画像", zFormat=".2f", colorTitle="标准化值"))

    chartColumns = st.columns([1, 1])
    with chartColumns[0]:
        yearSummary = frame.groupby("YearOfStudyCn", observed=False).agg(
            AI时长均值=("WeeklyGenAiHours", "mean"),
            平均GPA变化=("GpaChange", "mean"),
            技能保留均值=("SkillRetentionScore", "mean"),
            高倦怠比例=("HighBurnoutFlag", "mean"),
        ).reset_index()
        figObject = px.bar(
            yearSummary.melt(id_vars="YearOfStudyCn", var_name="指标", value_name="数值"),
            x="YearOfStudyCn",
            y="数值",
            color="指标",
            barmode="group",
            labels={"YearOfStudyCn": "年级"},
        )
        plotlyChart(figureLayout(figObject, "不同年级学生AI使用与学习结果差异", heightValue=500))
    with chartColumns[1]:
        interactionPivot = createPivotMean(frame, "TraditionalHoursBand", "GenAiHoursBand", "GpaChange")
        plotlyChart(createHeatmap(interactionPivot.round(3), "传统学习时长 × AI使用时长下的平均GPA变化", zFormat=".3f", colorTitle="平均GPA变化"))

    sampleFrame = makeSampleFrame(frame, 5000)
    figObject = px.scatter(
        sampleFrame,
        x="TraditionalStudyHours",
        y="WeeklyGenAiHours",
        color="GpaChange",
        size="SkillRetentionScore",
        hover_data=["MajorCategoryCn", "YearOfStudyCn", "PrimaryUseCaseCn", "BurnoutRiskLevelCn"],
        labels={"TraditionalStudyHours": "每周传统学习时长", "WeeklyGenAiHours": "每周AI使用时长", "GpaChange": "GPA变化", "SkillRetentionScore": "技能保留"},
    )
    plotlyChart(figureLayout(figObject, "传统学习投入与AI学习投入的交互分布（抽样显示）", heightValue=620))


def renderAiLiteracyPolicyTab(frame: pd.DataFrame):
    st.subheader("6. AI素养、制度政策与学习结果")
    renderSectionNote(
        "本页用于分析提示词能力、学校 AI 政策与学生学习结果之间的关系。它强调 AI 教育应用不仅是技术问题，也涉及学生 AI 素养和学校治理环境。"
    )

    chartColumns = st.columns([1, 1])
    with chartColumns[0]:
        skillOrderCn = ["初级", "中级", "高级"]
        figObject = px.violin(
            frame,
            x="PromptEngineeringSkillCn",
            y="GpaChange",
            color="PromptEngineeringSkillCn",
            box=True,
            points=False,
            category_orders={"PromptEngineeringSkillCn": skillOrderCn},
            labels={"PromptEngineeringSkillCn": "提示词能力", "GpaChange": "GPA变化"},
        )
        figObject.update_layout(showlegend=False)
        plotlyChart(figureLayout(figObject, "不同提示词能力学生的GPA变化分布", heightValue=500))
    with chartColumns[1]:
        figObject = px.violin(
            frame,
            x="PromptEngineeringSkillCn",
            y="SkillRetentionScore",
            color="PromptEngineeringSkillCn",
            box=True,
            points=False,
            category_orders={"PromptEngineeringSkillCn": skillOrderCn},
            labels={"PromptEngineeringSkillCn": "提示词能力", "SkillRetentionScore": "技能保留分数"},
        )
        figObject.update_layout(showlegend=False)
        plotlyChart(figureLayout(figObject, "不同提示词能力学生的技能保留分布", heightValue=500))

    chartColumns = st.columns([1, 1])
    with chartColumns[0]:
        policySummary = frame.groupby("InstitutionalPolicyCn", observed=False).agg(
            平均GPA变化=("GpaChange", "mean"),
            技能保留均值=("SkillRetentionScore", "mean"),
            AI依赖度均值=("PerceivedAiDependency", "mean"),
            高倦怠比例=("HighBurnoutFlag", "mean"),
        ).reset_index().sort_values("平均GPA变化", ascending=False)
        figObject = px.bar(
            policySummary.melt(id_vars="InstitutionalPolicyCn", var_name="指标", value_name="数值"),
            x="InstitutionalPolicyCn",
            y="数值",
            color="指标",
            barmode="group",
            labels={"InstitutionalPolicyCn": "学校AI政策"},
        )
        plotlyChart(figureLayout(figObject, "不同学校AI政策环境下的学习结果差异", heightValue=520))
    with chartColumns[1]:
        skillUsePivot = createPivotMean(frame, "PromptEngineeringSkillCn", "PrimaryUseCaseCn", "GpaChange")
        plotlyChart(createHeatmap(skillUsePivot.round(3), "提示词能力 × AI使用目的下的平均GPA变化", zFormat=".3f", colorTitle="平均GPA变化"))

    skillSummary = frame.groupby("PromptEngineeringSkillCn", observed=False).agg(
        样本数=("StudentId", "count"),
        AI时长均值=("WeeklyGenAiHours", "mean"),
        平均GPA变化=("GpaChange", "mean"),
        技能保留均值=("SkillRetentionScore", "mean"),
        AI依赖度均值=("PerceivedAiDependency", "mean"),
        高倦怠比例=("HighBurnoutFlag", "mean"),
    ).reset_index()
    st.markdown("#### 提示词能力分组统计")
    st.dataframe(skillSummary.round(4), use_container_width=True, hide_index=True)


def renderRiskTab(frame: pd.DataFrame):
    st.subheader("7. AI依赖、考试焦虑、技能保留与倦怠风险")
    renderSectionNote(
        "本页从心理风险角度分析生成式 AI 使用行为。重点观察高 AI 依赖、高考试焦虑、高 AI 学习占比和低技能保留是否形成重叠风险区。"
    )

    chartColumns = st.columns([1, 1])
    with chartColumns[0]:
        riskPivot = createPivotMean(frame, "AnxietyLevelDuringExams", "PerceivedAiDependency", "HighBurnoutFlag")
        plotlyChart(createHeatmap(riskPivot.round(3), "AI依赖度与考试焦虑水平下的高倦怠风险比例", zFormat=".1%", colorTitle="高倦怠比例"))
    with chartColumns[1]:
        riskMatrix = createPivotMean(frame, "DependencyBand", "GenAiShareBand", "HighBurnoutFlag")
        plotlyChart(createHeatmap(riskMatrix.round(3), "AI依赖分组 × AI学习占比分组下的高倦怠风险比例", zFormat=".1%", colorTitle="高倦怠比例"))

    chartColumns = st.columns([1, 1])
    with chartColumns[0]:
        burnoutSummary = frame.groupby("GenAiHoursBand", observed=False).agg(
            技能保留均值=("SkillRetentionScore", "mean"),
            高倦怠比例=("HighBurnoutFlag", "mean"),
        ).reset_index()
        hourOrder = ["0-2小时", "2-5小时", "5-10小时", "10-20小时", "20小时以上"]
        burnoutSummary = sortByKnownOrder(burnoutSummary, "GenAiHoursBand", hourOrder)
        figObject = make_subplots(specs=[[{"secondary_y": True}]])
        figObject.add_trace(go.Bar(x=burnoutSummary["GenAiHoursBand"].astype(str), y=burnoutSummary["技能保留均值"], name="技能保留均值"), secondary_y=False)
        figObject.add_trace(go.Scatter(x=burnoutSummary["GenAiHoursBand"].astype(str), y=burnoutSummary["高倦怠比例"], name="高倦怠比例", mode="lines+markers"), secondary_y=True)
        figObject.update_yaxes(title_text="技能保留均值", secondary_y=False)
        figObject.update_yaxes(title_text="高倦怠比例", secondary_y=True, tickformat=".0%")
        plotlyChart(figureLayout(figObject, "AI使用强度、技能保留与高倦怠风险", heightValue=500))
    with chartColumns[1]:
        figObject = px.box(
            frame,
            x="BurnoutRiskLevelCn",
            y="SkillRetentionScore",
            color="BurnoutRiskLevelCn",
            labels={"BurnoutRiskLevelCn": "倦怠风险等级", "SkillRetentionScore": "技能保留分数"},
        )
        figObject.update_layout(showlegend=False)
        plotlyChart(figureLayout(figObject, "不同倦怠风险等级下的技能保留分布", heightValue=500))

    sampleFrame = makeSampleFrame(frame, 5000)
    figObject = px.scatter(
        sampleFrame,
        x="PerceivedAiDependency",
        y="SkillRetentionScore",
        color="BurnoutRiskLevelCn",
        size="WeeklyGenAiHours",
        opacity=0.55,
        labels={"PerceivedAiDependency": "AI依赖度", "SkillRetentionScore": "技能保留分数", "BurnoutRiskLevelCn": "倦怠风险等级", "WeeklyGenAiHours": "AI使用时长"},
    )
    plotlyChart(figureLayout(figObject, "AI依赖度、技能保留与倦怠风险散点图（抽样显示）", heightValue=600))


def renderModelingTab(frame: pd.DataFrame):
    st.subheader("8. 相关性、回归、随机森林与聚类建模")
    renderSectionNote(
        "本页用于展示论文第 3.2 节的数据建模部分。模型不用于声称因果关系，而是用于识别与 GPA 变化、高倦怠风险和学习行为类型相关的重要特征。"
    )

    with st.spinner("正在计算相关性、回归、随机森林与聚类模型……"):
        numericColumns = [columnItem for columnItem in NUMERIC_COLUMNS + ["HighBurnoutFlag"] if columnItem in frame.columns]
        coefficientFrame, regressionScore = runRegressionModel(frame)
        gpaImportanceFrame, burnoutImportanceFrame, metricDict = runRandomForestModels(frame)
        projectionFrame, profileFrame, loadingFrame, silhouetteValue = runClusterModel(frame, clusterCount=4)

    metricColumns = st.columns(4)
    metricColumns[0].metric("线性回归 R²", formatNumber(regressionScore, 3))
    metricColumns[1].metric("随机森林 GPA R²", formatNumber(metricDict["GpaR2"], 3))
    metricColumns[2].metric("高倦怠分类准确率", formatPercent(metricDict["BurnoutAccuracy"]))
    metricColumns[3].metric("聚类轮廓系数", formatNumber(silhouetteValue, 3))

    plotlyChart(createCorrelationHeatmap(frame, numericColumns))

    chartColumns = st.columns([1, 1])
    with chartColumns[0]:
        displayFrame = coefficientFrame.sort_values("Coefficient")
        figObject = px.bar(
            displayFrame,
            x="Coefficient",
            y="Feature",
            orientation="h",
            labels={"Coefficient": "标准化回归系数", "Feature": "特征"},
        )
        figObject.add_vline(x=0, line_dash="dash")
        plotlyChart(figureLayout(figObject, "GPA变化线性回归系数解释", heightValue=620))
    with chartColumns[1]:
        figObject = px.bar(
            gpaImportanceFrame.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            labels={"Importance": "重要性", "Feature": "特征"},
        )
        plotlyChart(figureLayout(figObject, "随机森林模型：GPA变化特征重要性", heightValue=620))

    chartColumns = st.columns([1, 1])
    with chartColumns[0]:
        figObject = px.bar(
            burnoutImportanceFrame.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            labels={"Importance": "重要性", "Feature": "特征"},
        )
        plotlyChart(figureLayout(figObject, "随机森林模型：高倦怠风险特征重要性", heightValue=620))
    with chartColumns[1]:
        sampleProjection = makeSampleFrame(projectionFrame, 6000)
        figObject = px.scatter(
            sampleProjection,
            x="PcaOne",
            y="PcaTwo",
            color="ClusterName",
            opacity=0.65,
            labels={"PcaOne": "主成分1", "PcaTwo": "主成分2", "ClusterName": "学习行为类型"},
        )
        plotlyChart(figureLayout(figObject, "学生AI学习行为聚类 PCA 投影", heightValue=620))

    radarColumns = [
        "WeeklyGenAiHours",
        "TraditionalStudyHours",
        "GenAiStudyShare",
        "ToolDiversity",
        "PerceivedAiDependency",
        "AnxietyLevelDuringExams",
        "SkillRetentionScore",
        "GpaChange",
        "HighBurnoutFlag",
    ]
    radarFrame = profileFrame.copy()
    scaledRadar = radarFrame[radarColumns].copy()
    scaledRadar = (scaledRadar - scaledRadar.min()) / (scaledRadar.max() - scaledRadar.min()).replace(0, 1)
    scaledRadar["ClusterName"] = radarFrame["ClusterName"]
    figObject = go.Figure()
    radarLabels = [toChineseColumnName(columnItem) for columnItem in radarColumns]
    for _, rowItem in scaledRadar.iterrows():
        valuesList = [rowItem[columnItem] for columnItem in radarColumns]
        figObject.add_trace(go.Scatterpolar(r=valuesList + [valuesList[0]], theta=radarLabels + [radarLabels[0]], fill="toself", name=rowItem["ClusterName"]))
    figObject.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
    plotlyChart(figureLayout(figObject, "聚类学习行为画像雷达图（归一化指标）", heightValue=650))

    st.markdown("#### 聚类画像统计表")
    displayProfile = profileFrame.copy()
    displayProfile = displayProfile.rename(columns={columnItem: toChineseColumnName(columnItem) for columnItem in displayProfile.columns})
    st.dataframe(displayProfile.round(4), use_container_width=True, hide_index=True)

    with st.expander("查看模型输出明细表"):
        st.markdown("##### 回归系数")
        st.dataframe(coefficientFrame.round(5), use_container_width=True, hide_index=True)
        st.markdown("##### GPA变化随机森林重要性")
        st.dataframe(gpaImportanceFrame.round(5), use_container_width=True, hide_index=True)
        st.markdown("##### 高倦怠风险随机森林重要性")
        st.dataframe(burnoutImportanceFrame.round(5), use_container_width=True, hide_index=True)
        st.markdown("##### PCA载荷")
        st.dataframe(loadingFrame.round(4), use_container_width=True, hide_index=True)


def renderConclusionTab(frame: pd.DataFrame):
    st.subheader("9. 自动化结论摘要与论文写作辅助")
    renderSectionNote(
        "本页把交互筛选后的数据结果自动整理为论文可用的结论语句。可用于第 3.3 节‘案例分析结果与讨论’和第 5 章‘总结与展望’。"
    )

    hourSummary = frame.groupby("GenAiHoursBand", observed=False)["GpaChange"].mean().reset_index()
    if not hourSummary.empty:
        bestHourRow = hourSummary.loc[hourSummary["GpaChange"].idxmax()]
        worstHourRow = hourSummary.loc[hourSummary["GpaChange"].idxmin()]
    else:
        bestHourRow = pd.Series({"GenAiHoursBand": "--", "GpaChange": np.nan})
        worstHourRow = pd.Series({"GenAiHoursBand": "--", "GpaChange": np.nan})

    useCaseSummary = frame.groupby("PrimaryUseCaseCn", observed=False)["GpaChange"].mean().reset_index()
    if not useCaseSummary.empty:
        bestUseCaseRow = useCaseSummary.loc[useCaseSummary["GpaChange"].idxmax()]
        worstUseCaseRow = useCaseSummary.loc[useCaseSummary["GpaChange"].idxmin()]
    else:
        bestUseCaseRow = pd.Series({"PrimaryUseCaseCn": "--", "GpaChange": np.nan})
        worstUseCaseRow = pd.Series({"PrimaryUseCaseCn": "--", "GpaChange": np.nan})

    skillSummary = frame.groupby("PromptEngineeringSkillCn", observed=False).agg(
        平均GPA变化=("GpaChange", "mean"),
        技能保留均值=("SkillRetentionScore", "mean"),
    ).reset_index().sort_values("技能保留均值", ascending=False)
    topSkillText = skillSummary.iloc[0]["PromptEngineeringSkillCn"] if len(skillSummary) > 0 else "--"

    conclusionList = [
        f"筛选样本共 {len(frame):,} 条，AI 使用后 GPA 平均变化为 {frame['GpaChange'].mean():.3f}，技能保留均值为 {frame['SkillRetentionScore'].mean():.2f}。",
        f"从 AI 使用时长分组看，{bestHourRow['GenAiHoursBand']} 组的 GPA 平均变化最高（{bestHourRow['GpaChange']:.3f}），{worstHourRow['GenAiHoursBand']} 组最低（{worstHourRow['GpaChange']:.3f}），说明 AI 使用收益并非简单随时长线性增加。",
        f"从 AI 使用目的看，{bestUseCaseRow['PrimaryUseCaseCn']} 对应的 GPA 平均变化最高（{bestUseCaseRow['GpaChange']:.3f}），而 {worstUseCaseRow['PrimaryUseCaseCn']} 相对较低，提示 AI 的学习价值取决于具体使用方式。",
        f"提示词能力分组显示，{topSkillText} 学生在技能保留或 GPA 表现上更有优势，说明 AI 素养是影响学习收益的重要调节因素。",
        f"高倦怠风险比例为 {frame['HighBurnoutFlag'].mean():.1%}，AI 依赖度均值为 {frame['PerceivedAiDependency'].mean():.2f}，考试焦虑均值为 {frame['AnxietyLevelDuringExams'].mean():.2f}，需要关注高依赖与高焦虑叠加群体。",
    ]

    for indexValue, textItem in enumerate(conclusionList, start=1):
        st.markdown(f"**结论 {indexValue}：** {textItem}")

    paperText = "\n".join([f"（{indexValue}）{textItem}" for indexValue, textItem in enumerate(conclusionList, start=1)])
    st.text_area("可复制到论文中的结论文本", paperText, height=250)

    st.download_button(
        label="下载当前筛选样本的结论文本",
        data=paperText.encode("utf-8"),
        file_name="AiStudentAnalysisConclusion.txt",
        mime="text/plain",
    )

    st.markdown("#### 当前筛选数据导出")
    st.download_button(
        label="下载当前筛选后的数据 CSV",
        data=createDownloadBytes(frame),
        file_name="FilteredAiStudentImpactData.csv",
        mime="text/csv",
    )


# -----------------------------------------------------------------------------
# 主程序
# -----------------------------------------------------------------------------


def main():
    st.markdown(f"""
    <div class="hero-card">
        <h1>{APP_TITLE}</h1>
        <p>{APP_SUBTITLE}</p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("### 数据源设置")
    uploadedFile = st.sidebar.file_uploader("上传原始 CSV 数据文件", type=["csv"], help="支持 ai_student_impact_dataset 原始数据，也支持已经清洗后的增强数据。")
    defaultPath = findDefaultDataPath()

    try:
        if uploadedFile is not None:
            rawFrame = readUploadedFrame(uploadedFile)
            sourceName = uploadedFile.name
            sourceMode = "用户上传"
        elif defaultPath is not None:
            rawFrame = loadDefaultFrame(defaultPath)
            sourceName = defaultPath
            sourceMode = "本地默认文件"
        else:
            st.error("未找到默认数据文件。请在左侧上传原始 CSV 数据文件后继续。")
            st.stop()

        preparedFrame, missingAudit, outlierAudit, cleaningLog = prepareData(rawFrame)
    except Exception as errorObject:
        st.error(f"数据读取或清洗失败：{errorObject}")
        st.stop()

    overallFrame = preparedFrame.copy()

    st.sidebar.success(f"数据源：{sourceMode}")
    st.sidebar.caption(f"文件：{sourceName}")

    st.sidebar.markdown("### 全局筛选器")
    majorOptions = sorted(preparedFrame["MajorCategoryCn"].dropna().unique().tolist())
    yearOptions = sorted(preparedFrame["YearOfStudyCn"].dropna().unique().tolist())
    useCaseOptions = sorted(preparedFrame["PrimaryUseCaseCn"].dropna().unique().tolist())
    skillOptions = sorted(preparedFrame["PromptEngineeringSkillCn"].dropna().unique().tolist())
    policyOptions = sorted(preparedFrame["InstitutionalPolicyCn"].dropna().unique().tolist())
    burnoutOptions = sorted(preparedFrame["BurnoutRiskLevelCn"].dropna().unique().tolist())

    selectedMajors = st.sidebar.multiselect("专业类别", majorOptions, default=majorOptions)
    selectedYears = st.sidebar.multiselect("年级", yearOptions, default=yearOptions)
    selectedUseCases = st.sidebar.multiselect("AI主要用途", useCaseOptions, default=useCaseOptions)
    selectedSkills = st.sidebar.multiselect("提示词能力", skillOptions, default=skillOptions)
    selectedPolicies = st.sidebar.multiselect("学校AI政策", policyOptions, default=policyOptions)
    selectedBurnout = st.sidebar.multiselect("倦怠风险等级", burnoutOptions, default=burnoutOptions)

    maxAiHours = float(math.ceil(preparedFrame["WeeklyGenAiHours"].max()))
    maxTraditionalHours = float(math.ceil(preparedFrame["TraditionalStudyHours"].max()))
    selectedAiHours = st.sidebar.slider("每周AI使用时长范围", 0.0, maxAiHours, (0.0, maxAiHours), step=0.5)
    selectedTraditionalHours = st.sidebar.slider("每周传统学习时长范围", 0.0, maxTraditionalHours, (0.0, maxTraditionalHours), step=0.5)
    selectedGpaChange = st.sidebar.slider(
        "GPA变化范围",
        float(math.floor(preparedFrame["GpaChange"].min() * 10) / 10),
        float(math.ceil(preparedFrame["GpaChange"].max() * 10) / 10),
        (float(math.floor(preparedFrame["GpaChange"].min() * 10) / 10), float(math.ceil(preparedFrame["GpaChange"].max() * 10) / 10)),
        step=0.05,
    )

    filteredFrame = preparedFrame[
        preparedFrame["MajorCategoryCn"].isin(selectedMajors)
        & preparedFrame["YearOfStudyCn"].isin(selectedYears)
        & preparedFrame["PrimaryUseCaseCn"].isin(selectedUseCases)
        & preparedFrame["PromptEngineeringSkillCn"].isin(selectedSkills)
        & preparedFrame["InstitutionalPolicyCn"].isin(selectedPolicies)
        & preparedFrame["BurnoutRiskLevelCn"].isin(selectedBurnout)
        & preparedFrame["WeeklyGenAiHours"].between(selectedAiHours[0], selectedAiHours[1])
        & preparedFrame["TraditionalStudyHours"].between(selectedTraditionalHours[0], selectedTraditionalHours[1])
        & preparedFrame["GpaChange"].between(selectedGpaChange[0], selectedGpaChange[1])
    ].copy()

    if filteredFrame.empty:
        st.warning("当前筛选条件下没有样本，请放宽左侧筛选条件。")
        st.stop()

    st.caption(f"当前筛选样本：{len(filteredFrame):,} / {len(preparedFrame):,} 条。所有图表和模型均基于当前筛选条件动态更新。")

    tabList = st.tabs([
        "① 数据导入与清洗",
        "② 总体画像",
        "③ 学业成绩",
        "④ AI使用行为",
        "⑤ 群体差异",
        "⑥ AI素养与政策",
        "⑦ 心理风险",
        "⑧ 建模分析",
        "⑨ 结论与导出",
    ])

    with tabList[0]:
        renderDataImportTab(preparedFrame, missingAudit, outlierAudit, cleaningLog)
    with tabList[1]:
        renderOverviewTab(filteredFrame, overallFrame)
    with tabList[2]:
        renderAcademicOutcomeTab(filteredFrame)
    with tabList[3]:
        renderAiBehaviorTab(filteredFrame)
    with tabList[4]:
        renderGroupDifferenceTab(filteredFrame)
    with tabList[5]:
        renderAiLiteracyPolicyTab(filteredFrame)
    with tabList[6]:
        renderRiskTab(filteredFrame)
    with tabList[7]:
        if len(filteredFrame) < 100:
            st.warning("建模分析建议至少保留 100 条样本。当前筛选样本过少，请放宽筛选条件。")
        else:
            renderModelingTab(filteredFrame)
    with tabList[8]:
        renderConclusionTab(filteredFrame)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 使用建议")
    st.sidebar.write("1. 先查看数据导入与清洗页。")
    st.sidebar.write("2. 再按论文第3章顺序浏览总体画像、学业成绩、AI行为、群体差异和风险分析。")
    st.sidebar.write("3. 建模分析会随筛选条件重新计算，筛选样本过少时不建议解释模型结果。")
    st.sidebar.write("4. 结论与导出页可生成当前筛选条件下的论文结论文本。")


if __name__ == "__main__":
    main()
