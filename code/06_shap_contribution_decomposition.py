"""

Input data
----------
This script starts from the analysis-ready study-area-week dataset stored at
``data/analysis_ready_data.csv``. Raw-data acquisition, source provenance,
and variable construction are documented in ``data/README.md`` and the
manuscript/Supplementary Information. This script does not recreate the raw
source databases.

Paths are resolved relative to the repository root so the script can be run
on another computer without editing local drive paths.
"""

"""
流感预测研究 — SHAP 气象贡献占比分析
======================================================
输出内容：
  1. 全国尺度：四分区 + Overall 的【气象/非气象/时间周期/社会地理】贡献均值
  2. 州尺度：每个州在四分区下的各维度贡献均值
  3. 州尺度 Overall：每个州全样本贡献（供空间制图）
  4. 样本级明细 CSV (含25个特征的原始 SHAP 绝对值与百分比)
  5. 特征重要性排名 CSV (按物理分类标签归类)
数据分区与最优参数与调参/RFE/ROC/SHAP代码完全一致
======================================================
"""

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import shap
import json
import os

# ======================================================
# 1. 配置路径 (与 20260714 最新工作流完全一致)
# ======================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
input_path = PROJECT_ROOT / "data" / "analysis_ready_data.csv"
param_json_path = PROJECT_ROOT / "config" / "catboost_best_params.json"
output_dir = PROJECT_ROOT / "outputs" / "06_shap_contribution_decomposition"
output_dir.mkdir(parents=True, exist_ok=True)

# Map technical feature names to the human-readable names used in outputs.
FEATURE_RENAME_MAP = {
    "sp_max": "Maximum surface pressure",
    "sp_min": "Minimum surface pressure",
    "solar_rad_max": "Maximum solar radiation",
    "Temp_anomaly": "Temperature anomaly",
    "AH_anomaly": "Absolute humidity anomaly",
    "Temp_max_prior2": "Maximum temperature, 2 weeks prior",
    "Temp_min_prior5": "Minimum temperature, 5 weeks prior",
    "AH_mean_prior3": "Mean absolute humidity, 3 weeks prior",
    "AH_mean_prior4": "Mean absolute humidity, 4 weeks prior",
    "Precip_sum_prior5": "Total precipitation, 5 weeks prior",
    "Precip_sum_prior6": "Total precipitation, 6 weeks prior",
    "Solar_mean_prior5": "Mean solar radiation, 5 weeks prior",
    "Solar_mean_prior6": "Mean solar radiation, 6 weeks prior",
    "Temp_anomaly_prior1": "Temperature anomaly, 1 week prior",
    "Temp_anomaly_prior3": "Temperature anomaly, 3 weeks prior",
    "Temp_anomaly_prior4": "Temperature anomaly, 4 weeks prior",
    "Temp_anomaly_prior5": "Temperature anomaly, 5 weeks prior",
    "Temp_anomaly_prior6": "Temperature anomaly, 6 weeks prior",
    "RH_anomaly_prior1": "Relative humidity anomaly, 1 week prior",
    "RH_anomaly_prior6": "Relative humidity anomaly, 6 weeks prior",
    "AH_anomaly_prior2": "Absolute humidity anomaly, 2 weeks prior",
    "AH_anomaly_prior5": "Absolute humidity anomaly, 5 weeks prior",
    "Solar_anomaly_prior1": "Solar radiation anomaly, 1 week prior",
    "Solar_anomaly_prior3": "Solar radiation anomaly, 3 weeks prior",
    "Solar_anomaly_prior4": "Solar radiation anomaly, 4 weeks prior",
    "week_sin": "Calendar week sine",
    "week_cos": "Calendar week cosine",
    "hhs_region": "HHS region",
    "pop_den": "Population density"
}

y_col = "positivity_rate_P75_flag"

# ======================================================
# 2. 定义已重命名的特征分组 (共 29 个特征)
# ======================================================
features_all = [
    "Maximum surface pressure",
    "Minimum surface pressure",
    "Maximum solar radiation",
    "Temperature anomaly",
    "Absolute humidity anomaly",
    "Maximum temperature, 2 weeks prior",
    "Minimum temperature, 5 weeks prior",
    "Mean absolute humidity, 3 weeks prior",
    "Mean absolute humidity, 4 weeks prior",
    "Total precipitation, 5 weeks prior",
    "Total precipitation, 6 weeks prior",
    "Mean solar radiation, 5 weeks prior",
    "Mean solar radiation, 6 weeks prior",
    "Temperature anomaly, 1 week prior",
    "Temperature anomaly, 3 weeks prior",
    "Temperature anomaly, 4 weeks prior",
    "Temperature anomaly, 5 weeks prior",
    "Temperature anomaly, 6 weeks prior",
    "Relative humidity anomaly, 1 week prior",
    "Relative humidity anomaly, 6 weeks prior",
    "Absolute humidity anomaly, 2 weeks prior",
    "Absolute humidity anomaly, 5 weeks prior",
    "Solar radiation anomaly, 1 week prior",
    "Solar radiation anomaly, 3 weeks prior",
    "Solar radiation anomaly, 4 weeks prior",
    "Calendar week sine",
    "Calendar week cosine",
    "HHS region",
    "Population density",
]


features_meteo = [
    "Maximum surface pressure",
    "Minimum surface pressure",
    "Maximum solar radiation",
    "Temperature anomaly",
    "Absolute humidity anomaly",
    "Maximum temperature, 2 weeks prior",
    "Minimum temperature, 5 weeks prior",
    "Mean absolute humidity, 3 weeks prior",
    "Mean absolute humidity, 4 weeks prior",
    "Total precipitation, 5 weeks prior",
    "Total precipitation, 6 weeks prior",
    "Mean solar radiation, 5 weeks prior",
    "Mean solar radiation, 6 weeks prior",
    "Temperature anomaly, 1 week prior",
    "Temperature anomaly, 3 weeks prior",
    "Temperature anomaly, 4 weeks prior",
    "Temperature anomaly, 5 weeks prior",
    "Temperature anomaly, 6 weeks prior",
    "Relative humidity anomaly, 1 week prior",
    "Relative humidity anomaly, 6 weeks prior",
    "Absolute humidity anomaly, 2 weeks prior",
    "Absolute humidity anomaly, 5 weeks prior",
    "Solar radiation anomaly, 1 week prior",
    "Solar radiation anomaly, 3 weeks prior",
    "Solar radiation anomaly, 4 weeks prior"
]

features_calendar = [
    "Calendar week sine",
    "Calendar week cosine",
]

features_socio_geo = [
    "HHS region",
    "Population density",
]

# 合并得到 29 个全变量
all_variables = features_meteo + features_calendar + features_socio_geo

print(f"气象变量 (Meteorological): {len(features_meteo)} 个")
print(f"时间周期变量 (Calendar temporal): {len(features_calendar)} 个")
print(f"社会地理变量 (Socio-geographical): {len(features_socio_geo)} 个")
print(f"合计: {len(all_variables)} 个")

# 定义辅助分类器
def get_feature_subgroup(var_name):
    if var_name in features_calendar:
        return "Calendar temporal"
    elif var_name in features_socio_geo:
        return "Socio-geographical"
    else:
        return "Meteorological"

# ======================================================
# 3. 读取数据
# ======================================================
print("\n🔄 正在读取分析就绪数据...")
if not os.path.exists(input_path):
    raise FileNotFoundError(f"❌ 找不到分析就绪数据集，请检查路径：{input_path}")

data = pd.read_csv(input_path)
data = data.rename(columns={
    old: new for old, new in FEATURE_RENAME_MAP.items()
    if old in data.columns and new not in data.columns
})
data = data.sort_values("FID").reset_index(drop=True)

required_columns = set(
    ["FID", "REGION", "YEAR", "WEEK", y_col] + all_variables
)
missing_columns = sorted(required_columns.difference(data.columns))
if missing_columns:
    raise ValueError(
        f"analysis_ready_data.csv is missing required columns: {missing_columns}"
    )


# ======================================================
# 4. 二维四区时空双重划分 (时空隔离架构)
# ======================================================
print("✂️ 正在进行【二维四区】时空隔离划分...")

cities = data["REGION"].unique()
dev_cities, spat_eval_cities = train_test_split(cities, test_size=0.2, random_state=123)

dev_data    = data[data["REGION"].isin(dev_cities)]
spat_eval_df = data[data["REGION"].isin(spat_eval_cities)]

temp_eval_df = dev_data[dev_data["YEAR"] >= 2024]
history_df  = dev_data[dev_data["YEAR"] < 2024]

train_df, val_df = train_test_split(
    history_df, test_size=0.2, random_state=123, stratify=history_df[y_col]
)

print(f"  训练集 (Training):       {len(train_df)} 条")
print(f"  内部调参集 (Internal tuning):   {len(val_df)} 条")
print(f"  时间评估 (Temporal evaluation): {len(temp_eval_df)} 条")
print(f"  空间评估 (Spatial evaluation):  {len(spat_eval_df)} 条")

# ======================================================
# 5. 为全数据集打上互斥的分区标签
# ======================================================
data['Dataset_Type'] = 'Unknown'
data.loc[train_df.index,    'Dataset_Type'] = 'Training'
data.loc[val_df.index,      'Dataset_Type'] = 'Internal_Val'
data.loc[temp_eval_df.index, 'Dataset_Type'] = 'Temporal_Evaluation'
data.loc[spat_eval_df.index, 'Dataset_Type'] = 'Spatial_Evaluation'

print(f"\n  标签分布验证: {data['Dataset_Type'].value_counts().to_dict()}")

# ======================================================
# 6. 加载最优超参数
# ======================================================
with open(param_json_path, 'r', encoding='utf-8') as f:
    best_params = json.load(f)
best_params.update({'random_state': 123, 'verbose': False})
print(f"\n✅ 已加载最优参数: {best_params}")

# ======================================================
# 7. 训练全变量模型 (仅使用训练集)
# ======================================================
print("\n🚀 正在基于训练集构建 Full Model...")
X_train = train_df[all_variables]
y_train = train_df[y_col]

model = CatBoostClassifier(**best_params)
model.fit(X_train, y_train)

# ======================================================
# 8. 在全样本上计算 SHAP (TreeExplainer)
# ======================================================
print("🔢 正在计算全样本 SHAP 矩阵...")
explainer = shap.TreeExplainer(model)
X_all     = data[all_variables]
shap_values = explainer.shap_values(X_all)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

print(f"  SHAP 矩阵维度: {shap_values.shape}")

# ======================================================
# 9. 构建样本级 SHAP 绝对值 DataFrame
# ======================================================
shap_abs_df = pd.DataFrame(
    np.abs(shap_values),
    columns=[f"SHAP_{c}" for c in all_variables]
)

base_info = data[['FID', 'REGION', 'YEAR', 'WEEK', 'Dataset_Type', y_col]].copy().reset_index(drop=True)
combined_df = pd.concat([base_info, shap_abs_df], axis=1)

# ======================================================
# 10. 精细化计算：气象、时间、地理及非气象贡献占比 (%)
# ======================================================
met_cols       = [f"SHAP_{c}" for c in features_meteo]
calendar_cols  = [f"SHAP_{c}" for c in features_calendar]
sociogeo_cols  = [f"SHAP_{c}" for c in features_socio_geo]
non_met_cols   = calendar_cols + sociogeo_cols

# 绝对值求和
combined_df['Total_Abs_SHAP']     = shap_abs_df.sum(axis=1)
combined_df['Met_Abs_Sum']        = shap_abs_df[met_cols].sum(axis=1)
combined_df['Calendar_Abs_Sum']   = shap_abs_df[calendar_cols].sum(axis=1)
combined_df['SocioGeo_Abs_Sum']   = shap_abs_df[sociogeo_cols].sum(axis=1)
combined_df['NonMet_Abs_Sum']     = shap_abs_df[non_met_cols].sum(axis=1)

# 避免除以 0
denom = combined_df['Total_Abs_SHAP'].replace(0, np.nan)

# 转换为占比百分比 (%)
combined_df['Met_Contribution_Pct']      = (combined_df['Met_Abs_Sum']      / denom) * 100
combined_df['Calendar_Contribution_Pct'] = (combined_df['Calendar_Abs_Sum'] / denom) * 100
combined_df['SocioGeo_Contribution_Pct'] = (combined_df['SocioGeo_Abs_Sum'] / denom) * 100
combined_df['NonMet_Contribution_Pct']   = (combined_df['NonMet_Abs_Sum']   / denom) * 100

# ======================================================
# 11. 全国尺度统计（四分区 + Overall 汇总）
# ======================================================
target_cols = [
    'Met_Contribution_Pct', 'NonMet_Contribution_Pct',
    'Calendar_Contribution_Pct', 'SocioGeo_Contribution_Pct'
]

national_by_type = (
    combined_df.groupby('Dataset_Type')[target_cols]
    .mean()
    .reset_index()
)

overall_mean = combined_df[target_cols].mean().to_dict()
overall_row = pd.DataFrame([{**{'Dataset_Type': 'All_Data_Overall'}, **overall_mean}])

national_final = pd.concat([national_by_type, overall_row], ignore_index=True)
national_final[target_cols] = national_final[target_cols].round(2)

# 重命名列名，使其更具学术报告规范
rename_cols = {
    'Met_Contribution_Pct': 'Meteorological_Contribution(%)',
    'NonMet_Contribution_Pct': 'Non_Meteorological_Contribution(%)',
    'Calendar_Contribution_Pct': 'Calendar_Temporal_Contribution(%)',
    'SocioGeo_Contribution_Pct': 'Socio_Geographical_Contribution(%)'
}
national_final = national_final.rename(columns=rename_cols)

national_final.to_csv(
    os.path.join(output_dir, "SHAP_全国尺度贡献汇总.csv"),
    index=False, encoding="utf-8-sig"
)

print("\n📊 全国尺度各维度 SHAP 贡献汇总：")
print(national_final.to_string(index=False))

# ======================================================
# 12. 州尺度统计（按分区 × 州 及 Overall 供制图）
# ======================================================
# 12.1 分分区统计每一州
state_by_type = (
    combined_df.groupby(['Dataset_Type', 'REGION'])[target_cols]
    .mean()
    .reset_index()
    .round(2)
    .rename(columns=rename_cols)
)
state_by_type.to_csv(
    os.path.join(output_dir, "SHAP_州尺度贡献_分分区.csv"),
    index=False, encoding="utf-8-sig"
)

# 12.2 全样本合并（Overall）每一州，直接对接 ArcGIS / QGIS 空间制图
state_overall = (
    combined_df.groupby('REGION')[target_cols]
    .mean()
    .reset_index()
    .round(2)
    .rename(columns=rename_cols)
    .sort_values('Meteorological_Contribution(%)', ascending=False)
    .reset_index(drop=True)
)
state_overall.to_csv(
    os.path.join(output_dir, "SHAP_州尺度贡献_Overall（供制图）.csv"),
    index=False, encoding="utf-8-sig"
)

# ======================================================
# 13. 特征重要性排名（全局均值 |SHAP| 排名）
# ======================================================
feature_importance = pd.DataFrame({
    '变量':        all_variables,
    '分组':        [get_feature_subgroup(f) for f in all_variables],
    '均值_SHAP':   shap_abs_df.mean().values.round(6),
}).sort_values('均值_SHAP', ascending=False).reset_index(drop=True)

total_shap = feature_importance['均值_SHAP'].sum()
feature_importance['贡献占比(%)'] = (feature_importance['均值_SHAP'] / total_shap * 100).round(2)

feature_importance.to_csv(
    os.path.join(output_dir, "SHAP_特征重要性排名.csv"),
    index=False, encoding="utf-8-sig"
)

# ======================================================
# 14. 样本级明细 CSV 导出
# ======================================================
# 重命名样本级明细列，保持一致
combined_df = combined_df.rename(columns=rename_cols)
combined_df.to_csv(
    os.path.join(output_dir, "SHAP_样本级明细.csv"),
    index=False, encoding="utf-8-sig"
)

print(f"\n💾 所有精细化分析结果已保存至: {output_dir}")
print("🎉 SHAP 贡献占比与地理空间制图数据分析圆满完成！")
