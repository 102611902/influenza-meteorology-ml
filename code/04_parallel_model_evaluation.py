"""
Public-repository analysis script.

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
流感预测研究 — 四模型预测贡献分解与双重验证框架
======================================================
M1: Full model (29 variables)
M2: Meteorological model (25 variables)
M3: Calendar temporal model (2个时间特征 - 基线参考)
M4: Socio-geographical model (2个社会地理特征)
======================================================
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
# ======================================================
# 全局字体统一配置 (放在导入 matplotlib 之后)
# ======================================================
font_tnr = "Times New Roman"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [font_tnr]
plt.rcParams["axes.unicode_minus"] = False
# ======================================================
# 1. 读取数据
# ======================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "analysis_ready_data.csv"
PARAM_PATH = PROJECT_ROOT / "config" / "catboost_best_params.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "04_parallel_model_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("🔄 正在读取分析就绪数据集...")
data = pd.read_csv(DATA_PATH)
data = data.sort_values("FID").reset_index(drop=True)



y_col = "positivity_rate_P75_flag"
drop_cols = [
    'FID', 'REGION', 'YEAR', 'WEEK', 'week_date', 'total_specimens',
    'total_positive', 'positivity_rate', 'STUSPS',
    'positivity_rate_prior1', 'positivity_rate_prior2', 'positivity_rate_prior3',
    'Pop', 'ALAND_SQMI'
]

# ======================================================
# 2. 定义变量分组与模型配置 (M1 - M4)
# ======================================================
features_all = [
    "sp_max", "sp_min", "solar_rad_max", "Temp_anomaly",
    "AH_anomaly", "Temp_max_prior2", "Temp_min_prior5", "AH_mean_prior3",
    "AH_mean_prior4", "Precip_sum_prior5", "Precip_sum_prior6", "Solar_mean_prior5",
    "Solar_mean_prior6", "Temp_anomaly_prior1", "Temp_anomaly_prior3", "Temp_anomaly_prior4",
    "Temp_anomaly_prior5", "Temp_anomaly_prior6", "RH_anomaly_prior1", "RH_anomaly_prior6",
    "AH_anomaly_prior2", "AH_anomaly_prior5", "Solar_anomaly_prior1", "Solar_anomaly_prior3",
    "Solar_anomaly_prior4", "week_sin", "week_cos", "hhs_region",
    "pop_den"
]
features_meteo = [
    "sp_max", "sp_min", "solar_rad_max", "Temp_anomaly",
    "AH_anomaly", "Temp_max_prior2", "Temp_min_prior5", "AH_mean_prior3",
    "AH_mean_prior4", "Precip_sum_prior5", "Precip_sum_prior6", "Solar_mean_prior5",
    "Solar_mean_prior6", "Temp_anomaly_prior1", "Temp_anomaly_prior3", "Temp_anomaly_prior4",
    "Temp_anomaly_prior5", "Temp_anomaly_prior6", "RH_anomaly_prior1", "RH_anomaly_prior6",
    "AH_anomaly_prior2", "AH_anomaly_prior5", "Solar_anomaly_prior1", "Solar_anomaly_prior3",
    "Solar_anomaly_prior4"
]

features_calendar = ["week_sin", "week_cos"]

features_socio_geo = ["pop_den", "hhs_region"]

required_columns = set(
    ["FID", "REGION", "YEAR", y_col] + features_all
)
missing_columns = sorted(required_columns.difference(data.columns))
if missing_columns:
    raise ValueError(
        f"analysis_ready_data.csv is missing required columns: {missing_columns}"
    )


# 统一组织四模型的配置：名称、特征列表、线条颜色、线条类型
model_configs = [
    {"id": "M1", "name": "Full model", "features": features_all, "color": "#e78ac3", "ls": "-"},
    {"id": "M2", "name": "Meteorological model", "features": features_meteo, "color": "#2A9D36", "ls": "-"},
    {"id": "M3", "name": "Calendar-seasonality model", "features": features_calendar, "color": "#377eb8", "ls": "--"},
    {"id": "M4", "name": "Socio-geographical model", "features": features_socio_geo, "color": "#ff7f00", "ls": "-."}
]

print("📋 模型特征分组情况：")
for cfg in model_configs:
    print(f"  {cfg['id']} ({cfg['name']}): {len(cfg['features'])} 个变量")

# ======================================================
# 3. 二维四区时空双重划分
# ======================================================
print("\n✂️ 正在进行【二维四区】时空隔离划分...")

cities = data["REGION"].unique()
dev_cities, spat_eval_cities = train_test_split(cities, test_size=0.2, random_state=123)

dev_data    = data[data["REGION"].isin(dev_cities)]
spat_eval_df = data[data["REGION"].isin(spat_eval_cities)]

temp_eval_df = dev_data[dev_data["YEAR"] >= 2024]
history_df  = dev_data[dev_data["YEAR"] < 2024]

train_df, val_df = train_test_split(
    history_df, test_size=0.2, random_state=123, stratify=history_df[y_col]
)

y_train    = train_df[y_col]
y_val      = val_df[y_col]
y_temp_eval = temp_eval_df[y_col]
y_spat_eval = spat_eval_df[y_col]

print(f"  训练集:       {len(train_df)} 条")
print(f"  内部调参集:   {len(val_df)} 条")
print(f"  时间评估: {len(temp_eval_df)} 条")
print(f"  空间评估: {len(spat_eval_df)} 条")

# ======================================================
# 4. 加载调参阶段保存的最优超参数
# ======================================================
param_json_path = PARAM_PATH
with open(param_json_path, 'r', encoding='utf-8') as f:
    best_params = json.load(f)
best_params.update({'random_state': 123, 'verbose': False})
print(f"\n✅ 已加载最优参数: {best_params}")

# ======================================================
# 5. 训练通用函数与执行四模型训练
# ======================================================
def train_and_predict(feature_list, model_name):
    print(f"🚀 正在训练 {model_name}（{len(feature_list)} 个变量）...")
    model = CatBoostClassifier(**best_params)
    model.fit(train_df[feature_list], y_train)
    return {
        "train":    model.predict_proba(train_df[feature_list])[:, 1],
        "val":      model.predict_proba(val_df[feature_list])[:, 1],
        "temp_eval": model.predict_proba(temp_eval_df[feature_list])[:, 1],
        "spat_eval": model.predict_proba(spat_eval_df[feature_list])[:, 1],
    }

print("\n🏋️ 开始训练所有四个子模型...")
# 把所有的预测概率结果直接存储在配置字典里，方便后续读取
for cfg in model_configs:
    cfg["preds"] = train_and_predict(cfg["features"], f"{cfg['id']}-{cfg['name']}")

# ======================================================
# 6. 输出路径配置
# ======================================================
save_dir = OUTPUT_DIR

# ======================================================
# 7. 四个验证分区的属性定义
# ======================================================
partition_configs = [
    {
        "key":       "train",
        "y_true":    y_train,
        "title":     "ROC Curve – Training",
        "label":     "Training",
        "panel":     "a ",
        "filename":  "ROC_Training.png",
    },
    {
        "key":       "val",
        "y_true":    y_val,
        "title":     "ROC Curve – Internal tuning",
        "label":     "Internal tuning",
        "panel":     "b ",
        "filename":  "ROC_Internal_tuning.png",
    },
    {
        "key":       "temp_eval",
        "y_true":    y_temp_eval,
        "title":     "ROC Curve – Temporal evaluation",
        "label":     "Temporal evaluation",
        "panel":     "c ",
        "filename":  "ROC_Temporal_evaluation.png",
    },
    {
        "key":       "spat_eval",
        "y_true":    y_spat_eval,
        "title":     "ROC Curve – Spatial evaluation",
        "label":     "Spatial evaluation",
        "panel":     "d ",
        "filename":  "ROC_Spatial_evaluation.png",
    },
]

# ======================================================
# 8. 单图 ROC 绘图函数 (包含四条线)
# ======================================================
def plot_roc_single(part_cfg, save_path):
    font_tnr = "Times New Roman"
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(9, 7.5))

    # 循环绘制四个模型
    for m_cfg in model_configs:
        pred = m_cfg["preds"][part_cfg["key"]]
        fpr, tpr, _ = roc_curve(part_cfg["y_true"], pred)
        roc_auc = auc(fpr, tpr)

        # 图例中展示 ID + 名称 + AUC
        legend_label = f"{m_cfg['id']}: {m_cfg['name']} (AUC = {roc_auc:.3f})"
        ax.plot(fpr, tpr, lw=3, color=m_cfg["color"], linestyle=m_cfg["ls"], label=legend_label)

    ax.plot([0, 1], [0, 1], '--', color='#999999', linewidth=1.5, alpha=0.7)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=24, fontweight="bold", family=font_tnr)
    ax.set_ylabel("True Positive Rate", fontsize=24, fontweight="bold", family=font_tnr)

    # 修改前：
    # ax.set_title(part_cfg["title"], fontsize=18, fontweight="bold", family=font_tnr, pad=15)

    # 修改后（加 loc="left"）：
    ax.set_title(
        part_cfg["title"],
        fontsize=26,
        fontweight="bold",
        family=font_tnr,
        pad=15,
        loc="left",
    )
    # 因为有四条线，字号适当调小以保证不遮挡
    ax.legend(loc="lower right", frameon=False, prop={'family': font_tnr, 'weight': 'bold', 'size': 16})
    ax.tick_params(labelsize=24)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace(".png", ".pdf"), format='pdf', bbox_inches='tight')
    plt.close()

# ======================================================
# 9. 四合一 ROC 图（论文正文主图 - 包含四条线）
# ======================================================
def plot_roc_4panel():
    font_tnr = "Times New Roman"
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.flatten()

    for idx, part_cfg in enumerate(partition_configs):
        ax = axes[idx]

        for m_cfg in model_configs:
            pred = m_cfg["preds"][part_cfg["key"]]
            fpr, tpr, _ = roc_curve(part_cfg["y_true"], pred)
            roc_auc = auc(fpr, tpr)

            legend_label = f"{m_cfg['id']}: {m_cfg['name']} (AUC = {roc_auc:.3f})"
            ax.plot(fpr, tpr, lw=2.5, color=m_cfg["color"], linestyle=m_cfg["ls"], label=legend_label)

        ax.plot([0, 1], [0, 1], '--', color='#999999', linewidth=1.5, alpha=0.7)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        # 修改前：
        # ax.set_title(f"{part_cfg['panel']} {part_cfg['label']}",
        #              fontsize=16, fontweight="bold", family=font_tnr, pad=12)

        # 修改后（加 loc="left"）：
        ax.set_title(
            f"{part_cfg['panel']} {part_cfg['label']}",
            fontsize=26,
            fontweight="bold",
            family=font_tnr,
            pad=12,
            loc="left",
        )
        ax.set_xlabel("False Positive Rate", fontsize=24, fontweight="bold", family=font_tnr)
        ax.set_ylabel("True Positive Rate", fontsize=24, fontweight="bold", family=font_tnr)

        ax.legend(loc="lower right", frameon=False, prop={'family': font_tnr, 'weight': 'bold', 'size': 16})
        ax.tick_params(labelsize=24)
        ax.grid(False)

    plt.tight_layout(pad=3.0)
    combo_path = os.path.join(save_dir, "ROC_4Panel_Combined_AllModels.png")
    plt.savefig(combo_path, dpi=300, bbox_inches='tight')
    plt.savefig(combo_path.replace(".png", ".pdf"), format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✅ 四模型四合一 ROC 图已保存: {combo_path}")

# ======================================================
# 10. 执行绘图 + 汇总全部模型 AUC
# ======================================================
print("\n🎨 正在绘制四模型对比图并计算 AUC...")

auc_rows = []
for part_cfg in partition_configs:
    # 绘制单图
    single_path = os.path.join(save_dir, part_cfg["filename"])
    plot_roc_single(part_cfg, single_path)
    print(f"  ✅ 单图完成: {part_cfg['filename']}")

    # 统计该分区下各模型 AUC
    row_data = {"数据集": part_cfg["label"]}
    auc_dict = {}

    for m_cfg in model_configs:
        pred = m_cfg["preds"][part_cfg["key"]]
        fpr, tpr, _ = roc_curve(part_cfg["y_true"], pred)
        val_auc = round(auc(fpr, tpr), 4)

        row_data[f"{m_cfg['id']} AUC ({m_cfg['name']})"] = val_auc
        auc_dict[m_cfg['id']] = val_auc

    # 计算一些关键特征的差值作为参考
    row_data["差值 (M1 Full - M2 Meteo)"] = round(auc_dict["M1"] - auc_dict["M2"], 4)
    row_data["差值 (M2 Meteo - M3 Calendar)"] = round(auc_dict["M2"] - auc_dict["M3"], 4)

    auc_rows.append(row_data)

# 绘制组合图
plot_roc_4panel()

# ======================================================
# 11. 保存详细 AUC 汇总表
# ======================================================
auc_df = pd.DataFrame(auc_rows)
csv_save_path = os.path.join(save_dir, "AUC_四模型贡献分解汇总.csv")
auc_df.to_csv(csv_save_path, index=False, encoding="utf-8-sig")

print("\n📊 完整模型 AUC 汇总结果：")
print(auc_df.to_string(index=False))
print(f"\n💾 所有四模型对比结果与图表已保存至: {save_dir}")
print("🎉 恭喜！四模型气象贡献分解分析全面完成！")

