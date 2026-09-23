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
流感预测研究 — 时空联合 RFE 特征筛选
======================================================
数据分区与最优参数完全沿用调参代码（二维四区结构）
驱动指标 = mean(AUC_temporal_evaluation, AUC_spatial_evaluation)
======================================================
"""

from pathlib import Path

import pandas as pd
import numpy as np
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

# ======================================================
# 1. 读取数据（路径与调参代码完全一致）
# ======================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "analysis_ready_data.csv"
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "03_spatiotemporal_rfe"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("🔄 Step 1: 正在读取分析就绪数据集...")
data = pd.read_csv(DATA_PATH)
data = data.sort_values("FID").reset_index(drop=True)

y_col = "positivity_rate_P75_flag"
drop_cols = [
    'FID', 'REGION', 'YEAR', 'WEEK', 'week_date', 'total_specimens',
    'total_positive', 'positivity_rate', 'STUSPS',
    'positivity_rate_prior1', 'positivity_rate_prior2', 'positivity_rate_prior3',
    'Pop', 'ALAND_SQMI'
]

required_metadata = {"FID", "REGION", "YEAR", y_col}
missing_metadata = sorted(required_metadata.difference(data.columns))
if missing_metadata:
    raise ValueError(
        f"analysis_ready_data.csv is missing required columns: {missing_metadata}"
    )


# ======================================================
# 2. 二维四区时空双重划分（与调参代码完全一致，random_state=123）
# ======================================================
print("✂️ Step 2: 正在进行 [二维四区] 时空隔离划分...")

cities = data["REGION"].unique()
dev_cities, spat_eval_cities = train_test_split(cities, test_size=0.2, random_state=123)

dev_data    = data[data["REGION"].isin(dev_cities)]
spat_eval_df = data[data["REGION"].isin(spat_eval_cities)]

temp_eval_df = dev_data[dev_data["YEAR"] >= 2024]
history_df  = dev_data[dev_data["YEAR"] < 2024]

train_df, val_df = train_test_split(
    history_df, test_size=0.2, random_state=123, stratify=history_df[y_col]
)

# 全变量特征矩阵（保留完整列名用于 RFE 路径）
X_train_full    = train_df.drop(drop_cols + [y_col], axis=1, errors='ignore')
y_train         = train_df[y_col]

X_temp_eval_full = temp_eval_df.drop(drop_cols + [y_col], axis=1, errors='ignore')
y_temp_eval      = temp_eval_df[y_col]

X_spat_eval_full = spat_eval_df.drop(drop_cols + [y_col], axis=1, errors='ignore')
y_spat_eval      = spat_eval_df[y_col]

print(f"  训练集: {len(X_train_full)} 条 | "
      f"时间评估: {len(X_temp_eval_full)} 条 | "
      f"空间评估: {len(X_spat_eval_full)} 条")
print(f"  初始变量数: {X_train_full.shape[1]}")

# ======================================================
# 3. 加载调参保存的最优超参数
# ======================================================
param_json_path = CONFIG_DIR / "catboost_best_params.json"
with open(param_json_path, 'r', encoding='utf-8') as f:
    best_params = json.load(f)
best_params.update({'random_state': 123, 'verbose': False})

print(f"\n✅ 已加载最优调参结果: {best_params}")

# ======================================================
# 4. RFE 主循环 — 时空联合 AUC 驱动
# 每轮训练后：
#   (a) 计算 AUC_temporal_evaluation 与 AUC_spatial_evaluation
#   (b) 联合指标 = 均值（两者等权）
#   (c) 删去 feature_importance 最低的变量
# ======================================================
save_dir = OUTPUT_DIR

current_features = list(X_train_full.columns)
rfe_results = []

print(f"\n⚔️ Step 3: 启动时空联合 RFE（共 {len(current_features)} 轮）...\n")

while len(current_features) >= 1:

    model = CatBoostClassifier(**best_params)
    model.fit(X_train_full[current_features], y_train)

    auc_temp = roc_auc_score(
        y_temp_eval,
        model.predict_proba(X_temp_eval_full[current_features])[:, 1]
    )
    auc_spat = roc_auc_score(
        y_spat_eval,
        model.predict_proba(X_spat_eval_full[current_features])[:, 1]
    )
    # 联合指标：时间评估与空间评估等权平均
    auc_combined = (auc_temp + auc_spat) / 2.0

    record = {
        "n_features":       len(current_features),
        "AUC_temporal_evaluation": round(auc_temp,     4),
        "AUC_spatial_evaluation":  round(auc_spat,     4),
        "AUC_combined":     round(auc_combined, 4),
        "remaining_features": ",".join(current_features)
    }

    print(f"  n={len(current_features):3d} | "
          f"Temp={auc_temp:.4f} | Spat={auc_spat:.4f} | "
          f"Combined={auc_combined:.4f}")

    if len(current_features) == 1:
        rfe_results.append(record)
        break

    # 找到本轮重要性最低的变量并记录
    importance     = model.get_feature_importance()
    drop_feature   = (
        pd.Series(importance, index=current_features)
        .sort_values()
        .index[0]
    )
    record["Dropped_Feature"] = drop_feature
    rfe_results.append(record)
    current_features.remove(drop_feature)

# ======================================================
# 5. 构建 RFE 路径表（按变量数升序排列）
# ======================================================
rfe_df = (
    pd.DataFrame(rfe_results)
    .sort_values("n_features")
    .reset_index(drop=True)
)

# ======================================================
# 6. 最优模型选择
#    规则：联合 AUC 与最大值差距在 DELTA 范围内，取变量数最少的模型
#    DELTA = 0.01（相当于 1% AUC，比单一外部集更严格）
# ======================================================
DELTA_THRESHOLD = 0.01

auc_max      = rfe_df["AUC_combined"].max()
equivalent_df = rfe_df[
    (rfe_df["AUC_combined"] - auc_max).abs() <= DELTA_THRESHOLD
].copy()
selected_idx = equivalent_df["n_features"].idxmin()

rfe_df["Selected_Model"] = 0
rfe_df.loc[selected_idx, "Selected_Model"] = 1

selected_n   = int(rfe_df.loc[selected_idx, "n_features"])
selected_auc = float(rfe_df.loc[selected_idx, "AUC_combined"])
selected_auc_temp = float(rfe_df.loc[selected_idx, "AUC_temporal_evaluation"])
selected_auc_spat = float(rfe_df.loc[selected_idx, "AUC_spatial_evaluation"])

print(f"\n✅ 最优特征子集:")
print(f"   变量数        = {selected_n}")
print(f"   联合 AUC     = {selected_auc:.4f}（全路径最大 {auc_max:.4f}）")
print(f"   时间评估 AUC = {selected_auc_temp:.4f}")
print(f"   空间评估 AUC = {selected_auc_spat:.4f}")

# ======================================================
# 7. 提取并保存最终入模变量列表
# ======================================================
selected_features = rfe_df.loc[selected_idx, "remaining_features"].split(",")
print(f"\n📋 最终入模变量 ({selected_n} 个):")
for feat in selected_features:
    print(f"   - {feat}")

selected_features_path = CONFIG_DIR / "rfe_selected_features.csv"
pd.DataFrame({"selected_feature": selected_features}).to_csv(
    selected_features_path,
    index=False, encoding="utf-8-sig"
)

# ======================================================
# 8. 保存完整 RFE 路径审计表（论文补充材料 Table S4）
# ======================================================
rfe_df.to_csv(
    save_dir / "RFE_SpatioTemporal_Path.csv",
    index=False, encoding="utf-8-sig"
)
print(f"\n💾 RFE 路径审计表已保存至: {save_dir / 'RFE_SpatioTemporal_Path.csv'}")

# ======================================================
# 9. 双面板学术图（Times New Roman 版）
#    左图：时间评估 vs 空间评估 AUC 分离展示
#    右图：联合 AUC + 等价区间 + 选定点
# ======================================================
def plot_rfe_dual_panel(df, selected_n, auc_max, threshold, save_path):
    font_tnr = "Times New Roman"
    plt.rcParams["font.family"]  = "serif"
    plt.rcParams["font.serif"]   = ["Times New Roman"]
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ---------- 左图：时空分离路径 ----------
    ax = axes[0]
    ax.plot(df["n_features"], df["AUC_temporal_evaluation"],
            marker="o", markersize=5, linewidth=2.5,
            color="#1F77B4", label="Temporal evaluation AUC")
    ax.plot(df["n_features"], df["AUC_spatial_evaluation"],
            marker="s", markersize=5, linewidth=2.5,
            color="#2A9D36", label="Spatial evaluation AUC")
    ax.axvline(selected_n, color="#999999", linestyle="--",
               linewidth=2, alpha=0.7, label=f"Selected (n={selected_n})")
    ax.set_xlabel("Number of Predictors",
                  fontsize=16, fontweight="bold", family=font_tnr)
    ax.set_ylabel("AUC",
                  fontsize=16, fontweight="bold", family=font_tnr)
    ax.set_title("RFE Path: Temporal vs. Spatial Evaluation",
                 fontsize=14, fontweight="bold", family=font_tnr, pad=15)
    ax.legend(frameon=False,
              prop={'family': font_tnr, 'weight': 'bold', 'size': 13})
    ax.tick_params(labelsize=14)
    ax.grid(False)

    # ---------- 右图：联合 AUC + 等价区间 ----------
    ax = axes[1]
    ax.plot(df["n_features"], df["AUC_combined"],
            marker="o", markersize=5, linewidth=3,
            color="#D62728", label="Combined AUC (mean)")
    ax.axhline(auc_max, linestyle="--", color="#999999",
               linewidth=1.5, alpha=0.7)
    ax.fill_between(
        df["n_features"],
        auc_max - threshold, auc_max,
        color="#999999", alpha=0.12,
        label=f"Performance-Equivalent Region (Δ={threshold})"
    )
    ax.axvline(selected_n, color="navy", linestyle="--",
               linewidth=2.5, label=f"Selected Model (n={selected_n})")
    ax.set_xlabel("Number of Predictors",
                  fontsize=16, fontweight="bold", family=font_tnr)
    ax.set_ylabel("Combined AUC",
                  fontsize=16, fontweight="bold", family=font_tnr)
    ax.set_title("RFE Path: Spatiotemporal Combined Evaluation",
                 fontsize=14, fontweight="bold", family=font_tnr, pad=15)
    ax.legend(frameon=False,
              prop={'family': font_tnr, 'weight': 'bold', 'size': 12})
    ax.tick_params(labelsize=14)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace(".png", ".pdf"), format='pdf',
                bbox_inches='tight')
    print(f"📈 RFE 双面板图已保存至: {save_path}")
    plt.show()
    plt.close()


plot_rfe_dual_panel(
    rfe_df, selected_n, auc_max, DELTA_THRESHOLD,
    str(save_dir / "RFE_SpatioTemporal_DualPanel.png")
)

print("\n🎉 时空联合 RFE 全流程完成！")
print(f"  └── 最终变量数: {selected_n}")
print(f"  └── 时间评估 AUC: {selected_auc_temp:.4f}")
print(f"  └── 空间评估 AUC: {selected_auc_spat:.4f}")
print(f"  └── 联合 AUC:     {selected_auc:.4f}")
print(f"  └── 特征列表: {selected_features_path}")





