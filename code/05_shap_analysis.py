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
流感预测研究 — 四模型 SHAP 特征贡献分析 (学术无缝对齐优化版)
======================================================
1. 自动计算 M1、M2、M3、M4 四个模型的 SHAP 矩阵
2. 输出 4 个分区的统一对齐 SHAP summary dot 单图 (高度自适应)
3. 输出 4 个分区的统一对齐 SHAP 条形单图 (高度自适应，激活柱子厚度一致)
4. 输出学术级 2x2 四合一 SHAP Bar 组合对齐图 (无缝无突兀，无空行占位)
5. 输出学术级 2x2 四合一 SHAP Summary Dot 组合对齐图 (无缝无突兀，无空行占位)
6. 输出 M1 的【气象 vs 时间周期 vs 社会地理】三亚组贡献占比汇总 CSV
======================================================
"""

import json
from pathlib import Path
import os
from catboost import CatBoostClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split


# ======================================================
# 1. 读取数据
# ======================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "analysis_ready_data.csv"
PARAM_PATH = PROJECT_ROOT / "config" / "catboost_best_params.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "05_shap_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Map the technical analysis-data column names to the human-readable labels
# used in the SHAP figures. If the readable labels already exist, they are kept.
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

print("🔄 正在读取分析就绪数据集...")
data = pd.read_csv(DATA_PATH)
data = data.rename(columns={
    old: new for old, new in FEATURE_RENAME_MAP.items()
    if old in data.columns and new not in data.columns
})
data = data.sort_values("FID").reset_index(drop=True)

y_col = "positivity_rate_P75_flag"

# ======================================================
# 2. 定义变量分组与三亚组映射
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
    "Solar radiation anomaly, 4 weeks prior",
]

features_calendar = [
    "Calendar week sine",
    "Calendar week cosine",
]

features_socio_geo = [
    "HHS region",
    "Population density",
]

required_columns = set(
    ["FID", "REGION", "YEAR", y_col] + features_all
)
missing_columns = sorted(required_columns.difference(data.columns))
if missing_columns:
    raise ValueError(
        f"analysis_ready_data.csv is missing required columns: {missing_columns}"
    )



def get_feature_subgroup(var_name):
  if var_name in features_calendar:
    return "Calendar temporal"
  elif var_name in features_socio_geo:
    return "Socio-geographical"
  else:
    return "Meteorological"


color_subgroup = {
    "Meteorological": "#2A9D36",  # 绿色
    "Calendar temporal": "#377eb8",  # 蓝色
    "Socio-geographical": "#ff7f00",  # 橙色
}

model_configs = [
    {
        "id": "M1",
        "name": "Full model",
        "features": features_all,
        "color": "#e78ac3",
    },
    {
        "id": "M2",
        "name": "Meteorological model",
        "features": features_meteo,
        "color": "#2A9D36",
    },
    {
        "id": "M3",
        "name": "Calendar temporal (reference)",
        "features": features_calendar,
        "color": "#377eb8",
    },
    {
        "id": "M4",
        "name": "Socio-geographical model",
        "features": features_socio_geo,
        "color": "#ff7f00",
    },
]

# ======================================================
# 3. 二维四区时空双重划分
# ======================================================
print("✂️ 正在进行时空隔离划分...")
cities = data["REGION"].unique()
dev_cities, spat_eval_cities = train_test_split(
    cities, test_size=0.2, random_state=123
)

dev_data = data[data["REGION"].isin(dev_cities)]
history_df = dev_data[dev_data["YEAR"] < 2024]

train_df, val_df = train_test_split(
    history_df, test_size=0.2, random_state=123, stratify=history_df[y_col]
)
y_train = train_df[y_col]

# ======================================================
# 4. 加载最优超参数
# ======================================================
param_json_path = PARAM_PATH
with open(param_json_path, "r", encoding="utf-8") as f:
  best_params = json.load(f)
best_params.update({"random_state": 123, "verbose": False})

# ======================================================
# 5. 计算 M1-M4 的 SHAP 值
# ======================================================
print("\n🏋️ 开始并行计算四个模型的 SHAP 值...")
for m_cfg in model_configs:
  features = m_cfg["features"]
  X_tr = train_df[features]

  print(f"  🚀 正在训练 {m_cfg['id']} ({m_cfg['name']})...")
  model = CatBoostClassifier(**best_params)
  model.fit(X_tr, y_train)

  explainer = shap.TreeExplainer(model)
  shap_vals = explainer.shap_values(X_tr)
  if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]

  m_cfg["shap_values"] = shap_vals
  m_cfg["X_data"] = X_tr

save_dir = OUTPUT_DIR

font_size = 20
font_tnr = "Times New Roman"


def get_feature_color(f, model_id, default_color):
  if model_id == "M1":
    return color_subgroup[get_feature_subgroup(f)]
  return default_color


# ======================================================
# 6. 绘制 单个 SHAP Summary Dot 图
# ======================================================
print("\n🎨 正在生成单模型自适应 SHAP Summary Dot 图...")


def plot_shap_dot_single(m_cfg, save_path):
  plt.rcParams["font.family"] = font_tnr
  plt.rcParams["font.weight"] = "bold"

  features = m_cfg["features"]
  shap_vals = m_cfg["shap_values"]
  X_data = m_cfg["X_data"]

  fig_height = max(3.5, len(features) * 0.4 + 2.0)
  fig = plt.figure(figsize=(font_size, fig_height))

  shap.summary_plot(
      shap_vals,
      X_data,
      show=False,
      cmap=plt.get_cmap("coolwarm"),
      plot_type="dot",
      alpha=0.7,
      max_display=len(features),
  )

  ax = plt.gca()
  ax.set_xlabel(
      "SHAP Value (Impact on Model Output)",
      fontsize=font_size,
      fontweight="bold",
      family=font_tnr,
  )

  # 左上角对齐标题：使用 set_title 并指定 x=0, ha='left'，精准对齐 Y 轴最左侧边缘
  ax.set_title(
      f"{m_cfg['id']}: {m_cfg['name']}",
      fontsize=font_size + 2,
      fontweight="bold",
      family=font_tnr,
      pad=14,
      x=0.0,
      ha="left",
  )

  for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(font_size - 2)
    label.set_fontweight("bold")
    label.set_family(font_tnr)

  try:
    cb_ax = plt.gcf().axes[-1]
    cb_ax.set_ylabel(
        "Feature value",
        fontsize=font_size - 2,
        fontweight="bold",
        family=font_tnr,
    )
    for label in cb_ax.get_yticklabels():
      label.set_fontsize(font_size - 4)
      label.set_family(font_tnr)
  except Exception:
    pass

  plt.tight_layout()
  plt.savefig(save_path, dpi=300, bbox_inches="tight")
  plt.savefig(
      save_path.replace(".png", ".pdf"), format="pdf", bbox_inches="tight"
  )
  plt.close()
  plt.rcParams.update(plt.rcParamsDefault)


for m_cfg in model_configs:
  dot_name = f"SHAP_dot_{m_cfg['id']}_{m_cfg['name'].replace(' ', '_')}.png"
  plot_shap_dot_single(m_cfg, os.path.join(save_dir, dot_name))
  print(f"  ✅ 已保存 Dot 图: {dot_name}")

# ======================================================
# 7. 绘制 单个 SHAP Bar 图
# ======================================================
print("\n🎨 正在生成单模型自适应 SHAP Bar 图...")


def plot_shap_bar_single(m_cfg, save_path):
  plt.rcParams["font.family"] = font_tnr
  plt.rcParams["font.weight"] = "bold"

  features = m_cfg["features"]
  shap_vals = m_cfg["shap_values"]

  mean_abs = np.abs(shap_vals).mean(axis=0)
  df_im = pd.DataFrame({"Feature": features, "Importance": mean_abs})
  df_im = df_im.sort_values("Importance", ascending=True).reset_index(drop=True)

  fig_height = max(3.5, len(features) * 0.4 + 2.0)
  fig, ax = plt.subplots(figsize=(font_size, fig_height))

  bar_colors = [
      get_feature_color(f, m_cfg["id"], m_cfg["color"]) for f in df_im["Feature"]
  ]
  ax.barh(
      df_im["Feature"],
      df_im["Importance"],
      color=bar_colors,
      edgecolor="white",
      height=0.6,
  )

  ax.set_xlabel(
      "Mean |SHAP Value| (Global Importance)",
      fontsize=font_size,
      fontweight="bold",
      family=font_tnr,
  )

  # 左上角对齐标题
  ax.set_title(
      f"SHAP Feature Importance – {m_cfg['id']}: {m_cfg['name']}",
      fontsize=font_size + 2,
      fontweight="bold",
      family=font_tnr,
      pad=14,
      x=0.0,
      ha="left",
  )

  for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(font_size - 2)
    label.set_fontweight("bold")
    label.set_family(font_tnr)

  if m_cfg["id"] == "M1":
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(
            facecolor=color_subgroup["Meteorological"],
            label="Meteorological (21 vars)",
        ),
        Patch(
            facecolor=color_subgroup["Calendar temporal"],
            label="Calendar temporal (2 vars)",
        ),
        Patch(
            facecolor=color_subgroup["Socio-geographical"],
            label="Socio-geographical (2 vars)",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        prop={"family": font_tnr, "weight": "bold", "size": font_size - 2},
        loc="lower right",
    )

  ax.grid(False)
  plt.tight_layout()
  plt.savefig(save_path, dpi=300, bbox_inches="tight")
  plt.savefig(
      save_path.replace(".png", ".pdf"), format="pdf", bbox_inches="tight"
  )
  plt.close()
  plt.rcParams.update(plt.rcParamsDefault)


for m_cfg in model_configs:
  bar_name = f"SHAP_bar_{m_cfg['id']}_{m_cfg['name'].replace(' ', '_')}.png"
  plot_shap_bar_single(m_cfg, os.path.join(save_dir, bar_name))
  print(f"  ✅ 已保存 Bar 图: {bar_name}")

# ======================================================
# 8. 2x2 组合拼图：学术级对齐 SHAP 4-Panel Bar 图
# ======================================================
print("\n🎨 正在生成 2x2 柱状组合学术图 (动态行高)...")


def plot_shap_4panel_bar():
  plt.rcParams["font.family"] = font_tnr
  plt.rcParams["font.weight"] = "bold"

  fig = plt.figure(figsize=(36, 28))
  gs = gridspec.GridSpec(2, 2, height_ratios=[25, 4], hspace=0.35, wspace=0.35)

  axes = [
      fig.add_subplot(gs[0, 0]),
      fig.add_subplot(gs[0, 1]),
      fig.add_subplot(gs[1, 0]),
      fig.add_subplot(gs[1, 1]),
  ]
  panels = ["a ", "b ", "c ", "d "]

  for idx, m_cfg in enumerate(model_configs):
    ax = axes[idx]
    features = m_cfg["features"]
    shap_vals = m_cfg["shap_values"]

    mean_abs = np.abs(shap_vals).mean(axis=0)
    df_im = pd.DataFrame({"Feature": features, "Importance": mean_abs})
    df_im = df_im.sort_values("Importance", ascending=True).reset_index(
        drop=True
    )

    bar_colors = [
        get_feature_color(f, m_cfg["id"], m_cfg["color"])
        for f in df_im["Feature"]
    ]
    ax.barh(
        df_im["Feature"],
        df_im["Importance"],
        color=bar_colors,
        edgecolor="white",
        height=0.6,
    )

    # 精准左上角标题 (例如: A. M1: Full model)
    title_text = f"{panels[idx]} {m_cfg['id']}: {m_cfg['name']}"
    ax.set_title(
        title_text,
        fontsize=font_size,
        fontweight="bold",
        family=font_tnr,
        pad=14,
        x=0.0,
        ha="left",
    )

    ax.set_xlabel(
        "Mean |SHAP Value|", fontsize=font_size, fontweight="bold", family=font_tnr
    )

    ax.tick_params(axis="y", labelsize=font_size)
    ax.tick_params(axis="x", labelsize=font_size)
    for label in ax.get_yticklabels() + ax.get_xticklabels():
      label.set_fontfamily(font_tnr)
      label.set_fontweight("bold")
    ax.grid(False)

    if m_cfg["id"] == "M1":
      from matplotlib.patches import Patch

      legend_handles = [
          Patch(
              facecolor=color_subgroup["Meteorological"], label="Meteo"
          ),
          Patch(
              facecolor=color_subgroup["Calendar temporal"],
              label="Calendar",
          ),
          Patch(
              facecolor=color_subgroup["Socio-geographical"],
              label="Socio-geo",
          ),
      ]
      ax.legend(
          handles=legend_handles,
          frameon=False,
          prop={"family": font_tnr, "weight": "bold", "size": font_size},
          loc="lower right",
      )

  plt.tight_layout()
  combo_path = os.path.join(save_dir, "SHAP_bar_4Panel_Combined.png")
  plt.savefig(combo_path, dpi=300, bbox_inches="tight")
  plt.savefig(
      combo_path.replace(".png", ".pdf"), format="pdf", bbox_inches="tight"
  )
  plt.close()
  plt.rcParams.update(plt.rcParamsDefault)
  print(f"  ✅ 2x2 柱状学术拼图完成，已保存至: {combo_path}")


plot_shap_4panel_bar()

# ======================================================
# 9. 2x2 组合拼图：学术级对齐 SHAP 4-Panel Dot 图
# ======================================================
print("\n🎨 正在生成 2x2 散点组合学术图 (动态行高)...")


def plot_shap_4panel_dot():
  plt.rcParams["font.family"] = font_tnr
  plt.rcParams["font.weight"] = "bold"

  fig = plt.figure(figsize=(36, 28))
  gs = gridspec.GridSpec(2, 2, height_ratios=[25, 4], hspace=0.35, wspace=0.35)

  axes = [
      fig.add_subplot(gs[0, 0]),
      fig.add_subplot(gs[0, 1]),
      fig.add_subplot(gs[1, 0]),
      fig.add_subplot(gs[1, 1]),
  ]
  panels = ["A.", "B.", "C.", "D."]

  for idx, m_cfg in enumerate(model_configs):
    ax = axes[idx]
    plt.sca(ax)

    shap_vals = m_cfg["shap_values"]
    X_data = m_cfg["X_data"]

    shap.summary_plot(
        shap_vals,
        X_data,
        show=False,
        cmap=plt.get_cmap("coolwarm"),
        plot_type="dot",
        alpha=0.7,
        max_display=len(m_cfg["features"]),
        plot_size=None,
    )

    # 精准左上角标题 (例如: A. M1: Full model)
    title_text = f"{panels[idx]} {m_cfg['id']}: {m_cfg['name']}"
    ax.set_title(
        title_text,
        fontsize=font_size,
        fontweight="bold",
        family=font_tnr,
        pad=14,
        x=0.0,
        ha="left",
    )

    ax.set_xlabel(
        "SHAP Value (Impact on Model Output)",
        fontsize=font_size,
        fontweight="bold",
        family=font_tnr,
    )

    ax.tick_params(axis="y", labelsize=font_size)
    ax.tick_params(axis="x", labelsize=font_size)
    for label in ax.get_yticklabels() + ax.get_xticklabels():
      label.set_fontfamily(font_tnr)
      label.set_fontweight("bold")

    try:
      cb_ax = plt.gcf().axes[-1]
      cb_ax.set_ylabel(
          "Feature value", fontsize=font_size, fontweight="bold", family=font_tnr
      )
      cb_ax.tick_params(labelsize=font_size)
      for label in cb_ax.get_yticklabels():
        label.set_fontfamily(font_tnr)
    except Exception:
      pass

  plt.tight_layout()
  combo_dot_path = os.path.join(save_dir, "SHAP_dot_4Panel_Combined.png")
  plt.savefig(combo_dot_path, dpi=300, bbox_inches="tight")
  plt.savefig(
      combo_dot_path.replace(".png", ".pdf"), format="pdf", bbox_inches="tight"
  )
  plt.close()
  plt.rcParams.update(plt.rcParamsDefault)
  print(f"  ✅ 2x2 散点学术拼图完成，已保存至: {combo_dot_path}")


plot_shap_4panel_dot()

# ======================================================
# 10. 保存逐模型重要性表与 M1 分组贡献汇总 CSV
# ======================================================
print("\n📊 正在整理计算结果并保存 CSV 数据...")

for m_cfg in model_configs:
  features = m_cfg["features"]
  shap_vals = m_cfg["shap_values"]
  mean_abs = np.abs(shap_vals).mean(axis=0)

  df_im = pd.DataFrame({
      "变量": features,
      "均值_SHAP": np.round(mean_abs, 6),
      "物理分类": [get_feature_subgroup(f) for f in features],
  }).sort_values("均值_SHAP", ascending=False).reset_index(drop=True)

  total_model_shap = df_im["均值_SHAP"].sum()
  df_im["贡献占比(%)"] = np.round(
      (df_im["均值_SHAP"] / total_model_shap) * 100, 2
  )

  csv_name = (
      f"SHAP_逐变量重要性_{m_cfg['id']}_{m_cfg['name'].replace(' ', '_')}.csv"
  )
  df_im.to_csv(os.path.join(save_dir, csv_name), index=False, encoding="utf-8-sig")

# 整理 M1 三大亚组的贡献比例
m1_shap_vals = next(
    cfg["shap_values"] for cfg in model_configs if cfg["id"] == "M1"
)
m1_features = next(
    cfg["features"] for cfg in model_configs if cfg["id"] == "M1"
)
m1_mean_abs = np.abs(m1_shap_vals).mean(axis=0)

m1_df = pd.DataFrame({
    "变量": m1_features,
    "均值_SHAP": m1_mean_abs,
    "分组": [get_feature_subgroup(f) for f in m1_features],
})

total_m1_shap = m1_df["均值_SHAP"].sum()
group_summary = (
    m1_df.groupby("分组")["均值_SHAP"]
    .sum()
    .reset_index()
    .rename(columns={"均值_SHAP": "组内均值SHAP总和"})
)
group_summary["贡献占比(%)"] = np.round(
    (group_summary["组内均值SHAP总和"] / total_m1_shap) * 100, 2
)
group_summary = group_summary.sort_values(
    "贡献占比(%)", ascending=False
).reset_index(drop=True)

print("\n📊 M1 全模型三大物理亚组贡献汇总：")
print(group_summary.to_string(index=False))

group_summary["组内均值SHAP总和"] = group_summary["组内均值SHAP总和"].round(
    6
)
group_summary.to_csv(
    os.path.join(save_dir, "SHAP_M1_三大物理亚组贡献汇总.csv"),
    index=False,
    encoding="utf-8-sig",
)

print(f"\n💾 所有精美版 SHAP 图、组合图及报表已存至: {save_dir}")
print("🎉 学术级四模型 SHAP 分析排版对齐任务已全部高水准圆满完成！")
