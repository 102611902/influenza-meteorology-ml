"""
WM vs SM vs CM 提前 0-4 周预警分析
四分区全部评估：Training / Internal Val / Temporal External / Spatial External
数据划分方式与原始主模型代码完全一致
"""

import json
import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ══════════════════════════════════════════════════════════════
# 0. 配置
# ══════════════════════════════════════════════════════════════
DATA_PATH = r"E:\USFlu\20260804\0Data\flu_final_updated_file_renamed.csv"
PARAM_PATH = r"E:\USFlu\20260804\1Model\2TiaoCan\catboost_best_params.json"
OUT_DIR = r"E:\USFlu\20260828\1Figure6JiaSur"
OUT_CSV = os.path.join(OUT_DIR, "lead_k_results_all_partitions1.csv")

TARGET = "positivity_rate_P75_flag"
REGION = "REGION"
YEAR_COL = "YEAR"

# 25 个 WM 变量（气象+日历+地理）
WM_VARS = [
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

# 纯监测变量 (Surveillance Model, SM)
SM_VARS = [
    "positivity_rate_prior1",
    "positivity_rate_prior2",
    "positivity_rate_prior3",
]

# 组合模型变量 (Combined Model, CM)
CM_VARS = WM_VARS + SM_VARS

# ══════════════════════════════════════════════════════════════
# 1. 加载数据与超参数
# ══════════════════════════════════════════════════════════════
print(">>> 加载数据...")
df = pd.read_csv(DATA_PATH)
df = df.sort_values([REGION, YEAR_COL, "WEEK"]).reset_index(drop=True)
print(f"    {len(df)} 行 × {df.shape[1]} 列")

print(">>> 加载超参数...")
with open(PARAM_PATH, "r", encoding="utf-8") as f:
    best_params = json.load(f)
best_params.update({"random_seed": 123, "verbose": 0})
print(f"    {best_params}")

# 检查监测变量是否存在
RUN_SURVEILLANCE = all(v in df.columns for v in SM_VARS)
print(
    f"    监测变量: {'✓ 全部找到 (跑 WM, SM, CM)' if RUN_SURVEILLANCE else '⚠️  缺少，只跑 WM'}"
)

# ══════════════════════════════════════════════════════════════
# 2. 数据划分（与原始主模型代码完全一致）
#    sorted() 保证 unique() 顺序固定，random_state=123 保证可复现
# ══════════════════════════════════════════════════════════════
print("\n>>> 数据划分...")
cities = df[REGION].unique()
dev_cities, spat_ext_cities = train_test_split(
    cities, test_size=0.2, random_state=123
)
dev_states = list(dev_cities)
holdout_states = list(spat_ext_cities)

print(f"    开发州:   {len(dev_states)} 个")
print(f"    隔离州:   {len(holdout_states)} 个 → {sorted(holdout_states)}")

# ══════════════════════════════════════════════════════════════
# 3. 主循环：Lead k = 0, 1, 2, 3, 4
# ══════════════════════════════════════════════════════════════
print("\n>>> 开始 Lead k 分析...")
all_rows = []

for k in range(5):
    print(f"\n  ── Lead {k} wk {'─'*40}")

    # 3-1. 构造未来 k 周结局
    df_k = df.copy()
    df_k["future_flag"] = df_k.groupby(REGION)[TARGET].shift(-k)
    df_k = df_k.dropna(subset=["future_flag"])
    df_k["future_flag"] = df_k["future_flag"].astype(int)

    # 3-2. 切出四个分区（与主模型代码逻辑一致）
    dev_data = df_k[df_k[REGION].isin(dev_states)]
    spat_ext_df = df_k[df_k[REGION].isin(holdout_states)]
    temp_ext_df = dev_data[dev_data[YEAR_COL] >= 2024]
    history_df = dev_data[dev_data[YEAR_COL] < 2024]

    # 3-3. 内部 8:2 分层切分
    train_df, val_df = train_test_split(
        history_df,
        test_size=0.2,
        random_state=123,
        stratify=history_df["future_flag"],
    )

    y_train = train_df["future_flag"]
    y_val = val_df["future_flag"]
    y_temp_ext = temp_ext_df["future_flag"]
    y_spat_ext = spat_ext_df["future_flag"]

    print(
        f"    训练集:          {len(train_df):6d}  事件率={y_train.mean():.3f}"
    )
    print(
        f"    内部验证集:      {len(val_df):6d}  事件率={y_val.mean():.3f}"
    )
    print(
        f"    时间外部验证集:  {len(temp_ext_df):6d}  事件率={y_temp_ext.mean():.3f}"
    )
    print(
        f"    空间外部验证集:  {len(spat_ext_df):6d}  事件率={y_spat_ext.mean():.3f}"
    )

    # 3-4. 定义三个模型的评估分区数据
    eval_partitions_wm = {
        "Training": (train_df[WM_VARS], y_train),
        "Internal Val": (val_df[WM_VARS], y_val),
        "Temporal External": (temp_ext_df[WM_VARS], y_temp_ext),
        "Spatial External": (spat_ext_df[WM_VARS], y_spat_ext),
    }

    eval_partitions_sm = (
        {
            "Training": (train_df[SM_VARS], y_train),
            "Internal Val": (val_df[SM_VARS], y_val),
            "Temporal External": (temp_ext_df[SM_VARS], y_temp_ext),
            "Spatial External": (spat_ext_df[SM_VARS], y_spat_ext),
        }
        if RUN_SURVEILLANCE
        else {}
    )

    eval_partitions_cm = (
        {
            "Training": (train_df[CM_VARS], y_train),
            "Internal Val": (val_df[CM_VARS], y_val),
            "Temporal External": (temp_ext_df[CM_VARS], y_temp_ext),
            "Spatial External": (spat_ext_df[CM_VARS], y_spat_ext),
        }
        if RUN_SURVEILLANCE
        else {}
    )

    # 3-5. 训练 WM 模型
    wm = CatBoostClassifier(**best_params)
    wm.fit(train_df[WM_VARS], y_train)

    # 3-6. 训练 SM & CM 模型
    sm, cm = None, None
    if RUN_SURVEILLANCE:
        sm = CatBoostClassifier(**best_params)
        sm.fit(train_df[SM_VARS], y_train)

        cm = CatBoostClassifier(**best_params)
        cm.fit(train_df[CM_VARS], y_train)

    # 3-7. 四分区评估与结果记录
    for part_name, (X_wm, y_eval) in eval_partitions_wm.items():

        wm_auc = roc_auc_score(y_eval, wm.predict_proba(X_wm)[:, 1])

        row = {
            "Lead_weeks": k,
            "Partition": part_name,
            "n_eval": len(y_eval),
            "event_rate": round(float(y_eval.mean()), 4),
            "WM_AUC": round(wm_auc, 4),
            "SM_AUC": None,
            "CM_AUC": None,
            "Delta_CM_WM": None,
            "Delta_CM_SM": None,
            "WM_above_080": wm_auc >= 0.80,
        }

        if RUN_SURVEILLANCE:
            # SM 评估
            X_sm, _ = eval_partitions_sm[part_name]
            sm_auc = roc_auc_score(y_eval, sm.predict_proba(X_sm)[:, 1])
            row["SM_AUC"] = round(sm_auc, 4)

            # CM 评估
            X_cm, _ = eval_partitions_cm[part_name]
            cm_auc = roc_auc_score(y_eval, cm.predict_proba(X_cm)[:, 1])
            row["CM_AUC"] = round(cm_auc, 4)

            # 增量评估
            row["Delta_CM_WM"] = round(cm_auc - wm_auc, 4)
            row["Delta_CM_SM"] = round(cm_auc - sm_auc, 4)

        all_rows.append(row)

        line = f"    {part_name:22s}: WM={wm_auc:.4f}"
        if RUN_SURVEILLANCE:
            line += f"  SM={row['SM_AUC']:.4f}  CM={row['CM_AUC']:.4f}  Δ(CM-WM)={row['Delta_CM_WM']:+.4f}"
        print(line)

# ══════════════════════════════════════════════════════════════
# 4. 整理结果
# ══════════════════════════════════════════════════════════════
result_df = pd.DataFrame(all_rows)

part_order = [
    "Training",
    "Internal Val",
    "Temporal External",
    "Spatial External",
]
result_df["Partition"] = pd.Categorical(
    result_df["Partition"], categories=part_order, ordered=True
)
result_df = result_df.sort_values(["Lead_weeks", "Partition"]).reset_index(
    drop=True
)

# ══════════════════════════════════════════════════════════════
# 5. 打印汇总
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 92)
print(
    f"{'Lead':>5}  {'Partition':<20}  {'WM_AUC':>8}  {'SM_AUC':>8}  "
    f"{'CM_AUC':>8}  {'Δ(CM-WM)':>9}  {'Δ(CM-SM)':>9}  {'WM≥0.80':>7}"
)
print("-" * 92)
for _, r in result_df.iterrows():
    sm_str = f"{r['SM_AUC']:.4f}" if pd.notna(r["SM_AUC"]) else "   N/A"
    cm_str = f"{r['CM_AUC']:.4f}" if pd.notna(r["CM_AUC"]) else "   N/A"
    dl_wm_str = (
        f"{r['Delta_CM_WM']:+.4f}" if pd.notna(r["Delta_CM_WM"]) else "     N/A"
    )
    dl_sm_str = (
        f"{r['Delta_CM_SM']:+.4f}" if pd.notna(r["Delta_CM_SM"]) else "     N/A"
    )
    ok_str = "✓" if r["WM_above_080"] else "✗"

    print(
        f"{int(r['Lead_weeks']):>5}  {r['Partition']:<20}  "
        f"{r['WM_AUC']:.4f}  {sm_str}  {cm_str}  {dl_wm_str}  {dl_sm_str}  {ok_str:>7}"
    )
print("=" * 92)

# ══════════════════════════════════════════════════════════════
# 6. 保存
# ══════════════════════════════════════════════════════════════
os.makedirs(OUT_DIR, exist_ok=True)
result_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n✓ 结果已保存 → {OUT_CSV}")
print(f"  共 {len(result_df)} 行（5 Lead × 4 分区）")