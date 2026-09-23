# -*- coding: utf-8 -*-
"""
NRI / IDI / DeLong（解析法，正确实现）
======================================================
Reference:
  DeLong ER, DeLong DM, Clarke-Pearson DL.
  Biometrics. 1988;44(3):837-845.
======================================================
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from scipy import stats
from scipy.stats import mannwhitneyu
import os

# ── 配置 ─────────────────────────────────────────────────────
DATA_PATH   = r"E:\USFlu\20260804\0Data\flu_final_updated_file.csv"
OUT_PATH    = r"E:\USFlu\20260804\2Figure\5Table2\NRI_IDI_DeLong_full1.csv"
TARGET_COL  = 'positivity_rate_P75_flag'
REGION_COL  = 'REGION'
YEAR_COL    = 'YEAR'
RANDOM_SEED = 123
TRAIN_YEAR_MAX = 2023
NRI_THRESHOLD  = 0.5

# 最优超参数（Optuna TPESampler seed=123）
BEST_PARAMS = {
    'iterations':    1985,
    'learning_rate': 0.050684513404147966,
    'depth':         7,
    'l2_leaf_reg':   8.748434896659107,
    'random_seed':   RANDOM_SEED,
    'verbose':       0,
}

MET_VARS = [
    "sp_max", "sp_min", "solar_rad_max", "Temp_anomaly",
    "AH_anomaly", "Temp_max_prior2", "Temp_min_prior5",
    "AH_mean_prior3", "AH_mean_prior4",
    "Precip_sum_prior5", "Precip_sum_prior6",
    "Solar_mean_prior5", "Solar_mean_prior6",
    "Temp_anomaly_prior1", "Temp_anomaly_prior3",
    "Temp_anomaly_prior4", "Temp_anomaly_prior5", "Temp_anomaly_prior6",
    "RH_anomaly_prior1", "RH_anomaly_prior6",
    "AH_anomaly_prior2", "AH_anomaly_prior5",
    "Solar_anomaly_prior1", "Solar_anomaly_prior3", "Solar_anomaly_prior4",
]
CAL_VARS = ['week_sin', 'week_cos']
GEO_VARS = ['hhs_region', 'pop_den']
ALL_VARS = MET_VARS + CAL_VARS + GEO_VARS

# ── 1. 加载数据 ───────────────────────────────────────────────
print(">>> 加载数据...")
data = pd.read_csv(DATA_PATH)
print(f"    {data.shape[0]} 行 × {data.shape[1]} 列")
missing = [v for v in ALL_VARS + [TARGET_COL] if v not in data.columns]
if missing:
    raise ValueError(f"以下列未找到：{missing}")

# ── 2. 数据划分 ───────────────────────────────────────────────
print("\n>>> 数据分区...")
cities = data[REGION_COL].unique()
dev_cities, spat_ext_cities = train_test_split(
    cities, test_size=0.2, random_state=RANDOM_SEED
)
dev_states     = list(dev_cities)
holdout_states = list(spat_ext_cities)

df_dev  = data[data[REGION_COL].isin(dev_states)].copy()
df_spat = data[data[REGION_COL].isin(holdout_states)].copy()
df_hist = df_dev[df_dev[YEAR_COL] <= TRAIN_YEAR_MAX].copy()
df_temp = df_dev[df_dev[YEAR_COL] >  TRAIN_YEAR_MAX].copy()

X_hist = df_hist[ALL_VARS]; y_hist = df_hist[TARGET_COL]
X_train, X_val, y_train, y_val = train_test_split(
    X_hist, y_hist, test_size=0.2,
    random_state=RANDOM_SEED, stratify=y_hist
)
X_temp = df_temp[ALL_VARS]; y_temp = df_temp[TARGET_COL]
X_spat = df_spat[ALL_VARS]; y_spat = df_spat[TARGET_COL]

PARTITIONS = {
    'Training':          (X_train, y_train),
    'Internal Val':      (X_val,   y_val),
    'Temporal External': (X_temp,  y_temp),
    'Spatial External':  (X_spat,  y_spat),
}
for name, (X, y) in PARTITIONS.items():
    print(f"    {name:22s}: n={len(X):6d}  事件率={y.mean():.4f}")

# ── 3. 训练 M1–M4 ────────────────────────────────────────────
print("\n>>> 训练四个模型...")
MODEL_DEFS = {
    'M1': ('Full model',         ALL_VARS),
    'M2': ('Meteorological',     MET_VARS),
    'M3': ('Calendar temporal',  CAL_VARS),
    'M4': ('Socio-geographical', GEO_VARS),
}
trained = {}
for mname, (mlabel, feat_cols) in MODEL_DEFS.items():
    clf = CatBoostClassifier(**BEST_PARAMS)
    clf.fit(X_train[feat_cols], y_train)
    trained[mname] = (clf, feat_cols)
    aucs = {pn: round(roc_auc_score(y, clf.predict_proba(X[feat_cols])[:,1]),4)
            for pn,(X,y) in PARTITIONS.items()}
    print(f"    {mname}({len(feat_cols):2d}vars) {mlabel:22s}  "
          + "  ".join(f"{k[:8]}={v}" for k,v in aucs.items()))

preds = {
    mname: {pname: clf.predict_proba(X[feat_cols])[:,1]
            for pname,(X,y) in PARTITIONS.items()}
    for mname,(clf,feat_cols) in trained.items()
}

# ── 4. DeLong 解析法（DeLong 1988）───────────────────────────
def delong_roc_variance(y_true, p_score):
    """
    计算单个分类器的 AUC 和 placement values（V10, V01）。
    AUC = Mann-Whitney U / (n_pos × n_neg)
    V10[i] = P(score_pos_i > score_neg)  ── 正例 i 的 placement value
    V01[j] = P(score_pos > score_neg_j)  ── 负例 j 的 placement value
    """
    pos = p_score[y_true == 1]
    neg = p_score[y_true == 0]
    n_pos = len(pos); n_neg = len(neg)
    u, _ = mannwhitneyu(pos, neg, alternative='greater')
    auc = u / (n_pos * n_neg)
    # placement values（正确用全部正/负例计算）
    V10 = np.array([np.mean(p > neg) + 0.5 * np.mean(p == neg) for p in pos])
    V01 = np.array([np.mean(pos > n) + 0.5 * np.mean(pos == n) for n in neg])
    var = np.var(V10, ddof=1) / n_pos + np.var(V01, ddof=1) / n_neg
    return auc, var, V10, V01


def delong_test(y_true, p_new, p_ref):
    """
    比较两个相关 AUC 的 DeLong 解析检验。
    协方差通过正例/负例 placement values 的协方差矩阵估计。
    """
    y = np.asarray(y_true, float)
    pn = np.asarray(p_new, float)
    pr = np.asarray(p_ref, float)

    auc_new, var_new, V10n, V01n = delong_roc_variance(y, pn)
    auc_ref, var_ref, V10r, V01r = delong_roc_variance(y, pr)

    n_pos = int(y.sum()); n_neg = int((1 - y).sum())

    # 两个 AUC 之差的方差 = Var(AUC_new) + Var(AUC_ref) − 2·Cov(AUC_new, AUC_ref)
    cov = (np.cov(V10n, V10r, ddof=1)[0, 1] / n_pos
           + np.cov(V01n, V01r, ddof=1)[0, 1] / n_neg)
    var_diff = var_new + var_ref - 2 * cov
    se = np.sqrt(max(var_diff, 1e-14))

    d = auc_new - auc_ref
    z = d / se
    p = float(2 * (1 - stats.norm.cdf(abs(z))))

    return {
        'AUC_new':   round(float(auc_new), 4),
        'AUC_ref':   round(float(auc_ref), 4),
        'ΔAUC':      round(float(d), 4),
        'DeLong_Z':  round(float(z), 3),
        'p_value':   '<0.001' if p < 0.001 else round(p, 4),
    }

# ── 5. NRI / IDI ─────────────────────────────────────────────
def calc_nri(y, p_ref, p_new, thr=NRI_THRESHOLD):
    y, p_ref, p_new = map(lambda x: np.asarray(x, float), [y, p_ref, p_new])
    ev = y == 1; ne = y == 0
    nri_e = (((p_new > thr) & (p_ref <= thr) & ev).sum()
             - ((p_new <= thr) & (p_ref > thr) & ev).sum()) / ev.sum()
    nri_n = (((p_new <= thr) & (p_ref > thr) & ne).sum()
             - ((p_new > thr) & (p_ref <= thr) & ne).sum()) / ne.sum()
    return {'NRI': round(float(nri_e + nri_n), 4),
            'NRI_events': round(float(nri_e), 4),
            'NRI_nonevents': round(float(nri_n), 4)}


def calc_idi(y, p_ref, p_new):
    y, p_ref, p_new = map(lambda x: np.asarray(x, float), [y, p_ref, p_new])
    ev = y == 1; ne = y == 0
    return {'IDI': round(float(
        (p_new[ev].mean() - p_new[ne].mean())
        - (p_ref[ev].mean() - p_ref[ne].mean())), 4)}

# ── 6. 主循环 ─────────────────────────────────────────────────
COMPARISONS = [
    ('M2 vs M3', 'M2', 'M3', 'Meteorological vs Calendar baseline'),
    ('M1 vs M3', 'M1', 'M3', 'Full model vs Calendar baseline'),
    ('M4 vs M3', 'M4', 'M3', 'Socio-geographical vs Calendar baseline'),
    ('M1 vs M2', 'M1', 'M2', 'Full model vs Meteorological model'),
]

print("\n>>> 计算 DeLong / NRI / IDI...")
rows = []
for pname, (X, y) in PARTITIONS.items():
    y_arr = np.asarray(y, float)
    for comp_label, new_m, ref_m, comp_desc in COMPARISONS:
        pn = preds[new_m][pname]
        pr = preds[ref_m][pname]
        dl = delong_test(y_arr, pn, pr)
        rows.append({
            'Partition':   pname,
            'Comparison':  comp_label,
            'Description': comp_desc,
            **dl,
            **calc_nri(y_arr, pr, pn),
            **calc_idi(y_arr, pr, pn),
        })

result_df = pd.DataFrame(rows)

# ── 7. 自检 ──────────────────────────────────────────────────
print("\n>>> 自检...")
chk = (result_df['AUC_new'] - result_df['AUC_ref'] - result_df['ΔAUC']).abs()
assert chk.max() < 1e-3, f"ΔAUC 自洽性检验失败，最大偏差 {chk.max():.6f}"
print(f"    ✓ ΔAUC 自洽（最大偏差 {chk.max():.6f}）")

# ── 8. 输出 ───────────────────────────────────────────────────
for pname in PARTITIONS:
    sub = result_df[result_df['Partition'] == pname]
    print(f"\n{'─'*90}\n  {pname}\n{'─'*90}")
    print(sub[['Comparison','AUC_ref','AUC_new','ΔAUC',
               'DeLong_Z','p_value','NRI','IDI']].to_string(index=False))

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
result_df.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
print(f"\n✓ 结果已保存 → {OUT_PATH}")
print("""
【表2注释同步更新】
DeLong Z: Z statistic from the DeLong analytic test (DeLong et al.,
Biometrics, 1988;44:837-845) comparing two correlated AUCs derived
from the same test partition; p-values are two-sided.
删除原注中的 "bootstrap" 和 "n=2,000 resampling iterations"。
超参数更新为：iterations=1,985; learning_rate=0.051;
              depth=7; l2_leaf_reg=8.748
""")