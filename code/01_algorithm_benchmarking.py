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

from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# ==========================================
# 找到你原本导入 GBDT 的地方，替换为下面这个：
# ==========================================
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# 替换为这个：现代化支持 NaN 且速度飞快的直方图 GBDT
from sklearn.ensemble import HistGradientBoostingClassifier

# ======================================================
# 1. 读取数据与预处理
# ======================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "analysis_ready_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "01_algorithm_benchmarking"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("🔄 Step 1: 正在读取分析就绪数据集...")
data = pd.read_csv(DATA_PATH)
data = data.sort_values("FID").reset_index(drop=True)

y_col = "positivity_rate_P75_flag"
drop_cols = ['FID', 'REGION', 'YEAR', 'WEEK', 'week_date', 'total_specimens',
             'total_positive', 'positivity_rate', 'STUSPS', 'positivity_rate_prior1',
             'positivity_rate_prior2', 'positivity_rate_prior3', 'Pop', 'ALAND_SQMI']

required_metadata = {"FID", "REGION", "YEAR", y_col}
missing_metadata = sorted(required_metadata.difference(data.columns))
if missing_metadata:
    raise ValueError(
        f"analysis_ready_data.csv is missing required columns: {missing_metadata}"
    )


# ======================================================
# 2. “二维四区” 时空评估划分（全变量大盘）
# ======================================================
print("✂️ Step 2: 正在进行 [二维四区] 时空隔离划分...")

# 【空间地理隔离】
cities = data["REGION"].unique()
dev_cities, spat_eval_cities = train_test_split(cities, test_size=0.2, random_state=123)

dev_data = data[data["REGION"].isin(dev_cities)]
spat_eval_df = data[data["REGION"].isin(spat_eval_cities)]

# 【时间轴隔离】
temp_eval_df = dev_data[dev_data["YEAR"] >= 2024]
history_df = dev_data[dev_data["YEAR"] < 2024]

# 【历史大盘内部随机 8:2 划分训练与内部调参】
train_df, val_df = train_test_split(history_df, test_size=0.2, random_state=123, stratify=history_df[y_col])

# 提取 4 个分析分区的特征矩阵 X 与标签 y (此时为全变量)
X_train = train_df.drop(drop_cols + [y_col], axis=1, errors='ignore')
y_train = train_df[y_col]

X_val = val_df.drop(drop_cols + [y_col], axis=1, errors='ignore')
y_val = val_df[y_col]

X_temp_eval = temp_eval_df.drop(drop_cols + [y_col], axis=1, errors='ignore')
y_temp_eval = temp_eval_df[y_col]

X_spat_eval = spat_eval_df.drop(drop_cols + [y_col], axis=1, errors='ignore')
y_spat_eval = spat_eval_df[y_col]

# ======================================================
# 3. 初始化四大梯度提升算法 (完美适配 NaN 升级版)
# ======================================================
models = {
    "CatBoost": CatBoostClassifier(random_state=123, verbose=False),

    # 顺便删掉了 use_label_encoder=False，彻底解决那个黄色 Warning 警告
    "XGBoost": XGBClassifier(random_state=123, eval_metric='logloss'),

    "LightGBM": LGBMClassifier(random_state=123, verbosity=-1),

    # 替换为现代化的、支持缺失值的 HistGBDT
    "GBDT": HistGradientBoostingClassifier(random_state=123)
}


# ======================================================
# 4. 核心比对循环：计算各算法在四个分区的 AUC
# ======================================================
print("\n⚔️ Step 3: 开始进行四大算法基准测试（Benchmark）...")

results_list = []

for name, model in models.items():
    print(f" ➔ 正在训练并评估算法: {name}...")

    # 在全变量训练集上拟合模型
    model.fit(X_train, y_train)

    # 预测四个分区的概率值
    p_train = model.predict_proba(X_train)[:, 1]
    p_val = model.predict_proba(X_val)[:, 1]
    p_temp = model.predict_proba(X_temp_eval)[:, 1]
    p_spat = model.predict_proba(X_spat_eval)[:, 1]

    # 计算各自的 AUC 指标
    auc_train = auc(*roc_curve(y_train, p_train)[:2])
    auc_val = auc(*roc_curve(y_val, p_val)[:2])
    auc_temp = auc(*roc_curve(y_temp_eval, p_temp)[:2])
    auc_spat = auc(*roc_curve(y_spat_eval, p_spat)[:2])

    # 记录结果
    results_list.append({
        "Model": name,
        "AUC_Training": round(auc_train, 4),
        "AUC_Internal_Val": round(auc_val, 4),
        "AUC_Temporal_Evaluation": round(auc_temp, 4),
        "AUC_Spatial_Evaluation": round(auc_spat, 4)
    })

# 转化为 DataFrame 并按最具挑战性的空间评估 AUC 降序排列
benchmark_df = pd.DataFrame(results_list).sort_values(by="AUC_Spatial_Evaluation", ascending=False)

print("\n📊 四大算法时空双重验证基准测试结果：")
print(benchmark_df.to_string(index=False))

# ======================================================
# 5. 将对比表格保存到你的项目目录
# ======================================================
output_path = OUTPUT_DIR / "Model_Selection_Benchmark.csv"
benchmark_df.to_csv(output_path, index=False)
print(f"\n💾 基准测试表格已成功导出至: {output_path}，可直接用于论文附录 Table S3！")
