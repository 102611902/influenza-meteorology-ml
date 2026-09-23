import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
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
# 使用 TkAgg 后端确保图形能正常弹出
matplotlib.use('TkAgg')

# ======================================================
# 1. 读取数据与预处理
# ======================================================
print("🔄 Step 1: 正在读取并载入原始流感数据集...")
data = pd.read_csv(r"E:\USFlu\20260804\0Data\flu_final_updated_file.csv")
data = data.sort_values("FID").reset_index(drop=True)

y_col = "positivity_rate_P75_flag"
drop_cols = ['FID', 'REGION', 'YEAR', 'WEEK', 'week_date', 'total_specimens',
             'total_positive', 'positivity_rate', 'STUSPS', 'positivity_rate_prior1',
             'positivity_rate_prior2', 'positivity_rate_prior3', 'Pop', 'ALAND_SQMI']

# ======================================================
# 2. “二维四区” 时空双重独立划分（全变量大盘）
# ======================================================
print("✂️ Step 2: 正在进行 [二维四区] 时空隔离划分...")

# 【空间地理隔离】
cities = data["REGION"].unique()
dev_cities, spat_ext_cities = train_test_split(cities, test_size=0.2, random_state=123)

dev_data = data[data["REGION"].isin(dev_cities)]
spat_ext_df = data[data["REGION"].isin(spat_ext_cities)]

# 【时间轴隔离】
temp_ext_df = dev_data[dev_data["YEAR"] >= 2024]
history_df = dev_data[dev_data["YEAR"] < 2024]

# 【历史大盘内部随机 8:2 划分训练与内部验证】
train_df, val_df = train_test_split(history_df, test_size=0.2, random_state=123, stratify=history_df[y_col])

# 提取 4 个独立分区的特征矩阵 X 与标签 y (此时为全变量)
X_train = train_df.drop(drop_cols + [y_col], axis=1, errors='ignore')
y_train = train_df[y_col]

X_val = val_df.drop(drop_cols + [y_col], axis=1, errors='ignore')
y_val = val_df[y_col]

X_temp_ext = temp_ext_df.drop(drop_cols + [y_col], axis=1, errors='ignore')
y_temp_ext = temp_ext_df[y_col]

X_spat_ext = spat_ext_df.drop(drop_cols + [y_col], axis=1, errors='ignore')
y_spat_ext = spat_ext_df[y_col]

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
    p_temp = model.predict_proba(X_temp_ext)[:, 1]
    p_spat = model.predict_proba(X_spat_ext)[:, 1]

    # 计算各自的 AUC 指标
    auc_train = auc(*roc_curve(y_train, p_train)[:2])
    auc_val = auc(*roc_curve(y_val, p_val)[:2])
    auc_temp = auc(*roc_curve(y_temp_ext, p_temp)[:2])
    auc_spat = auc(*roc_curve(y_spat_ext, p_spat)[:2])

    # 记录结果
    results_list.append({
        "Model": name,
        "AUC_Training": round(auc_train, 4),
        "AUC_Internal_Val": round(auc_val, 4),
        "AUC_Temporal_Ext": round(auc_temp, 4),
        "AUC_Spatial_Ext": round(auc_spat, 4)
    })

# 转化为 DataFrame 并按最具挑战性的空间外部验证 AUC 降序排列
benchmark_df = pd.DataFrame(results_list).sort_values(by="AUC_Spatial_Ext", ascending=False)

print("\n📊 四大算法时空双重验证基准测试结果：")
print(benchmark_df.to_string(index=False))

# ======================================================
# 5. 将对比表格保存到你的项目目录
# ======================================================
output_path = r"E:\USFlu\20260804\1Model\1SelectModel\Model_Selection_Benchmark.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
benchmark_df.to_csv(output_path, index=False)
print(f"\n💾 基准测试表格已成功导出至: {output_path}，可直接用于论文附录 Table S3！")
