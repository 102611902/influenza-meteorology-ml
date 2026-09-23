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
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json  # 用于结构化保存和读取最优超参数
from catboost import CatBoostClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold
import optuna
from optuna.samplers import TPESampler

# 关闭 Optuna 冗长的打印日志，只保留警告，让控制台更干净
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Use a non-interactive backend for headless/server environments.

# ======================================================
# 1. 读取数据与预处理
# ======================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "analysis_ready_data.csv"
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "02_catboost_hyperparameter_tuning"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
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
# 2. “二维四区” 时空评估划分
# ======================================================
print("✂️ Step 2: 正在进行 [二维四区] 时空隔离划分...")

# 【第一步：空间地理隔离】
cities = data["REGION"].unique()
dev_cities, spat_eval_cities = train_test_split(cities, test_size=0.2, random_state=123)

dev_data = data[data["REGION"].isin(dev_cities)]
X_spat_eval = data[data["REGION"].isin(spat_eval_cities)]
y_spat_eval = X_spat_eval[y_col]
X_spat_eval = X_spat_eval.drop(drop_cols + [y_col], axis=1, errors='ignore')

# 【第二步：时间轴隔离】
temp_eval_df = dev_data[dev_data["YEAR"] >= 2024]
y_temp_eval = temp_eval_df[y_col]
X_temp_eval = temp_eval_df.drop(drop_cols + [y_col], axis=1, errors='ignore')

history_df = dev_data[dev_data["YEAR"] < 2024]

# 【第三步：在历史大盘内部随机 8:2 划分最终训练与内部调参 (用于最终画图、定型和后续 RFE)】
train_df, val_df = train_test_split(history_df, test_size=0.2, random_state=123, stratify=history_df[y_col])

X_train = train_df.drop(drop_cols + [y_col], axis=1, errors='ignore')
y_train = train_df[y_col]
X_val = val_df.drop(drop_cols + [y_col], axis=1, errors='ignore')
y_val = val_df[y_col]

print(f"📊 数据集划分最终确认：")
print(f" └── 1. 训练集样本数 (10-23年随机80%): {len(X_train)} 条")
print(f" └── 2. 内部调参集样本数 (10-23年随机20%): {len(X_val)} 条")
print(f" └── 3. 时间评估集 (24-26年纯未来): {len(X_temp_eval)} 条")
print(f" └── 4. 空间评估集 (隔离20%州全时段): {len(X_spat_eval)} 条")

# ======================================================
# 3. 自动化超参数寻优 (补充材料记录版：保存每一个Trial的不同折数AUC)
# ======================================================
print("\n🤖 Step 3: 启动 Optuna + 5-Fold CV 广谱泛化调参系统...")

# 提取完整的历史大盘，供 Optuna 内部进行多考场动态轮考
X_hist_all = history_df.drop(drop_cols + [y_col], axis=1, errors='ignore')
y_hist_all = history_df[y_col]


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'random_state': 123,
        'verbose': False
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    cv_scores = []

    # 执行 5 折交叉循环验证
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_hist_all, y_hist_all), 1):
        X_tr, X_va = X_hist_all.iloc[train_idx], X_hist_all.iloc[val_idx]
        y_tr, y_va = y_hist_all.iloc[train_idx], y_hist_all.iloc[val_idx]

        tmp_model = CatBoostClassifier(**params)
        tmp_model.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=30,
            use_best_model=True
        )

        preds = tmp_model.predict_proba(X_va)[:, 1]
        fold_auc = auc(*roc_curve(y_va, preds)[:2])
        cv_scores.append(fold_auc)

        # 💡 核心改动：把当前参数在这一折的真实 AUC 塞进 Optuna 的单独记录表中
        trial.set_user_attr(f"Fold_{fold_idx}_AUC", fold_auc)

    return np.mean(cv_scores)

study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=123))

print("⏳ 正在进行 30 轮 5-Fold CV 贝叶斯联合演化，并像素级记录每一折的战况...")
study.optimize(objective, n_trials=30)

print("\n👑 广谱超参数最佳组合搜索完成！")
print("最佳 5 折平均 AUC 成绩:", study.best_value)
print("最优参数字典:", study.best_params)

# 📂 定义统一保存路径
param_save_dir = OUTPUT_DIR

# 💡 核心增补：提取包含“不同参数 + 不同折数 AUC”的完整审计大表（用于论文补充材料）
print("💾 正在构建并导出用于论文【补充材料】的超参数全折数审计大表...")
trials_df = study.trials_dataframe()

# 重命名列名以符合学术规范
rename_dict = {
    "number": "Trial_ID",
    "value": "Mean_AUC",
    "params_iterations": "iterations",
    "params_learning_rate": "learning_rate",
    "params_depth": "depth",
    "params_l2_leaf_reg": "l2_leaf_reg",
    "user_attrs_Fold_1_AUC": "Fold_1_AUC",
    "user_attrs_Fold_2_AUC": "Fold_2_AUC",
    "user_attrs_Fold_3_AUC": "Fold_3_AUC",
    "user_attrs_Fold_4_AUC": "Fold_4_AUC",
    "user_attrs_Fold_5_AUC": "Fold_5_AUC"
}
selected_cols = list(rename_dict.keys())
audit_table = trials_df[selected_cols].rename(columns=rename_dict)

# 按照平均分由高到低降序排列，方便审稿人一眼看到第一行就是最优解
audit_table = audit_table.sort_values(by="Mean_AUC", ascending=False).reset_index(drop=True)
audit_table.to_csv(os.path.join(param_save_dir, "Optuna_Hyperparameter_Tuning_Audit.csv"), index=False)
print(f"✅ 补充材料核心表格已保存至: {os.path.join(param_save_dir, 'Optuna_Hyperparameter_Tuning_Audit.csv')}")

# 单独保存一份最优参数字典 json 文件
param_json_path = CONFIG_DIR / "catboost_best_params.json"
with open(param_json_path, 'w', encoding='utf-8') as f:
    json.dump(study.best_params, f, indent=4, ensure_ascii=False)

# ======================================================
# 4. 锁定最佳模型并进行最终多维验证预测 (用回你的 8:2 划分大盘)
# ======================================================
print("\n🔒 Step 4: 正在固化最佳超参数，构建最终定型模型...")
best_params = study.best_params
best_params.update({'random_state': 123, 'verbose': False})

final_model = CatBoostClassifier(**best_params)
# 用全考场锤炼出的最优参数在你的经典 X_train 上进行终极拟合
final_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50,
    use_best_model=True
)

# 最终预测四个分区的概率值
proba_train = final_model.predict_proba(X_train)[:, 1]
proba_val = final_model.predict_proba(X_val)[:, 1]
proba_temp_eval = final_model.predict_proba(X_temp_eval)[:, 1]
proba_spat_eval = final_model.predict_proba(X_spat_eval)[:, 1]


# ======================================================
# 5. 专业学术风格的四曲线 ROC 绘图函数 (Times New Roman 版)
# ======================================================
def plot_professional_roc(data_dict, title, save_path):
    print("🎨 Step 5: 正在绘制符合二区顶刊标准的学术风 ROC 曲线...")
    font_tnr = "Times New Roman"
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(8.5, 7.5))
    colors_map = {
        "Training Set": "#e78ac3",
        "Internal tuningidation": "#FF7F26",
        "Temporal evaluation": "#1F77B4",
        "Spatial evaluation": "#2A9D36"
    }

    for name, (y_true, y_pred) in data_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=3.5, color=colors_map[name],
                 label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], '--', color='#999999', linewidth=1.5, alpha=0.7)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate", fontsize=20, fontweight="bold", family=font_tnr)
    plt.ylabel("True Positive Rate", fontsize=20, fontweight="bold", family=font_tnr)
    plt.title(title, fontsize=18, fontweight="bold", family=font_tnr, pad=20)
    plt.legend(loc="lower right", frameon=False, prop={'family': font_tnr, 'weight': 'bold', 'size': 15})
    plt.xticks(fontsize=18, family=font_tnr)
    plt.yticks(fontsize=18, family=font_tnr)
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace(".png", ".pdf"), format='pdf', bbox_inches='tight')
    plt.show()
    plt.close()


data_to_plot = {
    "Training Set": (y_train, proba_train),
    "Internal tuningidation": (y_val, proba_val),
    "Temporal evaluation": (y_temp_eval, proba_temp_eval),
    "Spatial evaluation": (y_spat_eval, proba_spat_eval)
}

plot_professional_roc(
    data_to_plot,
    "ROC Curves: Spatiotemporal Evaluation (CV Optimized)",
    os.path.join(param_save_dir, "ROC_SpatioTemporal_Professional.png")
)

# ======================================================
# 6. 保存调参后的多维度指标结果与特征重要性
# ======================================================
print("💾 Step 6: 正在将调参后的最优统计指标与特征重要性写入磁盘...")

auc_train = auc(*roc_curve(y_train, proba_train)[:2])
auc_val = auc(*roc_curve(y_val, proba_val)[:2])
auc_temp_eval = auc(*roc_curve(y_temp_eval, proba_temp_eval)[:2])
auc_spat_eval = auc(*roc_curve(y_spat_eval, proba_spat_eval)[:2])

summary_df = pd.DataFrame([{
    "AUC_train": auc_train,
    "AUC_internal_val": auc_val,
    "AUC_temporal_evaluation": auc_temp_eval,
    "AUC_spatial_evaluation": auc_spat_eval,
    "Best_Tuned_Params": str(best_params),
    "Samples_train": len(X_train),
    "Samples_internal_val": len(X_val),
    "Samples_temporal_evaluation": len(X_temp_eval),
    "Samples_spatial_evaluation": len(X_spat_eval)
}])
summary_df.to_csv(os.path.join(param_save_dir, "Evaluation_Summary_SpatioTemporal.csv"), index=False)

importances_df = pd.DataFrame({
    '特征名称': X_train.columns,
    '特征重要性': final_model.feature_importances_
}).sort_values('特征重要性', ascending=False)

importances_df.to_excel(os.path.join(param_save_dir, "ALL76CatBoostP75_SpatioTemporal.xlsx"), index=False)

print("\n================== 🎉 自动化调参任务圆满完成 ==================")
print(f"✅ 完美的补充材料大表已成功导出: {os.path.join(param_save_dir, 'Optuna_Hyperparameter_Tuning_Audit.csv')}")
print("✅ 新一轮时空外推 ROC 图与特征重要性 Excel 已全部刷新，请查看成果！")
