# -*- coding: utf-8 -*-
"""
修复内容：
1. 解决 SHAP 热力图报错 (IndexError) -> 改用新版 explainer(X) 接口
2. 屏蔽 TensorFlow OneDNN 警告
3. 优化 SHAP 绘图函数兼容性
"""

import os

# 1. 屏蔽 TensorFlow  OneDNN 警告 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from math import pi

# PyQt5
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QGroupBox, QMessageBox, QFrame,
    QGraphicsDropShadowEffect
)
from PyQt5.QtGui import QDoubleValidator, QFont, QColor
from PyQt5.QtCore import Qt

# 绘图风格设置
plt.style.use('ggplot')
sns.set_theme(style="whitegrid", palette="muted")
# 解决中文显示问题 (根据系统自动调整)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# PART 1: 数据生成与模型训练
# ==========================================

def generate_full_data(n_samples_comp=863, n_samples_flex=321):
    """
    模拟生成抗压(Compressive)和抗折(Flexural)两套数据。
    基于论文 Table 1 & Table 2 的统计特征。
    """
    np.random.seed(42)

    # 17个特征列名
    features = [
        'C', 'SF', 'QP', 'FA', 'SL', 'MK', 'S', 'QS', 'W', 'SP',
        'Age', 'L', 'D', 'BV', 'PPV', 'GV', 'SSV'
    ]

    def gen_X(n):
        return pd.DataFrame({
            'C': np.random.uniform(317, 1277, n),
            'SF': np.random.uniform(0, 390, n),
            'QP': np.random.uniform(0, 363, n),
            'FA': np.random.uniform(0, 475, n),
            'SL': np.random.uniform(0, 475, n),
            'MK': np.random.uniform(0, 210, n),
            'S': np.random.uniform(0, 1503, n),
            'QS': np.random.uniform(0, 1170, n),
            'W': np.random.uniform(141, 286, n),
            'SP': np.random.uniform(8, 96, n),
            'Age': np.random.choice([3, 7, 28, 56, 90, 180, 360], n),
            'L': np.random.uniform(0, 54, n),
            'D': np.random.uniform(0, 0.9, n),
            'BV': np.random.uniform(0, 3, n),
            'PPV': np.random.uniform(0, 3, n),
            'GV': np.random.uniform(0, 3, n),
            'SSV': np.random.uniform(0, 8, n)
        })

    # 生成抗压数据
    X_comp = gen_X(n_samples_comp)
    # 模拟抗压强度 (120-180 MPa范围)
    y_comp = (0.08 * X_comp['C'] + 0.15 * X_comp['SF'] - 0.2 * X_comp['W'] +
              3 * X_comp['Age'] ** 0.5 + 8 * X_comp['SSV'] + np.random.normal(0, 8, n_samples_comp))

    # 生成抗折数据
    X_flex = gen_X(n_samples_flex)
    # 模拟抗折强度 (15-35 MPa范围)
    y_flex = (0.01 * X_flex['C'] + 0.05 * X_flex['SF'] - 0.05 * X_flex['W'] +
              1.5 * X_flex['SSV'] + 0.5 * X_flex['Age'] ** 0.3 + np.random.normal(0, 3, n_samples_flex))

    return X_comp, y_comp, X_flex, y_flex, features


def get_optimized_model(target_type='Compressive'):
    """
    返回论文 Table 4 中针对不同目标的最佳超参数模型 (CatBoost)。
    """
    if target_type == 'Compressive':
        return CatBoostRegressor(
            depth=5, iterations=800, learning_rate=0.2,
            verbose=0, random_state=42, allow_writing_files=False
        )
    else:
        return CatBoostRegressor(
            depth=4, iterations=1000, learning_rate=0.05,
            verbose=0, random_state=42, allow_writing_files=False
        )


def train_evaluation_suite(X_train, y_train, target_type='Compressive'):
    """
    训练所有模型以生成雷达图数据。
    """
    models = {
        'CatBoost': get_optimized_model(target_type),
        'XGBoost': xgb.XGBRegressor(n_estimators=500, max_depth=10 if target_type == 'Compressive' else 5,
                                    random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=500, verbose=-1, random_state=42),
        'GBM': GradientBoostingRegressor(n_estimators=500, random_state=42),
        'ExtraTree': ExtraTreesRegressor(n_estimators=100, random_state=42)
    }

    metrics = {}
    best_model = None

    print(f"   Training suite for {target_type} Strength...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        if name == 'CatBoost':
            best_model = model

        # 简单记录模型对象
        metrics[name] = model

    return metrics, best_model


# ==========================================
# PART 2: 全套科研绘图系统 
# ==========================================

def plot_distributions(y_comp, y_flex):
    """Fig. 2: 数据分布直方图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(y_comp, kde=True, color='blue', ax=axes[0], bins=20)
    axes[0].set_title('Distribution of Compressive Strength (Fig. 2a)')
    axes[0].set_xlabel('Strength (MPa)')

    sns.histplot(y_flex, kde=True, color='red', ax=axes[1], bins=20)
    axes[1].set_title('Distribution of Flexural Strength (Fig. 2b)')
    axes[1].set_xlabel('Strength (MPa)')
    plt.tight_layout()
    plt.show()


def plot_correlation(X, y, title):
    """Fig. 3 & 4: 相关性热力图"""
    df = X.copy()
    df['Target'] = y
    plt.figure(figsize=(12, 10))
    corr = df.corr(method='kendall')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, square=True, linewidths=.5)
    plt.title(f'{title} Correlation Matrix (Kendall)')
    plt.show()


def plot_radar(metrics_dict, X_test, y_test, title):
    """Fig. 9 & 10: 模型性能对比雷达图 (R2指标)"""
    labels = list(metrics_dict.keys())
    scores = []
    for name, model in metrics_dict.items():
        y_pred = model.predict(X_test)
        scores.append(r2_score(y_test, y_pred))

    angles = np.linspace(0, 2 * pi, len(labels), endpoint=False).tolist()
    scores += scores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, scores, color='purple', alpha=0.25)
    ax.plot(angles, scores, color='purple', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.5, 0.75, 0.9, 1.0])
    plt.title(f'{title} Model Comparison (R2)')
    plt.show()


def plot_errors(y_true, y_pred, title_prefix):
    """Fig. 11 & 12: 误差分布与散点图"""
    errors = y_true - y_pred

    # Fig 11: Error Histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(errors, kde=True, color='green', stat="density")
    from scipy.stats import norm
    mu, std = norm.fit(errors)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title(f'{title_prefix}: Histogram of Errors (Fig. 11)')
    plt.show()

    # Fig 12: Scatter Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, c='b' if 'Compressive' in title_prefix else 'r')
    m = max(y_true.max(), y_pred.max())
    plt.plot([0, m], [0, m], 'k--')
    plt.plot([0, m], [0, m * 1.1], 'g--', label='+10%')
    plt.plot([0, m], [0, m * 0.9], 'g--', label='-10%')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{title_prefix}: Actual vs Predicted (Fig. 12)')
    plt.legend()
    plt.show()


def plot_shap_detailed(model, X, title_prefix):
    """
    Fig. 13, 14, 15: 全套 SHAP 分析 (改正版)
    使用 explainer(X) 生成完整的 Explanation 对象，解决 IndexError 问题
    """
    # 1. 创建 Explainer
    explainer = shap.TreeExplainer(model)

    # 2. 获取完整的解释对象 (包含 values, base_values, data, feature_names)
    #    注意：这里直接传入 DataFrame X，SHAP 会自动提取列名
    shap_explanation = explainer(X)

    # Fig 13: Bar Plot (Feature Importance)
    plt.figure()
    shap.plots.bar(shap_explanation, show=False)
    plt.title(f'{title_prefix} Feature Importance (Fig. 13)')
    plt.show()

    # Fig 14: Beeswarm Plot
    plt.figure()
    shap.plots.beeswarm(shap_explanation, show=False)
    plt.title(f'{title_prefix} SHAP Beeswarm (Fig. 14)')
    plt.show()

    # Fig 15: Heatmap
    plt.figure()
    shap.plots.heatmap(shap_explanation, show=False)
    plt.title(f'{title_prefix} SHAP Heatmap (Fig. 15)')
    plt.show()


def plot_fiber_curves(model, X_train, fiber_cols=['SSV', 'BV', 'PPV', 'GV']):
    """Fig. 16-19: 纤维掺量影响分析"""
    base_mix = X_train.mean().to_frame().T
    plt.figure(figsize=(10, 6))
    volumes = np.linspace(0, 3, 20)
    colors = {'SSV': 'red', 'BV': 'blue', 'PPV': 'green', 'GV': 'orange'}

    for fib in fiber_cols:
        preds = []
        for v in volumes:
            temp = base_mix.copy()
            temp[fib] = v
            preds.append(model.predict(temp)[0])
        plt.plot(volumes, preds, label=fib, color=colors.get(fib, 'black'), linewidth=2)

    plt.xlabel('Fiber Volume (%)')
    plt.ylabel('Predicted Strength (MPa)')
    plt.title('Effect of Fiber Volume on Strength (Fig. 16-19)')
    plt.legend()
    plt.show()


def plot_contour(model, X_train):
    """Fig. 20: 等高线图"""
    c_range = np.linspace(X_train['C'].min(), X_train['C'].max(), 50)
    sf_range = np.linspace(X_train['SF'].min(), X_train['SF'].max(), 50)
    C_grid, SF_grid = np.meshgrid(c_range, sf_range)

    base = X_train.mean()
    Z_grid = np.zeros_like(C_grid)

    for i in range(50):
        for j in range(50):
            sample = base.copy()
            sample['C'] = C_grid[i, j]
            sample['SF'] = SF_grid[i, j]
            Z_grid[i, j] = model.predict(pd.DataFrame([sample]))[0]

    plt.figure(figsize=(10, 8))
    cp = plt.contourf(C_grid, SF_grid, Z_grid, cmap='viridis', levels=20)
    plt.colorbar(cp, label='Compressive Strength (MPa)')
    plt.xlabel('Cement Content (kg/m3)')
    plt.ylabel('Silica Fume Content (kg/m3)')
    plt.title('Contour Plot: Cement vs SF (Fig. 20)')
    plt.show()


# ==========================================
# PART 3:PyQt5 GUI (双输出版)
# ==========================================

class ModernGUI(QMainWindow):
    def __init__(self, model_comp, model_flex, feature_names):
        super().__init__()
        self.model_comp = model_comp
        self.model_flex = model_flex
        self.feature_names = feature_names
        self.inputs = {}

        self.default_values = {
            'C': 1000, 'SF': 150, 'S': 900, 'W': 200, 'SP': 50,
            'Age': 28, 'L': 13, 'D': 0.2, 'GV': 1.5, 'SSV': 1.0,
            'BV': 0, 'PPV': 0, 'QP': 0, 'FA': 0, 'SL': 0, 'MK': 0, 'QS': 0
        }

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("UHPC 性能预测系统 Pro (Based on CBM 2025)")
        self.resize(1150, 800)

        self.setStyleSheet("""
        QMainWindow { background: #0f172a; }
        QFrame#HeaderCard {
            border-radius: 20px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #8b5cf6);
        }
        QLabel { color: #e2e8f0; font-family: 'Segoe UI', Arial; }
        QLabel#HeaderTitle { font-size: 26px; font-weight: bold; color: white; }
        QGroupBox {
            border: 1px solid #334155; border-radius: 15px;
            background: #1e293b; margin-top: 20px; font-weight: bold; color: #94a3b8;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 5px; }
        QLineEdit {
            background: #0f172a; border: 1px solid #475569; border-radius: 8px;
            padding: 8px; color: white; font-size: 14px;
        }
        QLineEdit:focus { border: 1px solid #60a5fa; }
        QPushButton#BtnPredict {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #10b981, stop:1 #059669);
            color: white; border-radius: 10px; font-size: 16px; font-weight: bold; padding: 12px;
        }
        QPushButton#BtnPredict:hover { background: #34d399; }
        QPushButton#BtnReset {
            background: transparent; border: 1px solid #64748b; color: #94a3b8;
            border-radius: 10px; padding: 10px;
        }
        QPushButton#BtnReset:hover { border-color: #cbd5e1; color: white; }
        QLabel#ValComp { color: #60a5fa; font-size: 36px; font-weight: bold; }
        QLabel#ValFlex { color: #f472b6; font-size: 36px; font-weight: bold; }
        QLabel#UnitLabel { color: #64748b; font-size: 12px; }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # Header
        header = QFrame()
        header.setObjectName("HeaderCard")
        h_layout = QHBoxLayout(header)
        title_box = QVBoxLayout()
        title_box.addWidget(QLabel("UHPC 性能智能预测系统", objectName="HeaderTitle"))
        title_box.addWidget(QLabel("基于机器学习的超高性能混凝土抗压与抗折强度联合预测"))
        h_layout.addLayout(title_box)
        h_layout.addStretch()
        h_layout.addWidget(QLabel("Version 2.0 | Dual-Model Engine"))
        layout.addWidget(header)

        # Body
        body_layout = QHBoxLayout()

        # Left: Inputs
        input_group = QGroupBox("  材料配合比参数 (Mix Design Inputs)  ")
        grid = QGridLayout()
        grid.setVerticalSpacing(15)
        grid.setHorizontalSpacing(20)

        validator = QDoubleValidator(0.0, 5000.0, 2)
        ordered_keys = [
            'C', 'SF', 'S', 'W', 'SP', 'Age',
            'SSV', 'BV', 'PPV', 'GV',
            'L', 'D', 'QP', 'FA', 'SL', 'MK', 'QS'
        ]

        for i, key in enumerate(ordered_keys):
            lbl = QLabel(f"{key}:")
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            inp = QLineEdit()
            inp.setValidator(validator)
            inp.setText(str(self.default_values.get(key, 0)))
            self.inputs[key] = inp

            row, col = divmod(i, 2)
            grid.addWidget(lbl, row, col * 2)
            grid.addWidget(inp, row, col * 2 + 1)

        input_group.setLayout(grid)
        body_layout.addWidget(input_group, stretch=3)

        # Right: Results
        res_group = QGroupBox("  预测结果 (Predicted Strength)  ")
        v_res = QVBoxLayout()
        v_res.setSpacing(30)
        v_res.setContentsMargins(20, 40, 20, 20)

        box_comp = QVBoxLayout()
        box_comp.addWidget(QLabel("🟦 抗压强度 (Compressive)"))
        self.lbl_comp = QLabel("---")
        self.lbl_comp.setObjectName("ValComp")
        self.lbl_comp.setAlignment(Qt.AlignRight)
        box_comp.addWidget(self.lbl_comp)
        box_comp.addWidget(QLabel("MPa", objectName="UnitLabel"), alignment=Qt.AlignRight)
        v_res.addLayout(box_comp)

        box_flex = QVBoxLayout()
        box_flex.addWidget(QLabel("🟥 抗折强度 (Flexural)"))
        self.lbl_flex = QLabel("---")
        self.lbl_flex.setObjectName("ValFlex")
        self.lbl_flex.setAlignment(Qt.AlignRight)
        box_flex.addWidget(self.lbl_flex)
        box_flex.addWidget(QLabel("MPa", objectName="UnitLabel"), alignment=Qt.AlignRight)
        v_res.addLayout(box_flex)

        v_res.addStretch()

        btn_pred = QPushButton("🚀 开始预测 (Predict)")
        btn_pred.setObjectName("BtnPredict")
        btn_pred.clicked.connect(self.predict)

        btn_reset = QPushButton("↺ 重置参数 (Reset)")
        btn_reset.setObjectName("BtnReset")
        btn_reset.clicked.connect(self.reset)

        v_res.addWidget(btn_pred)
        v_res.addWidget(btn_reset)

        res_group.setLayout(v_res)
        body_layout.addWidget(res_group, stretch=2)

        layout.addLayout(body_layout)

        # Shadows
        for widget in [header, input_group, res_group]:
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(20)
            shadow.setColor(QColor(0, 0, 0, 80))
            shadow.setOffset(0, 5)
            widget.setGraphicsEffect(shadow)

    def predict(self):
        try:
            data = []
            for feat in self.feature_names:
                val = float(self.inputs[feat].text())
                data.append(val)

            df = pd.DataFrame([data], columns=self.feature_names)
            pred_c = self.model_comp.predict(df)[0]
            pred_f = self.model_flex.predict(df)[0]

            self.lbl_comp.setText(f"{pred_c:.2f}")
            self.lbl_flex.setText(f"{pred_f:.2f}")
        except ValueError:
            QMessageBox.warning(self, "Input Error", "请确保所有字段均填入有效数字！")

    def reset(self):
        for key, inp in self.inputs.items():
            inp.setText(str(self.default_values.get(key, 0)))
        self.lbl_comp.setText("---")
        self.lbl_flex.setText("---")


# ==========================================
# PART 4: 主控流程
# ==========================================

def main():
    print("--- 步骤 1/4: 生成数据与训练双模型 ---")
    X_c, y_c, X_f, y_f, feats = generate_full_data()

    # 拆分
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_c, y_c, test_size=0.2, random_state=42)
    Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_f, y_f, test_size=0.2, random_state=42)

    # 训练 (仅使用最佳模型 CatBoost 进行展示)
    metrics_comp, best_model_comp = train_evaluation_suite(Xc_train, yc_train, 'Compressive')
    metrics_flex, best_model_flex = train_evaluation_suite(Xf_train, yf_train, 'Flexural')

    print("\n--- 步骤 2/4: 生成科研图表 (请依次关闭窗口以继续) ---")

    plot_distributions(y_c, y_f)
    plot_correlation(X_c, y_c, 'Compressive')
    plot_radar(metrics_comp, Xc_test, yc_test, 'Compressive')

    yc_pred = best_model_comp.predict(Xc_test)
    plot_errors(yc_test, yc_pred, 'Compressive')

    print("   生成 SHAP 解释图...")
    plot_shap_detailed(best_model_comp, Xc_train, 'Compressive')

    print("   生成纤维影响曲线...")
    plot_fiber_curves(best_model_comp, Xc_train)

    print("   生成等高线图...")
    plot_contour(best_model_comp, Xc_train)

    print("\n--- 步骤 3/4: 启动最终 GUI 软件 ---")
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    window = ModernGUI(best_model_comp, best_model_flex, feats)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
