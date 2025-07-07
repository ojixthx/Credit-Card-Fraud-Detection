#!/usr/bin/env python3
# credit_card_fraud_pipeline.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, roc_curve, auc, confusion_matrix
)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification

sns.set_palette("husl")


def generate_sample_data(n_samples=50000, n_features=30, fraud_ratio=0.002):
    """Generate sample fraud detection data."""
    print("Generating sample credit card fraud data...")
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=20,
        n_redundant=5,
        n_clusters_per_class=1,
        weights=[1 - fraud_ratio, fraud_ratio],
        flip_y=0.01,
        random_state=42
    )
    feature_names = [f'V{i}' for i in range(1, 29)]
    np.random.seed(42)
    time_feature = np.random.randint(0, 86400, n_samples)
    amount_feature = np.random.lognormal(mean=3, sigma=1.5, size=n_samples)

    df = pd.DataFrame(X[:, :28], columns=feature_names)
    df['Time'] = time_feature
    df['Amount'] = amount_feature
    df['Class'] = y

    fraud_mask = df['Class'] == 1
    df.loc[fraud_mask, 'V1'] += np.random.normal(2, 0.5, fraud_mask.sum())
    df.loc[fraud_mask, 'V4'] += np.random.normal(-2, 0.5, fraud_mask.sum())
    df.loc[fraud_mask, 'V11'] += np.random.normal(2, 0.5, fraud_mask.sum())
    df.loc[fraud_mask, 'V14'] += np.random.normal(-2, 0.5, fraud_mask.sum())

    print(f"Generated dataset with {len(df)} samples")
    print(f"Fraud ratio: {df['Class'].mean():.4f}")
    return df


def explore_data(df):
    """Comprehensive data exploration."""
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\nData Types:")
    print(df.dtypes.value_counts())
    print("\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found!")
    else:
        print(missing[missing > 0])
    print("\nClass Distribution:")
    class_counts = df['Class'].value_counts()
    print(class_counts)
    print(f"Fraud percentage: {class_counts[1] / len(df) * 100:.4f}%")
    print("\nBasic Statistics:")
    print(df.describe())
    return df


def visualize_class_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.countplot(data=df, x='Class', ax=axes[0])
    axes[0].set_title('Class Distribution')
    axes[0].set_xlabel('Class (0: Non-Fraud, 1: Fraud)')
    class_counts = df['Class'].value_counts()
    axes[1].pie(class_counts.values, labels=['Non-Fraud', 'Fraud'],
                autopct='%1.4f%%', startangle=90)
    axes[1].set_title('Class Distribution (Percentage)')
    plt.tight_layout()
    plt.show()


def analyze_features(df):
    print("=" * 60)
    print("FEATURE ANALYSIS")
    print("=" * 60)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x='Time', hue='Class', bins=50, alpha=0.7)
    plt.title('Transaction Time Distribution by Class')
    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x='Amount', bins=50, alpha=0.7)
    plt.title('Amount Distribution')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='Class', y='Amount')
    plt.title('Amount Distribution by Class')
    plt.subplot(1, 2, 2)
    fraud_amounts = df[df['Class'] == 1]['Amount']
    normal_amounts = df[df['Class'] == 0]['Amount']
    plt.hist(normal_amounts, bins=50, alpha=0.7, label='Normal', density=True)
    plt.hist(fraud_amounts, bins=50, alpha=0.7, label='Fraud', density=True)
    plt.xlabel('Amount')
    plt.ylabel('Density')
    plt.title('Amount Distribution by Class (Density)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_importance_by_class(df):
    v_features = [col for col in df.columns if col.startswith('V')]
    fraud = df[df['Class'] == 1]
    normal = df[df['Class'] == 0]
    feature_stats = []
    for feature in v_features:
        diff = abs(fraud[feature].mean() - normal[feature].mean())
        feature_stats.append((feature, diff))
    feature_stats.sort(key=lambda x: x[1], reverse=True)
    top = [f for f, _ in feature_stats[:8]]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for i, feature in enumerate(top):
        sns.histplot(normal[feature], label='Normal', ax=axes[i], alpha=0.7, stat='density')
        sns.histplot(fraud[feature], label='Fraud', ax=axes[i], alpha=0.7, stat='density')
        axes[i].set_title(f'{feature}')
        axes[i].legend()
    plt.tight_layout()
    plt.show()
    return top


def engineer_features(df):
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    df_eng = df.copy()
    df_eng['Hours'] = (df_eng['Time'] % 86400) // 3600
    df_eng['Hour_sin'] = np.sin(2 * np.pi * df_eng['Hours'] / 24)
    df_eng['Hour_cos'] = np.cos(2 * np.pi * df_eng['Hours'] / 24)
    df_eng['LogAmount'] = np.log1p(df_eng['Amount'])
    scaler = RobustScaler()
    df_eng['Amount_scaled'] = scaler.fit_transform(df_eng[['Amount']]).flatten()
    for feat, q in [('V1', 0.95), ('V4', 0.05), ('V11', 0.95), ('V14', 0.05)]:
        outlier_col = feat + '_outlier'
        if '>' in str(q):
            mask = df_eng[feat] > df_eng[feat].quantile(q)
        else:
            mask = df_eng[feat] < df_eng[feat].quantile(q)
        df_eng[outlier_col] = mask.astype(int)
    df_eng['V1_V4_interaction'] = df_eng['V1'] * df_eng['V4']
    df_eng['V11_V14_interaction'] = df_eng['V11'] * df_eng['V14']
    v_feats = [c for c in df_eng.columns if c.startswith('V')]
    df_eng['V_mean'] = df_eng[v_feats].mean(axis=1)
    df_eng['V_std'] = df_eng[v_feats].std(axis=1)
    df_eng['V_skew'] = df_eng[v_feats].skew(axis=1)
    df_eng['V_sum'] = df_eng[v_feats].sum(axis=1)
    df_eng.drop(['Time', 'Amount', 'Hours'], axis=1, inplace=True)
    return df_eng


def prepare_data(df, test_size=0.2, random_state=42):
    print("=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def apply_sampling_techniques(X_train, y_train):
    print("=" * 60)
    print("SAMPLING TECHNIQUES")
    print("=" * 60)
    techniques = {
        'Original': None,
        'SMOTE': SMOTE(random_state=42),
        'ADASYN': ADASYN(random_state=42),
        'UnderSampler': RandomUnderSampler(random_state=42)
    }
    res = {}
    for name, sampler in techniques.items():
        if sampler is None:
            res[name] = (X_train, y_train)
        else:
            Xr, yr = sampler.fit_resample(X_train, y_train)
            res[name] = (Xr, yr)
        print(f"{name}: {res[name][0].shape}, fraud ratio: {yr.sum()/len(yr):.4f}" if sampler else f"{name} unchanged")
    return res


def get_models():
    return {
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred, zero_division=0),
        'ROC_AUC': roc_auc_score(y_test, y_proba),
        'PR_AUC': auc(*precision_recall_curve(y_test, y_proba)[:2])
    }, y_pred, y_proba


def train_and_evaluate_models(resampled_data, X_test, y_test):
    print("=" * 60)
    print("TRAIN & EVALUATE")
    print("=" * 60)
    scaler = RobustScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    results = []
    preds = {}
    for samp, (Xr, yr) in resampled_data.items():
        Xr_scaled = scaler.fit_transform(Xr)
        for name, model in get_models().items():
            use_scaled = (name == 'LogisticRegression')
            X_train_final = Xr_scaled if use_scaled else Xr
            X_test_final = X_test_scaled if use_scaled else X_test
            model.fit(X_train_final, yr)
            res, y_pred, y_proba = evaluate_model(model, X_test_final, y_test)
            key = f"{name}_{samp}"
            results.append({'Model': key, **res})
            preds[key] = {'pred': y_pred, 'proba': y_proba}
            print(f"{key}: F1={res['F1']:.4f}, ROC_AUC={res['ROC_AUC']:.4f}")
    return pd.DataFrame(results), preds


def run_complete_pipeline():
    df = generate_sample_data()
    df = explore_data(df)
    visualize_class_distribution(df)
    analyze_features(df)
    plot_feature_importance_by_class(df)

    df_eng = engineer_features(df)
    X_train, X_test, y_train, y_test = prepare_data(df_eng)
    resampled = apply_sampling_techniques(X_train, y_train)
    results_df, preds = train_and_evaluate_models(resampled, X_test, y_test)

    print("\nSummary of results:\n", results_df)


if __name__ == "__main__":
    run_complete_pipeline()
