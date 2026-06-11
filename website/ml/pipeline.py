"""
ML Pipeline — Trains models, generates charts, and prepares the prediction pipeline.
Mirrors the logic from main.ipynb exactly, with added chart generation.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# ── Chart Theme ──
CHART_COLORS = {
    'bg': '#1e293b',
    'fg': '#f1f5f9',
    'accent': '#c9a84c',
    'grid': '#334155',
    'secondary': '#94a3b8',
    'palette': ['#c9a84c', '#3b82f6', '#22c55e', '#ef4444', '#a855f7', '#f97316']
}


def _setup_chart_style():
    """Configure matplotlib for dark-themed charts matching the website palette."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.facecolor': CHART_COLORS['bg'],
        'axes.facecolor': CHART_COLORS['bg'],
        'axes.edgecolor': CHART_COLORS['grid'],
        'axes.labelcolor': CHART_COLORS['fg'],
        'text.color': CHART_COLORS['fg'],
        'xtick.color': CHART_COLORS['secondary'],
        'ytick.color': CHART_COLORS['secondary'],
        'grid.color': CHART_COLORS['grid'],
        'grid.alpha': 0.3,
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
    })


def _save_chart(fig, charts_dir, name):
    """Save a chart figure as a high-quality PNG."""
    import matplotlib.pyplot as plt
    path = os.path.join(charts_dir, f'{name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def build_pipeline(data_dir, charts_dir, generate_charts=True):
    """
    Full ML pipeline: load data, clean, engineer features, train, evaluate, and generate charts.

    Returns a dict with:
        - pipeline: fitted preprocessing pipeline
        - models: dict of fitted models
        - results: dict of model evaluation metrics
        - feature_names: list of feature column names
        - num_cols / cat_cols: column lists
        - dataset_stats: summary statistics
    """
    if generate_charts:
        _setup_chart_style()

    # ════════════════════════════════════════════
    # 1. LOAD DATA
    # ════════════════════════════════════════════
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))

    dataset_stats = {
        'rows': len(train),
        'columns': train.shape[1],
        'missing_cols': int(train.isnull().any().sum()),
        'numeric_features': len(train.select_dtypes(include=['number']).columns),
        'categorical_features': len(train.select_dtypes(include=['object', 'str']).columns),
    }

    # Drop Id
    train.drop(columns=['Id'], inplace=True, errors='ignore')

    # ════════════════════════════════════════════
    # 2. GENERATE EDA CHARTS (before cleaning)
    # ════════════════════════════════════════════
    if generate_charts:
        _generate_eda_charts(train, charts_dir)

    # ════════════════════════════════════════════
    # 3. DATA CLEANING
    # ════════════════════════════════════════════
    # Separate target
    y = train['SalePrice']
    X = train.drop('SalePrice', axis=1)

    # Columns with missing values that mean "None" (no feature present)
    cols_none = [
        'Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
        'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType',
        'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature'
    ]
    for col in cols_none:
        if col in X.columns:
            X[col] = X[col].fillna('None')

    # Numeric fills
    if 'LotFrontage' in X.columns:
        X['LotFrontage'] = X.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
    if 'MasVnrArea' in X.columns:
        X['MasVnrArea'] = X['MasVnrArea'].fillna(0)
    if 'GarageYrBlt' in X.columns:
        X['GarageYrBlt'] = X['GarageYrBlt'].fillna(0)
    if 'Electrical' in X.columns:
        X['Electrical'] = X['Electrical'].fillna(X['Electrical'].mode()[0])

    # ════════════════════════════════════════════
    # 4. FEATURE ENGINEERING
    # ════════════════════════════════════════════
    X['HouseAge'] = X['YrSold'] - X['YearBuilt']
    X['RemodAge'] = X['YrSold'] - X['YearRemodAdd']
    X['TotalSF'] = X['GrLivArea'] + X['TotalBsmtSF']
    X['TotalBath'] = (
        X['FullBath'] + 0.5 * X['HalfBath'] +
        X['BsmtFullBath'] + 0.5 * X['BsmtHalfBath']
    )
    X['TotalPorchSF'] = (
        X['WoodDeckSF'] + X['OpenPorchSF'] +
        X['EnclosedPorch'] + X['3SsnPorch'] + X['ScreenPorch']
    )
    X['OverallScore'] = X['OverallQual'] * X['OverallCond']
    X['HasPool'] = (X['PoolArea'] > 0).astype(int)
    X['HasGarage'] = (X['GarageArea'] > 0).astype(int)
    X['HasFireplace'] = (X['Fireplaces'] > 0).astype(int)
    X['HasBasement'] = (X['TotalBsmtSF'] > 0).astype(int)

    # Log-transform target
    y_log = np.log1p(y)

    # ════════════════════════════════════════════
    # 5. PREPROCESSING PIPELINE
    # ════════════════════════════════════════════
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'str', 'category']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
    ])

    # ════════════════════════════════════════════
    # 6. TRAIN / TEST SPLIT & FIT
    # ════════════════════════════════════════════
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # Handle any remaining NaN after transform
    X_train_scaled = pd.DataFrame(X_train_scaled).fillna(
        pd.DataFrame(X_train_scaled).median()
    ).values
    X_test_scaled = pd.DataFrame(X_test_scaled).fillna(
        pd.DataFrame(X_test_scaled).median()
    ).values

    # ════════════════════════════════════════════
    # 7. TRAIN MODELS
    # ════════════════════════════════════════════
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, max_iter=5000),
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Reverse log transform for MAE / RMSE in dollar terms
        y_test_actual = np.expm1(y_test)
        y_pred_actual = np.expm1(y_pred)

        results[name] = {
            'r2': round(r2_score(y_test, y_pred), 4),
            'mae': round(mean_absolute_error(y_test_actual, y_pred_actual), 2),
            'rmse': round(np.sqrt(mean_squared_error(y_test_actual, y_pred_actual)), 2),
        }

    # ════════════════════════════════════════════
    # 8. GENERATE MODEL COMPARISON CHARTS
    # ════════════════════════════════════════════
    if generate_charts:
        _generate_model_charts(results, charts_dir)

        # Generate feature importance from Random Forest
        _generate_feature_importance_chart(
            models['Random Forest'], X_train_scaled, num_cols, cat_cols,
            preprocessor, charts_dir
        )

        # Generate results/insights charts
        _generate_results_charts(train, charts_dir)

    # Update dataset stats with best model info
    best_model_name = max(results, key=lambda k: results[k]['r2'])
    dataset_stats['models_tested'] = len(models)
    dataset_stats['best_model'] = best_model_name
    dataset_stats['best_r2'] = results[best_model_name]['r2']

    return {
        'pipeline': preprocessor,
        'models': models,
        'results': results,
        'feature_names': X.columns.tolist(),
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'dataset_stats': dataset_stats,
    }


# ════════════════════════════════════════════════════
# CHART GENERATION FUNCTIONS
# ════════════════════════════════════════════════════

def _generate_eda_charts(train, charts_dir):
    """Generate all EDA charts from the raw training data."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 1. SalePrice Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(train['SalePrice'], bins=50, color=CHART_COLORS['accent'],
                 edgecolor='none', alpha=0.85, ax=ax)
    ax.set_title('Sale Price Distribution', fontweight='bold')
    ax.set_xlabel('Sale Price ($)')
    ax.set_ylabel('Frequency')
    ax.grid(True, axis='y', alpha=0.3)
    _save_chart(fig, charts_dir, 'price_distribution')

    # 2. Correlation Heatmap (top 15 numeric features)
    numeric_df = train.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    top_features = corr['SalePrice'].abs().sort_values(ascending=False).head(15).index
    top_corr = corr.loc[top_features, top_features]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(top_corr, annot=True, fmt='.2f', cmap='YlOrBr',
                ax=ax, linewidths=0.5, square=True,
                cbar_kws={'shrink': 0.8},
                annot_kws={'size': 8, 'color': CHART_COLORS['fg']})
    ax.set_title('Correlation Heatmap — Top 15 Features', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    _save_chart(fig, charts_dir, 'correlation_heatmap')

    # 3. Missing Values
    missing = train.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(missing.index[:15], missing.values[:15], color=CHART_COLORS['accent'],
                       edgecolor='none', alpha=0.85)
        ax.set_title('Missing Values by Feature', fontweight='bold')
        ax.set_xlabel('Number of Missing Values')
        ax.invert_yaxis()
        ax.grid(True, axis='x', alpha=0.3)
        for bar, val in zip(bars, missing.values[:15]):
            ax.text(val + 5, bar.get_y() + bar.get_height()/2,
                    f'{val}', va='center', fontsize=9, color=CHART_COLORS['fg'])
        _save_chart(fig, charts_dir, 'missing_values')

    # 4. Price vs Living Area
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(train['GrLivArea'], train['SalePrice'],
               c=CHART_COLORS['accent'], alpha=0.4, s=15, edgecolors='none')
    ax.set_title('Sale Price vs. Living Area', fontweight='bold')
    ax.set_xlabel('Above Ground Living Area (sq ft)')
    ax.set_ylabel('Sale Price ($)')
    ax.grid(True, alpha=0.3)
    _save_chart(fig, charts_dir, 'price_vs_area')

    # 5. Price by Overall Quality
    fig, ax = plt.subplots(figsize=(10, 6))
    quality_median = train.groupby('OverallQual')['SalePrice'].median()
    ax.bar(quality_median.index, quality_median.values,
           color=CHART_COLORS['accent'], edgecolor='none', alpha=0.85)
    ax.set_title('Median Sale Price by Overall Quality', fontweight='bold')
    ax.set_xlabel('Overall Quality (1-10)')
    ax.set_ylabel('Median Sale Price ($)')
    ax.grid(True, axis='y', alpha=0.3)
    _save_chart(fig, charts_dir, 'price_by_quality')


def _generate_model_charts(results, charts_dir):
    """Generate model comparison charts."""
    import matplotlib.pyplot as plt
    names = list(results.keys())
    r2_scores = [results[n]['r2'] for n in names]
    mae_scores = [results[n]['mae'] for n in names]
    rmse_scores = [results[n]['rmse'] for n in names]

    # R² Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, r2_scores, color=CHART_COLORS['palette'][:len(names)],
                  edgecolor='none', alpha=0.85)
    ax.set_title('R² Score Comparison', fontweight='bold')
    ax.set_ylabel('R² Score')
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, r2_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10,
                fontweight='bold', color=CHART_COLORS['fg'])
    plt.xticks(rotation=15, ha='right')
    _save_chart(fig, charts_dir, 'model_r2_comparison')

    # MAE Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, mae_scores, color=CHART_COLORS['palette'][:len(names)],
                  edgecolor='none', alpha=0.85)
    ax.set_title('Mean Absolute Error Comparison', fontweight='bold')
    ax.set_ylabel('MAE ($)')
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, mae_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'${val:,.0f}', ha='center', va='bottom', fontsize=10,
                fontweight='bold', color=CHART_COLORS['fg'])
    plt.xticks(rotation=15, ha='right')
    _save_chart(fig, charts_dir, 'model_mae_comparison')

    # RMSE Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, rmse_scores, color=CHART_COLORS['palette'][:len(names)],
                  edgecolor='none', alpha=0.85)
    ax.set_title('Root Mean Squared Error Comparison', fontweight='bold')
    ax.set_ylabel('RMSE ($)')
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, rmse_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'${val:,.0f}', ha='center', va='bottom', fontsize=10,
                fontweight='bold', color=CHART_COLORS['fg'])
    plt.xticks(rotation=15, ha='right')
    _save_chart(fig, charts_dir, 'model_rmse_comparison')


def _generate_feature_importance_chart(rf_model, X_train, num_cols, cat_cols, preprocessor, charts_dir):
    """Generate feature importance chart from Random Forest model."""
    try:
        import matplotlib.pyplot as plt
        importances = rf_model.feature_importances_
        feature_names = []

        # Try getting feature names directly from preprocessor
        if hasattr(preprocessor, 'get_feature_names_out'):
            try:
                feature_names = list(preprocessor.get_feature_names_out())
            except Exception:
                pass

        # If that failed or is empty, try manual building
        if not feature_names:
            feature_names = list(num_cols)
            try:
                if hasattr(preprocessor.named_transformers_['cat'], 'named_steps'):
                    encoder = preprocessor.named_transformers_['cat'].named_steps.get('encoder')
                    if encoder and hasattr(encoder, 'get_feature_names_out'):
                        # Call without arguments to avoid length mismatches
                        cat_feature_names = encoder.get_feature_names_out().tolist()
                        feature_names.extend(cat_feature_names)
            except Exception:
                pass

        # If counts still don't match, use generic names
        if len(feature_names) != len(importances):
            feature_names = [f'Feature {i}' for i in range(len(importances))]

        # Sort and take top 20
        indices = np.argsort(importances)[::-1][:20]
        top_names = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(top_names)), top_importances[::-1],
                color=CHART_COLORS['accent'], edgecolor='none', alpha=0.85)
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names[::-1], fontsize=9)
        ax.set_title('Top 20 Feature Importances (Random Forest)', fontweight='bold')
        ax.set_xlabel('Importance')
        ax.grid(True, axis='x', alpha=0.3)
        _save_chart(fig, charts_dir, 'feature_importance')
    except Exception:
        # Silently skip if feature importance generation fails
        pass


def _generate_results_charts(train, charts_dir):
    """Generate charts for the Results & Insights page."""
    import matplotlib.pyplot as plt

    # Price by Neighborhood (top 10)
    fig, ax = plt.subplots(figsize=(12, 6))
    neighborhood_median = train.groupby('Neighborhood')['SalePrice'].median().sort_values(ascending=False).head(10)
    ax.bar(neighborhood_median.index, neighborhood_median.values,
           color=CHART_COLORS['accent'], edgecolor='none', alpha=0.85)
    ax.set_title('Top 10 Neighborhoods by Median Sale Price', fontweight='bold')
    ax.set_xlabel('Neighborhood')
    ax.set_ylabel('Median Sale Price ($)')
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    _save_chart(fig, charts_dir, 'price_by_neighborhood')

    # Year Built vs Price
    fig, ax = plt.subplots(figsize=(10, 6))
    yearly_median = train.groupby('YearBuilt')['SalePrice'].median()
    ax.plot(yearly_median.index, yearly_median.values,
            color=CHART_COLORS['accent'], linewidth=1.5, alpha=0.85)
    ax.fill_between(yearly_median.index, yearly_median.values,
                    color=CHART_COLORS['accent'], alpha=0.1)
    ax.set_title('Median Sale Price by Year Built', fontweight='bold')
    ax.set_xlabel('Year Built')
    ax.set_ylabel('Median Sale Price ($)')
    ax.grid(True, alpha=0.3)
    _save_chart(fig, charts_dir, 'price_by_year')
