import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
from xgboost import XGBRegressor
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_features():
    return pd.read_parquet('data/reachability-all-features-normalized.parquet')

def load_label_data():
    return pd.read_parquet('data/line_labels_normalized.parquet')

def add_labels_to_features(features, labels):
    return pd.merge(features, labels, on=['game_id', 'play_id', 'nfl_id'], how='inner')

def main():
    print("Loading data...")
    features = load_features()
    labels = load_label_data()

    print("Merging...")
    df = add_labels_to_features(features, labels)

    print('Values of alignment: ', df['alignment'].value_counts())
    print('Number of rows with alignment = Unknown: ', df[df['alignment'].isna()].shape[0])
    print('Number of rows with coverage_responsibility = Unknown: ', df[df['coverage_responsibility'].isna()].shape[0])
    print('Values of coverage_responsibility: ', df['coverage_responsibility'].value_counts())
    df = df[df['alignment'].notna() & df['coverage_responsibility'].notna()]

    print("Final shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # ---------------------------
    #  Select defender features
    # ---------------------------

    numeric_features = [
        'x','y','s','a','dir','o',
        'player_average_speed','player_max_speed',
        'player_average_acceleration','player_max_acceleration',
        'dir_alignment',
    ]

    categorical_features = ['coverage_responsibility', 'alignment']

    # Handle categorical missing
    for col in categorical_features:
        df[col] = df[col].fillna('unknown')

    # One-hot encode
    df_cat = pd.get_dummies(df[categorical_features], prefix=categorical_features)
    df_num = df[numeric_features]

    def_features = pd.concat([df_num, df_cat], axis=1).astype(float)

    # Target-side features
    target_features = df[['num_frames_output', 'ball_land_x', 'ball_land_y', 'dist_to_ball_land']]

    # Combine everything into one XGBoost input
    X = pd.concat([def_features, target_features], axis=1).astype(float)

    y = df['distance_to_line'].astype(float)

    print("Feature matrix:", X.shape)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------------------
    # XGBoost model
    # ---------------------------
    print("Training XGBoost...")

    model = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='reg:squarederror',
        tree_method='hist'   # fast and stable
    )

    model.fit(X_train, y_train)

    # ---------------------------
    # Evaluate
    # ---------------------------
    print("Evaluating...")

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n================ RESULTS ================")
    print(f"MAE :  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")
    print("=========================================")

    # Generate predictions for all data points
    print("\nGenerating predictions for all data points...")
    all_preds = model.predict(X)
    
    # Save predictions for all data points
    out = pd.DataFrame({
        'game_id': df['game_id'].values,
        'play_id': df['play_id'].values,
        'nfl_id': df['nfl_id'].values,
        'actual': y.values,
        'predicted': all_preds
    })
    out.to_csv("xgb_predictions.csv", index=False)
    print("Predictions saved to xgb_predictions.csv")

    # ---------------------------
    # Feature importance plot
    # ---------------------------
    print("\nGenerating feature importance plot...")
    
    # Get feature importances
    feature_importance = (
        pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
    )

    # Top N
    top_n = min(20, len(feature_importance))
    top_features = feature_importance.head(top_n)

    # Define feature categories and colors
    def categorize_feature(feature_name):
        """Categorize feature by name"""
        if 'coverage_responsibility' in feature_name:
            return 'Coverage', '#4169E1'  # Blue
        elif 'alignment' in feature_name:
            return 'Alignment', '#DC143C'  # Crimson
        elif feature_name.startswith('player_'):
            return 'Player Stats', '#32CD32'  # Lime green
        elif feature_name in ['x', 'y', 's', 'a', 'dir', 'o', 'dir_alignment']:
            return 'Numeric', '#FF8C00'  # Dark orange
        elif 'ball_land' in feature_name or feature_name in ['dist_to_ball_land', 'num_frames_output']:
            return 'Target/Ball', '#9370DB'  # Medium purple
        else:
            return 'Other', '#808080'  # Gray

    # Assign categories and colors
    top_features = top_features.copy()
    category_color_pairs = top_features['feature'].apply(categorize_feature)
    top_features['category'] = [pair[0] for pair in category_color_pairs]
    top_features['color'] = [pair[1] for pair in category_color_pairs]

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('white')

    # Create bar plot with colors by category
    y_positions = range(top_n)
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.barh(i, row['importance'], height=0.7, color=row['color'], alpha=0.85, edgecolor='#333333', linewidth=0.5)

    # Set labels and styles
    ax.set_yticks(y_positions)
    ax.set_yticklabels(top_features["feature"].values, fontsize=11, color='#333333')
    ax.set_xlabel("Feature Importance", fontsize=12, fontweight='bold', color='#333333')
    ax.set_title(f"JUDGE Distance Model: Top {top_n} Features", fontsize=18, fontweight='bold', pad=10, color='#1a1a1a')

    # Most important at top
    ax.invert_yaxis()

    # Grid and styling
    ax.grid(axis="x", linestyle='-', linewidth=0.5, color='gray', alpha=0.2)
    ax.set_axisbelow(True)
    ax.set_facecolor('#f8f8f8')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color('#cccccc')
    ax.spines["bottom"].set_color('#cccccc')

    ax.tick_params(colors='#333333', labelsize=10)
    ax.tick_params(axis='y', left=False)
    
    # Get unique categories and their colors
    category_colors = top_features.groupby('category')['color'].first()
    legend_elements = [Patch(facecolor=color, alpha=0.85, edgecolor='#333333', linewidth=0.5, label=cat) 
                       for cat, color in category_colors.items()]
    
    legend = ax.legend(handles=legend_elements, loc='lower right', fontsize=11, 
                      framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')

    plt.tight_layout()
    plt.savefig("xgb_feature_importance.png", dpi=150, bbox_inches="tight", facecolor='white')
    plt.close()

    print("Feature importance plot saved to xgb_feature_importance.png")

    
    # Also print top features
    print(f"\nTop {top_n} Most Important Features:")
    print("=" * 50)
    for idx, row in top_features.iterrows():
        print(f"{row['feature']:40s} {row['importance']:.6f}")

    # ---------------------------
    # SHAP VALUES (Native XGBoost)
    # ---------------------------
    # print("\nComputing SHAP values with native XGBoost...")

    # # Convert training data to DMatrix
    # dtrain = xgb.DMatrix(
    #     X_train.values,
    #     feature_names=list(X_train.columns)
    # )

    # # Native SHAP from booster
    # booster = model.get_booster()
    # shap_values = booster.predict(dtrain, pred_contribs=True)

    # # Remove the last column (bias term)
    # shap_values = shap_values[:, :-1]

    # # Sample for plotting (optional)
    # sample_size = min(2000, len(X_train))
    # X_sample = X_train.iloc[:sample_size]
    # shap_sample = shap_values[:sample_size]

    # # SHAP summary plot
    # plt.figure(figsize=(10, 6))
    # shap.summary_plot(shap_sample, X_sample, show=False)
    # plt.tight_layout()
    # plt.savefig("shap-stuff/shap_summary.png", dpi=150, bbox_inches='tight')

    # print("SHAP summary plot saved to shap-stuff/shap_summary.png")

if __name__ == "__main__":
    main()
