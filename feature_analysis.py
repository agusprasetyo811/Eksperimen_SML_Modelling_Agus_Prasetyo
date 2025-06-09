# feature_analysis_dagshub_integrated.py
# Feature Analysis dengan Full DagsHub Integration

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# DagsHub Integration
from dotenv import load_dotenv
load_dotenv()

print("="*60)
print("FEATURE ANALYSIS - DAGSHUB INTEGRATED")
print("="*60)

output_dir = "MLProject/output/feature_analysis"

# ============================================================================
# 0. DAGSHUB & MLFLOW SETUP
# ============================================================================

def setup_dagshub_mlflow():
    """Setup DagsHub dan MLflow tracking sama seperti modelling script"""
    print("\n0. Setting up DagsHub & MLflow...")
    
    try:
        # DagsHub configuration dari environment variables
        dagshub_url = os.getenv('DAGSHUB_REPO_URL')
        dagshub_username = os.getenv('DAGSHUB_USERNAME') 
        dagshub_token = os.getenv('DAGSHUB_TOKEN')
        
        if not dagshub_url:
            # Fallback ke URL yang sudah diketahui bekerja
            dagshub_url = "https://dagshub.com/agusprasetyo811/kredit_pinjaman"
            print(f"   Using known working DagsHub URL: {dagshub_url}")
        
        # Set MLflow tracking URI ke DagsHub
        mlflow.set_tracking_uri(dagshub_url + ".mlflow")
        
        # Set credentials jika tersedia
        if dagshub_username and dagshub_token:
            os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_username
            os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
            print("✓ DagsHub credentials configured from environment")
        else:
            print("DagsHub credentials not found in .env file")
            print("   Will attempt connection without explicit credentials")
        
        # Set experiment (menggunakan yang sama dengan modeling)
        experiment_name = "credit-approval-prediction"
        
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"✓ Created new experiment: {experiment_name}")
        except mlflow.exceptions.MlflowException:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                experiment_id = experiment.experiment_id
                print(f"✓ Using existing experiment: {experiment_name}")
            else:
                print("Using default experiment")
                experiment_id = "0"
        
        mlflow.set_experiment(experiment_name)
        
        print(f"✓ MLflow tracking configured")
        print(f"   Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"   DagsHub URL: {dagshub_url}")
        print(f"   Experiment ID: {experiment_id}")
        
        return dagshub_url, experiment_id
        
    except Exception as e:
        print(f"DagsHub setup failed: {e}")
        print("   Using local tracking as fallback...")
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("feature-analysis-local")
        return None, "0"

# ============================================================================
# 1. DATA & MODEL LOADING
# ============================================================================

def load_best_model_and_data():
    """Load model terbaik dari DagsHub dan data"""
    print("\n1. Loading Best Model and Data from DagsHub...")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data first
    try:
        X_train = pd.read_csv('final_dataset/X_train.csv')
        X_test = pd.read_csv('final_dataset/X_test.csv')
        y_train = pd.read_csv('final_dataset/y_train.csv')['disetujui_encoded']
        y_test = pd.read_csv('final_dataset/y_test.csv')['disetujui_encoded']
        print(f"✓ Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        print(f"   Features: {X_train.shape[1]}, Target distribution: {np.bincount(y_train)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None
    
    # Try to load best model dari DagsHub/MLflow
    model = None
    best_run_info = None
    
    try:
        print("   Searching for best model in DagsHub experiment...")
        experiment = mlflow.get_experiment_by_name("credit-approval-prediction")
        
        if experiment:
            # Search untuk Random Forest model dengan F1-score tertinggi
            runs_df = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="params.model_name = 'Random Forest'",
                order_by=["metrics.f1_score DESC"],
                max_results=1
            )
            
            if not runs_df.empty:
                best_run = runs_df.iloc[0]
                best_run_id = best_run['run_id']
                f1_score = best_run.get('metrics.f1_score', 'N/A')
                accuracy = best_run.get('metrics.accuracy', 'N/A')
                
                print(f"✓ Found best Random Forest run in DagsHub:")
                print(f"   Run ID: {best_run_id[:8]}...")
                print(f"   F1-Score: {f1_score}")
                print(f"   Accuracy: {accuracy}")
                print(f"   Start Time: {best_run['start_time']}")
                
                # Try berbagai nama artifact model yang mungkin
                model_artifact_names = [
                    "model_random_forest",
                    "random_forest_model", 
                    "model",
                    "best_model",
                    "production_model"
                ]
                
                for artifact_name in model_artifact_names:
                    try:
                        model_uri = f"runs:/{best_run_id}/{artifact_name}"
                        model = mlflow.sklearn.load_model(model_uri)
                        print(f"✓ Model loaded from DagsHub artifact: {artifact_name}")
                        best_run_info = {
                            'run_id': best_run_id,
                            'f1_score': f1_score,
                            'accuracy': accuracy,
                            'artifact_name': artifact_name,
                            'source': 'dagshub'
                        }
                        break
                    except Exception as artifact_error:
                        continue
                        
                if model is None:
                    print("Could not load model from any DagsHub artifact")
            else:
                print("No Random Forest runs found in DagsHub experiment")
        else:
            print("Experiment 'credit-approval-prediction' not found in DagsHub")
            
    except Exception as e:
        print(f"DagsHub model loading failed: {e}")
    
    # Fallback: train new model jika tidak bisa load dari DagsHub
    if model is None:
        print("  Training new Random Forest model as fallback...")
        model = RandomForestClassifier(
            random_state=42, 
            n_estimators=100,
            max_depth=10
        )
        model.fit(X_train, y_train)
        
        # Quick evaluation
        from sklearn.metrics import f1_score, accuracy_score
        y_pred = model.predict(X_test)
        fallback_f1 = f1_score(y_test, y_pred)
        fallback_acc = accuracy_score(y_test, y_pred)
        
        print(f"✓ New model trained:")
        print(f"   F1-Score: {fallback_f1:.3f}")
        print(f"   Accuracy: {fallback_acc:.3f}")
        
        best_run_info = {
            'run_id': 'newly_trained',
            'f1_score': fallback_f1,
            'accuracy': fallback_acc,
            'artifact_name': 'fallback_model',
            'source': 'local_training'
        }
    
    return model, X_train, X_test, y_train, y_test, best_run_info

# ============================================================================
# 2. FEATURE IMPORTANCE ANALYSIS (ENHANCED)
# ============================================================================

def analyze_feature_importance(model, X_train, y_train):
    """Analisis feature importance dengan berbagai metode"""
    print("\n2. Comprehensive Feature Importance Analysis...")
    
    feature_names = X_train.columns
    n_features = len(feature_names)
    
    print(f"  Analyzing {n_features} features")
    print(f"  Model type: {type(model).__name__}")
    
    # 1. Built-in Feature Importance (Random Forest)
    print("   ⚡ Calculating Random Forest importance...")
    importance_rf = model.feature_importances_
    
    # 2. Permutation Importance (more robust)
    print("   Calculating permutation importance...")
    try:
        perm_importance = permutation_importance(
            model, X_train, y_train, 
            random_state=42, 
            n_repeats=10,  # More repeats for stability
            scoring='f1'   # Use F1 score for permutation
        )
        perm_mean = perm_importance.importances_mean
        perm_std = perm_importance.importances_std
        print(f"      ✓ Permutation importance completed with 10 repeats")
    except Exception as e:
        print(f"      Permutation importance failed: {e}")
        perm_mean = importance_rf.copy()  # Fallback
        perm_std = np.zeros_like(importance_rf)
    
    # 3. Correlation Analysis
    print("  Calculating feature correlations with target...")
    correlations = []
    for feature in feature_names:
        try:
            corr = np.corrcoef(X_train[feature], y_train)[0, 1]
            correlations.append(abs(corr) if not np.isnan(corr) else 0)
        except:
            correlations.append(0)
    
    # Create comprehensive DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'rf_importance': importance_rf,
        'perm_importance': perm_mean,
        'perm_std': perm_std,
        'correlation': correlations
    })
    
    # Sort by RF importance
    importance_df = importance_df.sort_values('rf_importance', ascending=False)
    importance_df['rf_rank'] = range(1, n_features + 1)
    
    # Calculate additional insights
    top_5_cumulative = importance_df.head(5)['rf_importance'].sum()
    top_10_cumulative = importance_df.head(10)['rf_importance'].sum()
    
    print("\nFeature Importance Rankings:")
    print("-" * 70)
    print(f"{'Rank':<5} {'Feature':<20} {'RF Imp':<10} {'Perm Imp':<12} {'Correlation':<12}")
    print("-" * 70)
    
    for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        print(f"{i:<5} {row['feature']:<20} {row['rf_importance']:<10.3f} {row['perm_importance']:<12.3f} {row['correlation']:<12.3f}")
    
    print(f"\nSummary Statistics:")
    print(f"   Top 5 features account for: {top_5_cumulative*100:.1f}% of total importance")
    print(f"   Top 10 features account for: {top_10_cumulative*100:.1f}% of total importance")
    print(f"   Most important feature: {importance_df.iloc[0]['feature']} ({importance_df.iloc[0]['rf_importance']:.3f})")
    
    # Save results
    importance_df.to_csv(f"{output_dir}/feature_importance_analysis.csv", index=False)
    print("✓ Saved: output/feature_analysis/feature_importance_analysis.csv")
    
    return importance_df

# ============================================================================
# 3. ADVANCED VISUALIZATIONS
# ============================================================================

def create_visualizations(importance_df, X_train, y_train):
    """Create comprehensive visualizations"""
    print("\n3. Creating Advanced Visualizations...")
    
    # Set professional style
    plt.style.use('default')
    sns.set_palette("viridis")
    
    # 1. Comprehensive Feature Dashboard
    print("   Creating feature importance dashboard...")
    fig = plt.figure(figsize=(16, 12))
    
    # Top 10 features
    top_10 = importance_df.head(10)
    
    # 1.1 RF Importance
    ax1 = plt.subplot(2, 3, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_10)))
    bars = ax1.barh(range(len(top_10)), top_10['rf_importance'], color=colors)
    ax1.set_yticks(range(len(top_10)))
    ax1.set_yticklabels(top_10['feature'])
    ax1.set_xlabel('Random Forest Importance')
    ax1.set_title('Top 10 Features - RF Importance')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=8)
    
    # 1.2 Permutation Importance
    ax2 = plt.subplot(2, 3, 2)
    ax2.barh(range(len(top_10)), top_10['perm_importance'], 
             color='lightcoral', alpha=0.8)
    ax2.set_yticks(range(len(top_10)))
    ax2.set_yticklabels(top_10['feature'])
    ax2.set_xlabel('Permutation Importance')
    ax2.set_title('Top 10 Features - Permutation Importance')
    ax2.grid(axis='x', alpha=0.3)
    
    # 1.3 Correlation Analysis
    ax3 = plt.subplot(2, 3, 3)
    top_10_corr = importance_df.nlargest(10, 'correlation')
    ax3.barh(range(len(top_10_corr)), top_10_corr['correlation'], 
             color='green', alpha=0.7)
    ax3.set_yticks(range(len(top_10_corr)))
    ax3.set_yticklabels(top_10_corr['feature'])
    ax3.set_xlabel('Absolute Correlation')
    ax3.set_title('Top 10 Features - Target Correlation')
    ax3.grid(axis='x', alpha=0.3)
    
    # 1.4 Importance vs Correlation Scatter
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(importance_df['rf_importance'], importance_df['correlation'], 
                         alpha=0.7, s=100, c=importance_df['rf_importance'], 
                         cmap='viridis')
    ax4.set_xlabel('RF Importance')
    ax4.set_ylabel('Correlation with Target')
    ax4.set_title('Importance vs Correlation')
    plt.colorbar(scatter, ax=ax4, label='RF Importance')
    
    # Add labels untuk top 5
    top_5 = importance_df.head(5)
    for _, row in top_5.iterrows():
        ax4.annotate(row['feature'], 
                    (row['rf_importance'], row['correlation']),
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.8)
    
    # 1.5 Cumulative Importance
    ax5 = plt.subplot(2, 3, 5)
    cumulative_importance = importance_df['rf_importance'].cumsum()
    ax5.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 
             'b-', marker='o', markersize=4)
    ax5.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80% threshold')
    ax5.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
    ax5.set_xlabel('Number of Features')
    ax5.set_ylabel('Cumulative Importance')
    ax5.set_title('Cumulative Feature Importance')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 1.6 Feature Distribution
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(importance_df['rf_importance'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax6.axvline(importance_df['rf_importance'].mean(), color='red', linestyle='--', 
                label=f'Mean: {importance_df["rf_importance"].mean():.3f}')
    ax6.set_xlabel('RF Importance')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Feature Importance Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: feature_analysis_dashboard.png")
    
    # 2. Feature Ranking Comparison
    print("   Creating ranking comparison...")
    plt.figure(figsize=(12, 8))
    
    # Prepare ranking data
    perm_ranks = importance_df.sort_values('perm_importance', ascending=False).reset_index(drop=True)
    perm_ranks['perm_rank'] = range(1, len(perm_ranks) + 1)
    
    # Merge rankings
    ranking_comparison = importance_df[['feature', 'rf_rank']].merge(
        perm_ranks[['feature', 'perm_rank']], on='feature'
    )
    
    # Top 15 features for comparison
    top_15 = ranking_comparison.head(15)
    x = np.arange(len(top_15))
    width = 0.35
    
    plt.bar(x - width/2, top_15['rf_rank'], width, label='RF Ranking', alpha=0.8, color='blue')
    plt.bar(x + width/2, top_15['perm_rank'], width, label='Permutation Ranking', alpha=0.8, color='red')
    
    plt.xlabel('Features')
    plt.ylabel('Ranking (1=Most Important)')
    plt.title('Feature Ranking Comparison - Top 15 Features')
    plt.xticks(x, [f[:8] + '...' if len(f) > 8 else f for f in top_15['feature']], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/feature_ranking_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: feature_ranking_comparison.png")
    
    # Create correlation DataFrame for return
    corr_df = pd.DataFrame({
        'feature': importance_df['feature'],
        'correlation': importance_df['correlation']
    }).sort_values('correlation', ascending=False)
    
    return corr_df

# ============================================================================
# 4. ENHANCED SHAP ANALYSIS
# ============================================================================

def enhanced_shap_analysis(model, X_train, X_test):
    """Enhanced SHAP analysis dengan multiple plots"""
    print("\n4. Enhanced SHAP Analysis...")
    
    try:
        import shap
        
        print("   Initializing SHAP explainer...")
        explainer = shap.TreeExplainer(model)
        
        # Sample data untuk balance speed dan accuracy
        sample_size = min(100, len(X_test))
        X_sample = X_test.sample(sample_size, random_state=42)
        
        print(f"   Calculating SHAP values for {sample_size} samples...")
        shap_values = explainer.shap_values(X_sample)
        
        # Handle binary classification
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]  # Use positive class
            expected_value = explainer.expected_value[1]
        else:
            expected_value = explainer.expected_value
        
        # 1. Summary Plot (Beeswarm)
        print("   Creating SHAP summary plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, 
                         feature_names=X_train.columns, show=False)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Bar Plot
        print("   Creating SHAP importance bar plot...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, 
                         feature_names=X_train.columns, 
                         plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_importance_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Waterfall Plot
        print("   Creating SHAP waterfall plot...")
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0], 
                base_values=expected_value,
                data=X_sample.iloc[0].values,
                feature_names=list(X_train.columns)
            ), show=False
        )
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_waterfall_individual.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✓ SHAP analysis completed successfully")
        
        # Return SHAP importance untuk logging
        shap_importance = np.abs(shap_values).mean(0)
        return shap_importance
        
    except ImportError:
        print("   SHAP not installed. Install with: pip install shap")
        return None
    except Exception as e:
        print(f"   SHAP analysis failed: {e}")
        return None

# ============================================================================
# 5. COMPREHENSIVE REPORT GENERATION
# ============================================================================

def generate_insights_report(importance_df, corr_df, best_run_info):
    """Generate comprehensive business insights report"""
    print("\n5. Generating Comprehensive Insights Report...")
    
    # Get key insights
    top_5_features = importance_df.head(5)
    top_importance = importance_df.iloc[0]
    top_5_cumulative = top_5_features['rf_importance'].sum()
    total_features = len(importance_df)
    
    report = f"""# FEATURE IMPORTANCE ANALYSIS REPORT
## Credit Approval Prediction Model - DagsHub Integration

### EXECUTIVE SUMMARY

This analysis identifies the most influential features for credit approval predictions using a Random Forest model. The analysis was conducted on **{total_features} features** with results tracked in DagsHub for reproducibility and collaboration.

**Key Finding**: The top feature **"{top_importance['feature']}"** contributes **{top_importance['rf_importance']*100:.1f}%** to model predictions.

### MODEL SOURCE INFORMATION

- **Model Source**: {best_run_info['source']}
- **Original Run ID**: {best_run_info['run_id']}
- **Model Performance**: F1-Score {best_run_info['f1_score']:.3f}, Accuracy {best_run_info.get('accuracy', 'N/A')}
- **Artifact Name**: {best_run_info['artifact_name']}
- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### TOP 5 MOST IMPORTANT FEATURES

"""
    
    for i, (_, row) in enumerate(top_5_features.iterrows(), 1):
        impact_level = "High" if row['rf_importance'] > 0.1 else "Medium" if row['rf_importance'] > 0.05 else "Low"
        stability = "Stable" if abs(row['rf_importance'] - row['perm_importance']) < 0.02 else "Variable"
        
        report += f"""**{i}. {row['feature']}**
- **Random Forest Importance**: {row['rf_importance']:.3f} ({row['rf_importance']*100:.1f}%)
- **Permutation Importance**: {row['perm_importance']:.3f} (±{row['perm_std']:.3f})
- **Target Correlation**: {row['correlation']:.3f}
- **Impact Level**: {impact_level}
- **Stability**: {stability}

"""
    
    report += f"""
### KEY INSIGHTS & METRICS

1. **Feature Concentration Analysis**:
   - Top 5 features account for **{top_5_cumulative*100:.1f}%** of total model importance
   - Model reliance: {'Highly concentrated' if top_5_cumulative > 0.8 else 'Moderately concentrated' if top_5_cumulative > 0.6 else 'Well distributed'}
   - Feature diversity: {total_features} total features analyzed

2. **Model Interpretability**:
   - **Most Predictive**: {top_importance['feature']} ({top_importance['rf_importance']*100:.1f}% contribution)
   - **Stability Score**: {'High' if abs(top_importance['rf_importance'] - top_importance['perm_importance']) < 0.02 else 'Medium'}
   - **Correlation Alignment**: {'Strong' if top_importance['correlation'] > 0.3 else 'Moderate' if top_importance['correlation'] > 0.1 else 'Weak'} correlation with target

3. **Feature Quality Assessment**:
   - **High-impact features** (importance > 0.1): {len(importance_df[importance_df['rf_importance'] > 0.1])}
   - **Medium-impact features** (0.05-0.1): {len(importance_df[(importance_df['rf_importance'] > 0.05) & (importance_df['rf_importance'] <= 0.1)])}
   - **Low-impact features** (< 0.05): {len(importance_df[importance_df['rf_importance'] <= 0.05])}

### BUSINESS RECOMMENDATIONS

#### Immediate Actions:
1. **Data Quality Monitoring**:
   - Implement real-time monitoring for top 5 features: {', '.join(top_5_features['feature'].tolist())}
   - Set up alerts for missing values or outliers in {top_importance['feature']}

2. **Feature Engineering Priorities**:
   - Explore interaction terms between {top_5_features.iloc[0]['feature']} and {top_5_features.iloc[1]['feature']}
   - Consider polynomial features for high-correlation variables

#### Strategic Initiatives:
3. **Model Optimization**:
   - Consider feature selection: Remove {len(importance_df[importance_df['rf_importance'] < 0.01])} features with importance < 0.01
   - Potential for model simplification using only top {len(importance_df[importance_df['rf_importance'] > 0.05])} features

4. **Risk Management**:
   - Monitor for concept drift in {top_importance['feature']}
   - Implement fallback rules if top features become unavailable

### TECHNICAL METHODOLOGY

- **Primary Method**: Random Forest built-in feature importance
- **Validation Method**: Permutation importance with 10 repeats
- **Stability Metric**: Standard deviation of permutation importance
- **Correlation Analysis**: Pearson correlation with binary target
- **Sample Size**: {len(top_5_features)} samples for analysis
- **Explainability**: SHAP TreeExplainer for individual predictions

### FILES GENERATED

- `feature_importance_analysis.csv` - Detailed importance scores and rankings
- `feature_analysis_dashboard.png` - Comprehensive visualization dashboard  
- `feature_ranking_comparison.png` - RF vs Permutation ranking comparison
- `shap_summary_beeswarm.png` - SHAP feature impact visualization
- `shap_importance_bar.png` - SHAP-based feature importance ranking
- `shap_waterfall_individual.png` - Individual prediction explanation

### REPRODUCIBILITY

This analysis is fully tracked in DagsHub:
- **Repository**: agusprasetyo811/kredit_pinjaman
- **Experiment**: credit-approval-prediction  
- **Tracking**: All metrics, parameters, and artifacts logged to MLflow
- **Version Control**: Analysis code and results version controlled

### NEXT STEPS

1. **Review this report** with domain experts to validate feature interpretations
2. **Implement monitoring** for top features in production pipeline
3. **Consider hyperparameter tuning** based on feature importance insights
4. **Plan feature engineering** experiments using identified interaction opportunities
5. **Schedule regular re-analysis** to detect feature importance drift

---

*Generated by Feature Analysis Pipeline with DagsHub Integration*  
*Report ID: feature_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}*
"""
    
    # Save report
    with open(f"{output_dir}/feature_analysis_report.md", "w") as f:
        f.write(report)
    
    print("✓ Comprehensive report saved: feature_analysis_report.md")
    print(f"\nKEY BUSINESS INSIGHT:")
    print(f"   '{top_importance['feature']}' is the most critical feature")
    print(f"   Top 5 features drive {top_5_cumulative*100:.1f}% of model decisions")
    print(f"   Monitor these features closely in production")

# ============================================================================
# 6. ENHANCED DAGSHUB LOGGING
# ============================================================================

def log_to_dagshub(importance_df, corr_df, best_run_info, shap_importance=None):
    """Comprehensive logging to DagsHub dengan detailed tracking"""
    print("\n6. Logging Comprehensive Results to DagsHub...")
    
    try:
        with mlflow.start_run(run_name=f"feature_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            
            # Log analysis metadata
            mlflow.log_param("analysis_type", "comprehensive_feature_importance")
            mlflow.log_param("source_model_run_id", best_run_info['run_id'])
            mlflow.log_param("source_model_source", best_run_info['source'])
            mlflow.log_param("source_model_artifact", best_run_info['artifact_name'])
            mlflow.log_param("analysis_timestamp", datetime.now().isoformat())
            mlflow.log_param("total_features", len(importance_df))
            
            # Log top features
            top_5_features = importance_df.head(5)['feature'].tolist()
            top_10_features = importance_df.head(10)['feature'].tolist()
            mlflow.log_param("top_5_features", str(top_5_features))
            mlflow.log_param("top_10_features", str(top_10_features))
            mlflow.log_param("most_important_feature", importance_df.iloc[0]['feature'])
            mlflow.log_param("least_important_feature", importance_df.iloc[-1]['feature'])
            
            # Log individual feature importance metrics
            print("   Logging individual feature metrics...")
            for _, row in importance_df.iterrows():
                feature_name = row['feature']
                mlflow.log_metric(f"rf_importance_{feature_name}", row['rf_importance'])
                mlflow.log_metric(f"perm_importance_{feature_name}", row['perm_importance'])
                mlflow.log_metric(f"perm_std_{feature_name}", row['perm_std'])
                mlflow.log_metric(f"correlation_{feature_name}", row['correlation'])
                mlflow.log_metric(f"rf_rank_{feature_name}", row['rf_rank'])
            
            # Log summary metrics
            print("   Logging summary metrics...")
            mlflow.log_metric("top_feature_importance", importance_df.iloc[0]['rf_importance'])
            mlflow.log_metric("top_5_cumulative_importance", importance_df.head(5)['rf_importance'].sum())
            mlflow.log_metric("top_10_cumulative_importance", importance_df.head(10)['rf_importance'].sum())
            mlflow.log_metric("mean_importance", importance_df['rf_importance'].mean())
            mlflow.log_metric("std_importance", importance_df['rf_importance'].std())
            mlflow.log_metric("median_importance", importance_df['rf_importance'].median())
            mlflow.log_metric("min_importance", importance_df['rf_importance'].min())
            mlflow.log_metric("max_importance", importance_df['rf_importance'].max())
            
            # Log feature concentration metrics
            high_impact_features = len(importance_df[importance_df['rf_importance'] > 0.1])
            medium_impact_features = len(importance_df[(importance_df['rf_importance'] > 0.05) & (importance_df['rf_importance'] <= 0.1)])
            low_impact_features = len(importance_df[importance_df['rf_importance'] <= 0.05])
            
            mlflow.log_metric("high_impact_features_count", high_impact_features)
            mlflow.log_metric("medium_impact_features_count", medium_impact_features) 
            mlflow.log_metric("low_impact_features_count", low_impact_features)
            
            # Log SHAP metrics if available
            if shap_importance is not None:
                print("   Logging SHAP metrics...")
                for i, feature in enumerate(importance_df['feature']):
                    mlflow.log_metric(f"shap_importance_{feature}", shap_importance[i])
            
            # Log source model performance for reference
            mlflow.log_metric("source_model_f1_score", float(best_run_info['f1_score']))
            if 'accuracy' in best_run_info:
                mlflow.log_metric("source_model_accuracy", float(best_run_info['accuracy']))
            
            # Log all artifacts
            print("   Logging artifacts to DagsHub...")
            artifacts_to_log = [
                f"{output_dir}/feature_importance_analysis.csv",
                f"{output_dir}/feature_analysis_dashboard.png",
                f"{output_dir}/feature_ranking_comparison.png",
                f"{output_dir}/feature_analysis_report.md"
            ]
            
            # Add SHAP artifacts if they exist
            shap_artifacts = [
                f"{output_dir}/shap_summary_beeswarm.png",
                f"{output_dir}/shap_importance_bar.png", 
                f"{output_dir}/shap_waterfall_individual.png"
            ]
            
            for artifact in artifacts_to_log + shap_artifacts:
                try:
                    if os.path.exists(artifact):
                        mlflow.log_artifact(artifact)
                        print(f"      ✓ Logged: {os.path.basename(artifact)}")
                except Exception as e:
                    print(f"      Failed to log {artifact}: {e}")
            
            # Log tags for better organization
            mlflow.set_tag("analysis_type", "feature_importance")
            mlflow.set_tag("integration", "dagshub")
            mlflow.set_tag("model_source", best_run_info['source'])
            mlflow.set_tag("feature_count", str(len(importance_df)))
            
            run_id = run.info.run_id
            print(f"Comprehensive results logged to DagsHub!")
            print(f"   Run ID: {run_id}")
            print(f"   DagsHub URL: https://dagshub.com/agusprasetyo811/kredit_pinjaman")
            print(f"   View in Experiments tab → {run.info.run_name}")
        
    except Exception as e:
        print(f"DagsHub logging failed: {e}")
        print("   Results are still saved locally in output/feature_analysis/")

# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution dengan full DagsHub integration"""
    
    # Setup DagsHub & MLflow
    dagshub_url, experiment_id = setup_dagshub_mlflow()
    
    # Load model and data
    model, X_train, X_test, y_train, y_test, best_run_info = load_best_model_and_data()
    
    if model is None:
        print("Cannot proceed without model and data")
        return
    
    # Feature importance analysis
    importance_df = analyze_feature_importance(model, X_train, y_train)
    
    # Create advanced visualizations
    corr_df = create_visualizations(importance_df, X_train, y_train)
    
    # Enhanced SHAP analysis
    shap_importance = enhanced_shap_analysis(model, X_train, X_test)
    
    # Generate comprehensive report
    generate_insights_report(importance_df, corr_df, best_run_info)
    
    # Log everything to DagsHub
    log_to_dagshub(importance_df, corr_df, best_run_info, shap_importance)
    
    print(f"\n" + "="*70)
    print("FEATURE ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nLocal Files Generated:")
    print("   output/feature_analysis/feature_importance_analysis.csv")
    print("   output/feature_analysis/feature_analysis_dashboard.png") 
    print("   output/feature_analysis/feature_ranking_comparison.png")
    print("   output/feature_analysis/feature_analysis_report.md")
    print("   output/feature_analysis/shap_*.png (if SHAP installed)")
    
    print(f"\nDagsHub Integration:")
    print(f"   Repository: https://dagshub.com/agusprasetyo811/kredit_pinjaman")
    print(f"   Experiments: Click 'Experiments' tab → Look for 'feature_analysis_*' runs")
    print(f"   Metrics: All feature importance scores logged individually") 
    print(f"   Artifacts: All visualizations and reports available for download")
    
    print(f"\nKey Insights:")
    top_feature = importance_df.iloc[0]
    print(f"   Most Important: {top_feature['feature']} ({top_feature['rf_importance']*100:.1f}%)")
    print(f"   Top 5 Features: {', '.join(importance_df.head(5)['feature'].tolist())}")
    print(f"   Cumulative Impact: {importance_df.head(5)['rf_importance'].sum()*100:.1f}%")
    
    print(f"\nNext Steps:")
    print("   1. Review feature_analysis_report.md for business insights")
    print("   2. Check DagsHub dashboard for interactive results")
    print("   3. Consider hyperparameter tuning based on findings")
    print("   4. Implement feature monitoring in production")

if __name__ == "__main__":
    main()