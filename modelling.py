# modelling.py
# Tahap 1: Setup dan Loading Data untuk Credit Approval Prediction
# Integrated with MLflow and DagsHub (Fixed Version)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

# MLflow imports
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
from datetime import datetime

print("="*60)
print("CREDIT APPROVAL PREDICTION - MODELLING STAGE")
print("="*60)

# ============================================================================
# 0. MLFLOW SETUP
# ============================================================================

def setup_mlflow():
    """Setup MLflow tracking dengan DagsHub"""
    print("\n0. Setting up MLflow with DagsHub...")
    
    try:
        # DagsHub configuration
        dagshub_url = os.getenv('DAGSHUB_REPO_URL')
        
        if not dagshub_url:
            print("⚠️  DAGSHUB_REPO_URL not found in environment variables")
            print("   Using fallback URL...")
            dagshub_url = "https://dagshub.com/your-username/your-repo"
        
        # Set MLflow tracking URI ke DagsHub
        mlflow.set_tracking_uri(dagshub_url + ".mlflow")
        
        # Optional: Set DagsHub credentials sebagai environment variables
        username = os.getenv('DAGSHUB_USERNAME')
        token = os.getenv('DAGSHUB_TOKEN')
        
        if username and token:
            os.environ['MLFLOW_TRACKING_USERNAME'] = username
            os.environ['MLFLOW_TRACKING_PASSWORD'] = token
            print("✓ DagsHub credentials configured")
        else:
            print("⚠️  DagsHub credentials not found, continuing anyway...")
        
        # Set experiment name
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
                print("⚠️  Using default experiment")
                experiment_id = "0"
                experiment_name = "Default"
        
        mlflow.set_experiment(experiment_name)
        
        print(f"✓ MLflow tracking configured")
        print(f"   Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"   Experiment: {experiment_name}")
        
        return experiment_id
        
    except Exception as e:
        print(f"⚠️  MLflow setup issue: {e}")
        print("   Continuing with basic setup...")
        
        # Fallback setup
        try:
            mlflow.set_experiment("credit-approval-prediction")
        except:
            pass
        
        return "0"

# ============================================================================
# 1. LOADING DATA
# ============================================================================

def load_data():
    """Load training dan testing data dari CSV files"""
    print("\n1. Loading Data...")
    
    try:
        # Load training data
        X_train = pd.read_csv('final_dataset/X_train.csv')
        y_train = pd.read_csv('final_dataset/y_train.csv')
        
        # Load testing data  
        X_test = pd.read_csv('final_dataset/X_test.csv')
        y_test = pd.read_csv('final_dataset/y_test.csv')
        
        # Convert y ke array 1D
        y_train = y_train['disetujui_encoded'].values
        y_test = y_test['disetujui_encoded'].values
        
        print(f"✓ Training data loaded: X_train {X_train.shape}, y_train {y_train.shape}")
        print(f"✓ Testing data loaded: X_test {X_test.shape}, y_test {y_test.shape}")
        
        # Log dataset info to MLflow (with error handling)
        try:
            mlflow.log_param("train_samples", X_train.shape[0])
            mlflow.log_param("test_samples", X_test.shape[0])
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("feature_names", str(list(X_train.columns)))
        except Exception as e:
            print(f"   (MLflow logging warning: {e})")
        
        return X_train, X_test, y_train, y_test
        
    except FileNotFoundError as e:
        print(f"✗ Error loading data: {e}")
        print("Pastikan file X_train.csv, X_test.csv, y_train.csv, y_test.csv ada di directory yang sama")
        return None, None, None, None

# ============================================================================
# 2. DATA EXPLORATION
# ============================================================================

def explore_data(X_train, X_test, y_train, y_test):
    """Explorasi basic data yang sudah di-split"""
    print("\n2. Data Exploration...")
    
    # Basic info
    print(f"\n Dataset Overview:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    print(f"   Total features: {X_train.shape[1]}")
    
    # Target distribution
    print(f"\nTarget Distribution (Training):")
    unique, counts = np.unique(y_train, return_counts=True)
    for val, count in zip(unique, counts):
        pct = (count/len(y_train))*100
        status = "Approved" if val == 1 else "Rejected"
        print(f"   {status} ({val}): {count} samples ({pct:.1f}%)")
        
        # Log target distribution to MLflow
        try:
            mlflow.log_param(f"target_{status.lower()}_count", count)
            mlflow.log_param(f"target_{status.lower()}_percentage", round(pct, 2))
        except:
            pass
    
    # Feature types
    print(f"\nFeature Types:")
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    boolean_features = X_train.select_dtypes(include=[bool]).columns.tolist()
    
    print(f"   Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"   Boolean features ({len(boolean_features)}): {boolean_features}")
    
    # Log feature types to MLflow
    try:
        mlflow.log_param("numeric_features_count", len(numeric_features))
        mlflow.log_param("boolean_features_count", len(boolean_features))
        mlflow.log_param("numeric_features", str(numeric_features))
        mlflow.log_param("boolean_features", str(boolean_features))
    except:
        pass
    
    # Check for missing values
    print(f"\nMissing Values Check:")
    missing_train = X_train.isnull().sum().sum()
    missing_test = X_test.isnull().sum().sum()
    print(f"   Training set: {missing_train} missing values")
    print(f"   Testing set: {missing_test} missing values")
    
    # Log missing values info
    try:
        mlflow.log_param("missing_values_train", missing_train)
        mlflow.log_param("missing_values_test", missing_test)
    except:
        pass
    
    return numeric_features, boolean_features

# ============================================================================
# 3. BASIC STATISTICS
# ============================================================================

def show_basic_stats(X_train, numeric_features):
    """Tampilkan statistik dasar untuk numeric features"""
    print("\n3. Basic Statistics...")
    
    if numeric_features:
        print(f"\nNumeric Features Statistics:")
        stats = X_train[numeric_features].describe()
        print(stats.round(2))
        
        # Log basic statistics to MLflow (limit to avoid too many params)
        try:
            for feature in numeric_features[:5]:  # Only log first 5 features to avoid param limit
                feature_stats = X_train[feature].describe()
                mlflow.log_param(f"{feature}_mean", round(feature_stats['mean'], 4))
                mlflow.log_param(f"{feature}_std", round(feature_stats['std'], 4))
        except:
            pass
    
    print(f"\nData loading dan exploration selesai!")
    print(f"   Ready untuk tahap modeling...")

# ============================================================================
# 4. MODEL INITIALIZATION
# ============================================================================

def initialize_models():
    """Initialize models yang akan digunakan"""
    print("\n4. Model Initialization...")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10)
    }
    
    print(f"✓ {len(models)} models initialized:")
    for name in models.keys():
        print(f"   - {name}")
    
    # Log model configurations
    try:
        mlflow.log_param("models_count", len(models))
        mlflow.log_param("model_types", str(list(models.keys())))
        mlflow.log_param("logistic_regression_max_iter", 1000)
        mlflow.log_param("random_forest_n_estimators", 100)
        mlflow.log_param("decision_tree_max_depth", 10)
        mlflow.log_param("random_state", 42)
    except:
        pass
    
    return models

# ============================================================================
# 5. MODEL TRAINING & PREDICTION (FIXED)
# ============================================================================

def train_and_predict(models, X_train, X_test, y_train, y_test):
    """Training semua models dan buat prediksi (Fixed version)"""
    print("\n5. Training Models...")
    
    trained_models = {}
    predictions = {}
    model_results = {}  # Store all results for each model
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Create individual run for each model
        with mlflow.start_run(run_name=f"{name.replace(' ', '_')}_{datetime.now().strftime('%H%M%S')}", nested=True):
            try:
                # Log model info
                mlflow.log_param("model_name", name)
                mlflow.log_param("model_type", type(model).__name__)
                
                # Log model specific parameters
                model_params = model.get_params()
                for param_name, param_value in model_params.items():
                    if isinstance(param_value, (int, float, str, bool)) and param_value is not None:
                        mlflow.log_param(f"model_{param_name}", param_value)
                
                # Training
                import time
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Prediction
                y_pred = model.predict(X_test)
                
                # Calculate all metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                # Calculate ROC AUC if possible
                try:
                    roc_auc = roc_auc_score(y_test, y_pred)
                except:
                    roc_auc = None
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Log all metrics immediately
                mlflow.log_metric("training_time_seconds", training_time)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("cv_mean_score", cv_mean)
                mlflow.log_metric("cv_std_score", cv_std)
                
                if roc_auc is not None:
                    mlflow.log_metric("roc_auc", roc_auc)
                
                # Log individual CV scores
                for i, score in enumerate(cv_scores):
                    mlflow.log_metric(f"cv_fold_{i+1}_score", score)
                
                # Log confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                cm_text = f"Confusion Matrix:\nTN: {cm[0,0]}, FP: {cm[0,1]}\nFN: {cm[1,0]}, TP: {cm[1,1]}"
                mlflow.log_text(cm_text, "confusion_matrix.txt")
                
                # Log classification report
                report = classification_report(y_test, y_pred, target_names=['Rejected', 'Approved'])
                mlflow.log_text(report, "classification_report.txt")
                
                # Log model artifact
                mlflow.sklearn.log_model(model, f"model_{name.lower().replace(' ', '_')}")
                
                # Store results
                trained_models[name] = model
                predictions[name] = y_pred
                model_results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'training_time': training_time,
                    'predictions': y_pred
                }
                
                print(f"✓ {name} training completed (Time: {training_time:.2f}s)")
                print(f"   Accuracy: {accuracy:.3f}, F1: {f1:.3f}, CV: {cv_mean:.3f}")
                
            except Exception as e:
                print(f"   ⚠️  MLflow logging issue for {name}: {e}")
                # Continue with basic training even if MLflow fails
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                trained_models[name] = model
                predictions[name] = y_pred
                
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                print(f"✓ {name} training completed (Accuracy: {accuracy:.3f}, F1: {f1:.3f})")
    
    return trained_models, predictions, model_results

# ============================================================================
# 6. MODEL EVALUATION (SIMPLIFIED)
# ============================================================================

def evaluate_models(predictions, y_test, model_results=None):
    """Evaluasi performa semua models (results already logged in training)"""
    print("\n6. Model Evaluation...")
    
    results = {}
    
    print(f"\nModel Performance Comparison:")
    print("-" * 80)
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 80)
    
    for name, y_pred in predictions.items():
        # Calculate metrics (or use cached results)
        if model_results and name in model_results:
            # Use cached results
            accuracy = model_results[name]['accuracy']
            precision = model_results[name]['precision']
            recall = model_results[name]['recall']
            f1 = model_results[name]['f1_score']
            roc_auc = model_results[name].get('roc_auc')
        else:
            # Calculate fresh
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            try:
                roc_auc = roc_auc_score(y_test, y_pred)
            except:
                roc_auc = None
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'predictions': y_pred
        }
        
        # Print results (metrics already logged in train_and_predict)
        print(f"{name:<20} {accuracy:<10.3f} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f}")
    
    print("-" * 80)
    return results

# ============================================================================
# 7. DETAILED EVALUATION (SIMPLIFIED)
# ============================================================================

def detailed_evaluation(results, y_test):
    """Evaluasi detail untuk setiap model (reports already logged)"""
    print("\n7. Detailed Evaluation...")
    
    for name, metrics in results.items():
        print(f"\n{name} - Detailed Report:")
        print("-" * 50)
        
        y_pred = metrics['predictions']
        
        # Classification report
        print("\nClassification Report:")
        report = classification_report(y_test, y_pred, target_names=['Rejected', 'Approved'])
        print(report)
        
        # Confusion Matrix
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"                 Predicted")
        print(f"               Rej  App")
        print(f"Actual   Rej   {cm[0,0]:3d}  {cm[0,1]:3d}")
        print(f"         App   {cm[1,0]:3d}  {cm[1,1]:3d}")
        
        # Note: Classification reports and confusion matrices already logged in train_and_predict

# ============================================================================
# 8. CROSS VALIDATION (RESULTS ALREADY LOGGED)
# ============================================================================

def cross_validation_analysis(models, X_train, y_train, model_results=None):
    """Cross-validation analysis (results already logged in training)"""
    print("\n8. Cross-Validation Analysis...")
    
    cv_results = {}
    
    print(f"\n🔄 5-Fold Cross-Validation Results:")
    print("-" * 60)
    print(f"{'Model':<20} {'Mean CV Score':<15} {'Std Dev':<10}")
    print("-" * 60)
    
    for name, model in models.items():
        if model_results and name in model_results:
            # Use cached CV results
            cv_mean = model_results[name]['cv_mean']
            cv_std = model_results[name]['cv_std']
            cv_scores = None  # Not stored individually
        else:
            # Calculate fresh CV scores
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        
        cv_results[name] = {
            'mean_score': cv_mean,
            'std_score': cv_std,
            'scores': cv_scores if cv_scores is not None else []
        }
        
        print(f"{name:<20} {cv_mean:<15.3f} {cv_std:<10.3f}")
    
    print("-" * 60)
    return cv_results

# ============================================================================
# 9. MODEL COMPARISON & RECOMMENDATION
# ============================================================================

def model_recommendation(results, cv_results, trained_models):
    """Berikan rekomendasi model terbaik"""
    print("\n9. Model Recommendation...")
    
    # Find best model based on F1-score (balance precision & recall)
    best_f1_model = max(results.items(), key=lambda x: x[1]['f1_score'])
    best_f1_name, best_f1_metrics = best_f1_model
    
    # Find best model based on CV score
    best_cv_model = max(cv_results.items(), key=lambda x: x[1]['mean_score'])
    best_cv_name, best_cv_metrics = best_cv_model
    
    print(f"\n🏆 Best Model Recommendations:")
    print("-" * 50)
    print(f"Best F1-Score: {best_f1_name} (F1: {best_f1_metrics['f1_score']:.3f})")
    print(f"Best CV Score: {best_cv_name} (CV: {best_cv_metrics['mean_score']:.3f})")
    
    # Start a summary run to log best models
    try:
        with mlflow.start_run(run_name=f"summary_{datetime.now().strftime('%H%M%S')}", nested=True):
            mlflow.log_param("best_f1_model", best_f1_name)
            mlflow.log_param("best_cv_model", best_cv_name)
            mlflow.log_metric("best_f1_score", best_f1_metrics['f1_score'])
            mlflow.log_metric("best_cv_score", best_cv_metrics['mean_score'])
            
            # Log the best model based on F1-score
            best_model = trained_models[best_f1_name]
            mlflow.sklearn.log_model(best_model, "best_model")
            
            # Create and log comparison table
            comparison_data = []
            for name in results.keys():
                comparison_data.append({
                    'model': name,
                    'accuracy': results[name]['accuracy'],
                    'precision': results[name]['precision'],
                    'recall': results[name]['recall'],
                    'f1_score': results[name]['f1_score'],
                    'cv_mean': cv_results[name]['mean_score'],
                    'cv_std': cv_results[name]['std_score']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_csv("model_comparison.csv", index=False)
            mlflow.log_artifact("model_comparison.csv")
            
            # Log summary metrics
            for name in results.keys():
                mlflow.log_metric(f"{name}_accuracy", results[name]['accuracy'])
                mlflow.log_metric(f"{name}_f1_score", results[name]['f1_score'])
                mlflow.log_metric(f"{name}_cv_score", cv_results[name]['mean_score'])
            
            print(f"✓ Summary logged to MLflow")
    
    except Exception as e:
        print(f"⚠️  Summary logging issue: {e}")
    
    # Overall recommendation
    if best_f1_name == best_cv_name:
        print(f"\nRECOMMENDED MODEL: {best_f1_name}")
        print("   Konsisten terbaik di both metrics!")
    else:
        print(f"\nPertimbangan:")
        print(f"   - {best_f1_name}: Terbaik untuk balance precision/recall")
        print(f"   - {best_cv_name}: Terbaik untuk generalization")
    
    return best_f1_name, best_cv_name

# ============================================================================
# 10. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Setup MLflow tracking
    experiment_id = setup_mlflow()
    
    # Start main experiment run
    with mlflow.start_run(run_name=f"credit_approval_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        try:
            # Log experiment metadata
            mlflow.log_param("experiment_date", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            mlflow.log_param("experiment_type", "credit_approval_prediction")
            mlflow.log_param("data_split_method", "predefined_train_test")
        except:
            pass
        
        # Tahap 1: Load & Explore Data
        X_train, X_test, y_train, y_test = load_data()
        
        if X_train is not None:
            # Data exploration
            numeric_features, boolean_features = explore_data(X_train, X_test, y_train, y_test)
            show_basic_stats(X_train, numeric_features)
            
            print(f"\n" + "="*60)
            print("TAHAP 2: MODEL TRAINING & EVALUATION")
            print("="*60)
            
            # Tahap 2: Model Training & Evaluation (Fixed)
            models = initialize_models()
            trained_models, predictions, model_results = train_and_predict(models, X_train, X_test, y_train, y_test)
            results = evaluate_models(predictions, y_test, model_results)
            detailed_evaluation(results, y_test)
            cv_results = cross_validation_analysis(models, X_train, y_train, model_results)
            best_f1, best_cv = model_recommendation(results, cv_results, trained_models)
            
            # Log final experiment summary
            try:
                mlflow.log_param("best_model_final", best_f1)
                mlflow.log_metric("experiment_success", 1)
            except:
                pass
            
            print(f"\n" + "="*60)
            print("TAHAP 2 SELESAI - MODEL TRAINING COMPLETED")
            print("="*60)
            print("\nMLflow Tracking:")
            print(f"   Experiment logged to: {mlflow.get_tracking_uri()}")
            print(f"   Check your DagsHub repository for results")
            print("\nNext steps:")
            print("- Tahap 3: Feature Importance Analysis")
            print("- Tahap 4: Model Visualization") 
            print("- Tahap 5: Model Saving & Deployment")

        else:
            print("\nGagal load data. Cek file CSV Anda.")
            try:
                mlflow.log_param("experiment_status", "failed_data_loading")
            except:
                pass