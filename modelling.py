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

import os
import sys
import time
from datetime import datetime
import joblib

print("="*60)
print("CREDIT APPROVAL PREDICTION - MODELLING STAGE")
print("="*60)

output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"✓ Output directory created: {output_dir}")
else:
    print(f"✓ Output directory already exists: {output_dir}")

# ============================================================================
# 0. ROBUST MLFLOW SETUP WITH DAGSHUB INTEGRATION
# ============================================================================

def setup_mlflow_robust():
    """Setup MLflow dengan robust fallback untuk DagsHub connection issues"""
    print("\n0. Setting up MLflow with DagsHub (Robust Version)...")
    
    # Import MLflow
    try:
        import mlflow
        import mlflow.sklearn
        print("✓ MLflow imported successfully")
    except ImportError:
        print("MLflow not installed. Install with: pip install mlflow")
        return None, False, "none"
    
    # Get DagsHub credentials
    dagshub_url = os.getenv('DAGSHUB_REPO_URL')
    dagshub_username = os.getenv('DAGSHUB_USERNAME') 
    dagshub_token = os.getenv('DAGSHUB_TOKEN')
    
    # Get experiment name from environment (MLflow Projects parameter)
    experiment_name = os.getenv('EXPERIMENT_NAME', 'credit-approval-prediction')
    model_name = os.getenv('MODEL_NAME', 'credit_approval_model')
    
    print(f"Configuration check:")
    print(f"  DagsHub URL: {'✓' if dagshub_url else '❌'} {dagshub_url}")
    print(f"  Username: {'✓' if dagshub_username else '❌'} {dagshub_username}")
    print(f"  Token: {'✓' if dagshub_token else '❌'}")
    print(f"  Experiment: {experiment_name}")
    print(f"  Model Name: {model_name}")
    
    # Strategy 1: Try DagsHub with timeout and proper error handling
    if dagshub_url and dagshub_username and dagshub_token:
        try:
            print("\nAttempting DagsHub connection...")
            
            # Import and initialize DagsHub
            try:
                import dagshub
                dagshub.init(
                    repo_owner='agusprasetyo811',
                    repo_name='kredit_pinjaman_1',
                    mlflow=True
                )
                print("✓ DagsHub initialized")
            except Exception as e:
                print(f"DagsHub init warning: {e}")
            
            # Setup environment variables
            os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_username
            os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
            
            # Configure tracking URI
            tracking_uri = dagshub_url + ".mlflow"
            mlflow.set_tracking_uri(tracking_uri)
            print(f"✓ Tracking URI set: {tracking_uri}")
            
            # Test connection with timeout
            def test_dagshub_connection():
                """Test DagsHub connection with simple operation"""
                try:
                    # Set timeout for HTTP requests
                    import socket
                    socket.setdefaulttimeout(15)  # 15 second timeout
                    
                    # Try to get tracking URI info
                    current_uri = mlflow.get_tracking_uri()
                    print(f"   Testing connection to: {current_uri}")
                    
                    # Try to create/get experiment (quick test)
                    try:
                        exp = mlflow.get_experiment_by_name(experiment_name)
                        if exp:
                            mlflow.set_experiment(experiment_name)
                            print(f"✓ Using existing experiment: {experiment_name}")
                        else:
                            mlflow.create_experiment(experiment_name)
                            print(f"✓ Created new experiment: {experiment_name}")
                    except Exception as e:
                        print(f"   Experiment setup issue: {e}")
                        # Try with default experiment
                        mlflow.set_experiment("Default")
                    
                    # Test with a simple run (simplified test)
                    try:
                        with mlflow.start_run(run_name="connection_test", nested=False):
                            mlflow.log_param("test_param", "connection_ok")
                            mlflow.log_metric("test_metric", 1.0)
                            run_id = mlflow.active_run().info.run_id
                            print(f"✓ Test run created: {run_id[:8]}...")
                    except Exception as e:
                        print(f"   Simple connection test: {e}")
                        # This is expected if DagsHub has issues, continue anyway
                    
                    socket.setdefaulttimeout(None)  # Reset timeout
                    return True
                    
                except Exception as e:
                    socket.setdefaulttimeout(None)  # Reset timeout
                    print(f"   Connection test failed: {e}")
                    print("   This is likely a DagsHub server issue, falling back to local...")
                    return False
            
            # Test connection
            if test_dagshub_connection():
                print("✅ DagsHub MLflow connection successful!")
                return mlflow, True, "dagshub"
            else:
                raise Exception("DagsHub connection test failed - server issues")
                
        except Exception as e:
            print(f"❌ DagsHub setup failed: {e}")
            print("🔄 Falling back to local MLflow...")
    
    # Strategy 2: Fallback to local MLflow
    try:
        print("\n🔄 Setting up local MLflow...")
        
        # Use local file-based tracking
        local_mlruns_dir = "./mlruns"
        os.makedirs(local_mlruns_dir, exist_ok=True)
        
        local_uri = f"file://{os.path.abspath(local_mlruns_dir)}"
        mlflow.set_tracking_uri(local_uri)
        
        # Setup experiment
        try:
            exp = mlflow.get_experiment_by_name(experiment_name)
            if exp:
                mlflow.set_experiment(experiment_name)
            else:
                mlflow.create_experiment(experiment_name)
        except:
            mlflow.set_experiment("Default")
        
        print(f"✅ Local MLflow setup successful!")
        print(f"   Tracking URI: {local_uri}")
        print(f"   Experiment: {experiment_name}")
        print(f"   MLruns directory: {local_mlruns_dir}")
        print(f"   Web UI: run 'mlflow ui' in terminal")
        
        return mlflow, True, "local"
        
    except Exception as e:
        print(f"❌ Local MLflow setup failed: {e}")
        print("⚠️ Continuing without MLflow tracking...")
        return None, False, "none"

# ============================================================================
# 1. LOADING DATA
# ============================================================================

def load_data():
    """Load training dan testing data dari CSV files"""
    print("\n1. Loading Data...")
    
    try:
        # Check if files exist
        required_files = [
            'final_dataset/X_train.csv',
            'final_dataset/y_train.csv', 
            'final_dataset/X_test.csv',
            'final_dataset/y_test.csv'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            return None, None, None, None
        
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
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None, None

# ============================================================================
# 2. DATA EXPLORATION WITH SAFE MLFLOW LOGGING
# ============================================================================

def explore_data(X_train, X_test, y_train, y_test, mlflow=None):
    """Explorasi basic data yang sudah di-split dengan safe MLflow logging"""
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
    
    # Feature types
    print(f"\nFeature Types:")
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    boolean_features = X_train.select_dtypes(include=[bool]).columns.tolist()
    
    print(f"   Numeric features ({len(numeric_features)}): {numeric_features[:5]}...")
    print(f"   Boolean features ({len(boolean_features)}): {boolean_features[:5]}...")
    
    # Check for missing values
    print(f"\nMissing Values Check:")
    missing_train = X_train.isnull().sum().sum()
    missing_test = X_test.isnull().sum().sum()
    print(f"   Training set: {missing_train} missing values")
    print(f"   Testing set: {missing_test} missing values")
    
    # Safe MLflow logging
    if mlflow:
        try:
            mlflow.log_param("train_samples", X_train.shape[0])
            mlflow.log_param("test_samples", X_test.shape[0])
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("numeric_features_count", len(numeric_features))
            mlflow.log_param("boolean_features_count", len(boolean_features))
            mlflow.log_param("missing_values_train", missing_train)
            mlflow.log_param("missing_values_test", missing_test)
            
            # Log target distribution
            approval_rate = counts[1] / len(y_train) if len(counts) > 1 else 0
            mlflow.log_param("approval_rate", round(approval_rate, 3))
            
        except Exception as e:
            print(f"   MLflow logging warning: {e}")
    
    return numeric_features, boolean_features

# ============================================================================
# 3. MODEL INITIALIZATION
# ============================================================================

def initialize_models():
    """Initialize models yang akan digunakan"""
    print("\n3. Model Initialization...")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10)
    }
    
    print(f"✓ {len(models)} models initialized:")
    for name in models.keys():
        print(f"   - {name}")
    
    return models

# ============================================================================
# 4. SAFE MODEL TRAINING WITH MLFLOW
# ============================================================================

def train_and_predict_safe(models, X_train, X_test, y_train, y_test, mlflow=None, mlflow_available=False):
    """Training semua models dengan safe MLflow logging"""
    print("\n4. Training Models...")
    
    trained_models = {}
    predictions = {}
    model_results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Create individual run only if MLflow is available
        if mlflow_available and mlflow:
            try:
                with mlflow.start_run(run_name=f"{name.replace(' ', '_')}_{datetime.now().strftime('%H%M%S')}", nested=True):
                    result = train_single_model(name, model, X_train, X_test, y_train, y_test, mlflow, True)
            except Exception as e:
                print(f"   MLflow nested run failed: {e}")
                print("   Continuing with basic training...")
                result = train_single_model(name, model, X_train, X_test, y_train, y_test, mlflow, False)
        else:
            result = train_single_model(name, model, X_train, X_test, y_train, y_test, mlflow, False)
        
        if result:
            trained_models[name] = result['model']
            predictions[name] = result['predictions']
            model_results[name] = result['metrics']
    
    return trained_models, predictions, model_results

def train_single_model(name, model, X_train, X_test, y_train, y_test, mlflow=None, log_to_mlflow=False):
    """Train single model dengan safe logging"""
    try:
        # Training
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Prediction
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # ROC AUC (safe calculation)
        try:
            roc_auc = roc_auc_score(y_test, y_pred)
        except:
            roc_auc = None
        
        # Cross-validation (safe)
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except Exception as e:
            print(f"     CV calculation failed: {e}")
            cv_mean, cv_std = 0, 0
            cv_scores = []
        
        # Safe MLflow logging
        if log_to_mlflow and mlflow:
            try:
                mlflow.log_param("model_name", name)
                mlflow.log_param("model_type", type(model).__name__)
                mlflow.log_metric("training_time_seconds", training_time)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("cv_mean_score", cv_mean)
                mlflow.log_metric("cv_std_score", cv_std)
                
                if roc_auc is not None:
                    mlflow.log_metric("roc_auc", roc_auc)
                
                # Log model
                mlflow.sklearn.log_model(model, f"model_{name.lower().replace(' ', '_')}")
                
            except Exception as e:
                print(f"     MLflow logging failed: {e}")
        
        print(f"✓ {name} completed - Acc: {accuracy:.3f}, F1: {f1:.3f}, CV: {cv_mean:.3f}")
        
        return {
            'model': model,
            'predictions': y_pred,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'training_time': training_time
            }
        }
        
    except Exception as e:
        print(f"❌ Training failed for {name}: {e}")
        return None

# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================

def evaluate_models(predictions, y_test, model_results=None):
    """Evaluasi performa semua models"""
    print("\n5. Model Evaluation...")
    
    results = {}
    
    print(f"\nModel Performance Comparison:")
    print("-" * 80)
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 80)
    
    for name, y_pred in predictions.items():
        if model_results and name in model_results:
            # Use cached results
            metrics = model_results[name]
            accuracy = metrics['accuracy']
            precision = metrics['precision']
            recall = metrics['recall']
            f1 = metrics['f1_score']
            roc_auc = metrics.get('roc_auc')
        else:
            # Calculate fresh
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            try:
                roc_auc = roc_auc_score(y_test, y_pred)
            except:
                roc_auc = None
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'predictions': y_pred
        }
        
        print(f"{name:<20} {accuracy:<10.3f} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f}")
    
    print("-" * 80)
    return results

# ============================================================================
# 6. MODEL RECOMMENDATION & SAVING
# ============================================================================

def model_recommendation_and_save(results, trained_models, mlflow=None, mlflow_available=False):
    """Rekomendasi model terbaik dan save model"""
    print("\n6. Model Recommendation & Saving...")
    
    # Find best model based on F1-score
    best_f1_model = max(results.items(), key=lambda x: x[1]['f1_score'])
    best_f1_name, best_f1_metrics = best_f1_model
    
    print(f"\nBest Model: {best_f1_name}")
    print(f"  F1-Score: {best_f1_metrics['f1_score']:.3f}")
    print(f"  Accuracy: {best_f1_metrics['accuracy']:.3f}")
    print(f"  Precision: {best_f1_metrics['precision']:.3f}")
    print(f"  Recall: {best_f1_metrics['recall']:.3f}")
    
    # Save model locally
    try:
        best_model = trained_models[best_f1_name]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"best_model_{timestamp}.pkl"
        filepath = os.path.join(output_dir, filename)
        
        joblib.dump(best_model, filepath)
        print(f"✓ Model saved locally: {filepath}")
        
        # Save model info
        info_file = os.path.join(output_dir, f"model_info_{timestamp}.txt")
        with open(info_file, 'w') as f:
            f.write(f"Best Model: {best_f1_name}\n")
            f.write(f"F1-Score: {best_f1_metrics['f1_score']:.4f}\n")
            f.write(f"Accuracy: {best_f1_metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {best_f1_metrics['precision']:.4f}\n")
            f.write(f"Recall: {best_f1_metrics['recall']:.4f}\n")
            f.write(f"Training Date: {datetime.now()}\n")
        
        print(f"✓ Model info saved: {info_file}")
        
    except Exception as e:
        print(f"❌ Model saving failed: {e}")
    
    # Safe MLflow logging
    if mlflow_available and mlflow:
        try:
            with mlflow.start_run(run_name=f"best_model_{datetime.now().strftime('%H%M%S')}", nested=True):
                mlflow.log_param("best_model_name", best_f1_name)
                mlflow.log_metric("best_f1_score", best_f1_metrics['f1_score'])
                mlflow.log_metric("best_accuracy", best_f1_metrics['accuracy'])
                mlflow.log_metric("best_precision", best_f1_metrics['precision'])
                mlflow.log_metric("best_recall", best_f1_metrics['recall'])
                
                # Log best model
                best_model = trained_models[best_f1_name]
                mlflow.sklearn.log_model(best_model, "best_model")
                
                # Try to register model
                try:
                    model_name = os.getenv('MODEL_NAME', 'credit_approval_model')
                    run_id = mlflow.active_run().info.run_id
                    model_uri = f"runs:/{run_id}/best_model"
                    mlflow.register_model(model_uri, model_name)
                    print(f"✓ Model registered as: {model_name}")
                except Exception as e:
                    print(f"   Model registration warning: {e}")
                
        except Exception as e:
            print(f"   MLflow best model logging failed: {e}")
    
    return best_f1_name, best_f1_metrics

# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("Starting Credit Approval Modeling Pipeline...")
    
    # Setup MLflow
    mlflow, mlflow_available, tracking_type = setup_mlflow_robust()
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    if X_train is None:
        print("❌ Failed to load data. Exiting...")
        return
    
    # Start main experiment
    if mlflow_available and mlflow:
        try:
            with mlflow.start_run(run_name=f"credit_approval_main_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log experiment info
                mlflow.log_param("experiment_date", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                mlflow.log_param("tracking_type", tracking_type)
                mlflow.log_param("experiment_type", "credit_approval_modeling")
                
                # Run pipeline
                run_modeling_pipeline(X_train, X_test, y_train, y_test, mlflow, mlflow_available)
                
                mlflow.log_metric("experiment_success", 1)
                
        except Exception as e:
            print(f"MLflow main experiment failed: {e}")
            print("Continuing with basic pipeline...")
            run_modeling_pipeline(X_train, X_test, y_train, y_test, mlflow, False)
    else:
        run_modeling_pipeline(X_train, X_test, y_train, y_test, mlflow, False)
    
    print(f"\n" + "="*60)
    print("MODELING PIPELINE COMPLETED!")
    print("="*60)
    print(f"Tracking: {tracking_type}")
    if tracking_type == "local":
        print("Run 'mlflow ui' to view results in browser")
    elif tracking_type == "dagshub":
        print("Check your DagsHub repository for results")

def run_modeling_pipeline(X_train, X_test, y_train, y_test, mlflow, mlflow_available):
    """Run the complete modeling pipeline"""
    # Data exploration
    numeric_features, boolean_features = explore_data(X_train, X_test, y_train, y_test, mlflow)
    
    # Model training
    models = initialize_models()
    trained_models, predictions, model_results = train_and_predict_safe(
        models, X_train, X_test, y_train, y_test, mlflow, mlflow_available
    )
    
    # Model evaluation
    results = evaluate_models(predictions, y_test, model_results)
    
    # Best model recommendation and saving
    best_name, best_metrics = model_recommendation_and_save(
        results, trained_models, mlflow, mlflow_available
    )
    
    print(f"\nPipeline completed successfully!")
    print(f"Best model: {best_name} (F1: {best_metrics['f1_score']:.3f})")

if __name__ == "__main__":
    main()