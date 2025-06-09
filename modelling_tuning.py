# model_optimization_comprehensive.py
# Comprehensive Model Optimization berdasarkan Feature Analysis Results

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, cross_val_score, 
                                   StratifiedKFold, train_test_split)
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score,
                           roc_auc_score, classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import optuna
from datetime import datetime
import time
import os
import joblib
import warnings
warnings.filterwarnings('ignore')


# DagsHub Integration
from dotenv import load_dotenv
load_dotenv()

print("="*70)
print("COMPREHENSIVE MODEL OPTIMIZATION - DAGSHUB INTEGRATED")
print("="*70)

output_dir = "MLProject/output/models"

# ============================================================================
# 0. SETUP DAGSHUB & MLFLOW
# ============================================================================

def setup_dagshub_mlflow():
    """Setup DagsHub dan MLflow tracking"""
    print("\n0. Setting up DagsHub & MLflow...")
    
    try:
        # DagsHub configuration
        dagshub_url = os.getenv('DAGSHUB_REPO_URL', "https://dagshub.com/agusprasetyo811/kredit_pinjaman")
        dagshub_username = os.getenv('DAGSHUB_USERNAME')
        dagshub_token = os.getenv('DAGSHUB_TOKEN')
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(dagshub_url + ".mlflow")
        
        # Set credentials
        if dagshub_username and dagshub_token:
            os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_username
            os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
            print("✓ DagsHub credentials configured")
        
        # Set experiment
        experiment_name = "credit-approval-prediction"
        mlflow.set_experiment(experiment_name)
        
        print(f"✓ MLflow tracking configured")
        print(f"   Repository: {dagshub_url}")
        print(f"   Experiment: {experiment_name}")
        
        return dagshub_url
        
    except Exception as e:
        print(f"DagsHub setup issue: {e}")
        print("   Using local tracking...")
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("model-optimization-local")
        return None

# ============================================================================
# 1. DATA LOADING & PREPARATION
# ============================================================================

def load_and_prepare_data():
    """Load data dan prepare untuk optimization"""
    print("\n1. Loading and Preparing Data...")
    
    try:
        # Load data
        X_train = pd.read_csv('final_dataset/X_train.csv')
        X_test = pd.read_csv('final_dataset/X_test.csv')
        y_train = pd.read_csv('final_dataset/y_train.csv')['disetujui_encoded']
        y_test = pd.read_csv('final_dataset/y_test.csv')['disetujui_encoded']
        
        print(f"✓ Data loaded successfully")
        print(f"   Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"   Testing: {X_test.shape[0]} samples")
        print(f"   Target distribution: {np.bincount(y_train)} (train), {np.bincount(y_test)} (test)")
        
        # Create validation split dari training data
        X_train_opt, X_val, y_train_opt, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"   Optimization split: {X_train_opt.shape[0]} train, {X_val.shape[0]} validation")
        
        return X_train, X_test, y_train, y_test, X_train_opt, X_val, y_train_opt, y_val
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None, None, None, None

def get_baseline_performance(X_train, X_test, y_train, y_test):
    """Get baseline performance untuk comparison"""
    print("\n2. Getting Baseline Performance...")
    
    # Simple Random Forest baseline
    baseline_model = RandomForestClassifier(random_state=42, n_estimators=100)
    baseline_model.fit(X_train, y_train)
    
    y_pred_baseline = baseline_model.predict(X_test)
    baseline_f1 = f1_score(y_test, y_pred_baseline)
    baseline_acc = accuracy_score(y_test, y_pred_baseline)
    baseline_cv = cross_val_score(baseline_model, X_train, y_train, cv=5, scoring='f1').mean()
    
    print(f"Baseline Performance (Random Forest default):")
    print(f"   Test F1-Score: {baseline_f1:.4f}")
    print(f"   Test Accuracy: {baseline_acc:.4f}")
    print(f"   CV F1-Score: {baseline_cv:.4f}")
    
    return {
        'model': baseline_model,
        'f1': baseline_f1,
        'accuracy': baseline_acc,
        'cv_f1': baseline_cv
    }

# ============================================================================
# 2. ADVANCED RANDOM FOREST OPTIMIZATION
# ============================================================================

def optimize_random_forest_advanced(X_train, y_train, X_val, y_val):
    """Advanced Random Forest optimization dengan multiple strategies"""
    print("\n3. Advanced Random Forest Optimization...")
    
    results = {}
    
    # Strategy 1: Comprehensive Grid Search
    print("   Strategy 1: Comprehensive Grid Search...")
    
    param_grid_comprehensive = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6],
        'max_features': ['sqrt', 'log2', 0.3, 0.5, None],
        'bootstrap': [True, False],
        'class_weight': [None, 'balanced', 'balanced_subsample']
    }
    
    with mlflow.start_run(run_name=f"RF_GridSearch_{datetime.now().strftime('%H%M%S')}", nested=True):
        mlflow.log_param("optimization_strategy", "comprehensive_grid_search")
        mlflow.log_param("total_combinations", np.prod([len(v) for v in param_grid_comprehensive.values()]))
        
        # Use RandomizedSearchCV untuk efficiency
        rf_grid = RandomizedSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid_comprehensive,
            n_iter=200,  # Test 200 combinations
            cv=5,
            scoring='f1',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        start_time = time.time()
        rf_grid.fit(X_train, y_train)
        optimization_time = time.time() - start_time
        
        # Evaluate on validation set
        best_rf = rf_grid.best_estimator_
        y_val_pred = best_rf.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        # Log results
        mlflow.log_params(rf_grid.best_params_)
        mlflow.log_metric("best_cv_f1", rf_grid.best_score_)
        mlflow.log_metric("validation_f1", val_f1)
        mlflow.log_metric("validation_accuracy", val_acc)
        mlflow.log_metric("optimization_time_minutes", optimization_time/60)
        mlflow.sklearn.log_model(best_rf, "optimized_random_forest")
        
        results['comprehensive_grid'] = {
            'model': best_rf,
            'params': rf_grid.best_params_,
            'cv_f1': rf_grid.best_score_,
            'val_f1': val_f1,
            'val_acc': val_acc,
            'optimization_time': optimization_time
        }
        
        print(f"      ✓ Best CV F1: {rf_grid.best_score_:.4f}")
        print(f"      ✓ Validation F1: {val_f1:.4f}")
        print(f"      ✓ Time: {optimization_time/60:.1f} minutes")
    
    # Strategy 2: Bayesian Optimization dengan Optuna
    print("   Strategy 2: Bayesian Optimization (Optuna)...")
    
    def rf_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
        }
        
        model = RandomForestClassifier(random_state=42, **params)
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1')  # 3-fold for speed
        return cv_scores.mean()
    
    try:
        with mlflow.start_run(run_name=f"RF_Optuna_{datetime.now().strftime('%H%M%S')}", nested=True):
            mlflow.log_param("optimization_strategy", "bayesian_optuna")
            
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
            start_time = time.time()
            study.optimize(rf_objective, n_trials=100, show_progress_bar=True)
            optimization_time = time.time() - start_time
            
            # Train best model
            best_rf_optuna = RandomForestClassifier(random_state=42, **study.best_params)
            best_rf_optuna.fit(X_train, y_train)
            
            # Evaluate
            y_val_pred = best_rf_optuna.predict(X_val)
            val_f1 = f1_score(y_val, y_val_pred)
            val_acc = accuracy_score(y_val, y_val_pred)
            
            # Log results
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_cv_f1", study.best_value)
            mlflow.log_metric("validation_f1", val_f1)
            mlflow.log_metric("validation_accuracy", val_acc)
            mlflow.log_metric("optimization_time_minutes", optimization_time/60)
            mlflow.log_metric("n_trials", len(study.trials))
            mlflow.sklearn.log_model(best_rf_optuna, "optuna_random_forest")
            
            results['optuna'] = {
                'model': best_rf_optuna,
                'params': study.best_params,
                'cv_f1': study.best_value,
                'val_f1': val_f1,
                'val_acc': val_acc,
                'optimization_time': optimization_time
            }
            
            print(f"      ✓ Best CV F1: {study.best_value:.4f}")
            print(f"      ✓ Validation F1: {val_f1:.4f}")
            print(f"      ✓ Trials: {len(study.trials)}")
    
    except ImportError:
        print("      Optuna not installed. Skipping Bayesian optimization.")
        print("      Install with: pip install optuna")
    except Exception as e:
        print(f"      Optuna optimization failed: {e}")
    
    return results

# ============================================================================
# 3. MULTI-MODEL OPTIMIZATION
# ============================================================================

def optimize_multiple_models(X_train, y_train, X_val, y_val):
    """Optimize multiple model types"""
    print("\n4. Multi-Model Optimization...")
    
    models_config = {
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.15, 0.2],
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0]
            }
        },
        'Extra Trees': {
            'model': ExtraTreesClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'param_grid': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'class_weight': [None, 'balanced']
            }
        }
    }
    
    results = {}
    
    for model_name, config in models_config.items():
        print(f"   🔧 Optimizing {model_name}...")
        
        with mlflow.start_run(run_name=f"{model_name.replace(' ', '_')}_{datetime.now().strftime('%H%M%S')}", nested=True):
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("optimization_method", "randomized_search")
            
            # Scale features untuk Logistic Regression
            if model_name == 'Logistic Regression':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
            else:
                X_train_scaled = X_train
                X_val_scaled = X_val
            
            # Randomized search
            search = RandomizedSearchCV(
                config['model'],
                config['param_grid'],
                n_iter=50,
                cv=5,
                scoring='f1',
                n_jobs=-1,
                random_state=42
            )
            
            start_time = time.time()
            search.fit(X_train_scaled, y_train)
            optimization_time = time.time() - start_time
            
            # Evaluate
            best_model = search.best_estimator_
            y_val_pred = best_model.predict(X_val_scaled)
            val_f1 = f1_score(y_val, y_val_pred)
            val_acc = accuracy_score(y_val, y_val_pred)
            
            # Log results
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("best_cv_f1", search.best_score_)
            mlflow.log_metric("validation_f1", val_f1)
            mlflow.log_metric("validation_accuracy", val_acc)
            mlflow.log_metric("optimization_time_minutes", optimization_time/60)
            
            # Log model dan scaler jika ada
            if model_name == 'Logistic Regression':
                mlflow.sklearn.log_model(best_model, "optimized_model")
                # Note: In production, you'd want to save the scaler too
            else:
                mlflow.sklearn.log_model(best_model, "optimized_model")
                
            # Register model to MLflow Model Registry
            run_id = mlflow.active_run().info.run_id
            # register model 
            mlflow.register_model(
                model_uri=f"runs:/{run_id}/model",
                name=os.getenv('MODEL_NAME')
            )
            
            # Save model locally
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = os.getenv("MODEL_NAME", "model")  # default jika tidak ada
            filename = f"{model_name}_{timestamp}.pkl"
            joblib.dump(best_model, f"{output_dir}/{filename}")
            
            results[model_name] = {
                'model': best_model,
                'scaler': scaler if model_name == 'Logistic Regression' else None,
                'params': search.best_params_,
                'cv_f1': search.best_score_,
                'val_f1': val_f1,
                'val_acc': val_acc,
                'optimization_time': optimization_time
            }
            
            print(f"      ✓ CV F1: {search.best_score_:.4f}")
            print(f"      ✓ Val F1: {val_f1:.4f}")
            print(f"      ✓ Time: {optimization_time/60:.1f}min")
    
    return results

# ============================================================================
# 4. ENSEMBLE OPTIMIZATION
# ============================================================================

def create_ensemble_model(rf_results, multi_model_results, X_train, y_train, X_val, y_val):
    """Create ensemble model dari best performers"""
    print("\n5. Ensemble Model Creation...")
    
    # Collect best models
    all_models = {}
    
    # Add RF models
    if 'comprehensive_grid' in rf_results:
        all_models['RF_Grid'] = rf_results['comprehensive_grid']
    if 'optuna' in rf_results:
        all_models['RF_Optuna'] = rf_results['optuna']
    
    # Add other models
    all_models.update(multi_model_results)
    
    # Sort by validation F1
    sorted_models = sorted(all_models.items(), key=lambda x: x[1]['val_f1'], reverse=True)
    
    print(f"  Model Performance Ranking:")
    for i, (name, result) in enumerate(sorted_models, 1):
        print(f"   {i}. {name}: Val F1 = {result['val_f1']:.4f}")
    
    # Simple ensemble: Average predictions dari top 3 models
    top_3_models = sorted_models[:3]
    
    with mlflow.start_run(run_name=f"Ensemble_Top3_{datetime.now().strftime('%H%M%S')}", nested=True):
        mlflow.log_param("ensemble_method", "average_predictions")
        mlflow.log_param("n_models", len(top_3_models))
        mlflow.log_param("models_used", [name for name, _ in top_3_models])
        
        # Get predictions dari each model
        predictions = []
        for name, result in top_3_models:
            model = result['model']
            if result.get('scaler'):  # Logistic regression
                X_val_scaled = result['scaler'].transform(X_val)
                pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            else:
                pred_proba = model.predict_proba(X_val)[:, 1]
            predictions.append(pred_proba)
        
        # Average predictions
        ensemble_pred_proba = np.mean(predictions, axis=0)
        ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
        
        # Evaluate ensemble
        ensemble_f1 = f1_score(y_val, ensemble_pred)
        ensemble_acc = accuracy_score(y_val, ensemble_pred)
        ensemble_auc = roc_auc_score(y_val, ensemble_pred_proba)
        
        # Log ensemble results
        mlflow.log_metric("ensemble_validation_f1", ensemble_f1)
        mlflow.log_metric("ensemble_validation_accuracy", ensemble_acc)
        mlflow.log_metric("ensemble_validation_auc", ensemble_auc)
        
        # Log individual model contributions
        for i, (name, result) in enumerate(top_3_models):
            mlflow.log_metric(f"model_{i+1}_val_f1", result['val_f1'])
            mlflow.log_param(f"model_{i+1}_name", name)
        
        print(f"  Ensemble Results:")
        print(f"   Validation F1: {ensemble_f1:.4f}")
        print(f"   Validation Accuracy: {ensemble_acc:.4f}")
        print(f"   Validation AUC: {ensemble_auc:.4f}")
        
        # Compare dengan best individual model
        best_individual_f1 = top_3_models[0][1]['val_f1']
        improvement = ensemble_f1 - best_individual_f1
        print(f"   Improvement over best individual: {improvement:+.4f}")
        
        ensemble_result = {
            'models': top_3_models,
            'val_f1': ensemble_f1,
            'val_acc': ensemble_acc,
            'val_auc': ensemble_auc,
            'improvement': improvement
        }
        
        mlflow.log_metric("improvement_over_best", improvement)
    
    return ensemble_result

# ============================================================================
# 5. FINAL EVALUATION & MODEL SELECTION
# ============================================================================

def final_evaluation(all_results, baseline, X_test, y_test):
    """Final evaluation pada test set"""
    print("\n6. Final Evaluation on Test Set...")
    
    # Collect all models untuk evaluation
    models_to_evaluate = []
    
    # Add RF results
    rf_results = all_results.get('rf_optimization', {})
    if 'comprehensive_grid' in rf_results:
        models_to_evaluate.append(('RF_Comprehensive', rf_results['comprehensive_grid']))
    if 'optuna' in rf_results:
        models_to_evaluate.append(('RF_Optuna', rf_results['optuna']))
    
    # Add multi-model results
    multi_results = all_results.get('multi_model', {})
    for name, result in multi_results.items():
        models_to_evaluate.append((name, result))
    
    # Evaluate each model on test set
    test_results = []
    
    print(f"   Test Set Evaluation:")
    print("-" * 80)
    print(f"{'Model':<20} {'Test F1':<10} {'Test Acc':<10} {'Improvement':<12}")
    print("-" * 80)
    
    for model_name, result in models_to_evaluate:
        model = result['model']
        
        # Handle scaling jika needed
        if result.get('scaler'):
            X_test_scaled = result['scaler'].transform(X_test)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        test_f1 = f1_score(y_test, y_pred)
        test_acc = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred)
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Calculate improvement over baseline
        f1_improvement = test_f1 - baseline['f1']
        
        test_results.append({
            'name': model_name,
            'model': model,
            'test_f1': test_f1,
            'test_acc': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_auc': test_auc,
            'f1_improvement': f1_improvement,
            'validation_f1': result['val_f1']
        })
        
        print(f"{model_name:<20} {test_f1:<10.4f} {test_acc:<10.4f} {f1_improvement:+10.4f}")
    
    print("-" * 80)
    
    # Find best model
    best_model_result = max(test_results, key=lambda x: x['test_f1'])
    
    print(f"\nBEST MODEL: {best_model_result['name']}")
    print(f"   Test F1-Score: {best_model_result['test_f1']:.4f}")
    print(f"   Test Accuracy: {best_model_result['test_acc']:.4f}")
    print(f"   Test Precision: {best_model_result['test_precision']:.4f}")
    print(f"   Test Recall: {best_model_result['test_recall']:.4f}")
    print(f"   Test AUC: {best_model_result['test_auc']:.4f}")
    print(f"   Improvement over baseline: {best_model_result['f1_improvement']:+.4f}")
    
    # Log best model results ke MLflow
    with mlflow.start_run(run_name=f"BEST_MODEL_FINAL_{datetime.now().strftime('%H%M%S')}", nested=True):
        mlflow.log_param("model_name", best_model_result['name'])
        mlflow.log_param("optimization_completed", True)
        mlflow.log_metric("final_test_f1", best_model_result['test_f1'])
        mlflow.log_metric("final_test_accuracy", best_model_result['test_acc'])
        mlflow.log_metric("final_test_precision", best_model_result['test_precision'])
        mlflow.log_metric("final_test_recall", best_model_result['test_recall'])
        mlflow.log_metric("final_test_auc", best_model_result['test_auc'])
        mlflow.log_metric("improvement_over_baseline", best_model_result['f1_improvement'])
        mlflow.log_metric("baseline_f1", baseline['f1'])
        
        # Save best model
        mlflow.sklearn.log_model(best_model_result['model'], "production_ready_model")
        
        # Create comparison report
        comparison_data = []
        for result in test_results:
            comparison_data.append({
                'model': result['name'],
                'test_f1': result['test_f1'],
                'test_accuracy': result['test_acc'],
                'validation_f1': result['validation_f1'],
                'f1_improvement': result['f1_improvement']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(f"{output_dir}/model_optimization_results.csv", index=False)
        mlflow.log_artifact(f"{output_dir}/model_optimization_results.csv")
    
    return best_model_result, test_results

# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

def main():
    """Main optimization pipeline"""
    
    # Setup
    dagshub_url = setup_dagshub_mlflow()
    
    # Load data
    data_results = load_and_prepare_data()
    if data_results[0] is None:
        print("Cannot proceed without data")
        return
    
    X_train, X_test, y_train, y_test, X_train_opt, X_val, y_train_opt, y_val = data_results
    
    # Get baseline
    baseline = get_baseline_performance(X_train, X_test, y_train, y_test)
    
    # Start main optimization run
    with mlflow.start_run(run_name=f"Model_Optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("optimization_date", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        mlflow.log_param("baseline_f1", baseline['f1'])
        mlflow.log_param("target_improvement", "0.025+")  # Target: +2.5% F1
        mlflow.log_param("total_samples", len(X_train))
        mlflow.log_param("features", X_train.shape[1])
        
        all_results = {}
        
        # Random Forest optimization
        print(f"\nRANDOM FOREST OPTIMIZATION")
        print("=" * 50)
        rf_results = optimize_random_forest_advanced(X_train_opt, y_train_opt, X_val, y_val)
        all_results['rf_optimization'] = rf_results
        
        # Multi-model optimization  
        print(f"\nMULTI-MODEL OPTIMIZATION")
        print("=" * 50)
        multi_results = optimize_multiple_models(X_train_opt, y_train_opt, X_val, y_val)
        all_results['multi_model'] = multi_results
        
        # Ensemble creation
        print(f"\nENSEMBLE CREATION")
        print("=" * 50)
        ensemble_result = create_ensemble_model(rf_results, multi_results, X_train_opt, y_train_opt, X_val, y_val)
        all_results['ensemble'] = ensemble_result
        
        # Final evaluation
        print(f"\nFINAL EVALUATION")
        print("=" * 50)
        best_model, all_test_results = final_evaluation(all_results, baseline, X_test, y_test)
        
        # Log summary
        mlflow.log_metric("optimization_success", 1)
        mlflow.log_param("best_model_name", best_model['name'])
        mlflow.log_metric("final_improvement", best_model['f1_improvement'])
        
    print(f"\n" + "="*70)
    print("MODEL OPTIMIZATION COMPLETED!")
    print("="*70)
    print(f"\nOPTIMIZATION SUMMARY:")
    print(f"   Baseline F1-Score: {baseline['f1']:.4f}")
    print(f"   Best F1-Score: {best_model['test_f1']:.4f}")
    print(f"   Total Improvement: {best_model['f1_improvement']:+.4f}")
    print(f"   Best Model: {best_model['name']}")
    
    target_met = best_model['f1_improvement'] >= 0.025
    print(f"\nTARGET ACHIEVEMENT:")
    print(f"   Target: +0.025 F1-Score improvement")
    print(f"   Achieved: {best_model['f1_improvement']:+.4f}")
    print(f"   Status: {'TARGET MET!' if target_met else 'Target not fully met, but progress made'}")
    
    print(f"\nDagsHub Integration:")
    if dagshub_url:
        print(f"   Repository: {dagshub_url}")
        print(f"   View results in Experiments tab")
        print(f"   All models and metrics tracked")
    
    print(f"\nNext Steps:")
    print("   1. Review best model performance in DagsHub")
    print("   2. Deploy production_ready_model untuk production")
    print("   3. Setup monitoring untuk model performance")
    print("   4. Consider A/B testing against current system")

if __name__ == "__main__":
    main()