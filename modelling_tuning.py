import pandas as pd
import numpy as np
import os
import sys
import time
import joblib
import warnings
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, cross_val_score, 
                                   StratifiedKFold, train_test_split)
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score,
                           roc_auc_score, classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.5, help="Regularization strength (alpha)")
parser.add_argument("--l1_ratio", type=float, default=0.1, help="ElasticNet mixing parameter (l1_ratio)")
args = parser.parse_args()

alpha = args.alpha
l1_ratio = args.l1_ratio

print("="*70)
print("COMPREHENSIVE MODEL OPTIMIZATION - ROBUST VERSION")
print("="*70)

# ============================================================================
# 0. ROBUST CONFIGURATION & SETUP
# ============================================================================

class Config:
    """Configuration class untuk centralized settings"""
    def __init__(self):
        # Directories
        self.output_dir = "output"
        self.data_dir = "final_dataset"
        
        # Model settings
        self.random_state = int(os.getenv('RANDOM_STATE', 42))
        self.test_size = 0.2
        self.cv_folds = 5
        
        # Enhanced optimization settings - dapat dioverride dari MLflow parameters
        self.n_iter_random = 100        
        self.n_iter_optuna = int(os.getenv('N_ITER_OPTUNA', 200))        
        self.grid_search_iter = int(os.getenv('GRID_SEARCH_ITER', 200))     
        
        # Target improvement
        self.target_improvement = 0.025
        
        # ROBUST SETTINGS - dapat dioverride dari MLflow parameters
        self.min_approval_rate = float(os.getenv('MIN_APPROVAL_RATE', 0.3))
        self.max_approval_rate = float(os.getenv('MAX_APPROVAL_RATE', 0.8))
        self.target_approval_rate = float(os.getenv('TARGET_APPROVAL_RATE', 0.6))
        
        # ADVANCED OPTIMIZATION SETTINGS - NEW
        self.use_advanced_features = True      # Enable advanced feature engineering
        self.use_feature_selection = True      # Enable feature selection
        self.use_ensemble_methods = True       # Enable ensemble optimization
        self.use_stacking = True               # Enable stacking ensemble
        self.feature_selection_k = 20          # Number of features to select
        self.ensemble_models = 5               # Number of models in ensemble
        
        # MLflow settings - dapat dioverride dari MLflow parameters  
        self.experiment_name = os.getenv('EXPERIMENT_NAME', 'credit-approval-prediction')
        self.model_name = os.getenv('MODEL_NAME', 'credit_approval_model')
        
        # DagsHub settings (corrected)
        self.dagshub_url = os.getenv('DAGSHUB_REPO_URL')
        self.dagshub_username = os.getenv('DAGSHUB_USERNAME')
        self.dagshub_token = os.getenv('DAGSHUB_TOKEN')
        
        print(f"Configuration loaded:")
        print(f"  DagsHub URL: {self.dagshub_url}")
        print(f"  Username: {'✓' if self.dagshub_username else '❌'}")
        print(f"  Token: {'✓' if self.dagshub_token else '❌'}")
        print(f"  Target Approval Rate: {self.target_approval_rate:.1%}")
        print(f"  Optuna Iterations: {self.n_iter_optuna}")
        print(f"  Random State: {self.random_state}")
        
        # Ensure directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)

def setup_logging():
    """Setup logging untuk debugging"""
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def setup_mlflow_with_robust_fallback(config):
    """Setup MLflow dengan super robust fallback untuk DagsHub issues"""
    try:
        import mlflow
        import mlflow.sklearn
        print("✓ MLflow imported successfully")
    except ImportError:
        print("MLflow not installed. Install with: pip install mlflow")
        return None, False, "none"
    
    # Strategy 1: Try DagsHub if credentials available
    if config.dagshub_url and config.dagshub_username and config.dagshub_token:
        print("Attempting DagsHub connection...")
        
        try:
            # Import and initialize DagsHub (corrected repo name)
            # try:
            #     import dagshub
            #     # Use the correct repo name from URL
            #     dagshub.init(
            #         repo_owner='agusprasetyo811',
            #         repo_name='kredit_pinjaman_1',  # Corrected from 'kredit_pinjaman'
            #         mlflow=True
            #     )
            #     print("✓ DagsHub initialized with correct repo name")
            # except Exception as e:
            #     print(f"DagsHub init warning: {e}")
            #     print("   Continuing with manual setup...")
            
            # Setup environment variables
            os.environ['MLFLOW_TRACKING_USERNAME'] = config.dagshub_username
            os.environ['MLFLOW_TRACKING_PASSWORD'] = config.dagshub_token
            
            # Configure tracking URI (ensure .mlflow suffix)  
            dagshub_tracking_uri = config.dagshub_url
            if not dagshub_tracking_uri.endswith('.mlflow'):
                dagshub_tracking_uri += '.mlflow'
            
            print(f"Setting tracking URI: {dagshub_tracking_uri}")
            mlflow.set_tracking_uri(dagshub_tracking_uri)
            
            # Test connection with improved timeout and error handling
            def test_dagshub_connection_robust():
                """Robust connection test with multiple fallback strategies"""
                try:
                    # Set short timeout for connection test
                    import socket
                    original_timeout = socket.getdefaulttimeout()
                    socket.setdefaulttimeout(15)  # 15 second timeout
                    
                    # Test 1: Check if we can reach the tracking URI
                    current_uri = mlflow.get_tracking_uri()
                    print(f"   Testing connection to: {current_uri}")
                    
                    # Test 2: Try to get or create experiment with timeout
                    experiment_name = config.experiment_name
                    
                    # Use shorter operation timeouts
                    import signal
                    def timeout_handler(signum, frame):
                        raise TimeoutError("Operation timed out")
                    
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(10)  # 10 second timeout for each operation
                    
                    try:
                        # Try to get existing experiment
                        exp = mlflow.get_experiment_by_name(experiment_name)
                        if exp and exp.lifecycle_stage != 'deleted':
                            mlflow.set_experiment(experiment_name)
                            print(f"✓ Using existing experiment: {experiment_name}")
                        else:
                            # Try to create new experiment
                            exp_id = mlflow.create_experiment(experiment_name)
                            print(f"✓ Created new experiment: {experiment_name}")
                        
                        signal.alarm(0)  # Cancel timeout
                        
                        # Test 3: Try a simple run to verify full functionality
                        signal.alarm(15)  # 15 seconds for run test
                        
                        with mlflow.start_run(run_name="dagshub_connection_test"):
                            mlflow.log_param("connection_test", "success")
                            mlflow.log_metric("test_metric", 1.0)
                            run_id = mlflow.active_run().info.run_id
                            print(f"✓ Test run successful: {run_id[:8]}...")
                        
                        signal.alarm(0)  # Cancel timeout
                        socket.setdefaulttimeout(original_timeout)  # Restore timeout
                        
                        return True
                        
                    except TimeoutError:
                        signal.alarm(0)
                        print("   Connection test timed out")
                        return False
                    except Exception as e:
                        signal.alarm(0)
                        print(f"   Experiment operation failed: {e}")
                        return False
                        
                except Exception as e:
                    socket.setdefaulttimeout(original_timeout)  # Restore timeout
                    print(f"   Connection test failed: {e}")
                    return False
            
            # Perform connection test
            if test_dagshub_connection_robust():
                print("DagsHub MLflow connection successful!")
                return mlflow, True, "dagshub"
            else:
                raise Exception("DagsHub connection test failed")
                
        except Exception as e:
            print(f"DagsHub setup failed: {e}")
            print("Falling back to local MLflow...")
    
    # Strategy 2: Fallback to local MLflow with enhanced setup
    try:
        print("Setting up local MLflow with enhanced configuration...")
        
        # Create local MLflow directory with better structure
        local_mlruns_dir = os.path.abspath("./mlruns")
        os.makedirs(local_mlruns_dir, exist_ok=True)
        
        # Use file:// URI format
        local_uri = f"file://{local_mlruns_dir}"
        mlflow.set_tracking_uri(local_uri)
        
        # Setup experiment
        experiment_name = config.experiment_name
        try:
            exp = mlflow.get_experiment_by_name(experiment_name)
            if exp and exp.lifecycle_stage != 'deleted':
                mlflow.set_experiment(experiment_name)
                print(f"✓ Using existing local experiment: {experiment_name}")
            else:
                exp_id = mlflow.create_experiment(experiment_name)
                print(f"✓ Created new local experiment: {experiment_name}")
        except Exception as e:
            print(f"   Experiment setup warning: {e}")
            # Fallback to default experiment
            mlflow.set_experiment("Default")
            print("✓ Using default experiment")
        
        # Test local setup
        try:
            with mlflow.start_run(run_name="local_setup_test"):
                mlflow.log_param("setup_test", "local_success")
                mlflow.log_metric("test_value", 1.0)
            print("✓ Local MLflow test successful")
        except Exception as e:
            print(f"   Local test warning: {e}")
        
        print(f"Local MLflow setup successful!")
        print(f"   Tracking URI: {local_uri}")
        print(f"   MLruns directory: {local_mlruns_dir}")
        print(f"   Web UI: run 'mlflow ui' in terminal to view results")
        
        return mlflow, True, "local"
        
    except Exception as e:
        print(f"Local MLflow setup failed: {e}")
        print("Continuing without MLflow tracking...")
        return None, False, "none"

def setup_optuna_optional():
    """Setup Optuna dengan optional import"""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        print("✓ Optuna available for Bayesian optimization")
        return optuna, True
    except ImportError:
        print("Optuna not installed. Bayesian optimization will be skipped.")
        print("   Install with: pip install optuna")
        return None, False

# ============================================================================
# 1. ROBUST DATA LOADING WITH CLASS BALANCE ANALYSIS
# ============================================================================

def load_and_validate_data(config, logger):
    """Load dan validate data dengan robust error handling dan class balance analysis"""
    logger.info("Loading and validating data...")
    
    # Check file existence
    required_files = [
        f'{config.data_dir}/X_train.csv',
        f'{config.data_dir}/X_test.csv', 
        f'{config.data_dir}/y_train.csv',
        f'{config.data_dir}/y_test.csv'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return None
    
    try:
        # Load data
        X_train = pd.read_csv(f'{config.data_dir}/X_train.csv')
        X_test = pd.read_csv(f'{config.data_dir}/X_test.csv')
        y_train = pd.read_csv(f'{config.data_dir}/y_train.csv')['disetujui_encoded']
        y_test = pd.read_csv(f'{config.data_dir}/y_test.csv')['disetujui_encoded']
        
        # Validate data shapes
        assert X_train.shape[0] == len(y_train), "X_train dan y_train shape mismatch"
        assert X_test.shape[0] == len(y_test), "X_test dan y_test shape mismatch"
        assert X_train.shape[1] == X_test.shape[1], "Feature count mismatch between train and test"
        
        # ============================================================================
        # CRITICAL: CLASS BALANCE ANALYSIS
        # ============================================================================
        
        # Analyze class distribution
        train_class_counts = np.bincount(y_train)
        test_class_counts = np.bincount(y_test)
        
        train_approval_rate = train_class_counts[1] / len(y_train) if len(train_class_counts) > 1 else 0
        test_approval_rate = test_class_counts[1] / len(y_test) if len(test_class_counts) > 1 else 0
        
        logger.info(f"CLASS BALANCE ANALYSIS:")
        logger.info(f"  Training approval rate: {train_approval_rate:.3f} ({train_class_counts[1] if len(train_class_counts) > 1 else 0}/{len(y_train)})")
        logger.info(f"  Test approval rate: {test_approval_rate:.3f} ({test_class_counts[1] if len(test_class_counts) > 1 else 0}/{len(y_test)})")
        
        # Check for extreme imbalance
        if train_approval_rate < 0.1:
            logger.warning(f"EXTREME CLASS IMBALANCE: Only {train_approval_rate:.1%} approvals!")
            logger.warning(f"   This will cause overly conservative models")
            logger.warning(f"   Consider data augmentation or class weight adjustment")
        elif train_approval_rate > 0.9:
            logger.warning(f"EXTREME CLASS IMBALANCE: {train_approval_rate:.1%} approvals!")
            logger.warning(f"   This will cause overly permissive models")
        elif train_approval_rate < 0.3 or train_approval_rate > 0.8:
            logger.warning(f"SIGNIFICANT CLASS IMBALANCE: {train_approval_rate:.1%} approvals")
            logger.warning(f"   Will apply balanced class weights in training")
        
        # Calculate optimal class weights based on business logic
        if len(train_class_counts) > 1:
            # Conservative approach: slightly favor rejection to be safe
            n_samples = len(y_train)
            n_rejected = train_class_counts[0]
            n_approved = train_class_counts[1]
            
            # Business-oriented class weights (slightly conservative)
            target_approval_rate = config.target_approval_rate
            
            if train_approval_rate < target_approval_rate:
                # Increase approval weight
                class_weight_ratio = (1 - target_approval_rate) / target_approval_rate
                optimal_class_weights = {0: 1.0, 1: class_weight_ratio}
            else:
                # Decrease approval weight
                class_weight_ratio = target_approval_rate / (1 - target_approval_rate)
                optimal_class_weights = {0: class_weight_ratio, 1: 1.0}
            
            logger.info(f"OPTIMAL CLASS WEIGHTS (targeting {target_approval_rate:.1%} approval):")
            logger.info(f"  Class 0 (Reject): {optimal_class_weights[0]:.3f}")
            logger.info(f"  Class 1 (Approve): {optimal_class_weights[1]:.3f}")
        else:
            optimal_class_weights = None
            logger.warning("Cannot calculate class weights - only one class present")
        
        # Check for missing values
        if X_train.isnull().sum().sum() > 0:
            logger.warning("Training data contains missing values")
        if X_test.isnull().sum().sum() > 0:
            logger.warning("Test data contains missing values")
        
        # Create validation split
        X_train_opt, X_val, y_train_opt, y_val = train_test_split(
            X_train, y_train, 
            test_size=config.test_size, 
            random_state=config.random_state, 
            stratify=y_train
        )
        
        logger.info(f"Data loaded successfully:")
        logger.info(f"  Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        logger.info(f"  Testing: {X_test.shape[0]} samples")
        logger.info(f"  Optimization split: {X_train_opt.shape[0]} train, {X_val.shape[0]} validation")
        
        return {
            'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
            'X_train_opt': X_train_opt, 'X_val': X_val, 'y_train_opt': y_train_opt, 'y_val': y_val,
            'class_weights': optimal_class_weights,
            'train_approval_rate': train_approval_rate,
            'test_approval_rate': test_approval_rate
        }
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

# ============================================================================
# 2. BUSINESS LOGIC VALIDATION
# ============================================================================

def validate_model_business_logic(model, X_val, y_val, config, logger, model_name="Model"):
    """Validate model follows business logic rules"""
    logger.info(f"Business logic validation for {model_name}...")
    
    try:
        # Make predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
        
        # Calculate approval rate
        approval_rate = np.mean(y_pred)
        
        # Business logic checks
        validation_results = {
            'approval_rate': approval_rate,
            'passes_business_logic': True,
            'warnings': [],
            'critical_issues': []
        }
        
        # Check 1: Approval rate sanity
        if approval_rate < config.min_approval_rate:
            validation_results['critical_issues'].append(f"Approval rate {approval_rate:.1%} too low (min: {config.min_approval_rate:.1%})")
            validation_results['passes_business_logic'] = False
        elif approval_rate > config.max_approval_rate:
            validation_results['critical_issues'].append(f"Approval rate {approval_rate:.1%} too high (max: {config.max_approval_rate:.1%})")
            validation_results['passes_business_logic'] = False
        elif approval_rate < 0.4:
            validation_results['warnings'].append(f"Approval rate {approval_rate:.1%} quite conservative")
        elif approval_rate > 0.75:
            validation_results['warnings'].append(f"Approval rate {approval_rate:.1%} quite permissive")
        
        # Check 2: Probability distribution sanity
        if y_pred_proba is not None:
            avg_max_prob = np.mean(np.max(y_pred_proba, axis=1))
            min_approval_prob = np.min(y_pred_proba[:, 1])
            max_approval_prob = np.max(y_pred_proba[:, 1])
            
            if avg_max_prob > 0.95:
                validation_results['warnings'].append(f"Model too confident (avg: {avg_max_prob:.3f})")
            if max_approval_prob < 0.1:
                validation_results['critical_issues'].append(f"No cases have >10% approval probability")
                validation_results['passes_business_logic'] = False
            if min_approval_prob > 0.9:
                validation_results['critical_issues'].append(f"All cases have >90% approval probability")
                validation_results['passes_business_logic'] = False
        
        # Log results
        if validation_results['passes_business_logic']:
            logger.info(f"  {model_name} passes business logic validation")
            logger.info(f"     Approval rate: {approval_rate:.1%}")
        else:
            logger.warning(f"  {model_name} FAILS business logic validation")
            for issue in validation_results['critical_issues']:
                logger.warning(f"     {issue}")
        
        for warning in validation_results['warnings']:
            logger.warning(f"     {warning}")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Business logic validation failed: {e}")
        return {'passes_business_logic': False, 'approval_rate': 0, 'error': str(e)}

# ============================================================================
# 3. BASELINE PERFORMANCE WITH VALIDATION
# ============================================================================

def get_baseline_performance(data, config, logger, mlflow=None):
    """Get baseline performance untuk comparison dengan business validation"""
    logger.info("Calculating baseline performance...")
    
    try:
        # Multiple baseline strategies
        baseline_configs = [
            {
                'name': 'Default RF',
                'model': RandomForestClassifier(random_state=config.random_state, n_estimators=100),
                'class_weight': None
            },
            {
                'name': 'Balanced RF', 
                'model': RandomForestClassifier(random_state=config.random_state, n_estimators=100, class_weight='balanced'),
                'class_weight': 'balanced'
            }
        ]
        
        # Add optimal class weight baseline if available
        if data.get('class_weights'):
            baseline_configs.append({
                'name': 'Optimal Weight RF',
                'model': RandomForestClassifier(random_state=config.random_state, n_estimators=100, class_weight=data['class_weights']),
                'class_weight': data['class_weights']
            })
        
        best_baseline = None
        best_baseline_score = 0
        
        for baseline_config in baseline_configs:
            logger.info(f"  Testing {baseline_config['name']}...")
            
            model = baseline_config['model']
            model.fit(data['X_train'], data['y_train'])
            
            # Predictions
            y_pred_baseline = model.predict(data['X_test'])
            
            # Metrics
            baseline_f1 = f1_score(data['y_test'], y_pred_baseline)
            baseline_acc = accuracy_score(data['y_test'], y_pred_baseline)
            baseline_cv = cross_val_score(
                model, data['X_train'], data['y_train'], 
                cv=config.cv_folds, scoring='f1'
            ).mean()
            
            # Business logic validation
            business_validation = validate_model_business_logic(
                model, data['X_val'], data['y_val'], config, logger, baseline_config['name']
            )
            
            logger.info(f"    F1: {baseline_f1:.4f}, Acc: {baseline_acc:.4f}, CV: {baseline_cv:.4f}")
            logger.info(f"    Approval Rate: {business_validation['approval_rate']:.1%}")
            logger.info(f"    Business Logic: {'PASS' if business_validation['passes_business_logic'] else 'FAIL'}")
            
            # Select best baseline that passes business logic
            if business_validation['passes_business_logic'] and baseline_f1 > best_baseline_score:
                best_baseline = {
                    'model': model,
                    'name': baseline_config['name'],
                    'f1': baseline_f1,
                    'accuracy': baseline_acc,
                    'cv_f1': baseline_cv,
                    'class_weight': baseline_config['class_weight'],
                    'business_validation': business_validation
                }
                best_baseline_score = baseline_f1
        
        if best_baseline is None:
            logger.warning("No baseline model passes business logic! Using default...")
            # Fallback to default
            model = RandomForestClassifier(random_state=config.random_state, n_estimators=100)
            model.fit(data['X_train'], data['y_train'])
            y_pred_baseline = model.predict(data['X_test'])
            
            best_baseline = {
                'model': model,
                'name': 'Default RF (Fallback)',
                'f1': f1_score(data['y_test'], y_pred_baseline),
                'accuracy': accuracy_score(data['y_test'], y_pred_baseline),
                'cv_f1': cross_val_score(model, data['X_train'], data['y_train'], cv=config.cv_folds, scoring='f1').mean(),
                'class_weight': None,
                'business_validation': validate_model_business_logic(model, data['X_val'], data['y_val'], config, logger, 'Default RF')
            }
        
        logger.info(f"Selected Baseline: {best_baseline['name']}")
        logger.info(f"  Test F1-Score: {best_baseline['f1']:.4f}")
        logger.info(f"  Test Accuracy: {best_baseline['accuracy']:.4f}")
        logger.info(f"  CV F1-Score: {best_baseline['cv_f1']:.4f}")
        logger.info(f"  Class Weight: {best_baseline['class_weight']}")
        
        # Safe MLflow logging
        if mlflow:
            try:
                with mlflow.start_run(run_name="Baseline_Performance", nested=True):
                    mlflow.log_param("model_type", best_baseline['name'])
                    mlflow.log_param("class_weight", str(best_baseline['class_weight']))
                    mlflow.log_metric("baseline_test_f1", best_baseline['f1'])
                    mlflow.log_metric("baseline_test_accuracy", best_baseline['accuracy'])
                    mlflow.log_metric("baseline_cv_f1", best_baseline['cv_f1'])
                    mlflow.log_metric("baseline_approval_rate", best_baseline['business_validation']['approval_rate'])
            except Exception as e:
                logger.warning(f"MLflow baseline logging failed: {e}")
        
        return best_baseline
        
    except Exception as e:
        logger.error(f"Error calculating baseline: {e}")
        return None

# ============================================================================
# 4. ROBUST RANDOM FOREST OPTIMIZATION
# ============================================================================

def optimize_random_forest(data, config, logger, mlflow=None, optuna=None):
    """Optimize Random Forest dengan business logic constraints"""
    logger.info("Starting Random Forest optimization with business constraints...")
    
    results = {}
    
    # Strategy 1: Conservative Randomized Search
    try:
        logger.info("  Strategy 1: Conservative Randomized Search...")
        
        # More conservative parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],  # Reduced range
            'max_depth': [10, 15, 20, 25],    # Avoid None untuk prevent overfitting
            'min_samples_split': [2, 5, 10],  # Conservative values
            'min_samples_leaf': [2, 4, 6],    # Prevent overfitting (min 2)
            'max_features': ['sqrt', 'log2'],  # Conservative feature selection
            'bootstrap': [True],               # Always use bootstrap
            'class_weight': [None, 'balanced'] # Test both approaches
        }
        
        # Add optimal class weight if available
        if data.get('class_weights'):
            param_grid['class_weight'].append(data['class_weights'])
        
        # Custom scoring function yang considers business logic
        def business_aware_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            approval_rate = np.mean(y_pred)
            f1 = f1_score(y, y_pred)
            
            # Penalty for extreme approval rates
            if approval_rate < config.min_approval_rate or approval_rate > config.max_approval_rate:
                return f1 * 0.5  # Heavy penalty
            elif approval_rate < 0.4 or approval_rate > 0.75:
                return f1 * 0.8  # Moderate penalty
            else:
                return f1
        
        rf_search = RandomizedSearchCV(
            RandomForestClassifier(random_state=config.random_state),
            param_grid,
            n_iter=config.grid_search_iter,
            cv=config.cv_folds,
            scoring=business_aware_scorer,  # Use business-aware scoring
            n_jobs=-1,
            random_state=config.random_state,
            verbose=0
        )
        
        start_time = time.time()
        rf_search.fit(data['X_train_opt'], data['y_train_opt'])
        optimization_time = time.time() - start_time
        
        # Evaluate on validation set
        best_rf = rf_search.best_estimator_
        y_val_pred = best_rf.predict(data['X_val'])
        val_f1 = f1_score(data['y_val'], y_val_pred)
        val_acc = accuracy_score(data['y_val'], y_val_pred)
        
        # Business logic validation
        business_validation = validate_model_business_logic(
            best_rf, data['X_val'], data['y_val'], config, logger, "RF_RandomizedSearch"
        )
        
        if business_validation['passes_business_logic']:
            results['randomized_search'] = {
                'model': best_rf,
                'params': rf_search.best_params_,
                'cv_f1': rf_search.best_score_,
                'val_f1': val_f1,
                'val_acc': val_acc,
                'optimization_time': optimization_time,
                'business_validation': business_validation
            }
            
            logger.info(f"    Best CV F1: {rf_search.best_score_:.4f}")
            logger.info(f"    Validation F1: {val_f1:.4f}")
            logger.info(f"    Approval Rate: {business_validation['approval_rate']:.1%}")
        else:
            logger.warning(f"    Model failed business validation, skipping...")
        
        # Safe MLflow logging
        if mlflow:
            try:
                with mlflow.start_run(run_name=f"RF_RandomizedSearch_{datetime.now().strftime('%H%M%S')}", nested=True):
                    mlflow.log_param("optimization_strategy", "conservative_randomized_search")
                    mlflow.log_params(rf_search.best_params_)
                    mlflow.log_metric("best_cv_f1", rf_search.best_score_)
                    mlflow.log_metric("validation_f1", val_f1)
                    mlflow.log_metric("validation_accuracy", val_acc)
                    mlflow.log_metric("approval_rate", business_validation['approval_rate'])
                    mlflow.log_metric("passes_business_logic", business_validation['passes_business_logic'])
                    mlflow.log_metric("optimization_time_minutes", optimization_time/60)
                    if business_validation['passes_business_logic']:
                        mlflow.sklearn.log_model(best_rf, "optimized_random_forest")
            except Exception as e:
                logger.warning(f"MLflow RF logging failed: {e}")
        
    except Exception as e:
        logger.error(f"Randomized search failed: {e}")
    
    # Strategy 2: Business-Constrained Optuna Optimization
    if optuna and data.get('class_weights'):
        try:
            logger.info("  Strategy 2: Business-Constrained Optuna Optimization...")
            
            def business_constrained_rf_objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'max_depth': trial.suggest_int('max_depth', 10, 25),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 8),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                    'bootstrap': True,
                    'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced', data['class_weights']])
                }
                
                model = RandomForestClassifier(random_state=config.random_state, **params)
                
                # Use smaller CV for speed but validate business logic
                cv_scores = cross_val_score(model, data['X_train_opt'], data['y_train_opt'], cv=3, scoring='f1')
                
                # Quick business logic check
                model.fit(data['X_train_opt'], data['y_train_opt'])
                y_val_pred = model.predict(data['X_val'])
                approval_rate = np.mean(y_val_pred)
                
                # Apply business constraints
                base_score = cv_scores.mean()
                if approval_rate < config.min_approval_rate or approval_rate > config.max_approval_rate:
                    return base_score * 0.3  # Heavy penalty
                elif approval_rate < 0.4 or approval_rate > 0.75:
                    return base_score * 0.7  # Moderate penalty
                else:
                    return base_score
            
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
            start_time = time.time()
            study.optimize(business_constrained_rf_objective, n_trials=config.n_iter_optuna, show_progress_bar=False)
            optimization_time = time.time() - start_time
            
            # Train best model
            best_rf_optuna = RandomForestClassifier(random_state=config.random_state, **study.best_params)
            best_rf_optuna.fit(data['X_train_opt'], data['y_train_opt'])
            
            # Evaluate
            y_val_pred = best_rf_optuna.predict(data['X_val'])
            val_f1 = f1_score(data['y_val'], y_val_pred)
            val_acc = accuracy_score(data['y_val'], y_val_pred)
            
            # Business logic validation
            business_validation = validate_model_business_logic(
                best_rf_optuna, data['X_val'], data['y_val'], config, logger, "RF_Optuna"
            )
            
            if business_validation['passes_business_logic']:
                results['optuna'] = {
                    'model': best_rf_optuna,
                    'params': study.best_params,
                    'cv_f1': study.best_value,
                    'val_f1': val_f1,
                    'val_acc': val_acc,
                    'optimization_time': optimization_time,
                    'business_validation': business_validation
                }
                
                logger.info(f"    Best CV F1: {study.best_value:.4f}")
                logger.info(f"    Validation F1: {val_f1:.4f}")
                logger.info(f"    Approval Rate: {business_validation['approval_rate']:.1%}")
            else:
                logger.warning(f"    Optuna model failed business validation, skipping...")
            
            # Safe MLflow logging
            if mlflow:
                try:
                    with mlflow.start_run(run_name=f"RF_Optuna_{datetime.now().strftime('%H%M%S')}", nested=True):
                        mlflow.log_param("optimization_strategy", "business_constrained_optuna")
                        mlflow.log_params(study.best_params)
                        mlflow.log_metric("best_cv_f1", study.best_value)
                        mlflow.log_metric("validation_f1", val_f1)
                        mlflow.log_metric("validation_accuracy", val_acc)
                        mlflow.log_metric("approval_rate", business_validation['approval_rate'])
                        mlflow.log_metric("passes_business_logic", business_validation['passes_business_logic'])
                        mlflow.log_metric("optimization_time_minutes", optimization_time/60)
                        mlflow.log_metric("n_trials", len(study.trials))
                        if business_validation['passes_business_logic']:
                            mlflow.sklearn.log_model(best_rf_optuna, "optuna_random_forest")
                except Exception as e:
                    logger.warning(f"MLflow Optuna logging failed: {e}")
        
        except Exception as e:
            logger.error(f"Optuna optimization failed: {e}")
    
    return results

# ============================================================================
# 5. MODEL EVALUATION & SELECTION WITH BUSINESS VALIDATION
# ============================================================================

def evaluate_all_models(rf_results, baseline, data, config, logger):
    """Evaluate all optimized models dan select best dengan business validation"""
    logger.info("Evaluating all optimized models with business validation...")
    
    # Collect all models that passed business validation
    all_models = {}
    
    # Add RF results
    for strategy, result in rf_results.items():
        if result.get('business_validation', {}).get('passes_business_logic', False):
            all_models[f'RF_{strategy}'] = result
    
    if not all_models:
        logger.error("NO MODELS PASSED BUSINESS VALIDATION!")
        logger.error("This indicates serious issues with optimization or data")
        return None, []
    
    # Evaluate on test set
    test_results = []
    
    logger.info("Test Set Evaluation (Business-Valid Models Only):")
    logger.info("-" * 90)
    logger.info(f"{'Model':<25} {'Test F1':<10} {'Test Acc':<10} {'Approval%':<10} {'Improvement':<12}")
    logger.info("-" * 90)
    
    for model_name, result in all_models.items():
        try:
            model = result['model']
            
            # Predict on test set
            y_pred = model.predict(data['X_test'])
            
            # Calculate metrics
            test_f1 = f1_score(data['y_test'], y_pred)
            test_acc = accuracy_score(data['y_test'], y_pred)
            test_precision = precision_score(data['y_test'], y_pred)
            test_recall = recall_score(data['y_test'], y_pred)
            test_approval_rate = np.mean(y_pred)
            
            # Calculate improvement
            f1_improvement = test_f1 - baseline['f1']
            
            # Final business validation on test set
            test_business_validation = validate_model_business_logic(
                model, data['X_test'], data['y_test'], config, logger, f"{model_name}_test"
            )
            
            test_results.append({
                'name': model_name,
                'model': model,
                'test_f1': test_f1,
                'test_acc': test_acc,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_approval_rate': test_approval_rate,
                'f1_improvement': f1_improvement,
                'validation_f1': result['val_f1'],
                'params': result['params'],
                'test_business_validation': test_business_validation
            })
            
            logger.info(f"{model_name:<25} {test_f1:<10.4f} {test_acc:<10.4f} {test_approval_rate:<9.1%} {f1_improvement:+10.4f}")
            
        except Exception as e:
            logger.error(f"Evaluation failed for {model_name}: {e}")
    
    logger.info("-" * 90)
    
    # Find best model (among those that pass business validation on test set)
    valid_test_results = [r for r in test_results if r['test_business_validation']['passes_business_logic']]
    
    if valid_test_results:
        best_model_result = max(valid_test_results, key=lambda x: x['test_f1'])
        
        logger.info(f"\nBEST BUSINESS-VALID MODEL: {best_model_result['name']}")
        logger.info(f"  Test F1-Score: {best_model_result['test_f1']:.4f}")
        logger.info(f"  Test Accuracy: {best_model_result['test_acc']:.4f}")
        logger.info(f"  Test Precision: {best_model_result['test_precision']:.4f}")
        logger.info(f"  Test Recall: {best_model_result['test_recall']:.4f}")
        logger.info(f"  Test Approval Rate: {best_model_result['test_approval_rate']:.1%}")
        logger.info(f"  Improvement over baseline: {best_model_result['f1_improvement']:+.4f}")
        
        return best_model_result, test_results
    else:
        logger.error("NO MODELS PASS BUSINESS VALIDATION ON TEST SET!")
        logger.warning("Falling back to best F1 model regardless of business validation")
        if test_results:
            best_model_result = max(test_results, key=lambda x: x['test_f1'])
            logger.warning(f"FALLBACK MODEL: {best_model_result['name']} (F1: {best_model_result['test_f1']:.4f})")
            return best_model_result, test_results
        else:
            return None, []

# ============================================================================
# 6. ROBUST MODEL SAVING
# ============================================================================

def save_best_model(best_model_result, config, logger, mlflow=None):
    """Save best model dan metadata dengan business validation info"""
    logger.info("Saving best model with business validation metadata...")
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_filename = f"{config.output_dir}/best_model_tuned.pkl"
        joblib.dump(best_model_result['model'], model_filename)
        logger.info(f"Model saved: {model_filename}")
        
        # Save comprehensive model info
        info_filename = f"{config.output_dir}/model_info_tuned_{timestamp}.txt"
        with open(info_filename, 'w') as f:
            f.write(f"Credit Approval Model Information (Tuned)\n")
            f.write(f"=========================================\n\n")
            f.write(f"Best Model: {best_model_result['name']}\n")
            f.write(f"Test F1-Score: {best_model_result['test_f1']:.4f}\n")
            f.write(f"Test Accuracy: {best_model_result['test_acc']:.4f}\n")
            f.write(f"Test Approval Rate: {best_model_result['test_approval_rate']:.1%}\n")
            f.write(f"Training Date: {datetime.now()}\n\n")
            
            f.write(f"Business Validation Results:\n")
            if 'test_business_validation' in best_model_result:
                bv = best_model_result['test_business_validation']
                f.write(f"  Passes Business Logic: {bv['passes_business_logic']}\n")
                f.write(f"  Approval Rate: {bv['approval_rate']:.1%}\n")
                if bv.get('warnings'):
                    f.write(f"  Warnings: {', '.join(bv['warnings'])}\n")
                if bv.get('critical_issues'):
                    f.write(f"  Critical Issues: {', '.join(bv['critical_issues'])}\n")
            
            f.write(f"\nBest Parameters:\n")
            for param, value in best_model_result['params'].items():
                f.write(f"  {param}: {value}\n")
        
        logger.info(f"Model info saved: {info_filename}")
        
        # Safe MLflow logging
        if mlflow:
            try:
                with mlflow.start_run(run_name=f"PRODUCTION_MODEL_{timestamp}", nested=True):
                    mlflow.log_param("model_name", best_model_result['name'])
                    mlflow.log_param("production_ready", True)
                    mlflow.log_param("business_logic_validated", best_model_result.get('test_business_validation', {}).get('passes_business_logic', False))
                    mlflow.log_metric("final_test_f1", best_model_result['test_f1'])
                    mlflow.log_metric("final_test_accuracy", best_model_result['test_acc'])
                    mlflow.log_metric("final_approval_rate", best_model_result['test_approval_rate'])
                    mlflow.log_metric("final_improvement", best_model_result['f1_improvement'])
                    
                    # Log model
                    mlflow.sklearn.log_model(best_model_result['model'], "production_model")
                    
                    # Log files
                    mlflow.log_artifact(model_filename)
                    mlflow.log_artifact(info_filename)
            except Exception as e:
                logger.warning(f"MLflow production logging failed: {e}")
        
        return model_filename
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return None

# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def main():
    """Main optimization pipeline dengan business validation"""
    # Setup
    config = Config()
    logger = setup_logging()
    mlflow, mlflow_available, tracking_type = setup_mlflow_with_robust_fallback(config)
    optuna, optuna_available = setup_optuna_optional()
    
    logger.info("Starting robust model optimization with business validation...")
    logger.info(f"MLflow tracking: {tracking_type}")
    logger.info(f"Optuna available: {optuna_available}")
    logger.info(f"Business constraints:")
    logger.info(f"  Target approval rate: {config.target_approval_rate:.1%}")
    logger.info(f"  Min approval rate: {config.min_approval_rate:.1%}")
    logger.info(f"  Max approval rate: {config.max_approval_rate:.1%}")
    
    # Load data
    data = load_and_validate_data(config, logger)
    if not data:
        logger.error("Failed to load data. Exiting...")
        return
    
    # Get baseline
    baseline = get_baseline_performance(data, config, logger, mlflow)
    if not baseline:
        logger.error("Failed to calculate baseline. Exiting...")
        return
    
    # Start main optimization
    if mlflow_available:
        try:
            with mlflow.start_run(run_name=f"Robust_Model_Optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Safe parameter logging
                try:
                    mlflow.log_param("tracking_type", tracking_type)
                    mlflow.log_param("optimization_date", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    mlflow.log_param("baseline_f1", baseline['f1'])
                    mlflow.log_param("baseline_approval_rate", baseline['business_validation']['approval_rate'])
                    mlflow.log_param("target_improvement", config.target_improvement)
                    mlflow.log_param("target_approval_rate", config.target_approval_rate)
                    mlflow.log_param("total_samples", len(data['X_train']))
                    mlflow.log_param("features", data['X_train'].shape[1])
                    mlflow.log_param("train_approval_rate", data['train_approval_rate'])
                except Exception as e:
                    logger.warning(f"MLflow param logging failed: {e}")
                
                # Run optimizations
                rf_results = optimize_random_forest(data, config, logger, mlflow, optuna)
                
                # Evaluate and select best
                best_model, all_results = evaluate_all_models(rf_results, baseline, data, config, logger)
                
                if best_model:
                    # Save model
                    model_file = save_best_model(best_model, config, logger, mlflow)
                    
                    # Final logging
                    try:
                        mlflow.log_metric("optimization_success", 1)
                        mlflow.log_param("best_model_name", best_model['name'])
                        mlflow.log_metric("final_improvement", best_model['f1_improvement'])
                        mlflow.log_metric("final_approval_rate", best_model['test_approval_rate'])
                    except Exception as e:
                        logger.warning(f"MLflow final logging failed: {e}")
                else:
                    logger.error("No valid models found")
                    try:
                        mlflow.log_metric("optimization_success", 0)
                    except:
                        pass
        except Exception as e:
            logger.error(f"MLflow main run failed: {e}")
            # Continue without MLflow
            rf_results = optimize_random_forest(data, config, logger)
            best_model, all_results = evaluate_all_models(rf_results, baseline, data, config, logger)
            
            if best_model:
                model_file = save_best_model(best_model, config, logger)
    else:
        # Run without MLflow
        rf_results = optimize_random_forest(data, config, logger)
        best_model, all_results = evaluate_all_models(rf_results, baseline, data, config, logger)
        
        if best_model:
            model_file = save_best_model(best_model, config, logger)
    
    # Print final summary
    if best_model:
        logger.info("=" * 70)
        logger.info("ROBUST MODEL OPTIMIZATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info(f"OPTIMIZATION SUMMARY:")
        logger.info(f"  Baseline F1-Score: {baseline['f1']:.4f}")
        logger.info(f"  Baseline Approval Rate: {baseline['business_validation']['approval_rate']:.1%}")
        logger.info(f"  Best F1-Score: {best_model['test_f1']:.4f}")
        logger.info(f"  Best Approval Rate: {best_model['test_approval_rate']:.1%}")
        logger.info(f"  Total Improvement: {best_model['f1_improvement']:+.4f}")
        logger.info(f"  Best Model: {best_model['name']}")
        logger.info(f"  MLflow Tracking: {tracking_type}")
        
        target_met = best_model['f1_improvement'] >= config.target_improvement
        business_valid = best_model.get('test_business_validation', {}).get('passes_business_logic', False)
        
        logger.info(f"\nVALIDATION RESULTS:")
        logger.info(f"  F1 Target (+{config.target_improvement:.3f}): {'ACHIEVED' if target_met else 'NOT MET'} ({best_model['f1_improvement']:+.4f})")
        logger.info(f"  Business Logic: {'VALID' if business_valid else 'INVALID'}")
        logger.info(f"  Approval Rate: {'HEALTHY' if config.min_approval_rate <= best_model['test_approval_rate'] <= config.max_approval_rate else 'OUT OF RANGE'} ({best_model['test_approval_rate']:.1%})")
        
        if target_met and business_valid:
            logger.info(f"\nOPTIMIZATION FULLY SUCCESSFUL!")
            logger.info(f"   Model ready for production deployment")
        elif business_valid:
            logger.info(f"\nBUSINESS VALIDATION PASSED")
            logger.info(f"   Model safe for production despite F1 target not fully met")
        else:
            logger.warning(f"\nBUSINESS VALIDATION FAILED")
            logger.warning(f"   Model may need further tuning before production")
            
        if tracking_type == "local":
            logger.info(f"\nView results: run 'mlflow ui' in terminal")
        elif tracking_type == "dagshub":
            logger.info(f"\nView results: {config.dagshub_url}")
            
    else:
        logger.error("Optimization failed - no valid models produced")
        logger.error("Consider:")
        logger.error("  1. Adjusting business constraints")
        logger.error("  2. Using different dataset")
        logger.error("  3. Manual hyperparameter tuning")

if __name__ == "__main__":
    main()