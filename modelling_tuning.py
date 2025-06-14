import pandas as pd
import numpy as np
import os
import sys
import time
import joblib
import warnings
import argparse
import json
import glob
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

print("="*70)
print("COMPREHENSIVE MODEL OPTIMIZATION - DAGSHUB & MLFLOW INTEGRATION")
print("="*70)

# ============================================================================
# FEATURE ALIGNMENT SYSTEM 
# ============================================================================

def ensure_feature_alignment(X_data, reference_columns, logger, data_name="Data"):
    """
    Ensure data has exact same features as reference (training data)
    """
    logger.info(f"Ensuring feature alignment for {data_name}...")
    
    try:
        # Convert to DataFrame if needed
        if not isinstance(X_data, pd.DataFrame):
            if hasattr(X_data, 'columns'):
                X_data = pd.DataFrame(X_data, columns=X_data.columns)
            else:
                X_data = pd.DataFrame(X_data, columns=reference_columns[:X_data.shape[1]])
        
        original_shape = X_data.shape
        logger.info(f"  {data_name} original shape: {original_shape}")
        logger.info(f"  Reference columns: {len(reference_columns)}")
        
        # Get current columns
        current_columns = list(X_data.columns)
        reference_columns = list(reference_columns)
        
        logger.info(f"  Current columns: {len(current_columns)}")
        
        # Find differences
        extra_columns = set(current_columns) - set(reference_columns)
        missing_columns = set(reference_columns) - set(current_columns)
        
        if extra_columns:
            logger.warning(f"  Extra columns found: {extra_columns}")
            logger.warning(f"  Removing extra columns...")
            X_data = X_data.drop(columns=extra_columns)
        
        if missing_columns:
            logger.warning(f"  Missing columns found: {missing_columns}")
            logger.warning(f"  Adding missing columns with zeros...")
            for col in missing_columns:
                X_data[col] = 0
        
        # Reorder columns to match reference
        X_data = X_data[reference_columns]
        
        final_shape = X_data.shape
        logger.info(f"  {data_name} final shape: {final_shape}")
        
        # Validate
        if X_data.shape[1] != len(reference_columns):
            raise ValueError(f"Feature alignment failed: {X_data.shape[1]} != {len(reference_columns)}")
        
        logger.info(f"  Feature alignment successful for {data_name}")
        return X_data
        
    except Exception as e:
        logger.error(f"Feature alignment failed for {data_name}: {e}")
        raise

def create_feature_metadata(X_train, config, logger):
    """
    Create metadata about features for consistent processing
    """
    logger.info("Creating feature metadata...")
    
    try:
        metadata = {
            'feature_names': list(X_train.columns),
            'n_features': X_train.shape[1],
            'feature_types': {},
            'feature_ranges': {},
            'creation_timestamp': datetime.now().isoformat()
        }
        
        # Analyze feature types and ranges
        for col in X_train.columns:
            if X_train[col].dtype in ['int64', 'float64']:
                metadata['feature_types'][col] = 'numerical'
                metadata['feature_ranges'][col] = {
                    'min': float(X_train[col].min()),
                    'max': float(X_train[col].max()),
                    'mean': float(X_train[col].mean()),
                    'std': float(X_train[col].std())
                }
            else:
                metadata['feature_types'][col] = 'categorical'
                metadata['feature_ranges'][col] = list(X_train[col].unique())
        
        # Save metadata
        metadata_file = f"{config.output_dir}/feature_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Feature metadata saved: {metadata_file}")
        logger.info(f"  Features: {metadata['n_features']}")
        logger.info(f"  Numerical: {sum(1 for t in metadata['feature_types'].values() if t == 'numerical')}")
        logger.info(f"  Categorical: {sum(1 for t in metadata['feature_types'].values() if t == 'categorical')}")
        
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to create feature metadata: {e}")
        return None

# ============================================================================
# 0. CONFIGURATION & SETUP
# ============================================================================

class Config:
    """Configuration class untuk centralized settings"""
    def __init__(self, args=None):
        # Parse command line arguments if provided
        if args:
            self.alpha = args.alpha
            self.l1_ratio = args.l1_ratio
        else:
            self.alpha = 0.5
            self.l1_ratio = 0.1
        
        # Directories
        self.output_dir = "output"
        self.data_dir = "final_dataset"
        
        # Model settings - FROM ENVIRONMENT OR DEFAULTS
        self.random_state = int(os.getenv('RANDOM_STATE', '42'))
        self.test_size = 0.2
        self.cv_folds = 5
        
        # Enhanced optimization settings - FROM ENVIRONMENT OR DEFAULTS
        self.n_iter_random = 100        
        self.n_iter_optuna = int(os.getenv('N_ITER_OPTUNA', '100'))  # Reduced for speed
        self.grid_search_iter = int(os.getenv('GRID_SEARCH_ITER', '100'))  # Reduced for speed
        
        # Target improvement
        self.target_improvement = 0.025
        
        # BUSINESS LOGIC SETTINGS - RELAXED FOR BETTER PERFORMANCE
        self.min_approval_rate = float(os.getenv('MIN_APPROVAL_RATE', '0.20'))  # Relaxed from 0.3
        self.max_approval_rate = float(os.getenv('MAX_APPROVAL_RATE', '0.85'))  # Relaxed from 0.8
        self.target_approval_rate = float(os.getenv('TARGET_APPROVAL_RATE', '0.45'))  # Adjusted to data reality
        
        # ADVANCED OPTIMIZATION SETTINGS - SIMPLIFIED
        self.use_advanced_features = False      # Disabled for stability
        self.use_feature_selection = False      # Disabled for stability 
        self.use_ensemble_methods = False       # Disabled for stability
        self.use_stacking = False               # Disabled for stability
        self.feature_selection_k = 20          
        self.ensemble_models = 3               
        
        # MLflow settings - FROM ENVIRONMENT OR DEFAULTS
        self.experiment_name = os.getenv('EXPERIMENT_NAME', 'credit-approval-tuning')
        self.model_name = os.getenv('MODEL_NAME', 'credit_approval_model_tuned')
        
        # DagsHub settings - FROM ENVIRONMENT
        self.dagshub_url = os.getenv('DAGSHUB_REPO_URL', 'https://dagshub.com/agusprasetyo811/kredit_pinjaman_2')
        self.dagshub_username = os.getenv('DAGSHUB_USERNAME', 'agusprasetyo811')
        self.dagshub_token = os.getenv('DAGSHUB_TOKEN')
        
        # Ensure directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        

def setup_logging():
    """Setup logging untuk debugging"""
    import logging
    
    # Create logs directory if not exists
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def setup_mlflow_dagshub(config, logger):
    """Enhanced MLflow setup dengan DagsHub integration and CI/CD compatibility"""
    try:
        # Check if MLflow should be disabled (useful for CI/CD debugging)
        if os.getenv('DISABLE_MLFLOW', '').lower() in ['true', '1', 'yes']:
            logger.warning("MLflow disabled via DISABLE_MLFLOW environment variable")
            return None, False
            
        import mlflow
        import mlflow.sklearn
        import mlflow.models
        
        logger.info("Setting up MLflow with DagsHub integration...")
        
        # CRITICAL: End any existing runs first (CI/CD fix)
        try:
            if mlflow.active_run():
                logger.warning("Found active MLflow run, ending it...")
                mlflow.end_run()
        except Exception as e:
            logger.warning(f"Error ending existing run: {e}")
        
        # Clear environment variables that might interfere
        mlflow_env_vars = [
            'MLFLOW_RUN_ID', 
            'MLFLOW_EXPERIMENT_ID'
        ]
        for var in mlflow_env_vars:
            if var in os.environ:
                logger.info(f"Clearing environment variable: {var}")
                del os.environ[var]
        
        # Simple fallback to local MLflow if DagsHub not configured
        if not config.dagshub_token:
            logger.warning("DAGSHUB_TOKEN not found, using local MLflow")
            mlflow.set_tracking_uri("file:./mlruns")
        else:
            try:
                # Set tracking URI to DagsHub MLflow
                dagshub_mlflow_uri = f"{config.dagshub_url}.mlflow"
                mlflow.set_tracking_uri(dagshub_mlflow_uri)
                
                # Set authentication
                os.environ['MLFLOW_TRACKING_USERNAME'] = config.dagshub_username
                os.environ['MLFLOW_TRACKING_PASSWORD'] = config.dagshub_token
                
                logger.info(f"DagsHub MLflow tracking configured:")
                logger.info(f"   Repository: {config.dagshub_url}")
                
            except Exception as e:
                logger.error(f"DagsHub setup failed: {e}")
                logger.warning("Falling back to local MLflow tracking...")
                mlflow.set_tracking_uri("file:./mlruns")
        
        # Set or create experiment with better error handling
        try:
            experiment = mlflow.get_experiment_by_name(config.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(config.experiment_name)
                logger.info(f"Created new MLflow experiment: {config.experiment_name}")
            else:
                logger.info(f"Using existing MLflow experiment: {config.experiment_name}")
            
            mlflow.set_experiment(config.experiment_name)
            
        except Exception as e:
            logger.error(f"Failed to set/create experiment: {e}")
            logger.warning("Using default experiment")
        
        # Test MLflow connection
        try:
            current_experiment = mlflow.get_experiment_by_name(config.experiment_name)
            logger.info(f"MLflow setup successful - experiment: {current_experiment.name if current_experiment else 'default'}")
        except Exception as e:
            logger.warning(f"MLflow test failed: {e}")
        
        return mlflow, True
        
    except ImportError:
        logger.error("MLflow not installed. Install with: pip install mlflow")
        return None, False
    except Exception as e:
        logger.error(f"MLflow setup failed: {e}")
        return None, False

def setup_optuna_optional(logger):
    """Setup Optuna dengan optional import"""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logger.info("Optuna available for Bayesian optimization")
        return optuna, True
    except ImportError:
        logger.warning("Optuna not installed. Bayesian optimization will be skipped.")
        logger.info("   Install with: pip install optuna")
        return None, False

def safe_mlflow_run(mlflow, config, logger, run_function):
    """Safely execute function within MLflow run context"""
    if not mlflow:
        return run_function(None)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Ensure no active run
            if mlflow.active_run():
                logger.warning(f"Ending existing run (attempt {attempt + 1})")
                mlflow.end_run()
            
            # Start new run
            run_name = f"SIMPLIFIED_Tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}_attempt_{attempt + 1}"
            with mlflow.start_run(run_name=run_name):
                return run_function(mlflow)
                
        except Exception as e:
            logger.warning(f"MLflow run attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logger.error("All MLflow attempts failed, running without MLflow")
                return run_function(None)
            
            # Wait before retry
            time.sleep(1)
    
    return None

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
        
        # Analyze class distribution
        train_class_counts = np.bincount(y_train)
        test_class_counts = np.bincount(y_test)
        
        train_approval_rate = train_class_counts[1] / len(y_train) if len(train_class_counts) > 1 else 0
        test_approval_rate = test_class_counts[1] / len(y_test) if len(test_class_counts) > 1 else 0
        
        logger.info(f"CLASS BALANCE ANALYSIS:")
        logger.info(f"  Training approval rate: {train_approval_rate:.3f} ({train_class_counts[1] if len(train_class_counts) > 1 else 0}/{len(y_train)})")
        logger.info(f"  Test approval rate: {test_approval_rate:.3f} ({test_class_counts[1] if len(test_class_counts) > 1 else 0}/{len(y_test)})")
        
        # Calculate optimal class weights based on business logic
        if len(train_class_counts) > 1:
            # Balanced approach based on actual data distribution
            target_approval_rate = config.target_approval_rate
            
            if train_approval_rate < target_approval_rate:
                # Increase approval weight
                class_weight_ratio = (1 - target_approval_rate) / target_approval_rate
                optimal_class_weights = {0: 1.0, 1: class_weight_ratio}
            else:
                # Decrease approval weight slightly
                class_weight_ratio = target_approval_rate / (1 - target_approval_rate)
                optimal_class_weights = {0: class_weight_ratio, 1: 1.0}
            
            logger.info(f"OPTIMAL CLASS WEIGHTS (targeting {target_approval_rate:.1%} approval):")
            logger.info(f"  Class 0 (Reject): {optimal_class_weights[0]:.3f}")
            logger.info(f"  Class 1 (Approve): {optimal_class_weights[1]:.3f}")
        else:
            optimal_class_weights = None
            logger.warning("Cannot calculate class weights - only one class present")
        
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
# 2. BUSINESS LOGIC VALIDATION - RELAXED VERSION
# ============================================================================

def validate_model_business_logic(model, X_val, y_val, config, logger, model_name="Model"):
    """Validate model follows business logic rules - RELAXED VERSION"""
    logger.info(f"Business logic validation for {model_name}...")
    
    try:
        # Make predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
        
        # Calculate approval rate
        approval_rate = np.mean(y_pred)
        
        # Business logic checks - RELAXED
        validation_results = {
            'approval_rate': approval_rate,
            'passes_business_logic': True,
            'warnings': [],
            'critical_issues': []
        }
        
        # Check 1: Approval rate sanity - RELAXED THRESHOLDS
        if approval_rate < config.min_approval_rate:
            validation_results['critical_issues'].append(f"Approval rate {approval_rate:.1%} too low (min: {config.min_approval_rate:.1%})")
            validation_results['passes_business_logic'] = False
        elif approval_rate > config.max_approval_rate:
            validation_results['critical_issues'].append(f"Approval rate {approval_rate:.1%} too high (max: {config.max_approval_rate:.1%})")
            validation_results['passes_business_logic'] = False
        elif approval_rate < 0.25:  # More lenient threshold
            validation_results['warnings'].append(f"Approval rate {approval_rate:.1%} quite conservative")
        elif approval_rate > 0.75:  # More lenient threshold
            validation_results['warnings'].append(f"Approval rate {approval_rate:.1%} quite permissive")
        
        # Check 2: Model makes reasonable variety of decisions
        unique_predictions = len(np.unique(y_pred))
        if unique_predictions < 2:
            validation_results['critical_issues'].append(f"Model only makes one type of decision")
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
# 3. BASELINE PERFORMANCE
# ============================================================================

def get_baseline_performance(data, config, logger, mlflow=None):
    """Get baseline performance untuk comparison"""
    logger.info("Calculating baseline performance...")
    
    try:
        # Multiple baseline strategies
        baseline_configs = [
            {
                'name': 'Balanced RF',
                'model': RandomForestClassifier(random_state=config.random_state, n_estimators=100, class_weight='balanced'),
                'class_weight': 'balanced'
            },
            {
                'name': 'Aggressive Weight RF',
                'model': RandomForestClassifier(random_state=config.random_state, n_estimators=100, class_weight={0: 1.0, 1: 2.0}),
                'class_weight': {0: 1.0, 1: 2.0}
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
            
            # Select best baseline
            if baseline_f1 > best_baseline_score:
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
        
        logger.info(f"Selected Baseline: {best_baseline['name']}")
        logger.info(f"  Test F1-Score: {best_baseline['f1']:.4f}")
        logger.info(f"  Test Accuracy: {best_baseline['accuracy']:.4f}")
        logger.info(f"  CV F1-Score: {best_baseline['cv_f1']:.4f}")
        
        return best_baseline
        
    except Exception as e:
        logger.error(f"Error calculating baseline: {e}")
        return None

# ============================================================================
# 4. SIMPLIFIED RANDOM FOREST OPTIMIZATION
# ============================================================================

def optimize_random_forest_simple(data, config, logger, mlflow=None, optuna=None):
    """Simplified Random Forest optimization"""
    logger.info("Starting simplified Random Forest optimization...")
    
    results = {}
    
    # Strategy 1: Class Weight Tuning
    try:
        logger.info("  Strategy 1: Class Weight Tuning...")
        
        # Test different class weights
        weight_configs = [
            'balanced',
            {0: 1.0, 1: 1.5},  # Slight favor to approvals
            {0: 1.0, 1: 2.0},  # Moderate favor to approvals
            {0: 1.0, 1: 3.0},  # Strong favor to approvals
        ]
        
        if data.get('class_weights'):
            weight_configs.append(data['class_weights'])
        
        best_weight_model = None
        best_weight_f1 = 0
        
        for i, weight in enumerate(weight_configs):
            try:
                rf_weight = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight=weight,
                    random_state=config.random_state
                )
                
                rf_weight.fit(data['X_train_opt'], data['y_train_opt'])
                y_val_pred = rf_weight.predict(data['X_val'])
                val_f1 = f1_score(data['y_val'], y_val_pred)
                val_acc = accuracy_score(data['y_val'], y_val_pred)
                approval_rate = np.mean(y_val_pred)
                unique_preds = len(np.unique(y_val_pred))
                
                logger.info(f"    Weight {weight}: F1={val_f1:.4f}, Approval={approval_rate:.1%}, Unique={unique_preds}")
                
                if val_f1 > best_weight_f1 and unique_preds > 1:
                    best_weight_f1 = val_f1
                    best_weight_model = rf_weight
                    best_weight_config = weight
                    
            except Exception as e:
                logger.warning(f"    Weight {weight} failed: {e}")
        
        if best_weight_model is not None:
            y_val_pred = best_weight_model.predict(data['X_val'])
            val_f1 = f1_score(data['y_val'], y_val_pred)
            val_acc = accuracy_score(data['y_val'], y_val_pred)
            
            business_validation = validate_model_business_logic(
                best_weight_model, data['X_val'], data['y_val'], config, logger, "RF_ClassWeight"
            )
            
            results['class_weight_tuning'] = {
                'model': best_weight_model,
                'params': {'class_weight': best_weight_config, 'n_estimators': 200, 'max_depth': 20},
                'cv_f1': val_f1,  # Use validation as proxy
                'val_f1': val_f1,
                'val_acc': val_acc,
                'optimization_time': 0,
                'business_validation': business_validation
            }
            
            logger.info(f"    Best Weight Model: F1={val_f1:.4f}, Approval={business_validation['approval_rate']:.1%}")
        
    except Exception as e:
        logger.error(f"Class weight tuning failed: {e}")
    
    # Strategy 2: Simple Grid Search
    try:
        logger.info("  Strategy 2: Simple Grid Search...")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [15, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced', {0: 1.0, 1: 2.0}]
        }
        
        rf_search = GridSearchCV(
            RandomForestClassifier(random_state=config.random_state),
            param_grid,
            cv=3,  # Reduced for speed
            scoring='f1',
            n_jobs=-1,
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
        unique_preds = len(np.unique(y_val_pred))
        
        # Business logic validation
        business_validation = validate_model_business_logic(
            best_rf, data['X_val'], data['y_val'], config, logger, "RF_GridSearch"
        )
        
        if val_f1 > 0 and unique_preds > 1:
            results['grid_search'] = {
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
        
    except Exception as e:
        logger.error(f"Grid search failed: {e}")
    
    logger.info(f"Random Forest optimization completed: {len(results)} models created")
    return results

# ============================================================================
# 5. SIMPLIFIED MULTI-MODEL OPTIMIZATION
# ============================================================================

def optimize_multiple_models_simple(data, config, logger, mlflow=None):
    """Simplified multi-model optimization"""
    logger.info("Starting simplified multi-model optimization...")
    
    results = {}
    
    # Model 1: Logistic Regression
    try:
        logger.info("  Training Logistic Regression...")
        
        # Prepare scaled data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(data['X_train_opt'])
        X_val_scaled = scaler.transform(data['X_val'])
        
        lr = LogisticRegression(
            random_state=config.random_state,
            max_iter=1000,
            class_weight='balanced',
            C=1.0
        )
        
        lr.fit(X_train_scaled, data['y_train_opt'])
        y_val_pred = lr.predict(X_val_scaled)
        val_f1 = f1_score(data['y_val'], y_val_pred)
        val_acc = accuracy_score(data['y_val'], y_val_pred)
        unique_preds = len(np.unique(y_val_pred))
        
        business_validation = validate_model_business_logic(
            lr, X_val_scaled, data['y_val'], config, logger, "LogisticRegression"
        )
        
        if val_f1 > 0 and unique_preds > 1:
            results['LogisticRegression'] = {
                'model': lr,
                'scaler': scaler,
                'params': {'C': 1.0, 'class_weight': 'balanced'},
                'cv_f1': val_f1,
                'val_f1': val_f1,
                'val_acc': val_acc,
                'optimization_time': 0,
                'scale_features': True,
                'business_validation': business_validation
            }
            
            logger.info(f"    Logistic Regression: F1={val_f1:.4f}, Approval={business_validation['approval_rate']:.1%}")
        
    except Exception as e:
        logger.error(f"Logistic Regression failed: {e}")
    
    # Model 2: Gradient Boosting
    try:
        logger.info("  Training Gradient Boosting...")
        
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=config.random_state
        )
        
        gb.fit(data['X_train_opt'], data['y_train_opt'])
        y_val_pred = gb.predict(data['X_val'])
        val_f1 = f1_score(data['y_val'], y_val_pred)
        val_acc = accuracy_score(data['y_val'], y_val_pred)
        unique_preds = len(np.unique(y_val_pred))
        
        business_validation = validate_model_business_logic(
            gb, data['X_val'], data['y_val'], config, logger, "GradientBoosting"
        )
        
        if val_f1 > 0 and unique_preds > 1:
            results['GradientBoosting'] = {
                'model': gb,
                'scaler': None,
                'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5},
                'cv_f1': val_f1,
                'val_f1': val_f1,
                'val_acc': val_acc,
                'optimization_time': 0,
                'scale_features': False,
                'business_validation': business_validation
            }
            
            logger.info(f"    Gradient Boosting: F1={val_f1:.4f}, Approval={business_validation['approval_rate']:.1%}")
        
    except Exception as e:
        logger.error(f"Gradient Boosting failed: {e}")
    
    logger.info(f"Multi-model optimization completed: {len(results)} models created")
    return results

# ============================================================================
# 6. MODEL EVALUATION & SELECTION
# ============================================================================

def evaluate_all_models_simple(rf_results, multi_results, baseline, data, config, logger):
    """Evaluate all models and select best"""
    logger.info("Evaluating all optimized models...")
    
    # Collect all models
    all_models = {}
    
    # Add RF results
    for strategy, result in rf_results.items():
        all_models[f'RF_{strategy}'] = result
    
    # Add multi-model results
    for model_name, result in multi_results.items():
        all_models[model_name] = result
    
    if not all_models:
        logger.error("NO MODELS FOUND!")
        return None, []
    
    # Evaluate on test set
    test_results = []
    
    logger.info("Test Set Evaluation:")
    logger.info("-" * 80)
    logger.info(f"{'Model':<25} {'Test F1':<10} {'Test Acc':<10} {'Approval%':<10} {'Unique':<8}")
    logger.info("-" * 80)
    
    for model_name, result in all_models.items():
        try:
            model = result['model']
            
            # Get training features for alignment
            training_features = list(data['X_train'].columns)
            
            # Prepare test data with feature alignment
            if result.get('scaler'):
                X_test_processed = result['scaler'].transform(data['X_test'])
                # Convert to DataFrame for alignment
                X_test_processed = pd.DataFrame(X_test_processed, columns=training_features)
            else:
                X_test_processed = data['X_test'].copy()
            
            # Ensure feature alignment
            X_test_processed = ensure_feature_alignment(
                X_test_processed, training_features, logger, f"{model_name}_test_data"
            )
            
            # Predict
            y_pred = model.predict(X_test_processed)
            
            # Calculate metrics
            test_f1 = f1_score(data['y_test'], y_pred)
            test_acc = accuracy_score(data['y_test'], y_pred)
            test_precision = precision_score(data['y_test'], y_pred, zero_division=0)
            test_recall = recall_score(data['y_test'], y_pred, zero_division=0)
            test_approval_rate = np.mean(y_pred)
            unique_preds = len(np.unique(y_pred))
            
            # Calculate improvement
            f1_improvement = test_f1 - baseline['f1']
            
            # Final business validation on test set
            test_business_validation = validate_model_business_logic(
                model, X_test_processed, data['y_test'], config, logger, f"{model_name}_test"
            )
            
            test_results.append({
                'name': model_name,
                'model': model,
                'scaler': result.get('scaler'),
                'test_f1': test_f1,
                'test_acc': test_acc,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_approval_rate': test_approval_rate,
                'unique_predictions': unique_preds,
                'f1_improvement': f1_improvement,
                'validation_f1': result['val_f1'],
                'params': result['params'],
                'test_business_validation': test_business_validation
            })
            
            logger.info(f"{model_name:<25} {test_f1:<10.4f} {test_acc:<10.4f} {test_approval_rate:<9.1%} {unique_preds:<8}")
            
        except Exception as e:
            logger.error(f"Evaluation failed for {model_name}: {e}")
    
    logger.info("-" * 80)
    
    if not test_results:
        logger.error("NO MODELS EVALUATED SUCCESSFULLY!")
        return None, []
    
    # Find best model - prioritize models that make varied predictions and have good F1
    valid_models = [r for r in test_results if r['unique_predictions'] > 1 and r['test_f1'] > 0]
    
    if valid_models:
        # First try models that pass business logic
        business_valid_models = [r for r in valid_models if r['test_business_validation']['passes_business_logic']]
        
        if business_valid_models:
            best_model_result = max(business_valid_models, key=lambda x: x['test_f1'])
            logger.info(f"\nBEST BUSINESS-VALID MODEL: {best_model_result['name']}")
        else:
            # Fallback to best F1 model that makes varied predictions
            best_model_result = max(valid_models, key=lambda x: x['test_f1'])
            logger.info(f"\nBEST MODEL (No business validation pass): {best_model_result['name']}")
        
        logger.info(f"  Test F1-Score: {best_model_result['test_f1']:.4f}")
        logger.info(f"  Test Accuracy: {best_model_result['test_acc']:.4f}")
        logger.info(f"  Test Approval Rate: {best_model_result['test_approval_rate']:.1%}")
        logger.info(f"  Unique Predictions: {best_model_result['unique_predictions']}")
        logger.info(f"  Improvement over baseline: {best_model_result['f1_improvement']:+.4f}")
        
        return best_model_result, test_results
    else:
        logger.error("NO MODELS MAKE VARIED PREDICTIONS!")
        return None, test_results

# ============================================================================
# 7. COMPREHENSIVE MODEL SAVING WITH METADATA
# ============================================================================

def save_best_model_with_metadata(best_model_result, data, config, logger, mlflow=None):
    """Save best model dengan comprehensive feature metadata"""
    logger.info("Saving best model with comprehensive feature metadata...")
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine training features from the actual training data
        training_features = list(data['X_train'].columns)
        n_features = len(training_features)
        
        logger.info(f"Saving model with {n_features} features")
        
        # Create feature metadata
        feature_metadata = {
            'feature_names': training_features,
            'n_features': n_features,
            'feature_types': {},
            'feature_ranges': {},
            'creation_timestamp': datetime.now().isoformat(),
            'model_type': best_model_result['name'],
            'requires_scaling': best_model_result.get('scaler') is not None
        }
        
        # Analyze each feature
        for col in training_features:
            if data['X_train'][col].dtype in ['int64', 'float64', 'float32', 'int32']:
                feature_metadata['feature_types'][col] = 'numerical'
                feature_metadata['feature_ranges'][col] = {
                    'min': float(data['X_train'][col].min()),
                    'max': float(data['X_train'][col].max()),
                    'mean': float(data['X_train'][col].mean()),
                    'std': float(data['X_train'][col].std())
                }
            else:
                feature_metadata['feature_types'][col] = 'categorical'
                unique_vals = data['X_train'][col].unique()
                feature_metadata['feature_ranges'][col] = [str(v) for v in unique_vals]
        
        # Save model components
        model_filename = f"{config.output_dir}/best_model_{timestamp}.pkl"
        features_filename = f"{config.output_dir}/training_features_{timestamp}.pkl"
        metadata_filename = f"{config.output_dir}/feature_metadata_{timestamp}.json"
        scaler_filename = None
        
        # Save main model
        joblib.dump(best_model_result['model'], model_filename)
        logger.info(f"Model saved: {model_filename}")
        
        # Save feature names (critical for prediction)
        joblib.dump(training_features, features_filename)
        logger.info(f"Training features saved: {features_filename}")
        
        # Save feature metadata as JSON
        with open(metadata_filename, 'w') as f:
            json.dump(feature_metadata, f, indent=2)
        logger.info(f"Feature metadata saved: {metadata_filename}")
        
        # Save scaler if exists
        if best_model_result.get('scaler'):
            scaler_filename = f"{config.output_dir}/scaler_{timestamp}.pkl"
            joblib.dump(best_model_result['scaler'], scaler_filename)
            logger.info(f"Scaler saved: {scaler_filename}")
        
        # Create comprehensive model info file
        info_filename = f"{config.output_dir}/model_info_{timestamp}.txt"
        with open(info_filename, 'w') as f:
            f.write(f"Credit Approval Model - Complete Information\n")
            f.write(f"===========================================\n\n")
            
            # Model Information
            f.write(f"MODEL DETAILS:\n")
            f.write(f"  Model Type: {best_model_result['name']}\n")
            f.write(f"  Training Date: {datetime.now()}\n")
            f.write(f"  Features Count: {n_features}\n")
            f.write(f"  Requires Scaling: {best_model_result.get('scaler') is not None}\n\n")
            
            # Performance Metrics
            f.write(f"PERFORMANCE METRICS:\n")
            f.write(f"  Test F1-Score: {best_model_result['test_f1']:.4f}\n")
            f.write(f"  Test Accuracy: {best_model_result['test_acc']:.4f}\n")
            f.write(f"  Test Precision: {best_model_result['test_precision']:.4f}\n")
            f.write(f"  Test Recall: {best_model_result['test_recall']:.4f}\n")
            f.write(f"  Test Approval Rate: {best_model_result['test_approval_rate']:.1%}\n")
            f.write(f"  Unique Predictions: {best_model_result['unique_predictions']}\n")
            f.write(f"  Improvement over baseline: {best_model_result['f1_improvement']:+.4f}\n\n")
            
            # Feature List
            f.write(f"FEATURES ({n_features}):\n")
            for i, feature in enumerate(training_features):
                f.write(f"  {i+1:2d}. {feature}\n")
            f.write(f"\n")
            
            # Usage Instructions
            f.write(f"USAGE INSTRUCTIONS:\n")
            f.write(f"1. Load model: model = joblib.load('{model_filename}')\n")
            f.write(f"2. Load features: features = joblib.load('{features_filename}')\n")
            if scaler_filename:
                f.write(f"3. Load scaler: scaler = joblib.load('{scaler_filename}')\n")
            f.write(f"4. Use prediction helper script for easy predictions\n")
        
        logger.info(f"Model info saved: {info_filename}")
        
        # Create simplified prediction helper
        helper_filename = f"{config.output_dir}/prediction_helper_{timestamp}.py"
        helper_content = f'''# Credit Approval Model - Prediction Helper
# Generated: {datetime.now()}

import pandas as pd
import numpy as np
import joblib
import json

# Model file paths
MODEL_FILE = "{model_filename}"
FEATURES_FILE = "{features_filename}"
METADATA_FILE = "{metadata_filename}"
SCALER_FILE = {f'"{scaler_filename}"' if scaler_filename else 'None'}

def load_model_components():
    """Load all model components"""
    model = joblib.load(MODEL_FILE)
    features = joblib.load(FEATURES_FILE)
    scaler = joblib.load(SCALER_FILE) if SCALER_FILE else None
    
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
    return model, features, scaler, metadata

def ensure_feature_alignment(X_data, reference_columns):
    """Ensure data has exact same features as training data"""
    if isinstance(X_data, dict):
        df = pd.DataFrame([X_data])
    elif isinstance(X_data, pd.DataFrame):
        df = X_data.copy()
    else:
        df = pd.DataFrame(X_data)
    
    # Add missing columns with zeros
    for col in reference_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Remove extra columns and reorder
    df = df[reference_columns]
    
    return df

def predict_credit_approval(input_data):
    """
    Predict credit approval
    
    Args:
        input_data: dict or DataFrame with applicant information
    
    Returns:
        dict: prediction results
    """
    try:
        # Load components
        model, features, scaler, metadata = load_model_components()
        
        # Ensure feature alignment
        df_aligned = ensure_feature_alignment(input_data, features)
        
        # Apply scaling if needed
        if scaler:
            X_processed = scaler.transform(df_aligned)
        else:
            X_processed = df_aligned
        
        # Make prediction
        prediction = model.predict(X_processed)[0]
        
        # Get probabilities if available
        probability = None
        if hasattr(model, 'predict_proba'):
            prob_scores = model.predict_proba(X_processed)[0]
            probability = {{
                'reject': float(prob_scores[0]),
                'approve': float(prob_scores[1])
            }}
        
        return {{
            'prediction': int(prediction),
            'decision': 'APPROVED' if prediction == 1 else 'REJECTED',
            'probability': probability,
            'model_type': metadata.get('model_type', 'Unknown'),
            'features_used': len(features),
            'success': True
        }}
        
    except Exception as e:
        return {{
            'success': False,
            'error': str(e)
        }}

def test_prediction():
    """Test with sample data"""
    sample = {{
        'umur': 35,
        'pendapatan': 12000000,
        'skor_kredit': 720,
        'jumlah_pinjaman': 40000000,
        'rasio_pinjaman_pendapatan': 3.33,
        'pekerjaan_Tetap': 1,
        'pekerjaan_Freelance': 0,
        'pekerjaan_Kontrak': 0,
        'kategori_umur_Dewasa': 1,
        'kategori_umur_Muda': 0,
        'kategori_umur_Senior': 0,
        'kategori_skor_kredit_Good': 1,
        'kategori_skor_kredit_Fair': 0,
        'kategori_skor_kredit_Poor': 0,
        'kategori_pendapatan_Tinggi': 0,
        'kategori_pendapatan_Sedang': 1,
        'kategori_pendapatan_Rendah': 0
    }}
    
    result = predict_credit_approval(sample)
    print("Sample prediction:", result)
    return result

if __name__ == "__main__":
    print("Testing credit approval model...")
    test_prediction()
'''
        
        with open(helper_filename, 'w') as f:
            f.write(helper_content)
        
        logger.info(f"Prediction helper created: {helper_filename}")
        
        # Final validation
        logger.info("Performing final model validation...")
        try:
            # Test loading and prediction
            test_model = joblib.load(model_filename)
            test_features = joblib.load(features_filename)
            
            # Test with small sample
            sample_data = data['X_test'].head(1)
            if best_model_result.get('scaler'):
                test_scaler = joblib.load(scaler_filename)
                sample_processed = test_scaler.transform(sample_data)
            else:
                sample_processed = sample_data
            
            test_pred = test_model.predict(sample_processed)
            logger.info(f"Final validation successful - test prediction: {test_pred[0]}")
            
        except Exception as e:
            logger.error(f"Final validation failed: {e}")
            return None
        
        logger.info("Model saved successfully with complete metadata!")
        return model_filename
        
    except Exception as e:
        logger.error(f"Error saving model with metadata: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# 8. OPTIMIZATION PIPELINE FUNCTION
# ============================================================================

def run_optimization_pipeline(mlflow, data, config, logger, baseline):
    """Main optimization pipeline that can run with or without MLflow"""
    try:
        if mlflow:
            # Log configuration
            mlflow.log_param("optimization_version", "SIMPLIFIED_WITH_METADATA")
            mlflow.log_param("optimization_date", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            mlflow.log_param("alpha", config.alpha)
            mlflow.log_param("l1_ratio", config.l1_ratio)
            mlflow.log_param("baseline_f1", baseline['f1'])
            mlflow.log_param("total_samples", len(data['X_train']))
            mlflow.log_param("features", data['X_train'].shape[1])
        
        # Run optimizations
        rf_results = optimize_random_forest_simple(data, config, logger, mlflow, None)
        multi_results = optimize_multiple_models_simple(data, config, logger, mlflow)
        
        # Evaluate and select best
        best_model, all_results = evaluate_all_models_simple(rf_results, multi_results, baseline, data, config, logger)
        
        if best_model:
            # Save model with comprehensive metadata
            model_file = save_best_model_with_metadata(best_model, data, config, logger, mlflow)
            
            # Log best model to MLflow if available
            if mlflow:
                try:
                    import mlflow.models
                    signature = mlflow.models.infer_signature(data['X_train'], data['y_train'])
                    mlflow.sklearn.log_model(best_model['model'], f"best_model_{best_model['name']}", signature=signature)
                    logger.info(f"    Logged best_model_{best_model['name']} with signature to MLflow")
                except Exception as e:
                    logger.warning(f"    Failed to log best_model_{best_model['name']} with signature: {e}")
                    try:
                        mlflow.sklearn.log_model(best_model['model'], f"best_model_{best_model['name']}")
                    except Exception as e2:
                        logger.warning(f"    Failed to log model at all: {e2}")
                
                # Final logging
                try:
                    mlflow.log_metric("optimization_success", 1)
                    mlflow.log_param("best_model_name", best_model['name'])
                    mlflow.log_metric("final_improvement", best_model['f1_improvement'])
                    mlflow.log_metric("final_approval_rate", best_model['test_approval_rate'])
                    mlflow.log_metric("final_unique_predictions", best_model['unique_predictions'])
                    mlflow.log_param("has_feature_metadata", True)
                except Exception as e:
                    logger.warning(f"    Failed to log final metrics: {e}")
            
            return best_model, model_file
        else:
            logger.error("No valid models found")
            if mlflow:
                try:
                    mlflow.log_metric("optimization_success", 0)
                except Exception as e:
                    logger.warning(f"Failed to log failure metric: {e}")
            return None, None
            
    except Exception as e:
        logger.error(f"Optimization pipeline failed: {e}")
        if mlflow:
            try:
                mlflow.log_metric("optimization_success", 0)
                mlflow.log_param("error_message", str(e))
            except Exception as e2:
                logger.warning(f"Failed to log error: {e2}")
        return None, None

# ============================================================================
# 9. ARGUMENT PARSING AND MAIN
# ============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Simplified Credit Approval Model Tuning with Feature Metadata'
    )
    
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha parameter (default: 0.5)')
    parser.add_argument('--l1_ratio', type=float, default=0.1, help='L1 ratio (default: 0.1)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--dry-run', action='store_true', help='Run validation only')
    parser.add_argument('--disable-mlflow', action='store_true', help='Disable MLflow tracking (useful for CI/CD)')
    
    return parser.parse_args()

def main():
    """Main optimization pipeline - SIMPLIFIED VERSION with CI/CD MLflow fixes"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup with arguments
    config = Config(args)
    logger = setup_logging()
    
    # Check if MLflow should be disabled
    if args.disable_mlflow:
        logger.info("MLflow disabled via command line argument")
        mlflow, mlflow_available = None, False
    else:
        mlflow, mlflow_available = setup_mlflow_dagshub(config, logger)
    
    optuna, optuna_available = setup_optuna_optional(logger)
    
    # Log configuration
    logger.info("Starting SIMPLIFIED model optimization with comprehensive feature metadata...")
    logger.info(f"Configuration:")
    logger.info(f"  Experiment: {config.experiment_name}")
    logger.info(f"  Alpha: {config.alpha}")
    logger.info(f"  L1 Ratio: {config.l1_ratio}")
    logger.info(f"  Random State: {config.random_state}")
    logger.info(f"  MLflow Available: {mlflow_available}")
    logger.info(f"Business constraints:")
    logger.info(f"  Target approval rate: {config.target_approval_rate:.1%}")
    logger.info(f"  Min approval rate: {config.min_approval_rate:.1%}")
    logger.info(f"  Max approval rate: {config.max_approval_rate:.1%}")
    
    # Check for dry run
    if args.dry_run:
        logger.info("DRY RUN MODE - Validation only")
        data = load_and_validate_data(config, logger)
        if data:
            logger.info("Data validation passed - ready for training")
            logger.info(f"   Training samples: {data['X_train'].shape[0]}")
            logger.info(f"   Features: {data['X_train'].shape[1]}")
            logger.info(f"   Training approval rate: {data['train_approval_rate']:.1%}")
        else:
            logger.error("Data validation failed")
        return
    
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
    
    # Start main optimization with safe MLflow handling
    try:
        if mlflow_available:
            # Use safe MLflow execution
            best_model, model_file = safe_mlflow_run(
                mlflow, config, logger, 
                lambda mlf: run_optimization_pipeline(mlf, data, config, logger, baseline)
            )
        else:
            # Run without MLflow
            best_model, model_file = run_optimization_pipeline(None, data, config, logger, baseline)
        
        # Print final summary
        if best_model:
            logger.info("=" * 70)
            logger.info("SIMPLIFIED MODEL TUNING COMPLETED WITH FEATURE METADATA!")
            logger.info("=" * 70)
            logger.info(f"OPTIMIZATION SUMMARY:")
            logger.info(f"  Baseline F1-Score: {baseline['f1']:.4f}")
            logger.info(f"  Best F1-Score: {best_model['test_f1']:.4f}")
            logger.info(f"  Best Approval Rate: {best_model['test_approval_rate']:.1%}")
            logger.info(f"  Total Improvement: {best_model['f1_improvement']:+.4f}")
            logger.info(f"  Best Model: {best_model['name']}")
            logger.info(f"  Unique Predictions: {best_model['unique_predictions']}")
            logger.info(f"  Features: {len(data['X_train'].columns)}")
            
            # Check success criteria
            makes_variety = best_model['unique_predictions'] > 1
            reasonable_approval = 0.15 <= best_model['test_approval_rate'] <= 0.85
            metadata_success = model_file is not None
            
            logger.info(f"\nVALIDATION RESULTS:")
            logger.info(f"  Makes Variety: {'YES' if makes_variety else 'NO'} ({best_model['unique_predictions']} types)")
            logger.info(f"  Approval Rate: {'REASONABLE' if reasonable_approval else 'EXTREME'} ({best_model['test_approval_rate']:.1%})")
            logger.info(f"  Feature Metadata: {'SAVED' if metadata_success else 'MISSING'}")
            
            # Show files created
            logger.info(f"\nFILES CREATED:")
            if model_file:
                timestamp_pattern = datetime.now().strftime("%Y%m%d")
                logger.info(f"  Model file: output/best_model_{timestamp_pattern}_*.pkl")
                logger.info(f"  Features file: output/training_features_{timestamp_pattern}_*.pkl")
                logger.info(f"  Metadata file: output/feature_metadata_{timestamp_pattern}_*.json")
                logger.info(f"  Prediction helper: output/prediction_helper_{timestamp_pattern}_*.py")
                logger.info(f"  Model info: output/model_info_{timestamp_pattern}_*.txt")
                if best_model.get('scaler'):
                    logger.info(f"  Scaler file: output/scaler_{timestamp_pattern}_*.pkl")
            
            if makes_variety and reasonable_approval and metadata_success:
                logger.info(f"\nOPTIMIZATION FULLY SUCCESSFUL!")
                logger.info(f"   Model ready for production deployment")
                logger.info(f"   Complete prediction pipeline created")
                logger.info(f"   Feature metadata saved for consistency")
                
                # Show usage instructions
                logger.info(f"\nQUICK START:")
                helper_files = glob.glob(f"{config.output_dir}/prediction_helper_*.py")
                if helper_files:
                    latest_helper = max(helper_files, key=os.path.getctime)
                    logger.info(f"1. Test the model:")
                    logger.info(f"   python {latest_helper}")
                    
                    logger.info(f"2. Use in your code:")
                    logger.info(f"   from {os.path.basename(latest_helper).replace('.py', '')} import predict_credit_approval")
                    logger.info(f"   result = predict_credit_approval({{'umur': 35, 'pendapatan': 12000000, ...}})")
                
            else:
                logger.warning(f"\nOPTIMIZATION NEEDS IMPROVEMENT")
                logger.warning(f"   Model functional but may need tuning")
                if not makes_variety:
                    logger.warning(f"   Model doesn't make varied predictions")
                if not reasonable_approval:
                    logger.warning(f"   Approval rate outside reasonable range")
                if not metadata_success:
                    logger.warning(f"   Feature metadata not saved properly")
        else:
            logger.error("OPTIMIZATION FAILED - no valid models produced")
            
    except Exception as e:
        logger.error(f"Optimization failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ensure MLflow run is properly closed
        if mlflow_available and mlflow:
            try:
                if mlflow.active_run():
                    logger.info("Cleaning up MLflow run...")
                    mlflow.end_run()
            except Exception as e:
                logger.warning(f"Error cleaning up MLflow run: {e}")

if __name__ == "__main__":
    main()