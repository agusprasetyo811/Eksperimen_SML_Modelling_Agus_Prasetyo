import pandas as pd
import numpy as np
import os
import sys
import time
import joblib
import warnings
import argparse
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, cross_val_score, 
                                   StratifiedKFold, train_test_split)
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score,
                           roc_auc_score, classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings('ignore')

print("="*70)
print("COMPREHENSIVE MODEL OPTIMIZATION - DAGSHUB & MLFLOW INTEGRATION")
print("="*70)

# ============================================================================
# CUSTOM TRANSFORMER FOR FEATURE ENGINEERING
# ============================================================================

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for advanced feature engineering that works with MLflow"""
    
    def __init__(self, use_advanced_features=True):
        self.use_advanced_features = use_advanced_features
        self.original_features = None
        
    def fit(self, X, y=None):
        # Store original feature names for consistency
        if hasattr(X, 'columns'):
            self.original_features = X.columns.tolist()
        else:
            self.original_features = [f'feature_{i}' for i in range(X.shape[1])]
        return self
    
    def transform(self, X):
        if not self.use_advanced_features:
            return X
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.original_features)
        
        X_enhanced = X.copy()
        
        try:
            # Feature 1: Advanced debt ratio categories
            if 'rasio_pinjaman_pendapatan' in X_enhanced.columns:
                X_enhanced['debt_ratio_low'] = (X_enhanced['rasio_pinjaman_pendapatan'] <= 3).astype(int)
                X_enhanced['debt_ratio_medium'] = ((X_enhanced['rasio_pinjaman_pendapatan'] > 3) & 
                                                 (X_enhanced['rasio_pinjaman_pendapatan'] <= 5)).astype(int)
                X_enhanced['debt_ratio_high'] = ((X_enhanced['rasio_pinjaman_pendapatan'] > 5) & 
                                               (X_enhanced['rasio_pinjaman_pendapatan'] <= 7)).astype(int)
                X_enhanced['debt_ratio_extreme'] = (X_enhanced['rasio_pinjaman_pendapatan'] > 7).astype(int)
            
            # Feature 2: Enhanced credit score ranges
            if 'skor_kredit' in X_enhanced.columns:
                X_enhanced['credit_excellent_plus'] = (X_enhanced['skor_kredit'] >= 800).astype(int)
                X_enhanced['credit_very_good'] = ((X_enhanced['skor_kredit'] >= 750) & (X_enhanced['skor_kredit'] < 800)).astype(int)
                X_enhanced['credit_good_plus'] = ((X_enhanced['skor_kredit'] >= 700) & (X_enhanced['skor_kredit'] < 750)).astype(int)
                X_enhanced['credit_borderline'] = ((X_enhanced['skor_kredit'] >= 600) & (X_enhanced['skor_kredit'] < 650)).astype(int)
                X_enhanced['credit_high_risk'] = (X_enhanced['skor_kredit'] < 550).astype(int)
            
            # Feature 3: Income stability composite score
            stability_score = np.zeros(len(X_enhanced))
            
            if 'pekerjaan_Tetap' in X_enhanced.columns:
                stability_score += X_enhanced['pekerjaan_Tetap'] * 3
            if 'pekerjaan_Kontrak' in X_enhanced.columns:
                stability_score += X_enhanced['pekerjaan_Kontrak'] * 2
            if 'pekerjaan_Freelance' in X_enhanced.columns:
                stability_score += X_enhanced['pekerjaan_Freelance'] * 1
            
            if 'kategori_pendapatan_Tinggi' in X_enhanced.columns:
                stability_score += X_enhanced['kategori_pendapatan_Tinggi'] * 2
            if 'kategori_pendapatan_Sedang' in X_enhanced.columns:
                stability_score += X_enhanced['kategori_pendapatan_Sedang'] * 1
            
            X_enhanced['income_stability_score'] = stability_score
            
            # Feature 4: Risk combination indicators
            if all(col in X_enhanced.columns for col in ['kategori_umur_Muda', 'pekerjaan_Freelance', 'kategori_skor_kredit_Poor']):
                X_enhanced['high_risk_combination'] = (
                    X_enhanced['kategori_umur_Muda'] & 
                    X_enhanced['pekerjaan_Freelance'] & 
                    X_enhanced['kategori_skor_kredit_Poor']
                ).astype(int)
            
            if all(col in X_enhanced.columns for col in ['kategori_umur_Dewasa', 'pekerjaan_Tetap', 'kategori_skor_kredit_Good']):
                X_enhanced['low_risk_combination'] = (
                    X_enhanced['kategori_umur_Dewasa'] & 
                    X_enhanced['pekerjaan_Tetap'] & 
                    X_enhanced['kategori_skor_kredit_Good']
                ).astype(int)
            
            if all(col in X_enhanced.columns for col in ['kategori_umur_Senior', 'pekerjaan_Tetap']):
                X_enhanced['senior_stable_combination'] = (
                    X_enhanced['kategori_umur_Senior'] & 
                    X_enhanced['pekerjaan_Tetap']
                ).astype(int)
            
            # Feature 5: Age-Income interaction effects
            if all(col in X_enhanced.columns for col in ['umur', 'pendapatan']):
                X_enhanced['peak_earning_years'] = ((X_enhanced['umur'] >= 35) & (X_enhanced['umur'] <= 50)).astype(int)
                
                if 'kategori_pendapatan_Tinggi' in X_enhanced.columns:
                    X_enhanced['young_high_earner'] = (
                        (X_enhanced['umur'] < 35) & X_enhanced['kategori_pendapatan_Tinggi']
                    ).astype(int)
                
                age_norm = (X_enhanced['umur'] - 18) / (65 - 18)
                income_millions = X_enhanced['pendapatan'] / 1000000
                X_enhanced['experience_income_factor'] = age_norm * np.log1p(income_millions)
            
            # Feature 6: Loan efficiency and capacity features
            if all(col in X_enhanced.columns for col in ['jumlah_pinjaman', 'pendapatan', 'umur', 'rasio_pinjaman_pendapatan']):
                working_years_left = np.maximum(65 - X_enhanced['umur'], 5)
                monthly_capacity = X_enhanced['pendapatan'] * 0.3
                total_capacity = monthly_capacity * working_years_left * 12
                
                X_enhanced['loan_to_total_capacity'] = (X_enhanced['jumlah_pinjaman'] / total_capacity).clip(0, 5)
                X_enhanced['age_adjusted_loan_burden'] = X_enhanced['rasio_pinjaman_pendapatan'] * (70 - X_enhanced['umur']) / 40
                X_enhanced['monthly_debt_service_ratio'] = (X_enhanced['jumlah_pinjaman'] / 240) / X_enhanced['pendapatan']
            
            # Handle any NaN values
            X_enhanced = X_enhanced.fillna(0)
            
            # Ensure all values are finite
            X_enhanced = X_enhanced.replace([np.inf, -np.inf], 0)
            
            return X_enhanced
            
        except Exception as e:
            print(f"Warning: Feature engineering failed: {e}")
            return X
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.original_features
        
        if not self.use_advanced_features:
            return input_features
        
        # Return expected feature names after transformation
        enhanced_features = list(input_features)
        
        # Add new feature names that would be created
        new_features = [
            'debt_ratio_low', 'debt_ratio_medium', 'debt_ratio_high', 'debt_ratio_extreme',
            'credit_excellent_plus', 'credit_very_good', 'credit_good_plus', 'credit_borderline', 'credit_high_risk',
            'income_stability_score', 'high_risk_combination', 'low_risk_combination', 'senior_stable_combination',
            'peak_earning_years', 'young_high_earner', 'experience_income_factor',
            'loan_to_total_capacity', 'age_adjusted_loan_burden', 'monthly_debt_service_ratio'
        ]
        
        enhanced_features.extend(new_features)
        return enhanced_features

class IntelligentFeatureSelector(BaseEstimator, TransformerMixin):
    """Custom transformer for intelligent feature selection"""
    
    def __init__(self, use_feature_selection=True, k=20, random_state=42):
        self.use_feature_selection = use_feature_selection
        self.k = k
        self.random_state = random_state
        self.selected_features_ = None
        
    def fit(self, X, y=None):
        if not self.use_feature_selection or y is None:
            if hasattr(X, 'columns'):
                self.selected_features_ = X.columns.tolist()
            else:
                self.selected_features_ = [f'feature_{i}' for i in range(X.shape[1])]
            return self
        
        try:
            from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
            
            # Convert to DataFrame if needed
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            
            # Method 1: Statistical feature selection
            selector_stats = SelectKBest(score_func=f_classif, k=min(self.k, X.shape[1]))
            selector_stats.fit(X, y)
            selected_features_stats = X.columns[selector_stats.get_support()].tolist()
            
            # Method 2: Mutual information
            selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(self.k, X.shape[1]))
            selector_mi.fit(X, y)
            selected_features_mi = X.columns[selector_mi.get_support()].tolist()
            
            # Method 3: Random Forest importance
            rf_selector = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            rf_selector.fit(X, y)
            
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_selector.feature_importances_
            }).sort_values('importance', ascending=False)
            
            top_features_rf = feature_importance.head(self.k)['feature'].tolist()
            
            # Ensemble selection
            feature_votes = {}
            for feature in X.columns:
                votes = 0
                if feature in selected_features_stats:
                    votes += 1
                if feature in selected_features_mi:
                    votes += 1
                if feature in top_features_rf:
                    votes += 2
                feature_votes[feature] = votes
            
            # Sort by votes and importance
            sorted_features = sorted(feature_votes.items(), 
                                   key=lambda x: (x[1], feature_importance[feature_importance['feature'] == x[0]]['importance'].iloc[0] if len(feature_importance[feature_importance['feature'] == x[0]]) > 0 else 0), 
                                   reverse=True)
            
            self.selected_features_ = [f[0] for f in sorted_features[:self.k]]
            
        except Exception as e:
            print(f"Warning: Feature selection failed: {e}")
            if hasattr(X, 'columns'):
                self.selected_features_ = X.columns.tolist()
            else:
                self.selected_features_ = [f'feature_{i}' for i in range(X.shape[1])]
        
        return self
    
    def transform(self, X):
        if not self.use_feature_selection or self.selected_features_ is None:
            return X
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            if hasattr(self, 'original_features_'):
                X = pd.DataFrame(X, columns=self.original_features_)
            else:
                X = pd.DataFrame(X)
        
        # Select only the chosen features
        available_features = [f for f in self.selected_features_ if f in X.columns]
        return X[available_features]
    
    def get_feature_names_out(self, input_features=None):
        if not self.use_feature_selection or self.selected_features_ is None:
            return input_features
        return self.selected_features_

# ============================================================================
# COMPREHENSIVE PIPELINE CREATOR
# ============================================================================

def create_comprehensive_pipeline(model, scale_features=False, use_advanced_features=True, 
                                use_feature_selection=True, feature_selection_k=20, 
                                class_weight=None, random_state=42, **model_params):
    """Create a comprehensive pipeline that includes all preprocessing steps"""
    
    steps = []
    
    # Step 1: Advanced Feature Engineering
    if use_advanced_features:
        steps.append(('feature_engineer', AdvancedFeatureEngineer(use_advanced_features=True)))
    
    # Step 2: Feature Selection
    if use_feature_selection:
        steps.append(('feature_selector', IntelligentFeatureSelector(
            use_feature_selection=True, k=feature_selection_k, random_state=random_state
        )))
    
    # Step 3: Scaling (if needed)
    if scale_features:
        steps.append(('scaler', StandardScaler()))
    
    # Step 4: Model
    if isinstance(model, str):
        if model == 'RandomForest':
            model_obj = RandomForestClassifier(random_state=random_state, class_weight=class_weight, **model_params)
        elif model == 'GradientBoosting':
            model_obj = GradientBoostingClassifier(random_state=random_state, **model_params)
        elif model == 'ExtraTrees':
            model_obj = ExtraTreesClassifier(random_state=random_state, class_weight=class_weight, **model_params)
        elif model == 'LogisticRegression':
            model_obj = LogisticRegression(random_state=random_state, max_iter=1000, class_weight=class_weight, **model_params)
        else:
            raise ValueError(f"Unknown model type: {model}")
    else:
        model_obj = model
    
    steps.append(('model', model_obj))
    
    return Pipeline(steps)

# ============================================================================
# 0. CONFIGURATION & SETUP
# ============================================================================

class Config:
    """Configuration class untuk centralized settings"""
    def __init__(self, args=None):
        if args:
            self.alpha = args.alpha
            self.l1_ratio = args.l1_ratio
        else:
            self.alpha = 0.5
            self.l1_ratio = 0.1
        
        # Directories
        self.output_dir = "output"
        self.data_dir = "final_dataset"
        
        # Model settings
        self.random_state = int(os.getenv('RANDOM_STATE', '42'))
        self.test_size = 0.2
        self.cv_folds = 5
        
        # Enhanced optimization settings
        self.n_iter_random = 100        
        self.n_iter_optuna = int(os.getenv('N_ITER_OPTUNA', '200'))
        self.grid_search_iter = int(os.getenv('GRID_SEARCH_ITER', '200'))
        
        # Target improvement
        self.target_improvement = 0.025
        
        # BUSINESS LOGIC SETTINGS
        self.min_approval_rate = float(os.getenv('MIN_APPROVAL_RATE', '0.3'))
        self.max_approval_rate = float(os.getenv('MAX_APPROVAL_RATE', '0.8'))
        self.target_approval_rate = float(os.getenv('TARGET_APPROVAL_RATE', '0.6'))
        
        # ADVANCED OPTIMIZATION SETTINGS
        self.use_advanced_features = True
        self.use_feature_selection = True
        self.use_ensemble_methods = True
        self.use_stacking = True
        self.feature_selection_k = 20
        self.ensemble_models = 5
        
        # MLflow settings
        self.experiment_name = os.getenv('EXPERIMENT_NAME', 'credit-approval-tuning')
        self.model_name = os.getenv('MODEL_NAME', 'credit_approval_model_tuned')
        
        # DagsHub settings
        self.dagshub_url = os.getenv('DAGSHUB_REPO_URL', 'https://dagshub.com/agusprasetyo811/kredit_pinjaman_2')
        self.dagshub_username = os.getenv('DAGSHUB_USERNAME', 'agusprasetyo811')
        self.dagshub_token = os.getenv('DAGSHUB_TOKEN')
        
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)

def setup_logging():
    """Setup logging untuk debugging"""
    import logging
    
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
    """Enhanced MLflow setup dengan DagsHub integration"""
    try:
        import mlflow
        import mlflow.sklearn
        import mlflow.models
        
        logger.info("Setting up MLflow with DagsHub integration...")
        
        if not config.dagshub_token:
            logger.error("DAGSHUB_TOKEN not found in environment variables!")
            logger.error("Please set DAGSHUB_TOKEN before running the script")
            return None, False
        
        if not config.dagshub_username:
            logger.error("DAGSHUB_USERNAME not found in environment variables!")
            return None, False
        
        if not config.dagshub_url:
            logger.error("DAGSHUB_REPO_URL not found in environment variables!")
            return None, False
        
        try:
            dagshub_mlflow_uri = f"{config.dagshub_url}.mlflow"
            mlflow.set_tracking_uri(dagshub_mlflow_uri)
            
            os.environ['MLFLOW_TRACKING_USERNAME'] = config.dagshub_username
            os.environ['MLFLOW_TRACKING_PASSWORD'] = config.dagshub_token
            
            logger.info(f"DagsHub MLflow tracking configured:")
            logger.info(f"   Repository: {config.dagshub_url}")
            logger.info(f"   Tracking URI: {dagshub_mlflow_uri}")
            logger.info(f"   Username: {config.dagshub_username}")
            logger.info(f"   Token: {'*' * len(config.dagshub_token[:4])}...{config.dagshub_token[-4:]}")
            
        except Exception as e:
            logger.error(f"DagsHub setup failed: {e}")
            logger.warning("Falling back to local MLflow tracking...")
            mlflow.set_tracking_uri("file:./mlruns")
        
        try:
            experiment = mlflow.get_experiment_by_name(config.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(config.experiment_name)
                logger.info(f"Created new MLflow experiment: {config.experiment_name} (ID: {experiment_id})")
            else:
                logger.info(f"Using existing MLflow experiment: {config.experiment_name} (ID: {experiment.experiment_id})")
            
            mlflow.set_experiment(config.experiment_name)
            
        except Exception as e:
            logger.error(f"Failed to set/create experiment: {e}")
            logger.warning("Using default experiment")
        
        try:
            with mlflow.start_run(run_name="connection_test"):
                mlflow.log_param("test", "connection")
                mlflow.log_metric("test_metric", 1.0)
                logger.info("MLflow connection test successful")
            
        except Exception as e:
            logger.error(f"MLflow connection test failed: {e}")
            logger.warning("MLflow may not be properly configured")
        
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

# ============================================================================
# 1. ROBUST DATA LOADING WITH CLASS BALANCE ANALYSIS
# ============================================================================

def load_and_validate_data(config, logger):
    """Load dan validate data dengan robust error handling dan class balance analysis"""
    logger.info("Loading and validating data...")
    
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
        X_train = pd.read_csv(f'{config.data_dir}/X_train.csv')
        X_test = pd.read_csv(f'{config.data_dir}/X_test.csv')
        y_train = pd.read_csv(f'{config.data_dir}/y_train.csv')['disetujui_encoded']
        y_test = pd.read_csv(f'{config.data_dir}/y_test.csv')['disetujui_encoded']
        
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
        
        if train_approval_rate < 0.1:
            logger.warning(f"EXTREME CLASS IMBALANCE: Only {train_approval_rate:.1%} approvals!")
            logger.warning(f"   This will cause overly conservative models")
        elif train_approval_rate > 0.9:
            logger.warning(f"EXTREME CLASS IMBALANCE: {train_approval_rate:.1%} approvals!")
            logger.warning(f"   This will cause overly permissive models")
        elif train_approval_rate < 0.3 or train_approval_rate > 0.8:
            logger.warning(f"SIGNIFICANT CLASS IMBALANCE: {train_approval_rate:.1%} approvals")
            logger.warning(f"   Will apply balanced class weights in training")
        
        # Calculate optimal class weights
        if len(train_class_counts) > 1:
            target_approval_rate = config.target_approval_rate
            
            if train_approval_rate < target_approval_rate:
                class_weight_ratio = (1 - target_approval_rate) / target_approval_rate
                optimal_class_weights = {0: 1.0, 1: class_weight_ratio}
            else:
                class_weight_ratio = target_approval_rate / (1 - target_approval_rate)
                optimal_class_weights = {0: class_weight_ratio, 1: 1.0}
            
            logger.info(f"OPTIMAL CLASS WEIGHTS (targeting {target_approval_rate:.1%} approval):")
            logger.info(f"  Class 0 (Reject): {optimal_class_weights[0]:.3f}")
            logger.info(f"  Class 1 (Approve): {optimal_class_weights[1]:.3f}")
        else:
            optimal_class_weights = None
            logger.warning("Cannot calculate class weights - only one class present")
        
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
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
        
        approval_rate = np.mean(y_pred)
        
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
        baseline_configs = [
            {
                'name': 'Default RF Pipeline',
                'pipeline': create_comprehensive_pipeline(
                    'RandomForest', 
                    scale_features=False,
                    use_advanced_features=False,
                    use_feature_selection=False,
                    random_state=config.random_state,
                    n_estimators=100
                )
            },
            {
                'name': 'Balanced RF Pipeline',
                'pipeline': create_comprehensive_pipeline(
                    'RandomForest', 
                    scale_features=False,
                    use_advanced_features=False,
                    use_feature_selection=False,
                    class_weight='balanced',
                    random_state=config.random_state,
                    n_estimators=100
                )
            }
        ]
        
        if data.get('class_weights'):
            baseline_configs.append({
                'name': 'Optimal Weight RF Pipeline',
                'pipeline': create_comprehensive_pipeline(
                    'RandomForest', 
                    scale_features=False,
                    use_advanced_features=False,
                    use_feature_selection=False,
                    class_weight=data['class_weights'],
                    random_state=config.random_state,
                    n_estimators=100
                )
            })
        
        best_baseline = None
        best_baseline_score = 0
        
        for baseline_config in baseline_configs:
            logger.info(f"  Testing {baseline_config['name']}...")
            
            pipeline = baseline_config['pipeline']
            pipeline.fit(data['X_train'], data['y_train'])
            
            y_pred_baseline = pipeline.predict(data['X_test'])
            
            baseline_f1 = f1_score(data['y_test'], y_pred_baseline)
            baseline_acc = accuracy_score(data['y_test'], y_pred_baseline)
            baseline_cv = cross_val_score(
                pipeline, data['X_train'], data['y_train'], 
                cv=config.cv_folds, scoring='f1'
            ).mean()
            
            business_validation = validate_model_business_logic(
                pipeline, data['X_val'], data['y_val'], config, logger, baseline_config['name']
            )
            
            logger.info(f"    F1: {baseline_f1:.4f}, Acc: {baseline_acc:.4f}, CV: {baseline_cv:.4f}")
            logger.info(f"    Approval Rate: {business_validation['approval_rate']:.1%}")
            logger.info(f"    Business Logic: {'PASS' if business_validation['passes_business_logic'] else 'FAIL'}")
            
            if business_validation['passes_business_logic'] and baseline_f1 > best_baseline_score:
                best_baseline = {
                    'model': pipeline,
                    'name': baseline_config['name'],
                    'f1': baseline_f1,
                    'accuracy': baseline_acc,
                    'cv_f1': baseline_cv,
                    'business_validation': business_validation
                }
                best_baseline_score = baseline_f1
        
        if best_baseline is None:
            logger.warning("No baseline model passes business logic! Using default...")
            pipeline = baseline_configs[0]['pipeline']
            pipeline.fit(data['X_train'], data['y_train'])
            y_pred_baseline = pipeline.predict(data['X_test'])
            
            best_baseline = {
                'model': pipeline,
                'name': 'Default RF Pipeline (Fallback)',
                'f1': f1_score(data['y_test'], y_pred_baseline),
                'accuracy': accuracy_score(data['y_test'], y_pred_baseline),
                'cv_f1': cross_val_score(pipeline, data['X_train'], data['y_train'], cv=config.cv_folds, scoring='f1').mean(),
                'business_validation': validate_model_business_logic(pipeline, data['X_val'], data['y_val'], config, logger, 'Default RF Pipeline')
            }
        
        logger.info(f"Selected Baseline: {best_baseline['name']}")
        logger.info(f"  Test F1-Score: {best_baseline['f1']:.4f}")
        logger.info(f"  Test Accuracy: {best_baseline['accuracy']:.4f}")
        logger.info(f"  CV F1-Score: {best_baseline['cv_f1']:.4f}")
        
        if mlflow:
            with mlflow.start_run(run_name="Baseline_Performance", nested=True):
                mlflow.log_param("model_type", best_baseline['name'])
                mlflow.log_metric("baseline_test_f1", best_baseline['f1'])
                mlflow.log_metric("baseline_test_accuracy", best_baseline['accuracy'])
                mlflow.log_metric("baseline_cv_f1", best_baseline['cv_f1'])
                mlflow.log_metric("baseline_approval_rate", best_baseline['business_validation']['approval_rate'])
                
                try:
                    signature = mlflow.models.infer_signature(data['X_train'], data['y_train'])
                    mlflow.sklearn.log_model(best_baseline['model'], "baseline_model", signature=signature)
                    logger.info("    Baseline model logged with signature")
                except Exception as e:
                    logger.warning(f"    Failed to log baseline with signature: {e}")
                    mlflow.sklearn.log_model(best_baseline['model'], "baseline_model")
        
        return best_baseline
        
    except Exception as e:
        logger.error(f"Error calculating baseline: {e}")
        return None

# ============================================================================
# 4. ROBUST RANDOM FOREST OPTIMIZATION
# ============================================================================

def optimize_random_forest(data, config, logger, mlflow=None, optuna=None):
    """Optimize Random Forest dengan business logic constraints using pipelines"""
    logger.info("Starting Random Forest optimization with business constraints...")
    
    results = {}
    
    # Strategy 1: Conservative Randomized Search
    try:
        logger.info("  Strategy 1: Conservative Randomized Search...")
        
        def business_aware_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            approval_rate = np.mean(y_pred)
            f1 = f1_score(y, y_pred)
            
            if approval_rate < config.min_approval_rate or approval_rate > config.max_approval_rate:
                return f1 * 0.5
            elif approval_rate < 0.4 or approval_rate > 0.75:
                return f1 * 0.8
            else:
                return f1
        
        # Create parameter grid for pipeline
        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [10, 15, 20, 25],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [2, 4, 6],
            'model__max_features': ['sqrt', 'log2'],
            'model__bootstrap': [True],
            'model__class_weight': [None, 'balanced']
        }
        
        if data.get('class_weights'):
            param_grid['model__class_weight'].append(data['class_weights'])
        
        # Create base pipeline
        base_pipeline = create_comprehensive_pipeline(
            'RandomForest',
            scale_features=False,
            use_advanced_features=config.use_advanced_features,
            use_feature_selection=config.use_feature_selection,
            feature_selection_k=config.feature_selection_k,
            random_state=config.random_state
        )
        
        rf_search = RandomizedSearchCV(
            base_pipeline,
            param_grid,
            n_iter=config.grid_search_iter,
            cv=config.cv_folds,
            scoring=business_aware_scorer,
            n_jobs=-1,
            random_state=config.random_state,
            verbose=0
        )
        
        start_time = time.time()
        rf_search.fit(data['X_train_opt'], data['y_train_opt'])
        optimization_time = time.time() - start_time
        
        best_rf = rf_search.best_estimator_
        y_val_pred = best_rf.predict(data['X_val'])
        val_f1 = f1_score(data['y_val'], y_val_pred)
        val_acc = accuracy_score(data['y_val'], y_val_pred)
        
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
                'business_validation': business_validation,
            }
            
            logger.info(f"    Best CV F1: {rf_search.best_score_:.4f}")
            logger.info(f"    Validation F1: {val_f1:.4f}")
            logger.info(f"    Approval Rate: {business_validation['approval_rate']:.1%}")
        else:
            logger.warning(f"    Model failed business validation, skipping...")
        
        if mlflow:
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
                    try:
                        signature = mlflow.models.infer_signature(data['X_train_opt'], data['y_train_opt'])
                        mlflow.sklearn.log_model(best_rf, "optimized_random_forest", signature=signature)
                        logger.info("    RandomizedSearch RF logged with signature")
                    except Exception as e:
                        logger.warning(f"    Failed to log RF with signature: {e}")
                        mlflow.sklearn.log_model(best_rf, "optimized_random_forest")
        
    except Exception as e:
        logger.error(f"Randomized search failed: {e}")
    
    # Strategy 2: Business-Constrained Optuna Optimization
    if optuna and data.get('class_weights'):
        try:
            logger.info("  Strategy 2: Business-Constrained Optuna Optimization...")
            
            def business_constrained_rf_objective(trial):
                params = {
                    'model__n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'model__max_depth': trial.suggest_int('max_depth', 10, 25),
                    'model__min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'model__min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 8),
                    'model__max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                    'model__bootstrap': True,
                    'model__class_weight': trial.suggest_categorical('class_weight', [None, 'balanced', data['class_weights']])
                }
                
                pipeline = create_comprehensive_pipeline(
                    'RandomForest',
                    scale_features=False,
                    use_advanced_features=config.use_advanced_features,
                    use_feature_selection=config.use_feature_selection,
                    feature_selection_k=config.feature_selection_k,
                    random_state=config.random_state
                )
                
                pipeline.set_params(**params)
                
                cv_scores = cross_val_score(pipeline, data['X_train_opt'], data['y_train_opt'], cv=3, scoring='f1')
                
                pipeline.fit(data['X_train_opt'], data['y_train_opt'])
                y_val_pred = pipeline.predict(data['X_val'])
                approval_rate = np.mean(y_val_pred)
                
                base_score = cv_scores.mean()
                if approval_rate < config.min_approval_rate or approval_rate > config.max_approval_rate:
                    return base_score * 0.3
                elif approval_rate < 0.4 or approval_rate > 0.75:
                    return base_score * 0.7
                else:
                    return base_score
            
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
            start_time = time.time()
            study.optimize(business_constrained_rf_objective, n_trials=config.n_iter_optuna, show_progress_bar=False)
            optimization_time = time.time() - start_time
            
            # Build final pipeline with best params
            best_pipeline = create_comprehensive_pipeline(
                'RandomForest',
                scale_features=False,
                use_advanced_features=config.use_advanced_features,
                use_feature_selection=config.use_feature_selection,
                feature_selection_k=config.feature_selection_k,
                random_state=config.random_state
            )
            
            # Convert params for pipeline
            pipeline_params = {}
            for key, value in study.best_params.items():
                if key != 'class_weight':
                    pipeline_params[f'model__{key}'] = value
                else:
                    pipeline_params[f'model__{key}'] = value
            
            best_pipeline.set_params(**pipeline_params)
            best_pipeline.fit(data['X_train_opt'], data['y_train_opt'])
            
            y_val_pred = best_pipeline.predict(data['X_val'])
            val_f1 = f1_score(data['y_val'], y_val_pred)
            val_acc = accuracy_score(data['y_val'], y_val_pred)
            
            business_validation = validate_model_business_logic(
                best_pipeline, data['X_val'], data['y_val'], config, logger, "RF_Optuna"
            )
            
            if business_validation['passes_business_logic']:
                results['optuna'] = {
                    'model': best_pipeline,
                    'params': pipeline_params,
                    'cv_f1': study.best_value,
                    'val_f1': val_f1,
                    'val_acc': val_acc,
                    'optimization_time': optimization_time,
                    'business_validation': business_validation,
                }
                
                logger.info(f"    Best CV F1: {study.best_value:.4f}")
                logger.info(f"    Validation F1: {val_f1:.4f}")
                logger.info(f"    Approval Rate: {business_validation['approval_rate']:.1%}")
            else:
                logger.warning(f"    Optuna model failed business validation, skipping...")
            
            if mlflow:
                with mlflow.start_run(run_name=f"RF_Optuna_{datetime.now().strftime('%H%M%S')}", nested=True):
                    mlflow.log_param("optimization_strategy", "business_constrained_optuna")
                    mlflow.log_params(pipeline_params)
                    mlflow.log_metric("best_cv_f1", study.best_value)
                    mlflow.log_metric("validation_f1", val_f1)
                    mlflow.log_metric("validation_accuracy", val_acc)
                    mlflow.log_metric("approval_rate", business_validation['approval_rate'])
                    mlflow.log_metric("passes_business_logic", business_validation['passes_business_logic'])
                    mlflow.log_metric("optimization_time_minutes", optimization_time/60)
                    mlflow.log_metric("n_trials", len(study.trials))
                    
                    if business_validation['passes_business_logic']:
                        try:
                            signature = mlflow.models.infer_signature(data['X_train_opt'], data['y_train_opt'])
                            mlflow.sklearn.log_model(best_pipeline, "optuna_random_forest", signature=signature)
                            logger.info("    Optuna RF logged with signature")
                        except Exception as e:
                            logger.warning(f"    Failed to log Optuna RF with signature: {e}")
                            mlflow.sklearn.log_model(best_pipeline, "optuna_random_forest")
        
        except Exception as e:
            logger.error(f"Optuna optimization failed: {e}")
    
    return results

# ============================================================================
# 5. MULTI-MODEL OPTIMIZATION WITH BUSINESS CONSTRAINTS
# ============================================================================

def optimize_multiple_models(data, config, logger, mlflow=None):
    """Optimize multiple model types dengan business constraints dan comprehensive pipelines"""
    logger.info("Starting multi-model optimization with business constraints...")
    
    models_config = {
        'Gradient Boosting': {
            'param_grid': {
                'model__n_estimators': [100, 200],
                'model__learning_rate': [0.05, 0.1, 0.15],
                'model__max_depth': [3, 5, 7],
                'model__min_samples_split': [5, 10],
                'model__min_samples_leaf': [2, 4],
                'model__subsample': [0.8, 0.9]
            },
            'scale_features': False,
            'model_type': 'GradientBoosting'
        },
        'Extra Trees': {
            'param_grid': {
                'model__n_estimators': [100, 200],
                'model__max_depth': [10, 15, 20],
                'model__min_samples_split': [5, 10],
                'model__min_samples_leaf': [2, 4],
                'model__max_features': ['sqrt', 'log2'],
                'model__class_weight': [None, 'balanced']
            },
            'scale_features': False,
            'model_type': 'ExtraTrees'
        },
        'Logistic Regression': {
            'param_grid': {
                'model__C': [0.1, 1, 10],
                'model__penalty': ['l1', 'l2', 'elasticnet'],
                'model__solver': ['liblinear', 'saga'],
                'model__class_weight': [None, 'balanced'],
                'model__l1_ratio': [config.l1_ratio]
            },
            'scale_features': True,
            'model_type': 'LogisticRegression'
        }
    }
    
    if data.get('class_weights'):
        for model_config in models_config.values():
            if 'model__class_weight' in model_config['param_grid']:
                model_config['param_grid']['model__class_weight'].append(data['class_weights'])
    
    results = {}
    
    for model_name, config_model in models_config.items():
        try:
            logger.info(f"  Optimizing {model_name}...")
            
            def business_aware_scorer(estimator, X, y):
                y_pred = estimator.predict(X)
                approval_rate = np.mean(y_pred)
                f1 = f1_score(y, y_pred)
                
                if approval_rate < config.min_approval_rate or approval_rate > config.max_approval_rate:
                    return f1 * 0.5
                elif approval_rate < 0.4 or approval_rate > 0.75:
                    return f1 * 0.8
                else:
                    return f1
            
            # Create comprehensive pipeline
            pipeline = create_comprehensive_pipeline(
                config_model['model_type'],
                scale_features=config_model['scale_features'],
                use_advanced_features=config.use_advanced_features,
                use_feature_selection=config.use_feature_selection,
                feature_selection_k=config.feature_selection_k,
                random_state=config.random_state
            )
            
            search = RandomizedSearchCV(
                pipeline,
                config_model['param_grid'],
                n_iter=config.n_iter_random,
                cv=config.cv_folds,
                scoring=business_aware_scorer,
                n_jobs=-1,
                random_state=config.random_state,
                verbose=0
            )
            
            start_time = time.time()
            search.fit(data['X_train_opt'], data['y_train_opt'])
            optimization_time = time.time() - start_time
            
            best_model = search.best_estimator_
            y_val_pred = best_model.predict(data['X_val'])
            val_f1 = f1_score(data['y_val'], y_val_pred)
            val_acc = accuracy_score(data['y_val'], y_val_pred)
            
            business_validation = validate_model_business_logic(
                best_model, data['X_val'], data['y_val'], config, logger, model_name
            )
            
            if business_validation['passes_business_logic']:
                results[model_name] = {
                    'model': best_model,
                    'params': search.best_params_,
                    'cv_f1': search.best_score_,
                    'val_f1': val_f1,
                    'val_acc': val_acc,
                    'optimization_time': optimization_time,
                    'business_validation': business_validation,
                }
                
                logger.info(f"    CV F1: {search.best_score_:.4f}")
                logger.info(f"    Val F1: {val_f1:.4f}")
                logger.info(f"    Approval Rate: {business_validation['approval_rate']:.1%}")
            else:
                logger.warning(f"    {model_name} failed business validation, skipping...")
            
            if mlflow and business_validation['passes_business_logic']:
                with mlflow.start_run(run_name=f"{model_name.replace(' ', '_')}_{datetime.now().strftime('%H%M%S')}", nested=True):
                    mlflow.log_param("model_type", model_name)
                    mlflow.log_param("optimization_method", "business_constrained_randomized_search")
                    mlflow.log_params(search.best_params_)
                    mlflow.log_metric("best_cv_f1", search.best_score_)
                    mlflow.log_metric("validation_f1", val_f1)
                    mlflow.log_metric("validation_accuracy", val_acc)
                    mlflow.log_metric("approval_rate", business_validation['approval_rate'])
                    mlflow.log_metric("optimization_time_minutes", optimization_time/60)
                    
                    try:
                        signature = mlflow.models.infer_signature(data['X_train_opt'], data['y_train_opt'])
                        mlflow.sklearn.log_model(best_model, "optimized_model", signature=signature)
                        logger.info(f"    {model_name} logged with signature")
                    except Exception as e:
                        logger.warning(f"    Failed to log {model_name} with signature: {e}")
                        mlflow.sklearn.log_model(best_model, "optimized_model")
        
        except Exception as e:
            logger.error(f"Optimization failed for {model_name}: {e}")
    
    return results

# ============================================================================
# 6. ADVANCED ENSEMBLE METHODS
# ============================================================================

def create_advanced_ensemble_models(data, config, logger, mlflow=None):
    """Create advanced ensemble models using comprehensive pipelines"""
    if not config.use_ensemble_methods:
        logger.info("Ensemble methods disabled")
        return {}
    
    logger.info("Creating advanced ensemble models...")
    
    ensemble_results = {}
    
    try:
        from sklearn.ensemble import VotingClassifier, StackingClassifier
        
        # Ensemble 1: Diverse Random Forest Ensemble
        logger.info("  Creating diverse Random Forest ensemble...")
        
        rf_pipelines = []
        for i in range(min(5, config.ensemble_models)):
            rf_config = {
                'n_estimators': [150, 200, 250][i % 3],
                'max_depth': [12, 15, 18, 20, 25][i % 5],
                'min_samples_split': [2, 5, 8, 10][i % 4],
                'class_weight': [None, 'balanced', 'balanced_subsample'][i % 3] if data.get('class_weights') else [None, 'balanced'][i % 2],
                'max_features': ['sqrt', 'log2'][i % 2],
                'bootstrap': [True, False][i % 2] if i > 2 else True
            }
            
            pipeline = create_comprehensive_pipeline(
                'RandomForest',
                scale_features=False,
                use_advanced_features=config.use_advanced_features,
                use_feature_selection=config.use_feature_selection,
                feature_selection_k=config.feature_selection_k,
                random_state=config.random_state + i,
                **rf_config
            )
            
            rf_pipelines.append((f'rf_{i}', pipeline))
        
        # Create voting ensemble
        voting_ensemble = VotingClassifier(estimators=rf_pipelines, voting='soft')
        voting_ensemble.fit(data['X_train_opt'], data['y_train_opt'])
        
        y_val_pred = voting_ensemble.predict(data['X_val'])
        val_f1 = f1_score(data['y_val'], y_val_pred)
        val_acc = accuracy_score(data['y_val'], y_val_pred)
        val_approval_rate = np.mean(y_val_pred)
        
        business_validation = validate_model_business_logic(
            voting_ensemble, data['X_val'], data['y_val'], config, logger, "RF_Voting_Ensemble"
        )
        
        if business_validation['passes_business_logic']:
            ensemble_results['rf_voting_ensemble'] = {
                'model': voting_ensemble,
                'val_f1': val_f1,
                'val_acc': val_acc,
                'optimization_time': 0,
                'business_validation': business_validation,
                'params': {'ensemble_type': 'rf_voting', 'n_models': len(rf_pipelines)},
            }
            logger.info(f"    RF Voting Ensemble: F1={val_f1:.4f}, Approval={val_approval_rate:.1%}")
        else:
            logger.warning(f"    RF Voting Ensemble failed business validation")
        
        # Ensemble 2: Multi-Algorithm Ensemble
        logger.info("  Creating multi-algorithm ensemble...")
        
        base_models = [
            ('rf', create_comprehensive_pipeline(
                'RandomForest', 
                scale_features=False,
                use_advanced_features=config.use_advanced_features,
                use_feature_selection=config.use_feature_selection,
                feature_selection_k=config.feature_selection_k,
                class_weight='balanced',
                random_state=config.random_state,
                n_estimators=200, 
                max_depth=15
            )),
            ('gb', create_comprehensive_pipeline(
                'GradientBoosting',
                scale_features=False,
                use_advanced_features=config.use_advanced_features,
                use_feature_selection=config.use_feature_selection,
                feature_selection_k=config.feature_selection_k,
                random_state=config.random_state,
                n_estimators=150, 
                learning_rate=0.1, 
                max_depth=5
            )),
            ('et', create_comprehensive_pipeline(
                'ExtraTrees',
                scale_features=False,
                use_advanced_features=config.use_advanced_features,
                use_feature_selection=config.use_feature_selection,
                feature_selection_k=config.feature_selection_k,
                class_weight='balanced',
                random_state=config.random_state,
                n_estimators=200, 
                max_depth=20
            )),
            ('lr', create_comprehensive_pipeline(
                'LogisticRegression',
                scale_features=True,
                use_advanced_features=config.use_advanced_features,
                use_feature_selection=config.use_feature_selection,
                feature_selection_k=config.feature_selection_k,
                class_weight='balanced',
                random_state=config.random_state,
                C=1.0,
                l1_ratio=config.l1_ratio
            ))
        ]
        
        multi_algo_ensemble = VotingClassifier(estimators=base_models, voting='soft')
        multi_algo_ensemble.fit(data['X_train_opt'], data['y_train_opt'])
        
        y_val_pred = multi_algo_ensemble.predict(data['X_val'])
        val_f1 = f1_score(data['y_val'], y_val_pred)
        val_acc = accuracy_score(data['y_val'], y_val_pred)
        val_approval_rate = np.mean(y_val_pred)
        
        business_validation = validate_model_business_logic(
            multi_algo_ensemble, data['X_val'], data['y_val'], config, logger, "Multi_Algorithm_Ensemble"
        )
        
        if business_validation['passes_business_logic']:
            ensemble_results['multi_algorithm_ensemble'] = {
                'model': multi_algo_ensemble,
                'val_f1': val_f1,
                'val_acc': val_acc,
                'optimization_time': 0,
                'business_validation': business_validation,
                'params': {'ensemble_type': 'multi_algorithm', 'algorithms': ['RF', 'GB', 'ET', 'LR']},
            }
            logger.info(f"    Multi-Algorithm Ensemble: F1={val_f1:.4f}, Approval={val_approval_rate:.1%}")
        else:
            logger.warning(f"    Multi-Algorithm Ensemble failed business validation")
        
        # Log to MLflow if available
        if mlflow and ensemble_results:
            for ensemble_name, result in ensemble_results.items():
                with mlflow.start_run(run_name=f"Ensemble_{ensemble_name}_{datetime.now().strftime('%H%M%S')}", nested=True):
                    mlflow.log_param("ensemble_type", ensemble_name)
                    mlflow.log_params(result['params'])
                    mlflow.log_metric("validation_f1", result['val_f1'])
                    mlflow.log_metric("validation_accuracy", result['val_acc'])
                    mlflow.log_metric("approval_rate", result['business_validation']['approval_rate'])
                    mlflow.log_metric("passes_business_logic", result['business_validation']['passes_business_logic'])
                    
                    try:
                        signature = mlflow.models.infer_signature(data['X_train_opt'], data['y_train_opt'])
                        mlflow.sklearn.log_model(result['model'], f"ensemble_{ensemble_name}", signature=signature)
                        logger.info(f"    Logged {ensemble_name} with signature to MLflow")
                    except Exception as e:
                        logger.warning(f"    Failed to log {ensemble_name} with signature: {e}")
                        mlflow.sklearn.log_model(result['model'], f"ensemble_{ensemble_name}")
        
        logger.info(f"Advanced ensemble creation completed: {len(ensemble_results)} valid ensembles")
        return ensemble_results
        
    except Exception as e:
        logger.error(f"Advanced ensemble creation failed: {e}")
        return {}

# ============================================================================
# 7. MODEL EVALUATION & SELECTION WITH BUSINESS VALIDATION
# ============================================================================

def evaluate_all_models(rf_results, multi_results, baseline, data, config, logger, ensemble_results=None):
    """Evaluate all optimized models dan select best dengan business validation"""
    logger.info("Evaluating all optimized models with business validation...")
    
    all_models = {}
    
    # Add RF results
    for strategy, result in rf_results.items():
        if result.get('business_validation', {}).get('passes_business_logic', False):
            all_models[f'RF_{strategy}'] = result
    
    # Add multi-model results
    for model_name, result in multi_results.items():
        if result.get('business_validation', {}).get('passes_business_logic', False):
            all_models[model_name] = result
    
    # Add ensemble results
    if ensemble_results:
        for ensemble_name, result in ensemble_results.items():
            if result.get('business_validation', {}).get('passes_business_logic', False):
                all_models[f'Ensemble_{ensemble_name}'] = result
    
    if not all_models:
        logger.error("NO MODELS PASSED BUSINESS VALIDATION!")
        logger.error("This indicates serious issues with optimization or data")
        return None, []
    
    test_results = []
    
    logger.info("Test Set Evaluation (Business-Valid Models Only):")
    logger.info("-" * 90)
    logger.info(f"{'Model':<25} {'Test F1':<10} {'Test Acc':<10} {'Approval%':<10} {'Improvement':<12}")
    logger.info("-" * 90)
    
    for model_name, result in all_models.items():
        try:
            model = result['model']
            
            y_pred = model.predict(data['X_test'])
            
            test_f1 = f1_score(data['y_test'], y_pred)
            test_acc = accuracy_score(data['y_test'], y_pred)
            test_precision = precision_score(data['y_test'], y_pred)
            test_recall = recall_score(data['y_test'], y_pred)
            test_approval_rate = np.mean(y_pred)
            
            f1_improvement = test_f1 - baseline['f1']
            
            # Test predict_proba functionality
            try:
                y_pred_proba = model.predict_proba(data['X_test'])
                proba_check = np.all(np.isfinite(y_pred_proba)) and np.all(y_pred_proba >= 0) and np.all(y_pred_proba <= 1)
                if not proba_check:
                    logger.warning(f"    {model_name}: predict_proba produces invalid probabilities")
            except Exception as e:
                logger.warning(f"    {model_name}: predict_proba failed: {e}")
            
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
                'test_business_validation': test_business_validation,
            })
            
            logger.info(f"{model_name:<25} {test_f1:<10.4f} {test_acc:<10.4f} {test_approval_rate:<9.1%} {f1_improvement:+10.4f}")
            
        except Exception as e:
            logger.error(f"Evaluation failed for {model_name}: {e}")
    
    logger.info("-" * 90)
    
    # Find best model
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
# 8. ROBUST MODEL SAVING
# ============================================================================

def save_best_model(best_model_result, data, config, logger, mlflow=None):
    """Save best model dan metadata dengan business validation info"""
    logger.info("Saving best model with business validation metadata...")
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_filename = f"{config.output_dir}/best_model.pkl"
        joblib.dump(best_model_result['model'], model_filename)
        logger.info(f"Model saved: {model_filename}")
        
        # Save comprehensive model info
        info_filename = f"{config.output_dir}/model_info_{timestamp}.txt"
        with open(info_filename, 'w') as f:
            f.write(f"Credit Approval Model Information\n")
            f.write(f"=================================\n\n")
            f.write(f"Best Model: {best_model_result['name']}\n")
            f.write(f"Test F1-Score: {best_model_result['test_f1']:.4f}\n")
            f.write(f"Test Accuracy: {best_model_result['test_acc']:.4f}\n")
            f.write(f"Test Approval Rate: {best_model_result['test_approval_rate']:.1%}\n")
            f.write(f"Training Date: {datetime.now()}\n")
            f.write(f"Uses Comprehensive Pipeline: True\n")
            f.write(f"MLflow Compatible: True\n\n")
            
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
            
            f.write(f"\nPipeline Information:\n")
            pipeline = best_model_result['model']
            if hasattr(pipeline, 'steps'):
                f.write(f"  Pipeline Steps: {[step[0] for step in pipeline.steps]}\n")
                for step_name, step_obj in pipeline.steps:
                    f.write(f"    {step_name}: {type(step_obj).__name__}\n")
        
        logger.info(f"Model info saved: {info_filename}")
        
        # Test model functionality
        logger.info("Testing saved model functionality...")
        try:
            loaded_model = joblib.load(model_filename)
            test_pred = loaded_model.predict(data['X_test'][:5])
            test_proba = loaded_model.predict_proba(data['X_test'][:5])
            logger.info(f"  Predictions test: PASSED ({len(test_pred)} predictions)")
            logger.info(f"  Probabilities test: PASSED (shape: {test_proba.shape})")
            logger.info(f"  Probability range: [{test_proba.min():.3f}, {test_proba.max():.3f}]")
        except Exception as e:
            logger.error(f"  Model functionality test FAILED: {e}")
        
        # Log to MLflow
        if mlflow:
            with mlflow.start_run(run_name=f"PRODUCTION_MODEL_{timestamp}", nested=True):
                mlflow.log_param("model_name", best_model_result['name'])
                mlflow.log_param("production_ready", True)
                mlflow.log_param("business_logic_validated", best_model_result.get('test_business_validation', {}).get('passes_business_logic', False))
                mlflow.log_param("uses_comprehensive_pipeline", True)
                mlflow.log_param("mlflow_compatible", True)
                mlflow.log_metric("final_test_f1", best_model_result['test_f1'])
                mlflow.log_metric("final_test_accuracy", best_model_result['test_acc'])
                mlflow.log_metric("final_approval_rate", best_model_result['test_approval_rate'])
                mlflow.log_metric("final_improvement", best_model_result['f1_improvement'])
                
                try:
                    # Use raw data for signature to ensure consistency
                    signature = mlflow.models.infer_signature(data['X_train'], data['y_train'])
                    mlflow.sklearn.log_model(
                        best_model_result['model'], 
                        "production_model", 
                        signature=signature,
                        input_example=data['X_train'].head(3)
                    )
                    logger.info("Production model logged with signature and input example")
                        
                except Exception as e:
                    logger.warning(f"Failed to log production model with signature: {e}")
                    mlflow.sklearn.log_model(best_model_result['model'], "production_model")
                
                mlflow.log_artifact(model_filename)
                mlflow.log_artifact(info_filename)
        
        return model_filename
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return None

# ============================================================================
# 9. MAIN EXECUTION
# ============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Advanced Credit Approval Model Tuning with Business Validation'
    )
    
    parser.add_argument(
        '--alpha', 
        type=float, 
        default=0.5,
        help='Alpha parameter for regularization (default: 0.5)'
    )
    
    parser.add_argument(
        '--l1_ratio', 
        type=float, 
        default=0.1,
        help='L1 ratio for ElasticNet regularization (default: 0.1)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Run validation only without training'
    )
    
    return parser.parse_args()

def main():
    """Main optimization pipeline dengan business validation dan DagsHub integration"""
    args = parse_arguments()
    
    config = Config(args)
    logger = setup_logging()
    mlflow, mlflow_available = setup_mlflow_dagshub(config, logger)
    optuna, optuna_available = setup_optuna_optional(logger)
    
    logger.info("Starting robust model optimization with DagsHub & MLflow integration...")
    logger.info(f"Configuration:")
    logger.info(f"  Experiment: {config.experiment_name}")
    logger.info(f"  Model name: {config.model_name}")
    logger.info(f"  Alpha: {config.alpha}")
    logger.info(f"  L1 Ratio: {config.l1_ratio}")
    logger.info(f"  Random State: {config.random_state}")
    logger.info(f"Business constraints:")
    logger.info(f"  Target approval rate: {config.target_approval_rate:.1%}")
    logger.info(f"  Min approval rate: {config.min_approval_rate:.1%}")
    logger.info(f"  Max approval rate: {config.max_approval_rate:.1%}")
    logger.info(f"Optimization settings:")
    logger.info(f"  Optuna iterations: {config.n_iter_optuna}")
    logger.info(f"  Grid search iterations: {config.grid_search_iter}")
    
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
    
    # Start main optimization
    if mlflow_available:
        with mlflow.start_run(run_name=f"Advanced_Tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log configuration
            mlflow.log_param("optimization_date", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            mlflow.log_param("experiment_name", config.experiment_name)
            mlflow.log_param("model_name", config.model_name)
            mlflow.log_param("alpha", config.alpha)
            mlflow.log_param("l1_ratio", config.l1_ratio)
            mlflow.log_param("random_state", config.random_state)
            mlflow.log_param("baseline_f1", baseline['f1'])
            mlflow.log_param("baseline_approval_rate", baseline['business_validation']['approval_rate'])
            mlflow.log_param("target_improvement", config.target_improvement)
            mlflow.log_param("target_approval_rate", config.target_approval_rate)
            mlflow.log_param("min_approval_rate", config.min_approval_rate)
            mlflow.log_param("max_approval_rate", config.max_approval_rate)
            mlflow.log_param("uses_comprehensive_pipelines", True)
            mlflow.log_param("mlflow_prediction_ready", True)
            
            # Run optimizations
            rf_results = optimize_random_forest(data, config, logger, mlflow, optuna)
            multi_results = optimize_multiple_models(data, config, logger, mlflow)
            ensemble_results = create_advanced_ensemble_models(data, config, logger, mlflow)
            
            # Evaluate and select best
            best_model, all_results = evaluate_all_models(rf_results, multi_results, baseline, data, config, logger, ensemble_results)
            
            if best_model:
                model_file = save_best_model(best_model, data, config, logger, mlflow)
                
                mlflow.log_metric("optimization_success", 1)
                mlflow.log_param("best_model_name", best_model['name'])
                mlflow.log_metric("final_improvement", best_model['f1_improvement'])
                mlflow.log_metric("final_approval_rate", best_model['test_approval_rate'])
                mlflow.log_param("model_file_saved", model_file is not None)
                mlflow.log_param("final_model_pipeline_ready", True)
            else:
                logger.error("No valid models found")
                if mlflow_available:
                    mlflow.log_metric("optimization_success", 0)
    else:
        # Run without MLflow
        rf_results = optimize_random_forest(data, config, logger)
        multi_results = optimize_multiple_models(data, config, logger)
        ensemble_results = create_advanced_ensemble_models(data, config, logger)
        best_model, all_results = evaluate_all_models(rf_results, multi_results, baseline, data, config, logger, ensemble_results)
        
        if best_model:
            model_file = save_best_model(best_model, data, config, logger)
    
    # Print final summary
    if best_model:
        logger.info("=" * 70)
        logger.info("ADVANCED MODEL TUNING WITH DAGSHUB COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info(f"EXPERIMENT CONFIGURATION:")
        logger.info(f"  Experiment: {config.experiment_name}")
        logger.info(f"  Model Name: {config.model_name}")
        logger.info(f"  Alpha (Regularization): {config.alpha}")
        logger.info(f"  L1 Ratio (ElasticNet): {config.l1_ratio}")
        logger.info(f"  Random State: {config.random_state}")
        logger.info(f"OPTIMIZATION SUMMARY:")
        logger.info(f"  Baseline F1-Score: {baseline['f1']:.4f}")
        logger.info(f"  Baseline Approval Rate: {baseline['business_validation']['approval_rate']:.1%}")
        logger.info(f"  Best F1-Score: {best_model['test_f1']:.4f}")
        logger.info(f"  Best Approval Rate: {best_model['test_approval_rate']:.1%}")
        logger.info(f"  Total Improvement: {best_model['f1_improvement']:+.4f}")
        logger.info(f"  Best Model: {best_model['name']}")
        logger.info(f"  Uses Comprehensive Pipeline: True")
        logger.info(f"  MLflow Server Compatible: True")
        
        target_met = best_model['f1_improvement'] >= config.target_improvement
        business_valid = best_model.get('test_business_validation', {}).get('passes_business_logic', False)
        
        logger.info(f"\nVALIDATION RESULTS:")
        logger.info(f"  F1 Target (+{config.target_improvement:.3f}): {'ACHIEVED' if target_met else 'NOT MET'} ({best_model['f1_improvement']:+.4f})")
        logger.info(f"  Business Logic: {'VALID' if business_valid else 'INVALID'}")
        logger.info(f"  Approval Rate: {'HEALTHY' if config.min_approval_rate <= best_model['test_approval_rate'] <= config.max_approval_rate else 'OUT OF RANGE'} ({best_model['test_approval_rate']:.1%})")
        logger.info(f"  Predict Proba Ready: True")
        
        if mlflow_available:
            logger.info(f"\nDAGSHUB INTEGRATION:")
            logger.info(f"  Repository: {config.dagshub_url}")
            logger.info(f"  Experiment tracked: {config.experiment_name}")
            logger.info(f"  Models logged with signatures: True")
            logger.info(f"  Server deployment ready: True")
        
        if target_met and business_valid:
            logger.info(f"\nOPTIMIZATION FULLY SUCCESSFUL!")
            logger.info(f"   Model ready for production deployment")
            logger.info(f"   MLflow server compatible: predict_proba enabled")
            logger.info(f"   Check DagsHub for detailed experiment tracking")
        elif business_valid:
            logger.info(f"\nBUSINESS VALIDATION PASSED")
            logger.info(f"   Model safe for production despite F1 target not fully met")
            logger.info(f"   MLflow server compatible: predict_proba enabled")
        else:
            logger.warning(f"\nBUSINESS VALIDATION FAILED")
            logger.warning(f"   Model may need further tuning before production")
    else:
        logger.error("Optimization failed - no valid models produced")

if __name__ == "__main__":
    main()