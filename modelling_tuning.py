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

warnings.filterwarnings('ignore')

print("="*70)
print("COMPREHENSIVE MODEL OPTIMIZATION - DAGSHUB & MLFLOW INTEGRATION")
print("="*70)

# ============================================================================
# PROBABILITY WRAPPER FOR MLFLOW - FIX FOR PROBABILITY ISSUE
# ============================================================================

class ProbabilityWrapper:
    """Custom wrapper to ensure MLflow returns probabilities instead of class predictions"""
    
    def __init__(self, model, scaler=None, return_class_1_prob=True):
        """
        Initialize probability wrapper
        
        Args:
            model: Trained sklearn model with predict_proba method
            scaler: Optional StandardScaler for preprocessing
            return_class_1_prob: If True, return probability of class 1 (approval)
                                If False, return full probability array
        """
        self.model = model
        self.scaler = scaler
        self.return_class_1_prob = return_class_1_prob
    
    def predict(self, X):
        """
        Predict method that returns probabilities instead of classes
        This is what MLflow serving will call
        """
        # Apply scaling if scaler is provided
        if self.scaler is not None:
            X_processed = self.scaler.transform(X)
        else:
            X_processed = X
        
        # Get probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_processed)
            
            if self.return_class_1_prob:
                # Return probability of class 1 (approval probability)
                return probabilities[:, 1]
            else:
                # Return full probability array
                return probabilities
        else:
            # Fallback to regular prediction if no predict_proba
            return self.model.predict(X_processed)
    
    def predict_proba(self, X):
        """Standard predict_proba method for compatibility"""
        if self.scaler is not None:
            X_processed = self.scaler.transform(X)
        else:
            X_processed = X
        
        return self.model.predict_proba(X_processed)

def create_mlflow_probability_model(model, scaler=None, return_class_1_prob=True):
    """
    Create MLflow-compatible model that returns probabilities
    
    Args:
        model: Trained sklearn model
        scaler: Optional scaler for preprocessing
        return_class_1_prob: Whether to return only class 1 probability
    
    Returns:
        Wrapped model ready for MLflow logging
    """
    return ProbabilityWrapper(model, scaler, return_class_1_prob)

# ============================================================================
# 2.5. ADVANCED FEATURE ENGINEERING - NEW ADDITION
# ============================================================================

def create_advanced_features(X_train, X_test, config, logger):
    """Create advanced engineered features untuk improve model performance"""
    if not config.use_advanced_features:
        logger.info("Advanced feature engineering disabled, using original features")
        return X_train, X_test
    
    logger.info("Creating advanced engineered features...")
    
    try:
        # Make copies
        X_train_enhanced = X_train.copy()
        X_test_enhanced = X_test.copy()
        
        # Feature 1: Advanced debt ratio categories
        def categorize_debt_ratio(df):
            if 'rasio_pinjaman_pendapatan' in df.columns:
                df['debt_ratio_low'] = (df['rasio_pinjaman_pendapatan'] <= 3).astype(int)
                df['debt_ratio_medium'] = ((df['rasio_pinjaman_pendapatan'] > 3) & 
                                         (df['rasio_pinjaman_pendapatan'] <= 5)).astype(int)
                df['debt_ratio_high'] = ((df['rasio_pinjaman_pendapatan'] > 5) & 
                                       (df['rasio_pinjaman_pendapatan'] <= 7)).astype(int)
                df['debt_ratio_extreme'] = (df['rasio_pinjaman_pendapatan'] > 7).astype(int)
            return df
        
        # Feature 2: Enhanced credit score ranges
        def create_enhanced_credit_ranges(df):
            if 'skor_kredit' in df.columns:
                df['credit_excellent_plus'] = (df['skor_kredit'] >= 800).astype(int)
                df['credit_very_good'] = ((df['skor_kredit'] >= 750) & (df['skor_kredit'] < 800)).astype(int)
                df['credit_good_plus'] = ((df['skor_kredit'] >= 700) & (df['skor_kredit'] < 750)).astype(int)
                df['credit_borderline'] = ((df['skor_kredit'] >= 600) & (df['skor_kredit'] < 650)).astype(int)
                df['credit_high_risk'] = (df['skor_kredit'] < 550).astype(int)
            return df
        
        # Feature 3: Income stability composite score
        def create_income_stability_score(df):
            stability_score = 0
            
            # Job stability component (0-3 points)
            if 'pekerjaan_Tetap' in df.columns:
                stability_score += df['pekerjaan_Tetap'] * 3  # Highest stability
            if 'pekerjaan_Kontrak' in df.columns:
                stability_score += df['pekerjaan_Kontrak'] * 2  # Medium stability
            if 'pekerjaan_Freelance' in df.columns:
                stability_score += df['pekerjaan_Freelance'] * 1  # Lowest stability
            
            # Income level component (0-2 points)
            if 'kategori_pendapatan_Tinggi' in df.columns:
                stability_score += df['kategori_pendapatan_Tinggi'] * 2
            if 'kategori_pendapatan_Sedang' in df.columns:
                stability_score += df['kategori_pendapatan_Sedang'] * 1
            
            df['income_stability_score'] = stability_score
            return df
        
        # Feature 4: Risk combination indicators
        def create_risk_combinations(df):
            # High-risk combination: Young + Freelance + Poor Credit
            if all(col in df.columns for col in ['kategori_umur_Muda', 'pekerjaan_Freelance', 'kategori_skor_kredit_Poor']):
                df['high_risk_combination'] = (
                    df['kategori_umur_Muda'] & 
                    df['pekerjaan_Freelance'] & 
                    df['kategori_skor_kredit_Poor']
                ).astype(int)
            
            # Low-risk combination: Adult + Permanent + Good Credit
            if all(col in df.columns for col in ['kategori_umur_Dewasa', 'pekerjaan_Tetap', 'kategori_skor_kredit_Good']):
                df['low_risk_combination'] = (
                    df['kategori_umur_Dewasa'] & 
                    df['pekerjaan_Tetap'] & 
                    df['kategori_skor_kredit_Good']
                ).astype(int)
            
            # Senior stability combination
            if all(col in df.columns for col in ['kategori_umur_Senior', 'pekerjaan_Tetap']):
                df['senior_stable_combination'] = (
                    df['kategori_umur_Senior'] & 
                    df['pekerjaan_Tetap']
                ).astype(int)
            
            return df
        
        # Feature 5: Age-Income interaction effects
        def create_age_income_interactions(df):
            if all(col in df.columns for col in ['umur', 'pendapatan']):
                # Peak earning years indicator (35-50)
                df['peak_earning_years'] = ((df['umur'] >= 35) & (df['umur'] <= 50)).astype(int)
                
                # Young high earner (potential future growth)
                if 'kategori_pendapatan_Tinggi' in df.columns:
                    df['young_high_earner'] = (
                        (df['umur'] < 35) & df['kategori_pendapatan_Tinggi']
                    ).astype(int)
                
                # Experience factor (age * income level indicator)
                age_norm = (df['umur'] - 18) / (65 - 18)  # Normalize age to 0-1
                income_millions = df['pendapatan'] / 1000000
                df['experience_income_factor'] = age_norm * np.log1p(income_millions)
            
            return df
        
        # Feature 6: Loan efficiency and capacity features
        def create_loan_efficiency_features(df):
            if all(col in df.columns for col in ['jumlah_pinjaman', 'pendapatan', 'umur']):
                # Working years left (assume retirement at 65)
                working_years_left = np.maximum(65 - df['umur'], 5)
                
                # Monthly debt service capacity (assume 30% of income)
                monthly_capacity = df['pendapatan'] * 0.3
                
                # Total repayment capacity over working life
                total_capacity = monthly_capacity * working_years_left * 12
                
                # Loan to total capacity ratio
                df['loan_to_total_capacity'] = (df['jumlah_pinjaman'] / total_capacity).clip(0, 5)
                
                # Loan burden relative to age
                df['age_adjusted_loan_burden'] = df['rasio_pinjaman_pendapatan'] * (70 - df['umur']) / 40
                
                # Debt service ratio (monthly)
                df['monthly_debt_service_ratio'] = (df['jumlah_pinjaman'] / 240) / df['pendapatan']  # Assume 20 year loan
            
            return df
        
        # Apply all feature engineering steps
        logger.info("  Creating debt ratio categories...")
        X_train_enhanced = categorize_debt_ratio(X_train_enhanced)
        X_test_enhanced = categorize_debt_ratio(X_test_enhanced)
        
        logger.info("  Creating enhanced credit score ranges...")
        X_train_enhanced = create_enhanced_credit_ranges(X_train_enhanced)
        X_test_enhanced = create_enhanced_credit_ranges(X_test_enhanced)
        
        logger.info("  Creating income stability scores...")
        X_train_enhanced = create_income_stability_score(X_train_enhanced)
        X_test_enhanced = create_income_stability_score(X_test_enhanced)
        
        logger.info("  Creating risk combination features...")
        X_train_enhanced = create_risk_combinations(X_train_enhanced)
        X_test_enhanced = create_risk_combinations(X_test_enhanced)
        
        logger.info("  Creating age-income interactions...")
        X_train_enhanced = create_age_income_interactions(X_train_enhanced)
        X_test_enhanced = create_age_income_interactions(X_test_enhanced)
        
        logger.info("  Creating loan efficiency features...")
        X_train_enhanced = create_loan_efficiency_features(X_train_enhanced)
        X_test_enhanced = create_loan_efficiency_features(X_test_enhanced)
        
        # Handle any NaN values created during feature engineering
        X_train_enhanced = X_train_enhanced.fillna(0)
        X_test_enhanced = X_test_enhanced.fillna(0)
        
        new_features_count = X_train_enhanced.shape[1] - X_train.shape[1]
        logger.info(f"Advanced feature engineering completed:")
        logger.info(f"   Original features: {X_train.shape[1]}")
        logger.info(f"   New features created: {new_features_count}")
        logger.info(f"   Total features: {X_train_enhanced.shape[1]}")
        
        return X_train_enhanced, X_test_enhanced
        
    except Exception as e:
        logger.error(f"Advanced feature engineering failed: {e}")
        logger.warning("Falling back to original features...")
        return X_train, X_test

def perform_intelligent_feature_selection(X_train, X_test, y_train, config, logger):
    """Perform intelligent feature selection using multiple methods"""
    if not config.use_feature_selection:
        logger.info("Feature selection disabled, using all features")
        return X_train, X_test, X_train.columns.tolist()
    
    logger.info("Performing intelligent feature selection...")
    
    try:
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
        
        # Method 1: Statistical feature selection (f_classif)
        selector_stats = SelectKBest(score_func=f_classif, k=min(config.feature_selection_k, X_train.shape[1]))
        selector_stats.fit(X_train, y_train)
        selected_features_stats = X_train.columns[selector_stats.get_support()].tolist()
        
        # Method 2: Mutual information feature selection
        selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(config.feature_selection_k, X_train.shape[1]))
        selector_mi.fit(X_train, y_train)
        selected_features_mi = X_train.columns[selector_mi.get_support()].tolist()
        
        # Method 3: Random Forest feature importance
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=config.random_state)
        rf_selector.fit(X_train, y_train)
        
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_selector.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_features_rf = feature_importance.head(config.feature_selection_k)['feature'].tolist()
        
        # Ensemble feature selection: Take union but prioritize features that appear in multiple methods
        feature_votes = {}
        for feature in X_train.columns:
            votes = 0
            if feature in selected_features_stats:
                votes += 1
            if feature in selected_features_mi:
                votes += 1
            if feature in top_features_rf:
                votes += 2  # RF gets double weight
            feature_votes[feature] = votes
        
        # Sort by votes and importance, take top K
        sorted_features = sorted(feature_votes.items(), key=lambda x: (x[1], feature_importance[feature_importance['feature'] == x[0]]['importance'].iloc[0] if len(feature_importance[feature_importance['feature'] == x[0]]) > 0 else 0), reverse=True)
        
        final_features = [f[0] for f in sorted_features[:config.feature_selection_k]]
        
        X_train_selected = X_train[final_features]
        X_test_selected = X_test[final_features]
        
        logger.info(f"Intelligent feature selection completed:")
        logger.info(f"   Original features: {X_train.shape[1]}")
        logger.info(f"   Selected features: {len(final_features)}")
        logger.info(f"   Feature reduction: {(1 - len(final_features)/X_train.shape[1])*100:.1f}%")
        
        # Log top selected features
        logger.info(f"   Top 10 selected features:")
        for i, feature in enumerate(final_features[:10]):
            importance = feature_importance[feature_importance['feature'] == feature]['importance'].iloc[0] if len(feature_importance[feature_importance['feature'] == feature]) > 0 else 0
            votes = feature_votes[feature]
            logger.info(f"     {i+1}. {feature}: importance={importance:.4f}, votes={votes}")
        
        return X_train_selected, X_test_selected, final_features
        
    except Exception as e:
        logger.error(f"Feature selection failed: {e}")
        logger.warning("Using all features...")
        return X_train, X_test, X_train.columns.tolist()

# ============================================================================
# 2.6. ADVANCED ENSEMBLE METHODS - NEW ADDITION
# ============================================================================

def create_advanced_ensemble_models(X_train, y_train, X_val, y_val, config, logger, mlflow=None):
    """Create advanced ensemble models for improved performance"""
    if not config.use_ensemble_methods:
        logger.info("Ensemble methods disabled")
        return {}
    
    logger.info("Creating advanced ensemble models...")
    
    ensemble_results = {}
    
    try:
        # Ensemble 1: Diverse Random Forest Ensemble
        logger.info("  Creating diverse Random Forest ensemble...")
        
        rf_configs = [
            {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5, 'class_weight': 'balanced', 'max_features': 'sqrt'},
            {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 10, 'class_weight': None, 'max_features': 'log2'},
            {'n_estimators': 150, 'max_depth': 12, 'min_samples_split': 8, 'class_weight': 'balanced_subsample', 'bootstrap': False},
            {'n_estimators': 250, 'max_depth': 18, 'min_samples_split': 6, 'criterion': 'entropy'},
            {'n_estimators': 180, 'max_depth': 25, 'min_samples_split': 4, 'max_features': 0.6}
        ]
        
        rf_models = []
        for i, rf_config in enumerate(rf_configs[:config.ensemble_models]):
            rf = RandomForestClassifier(random_state=config.random_state + i, **rf_config)
            rf.fit(X_train, y_train)
            rf_models.append((f'rf_{i}', rf))
        
        # Create voting ensemble
        from sklearn.ensemble import VotingClassifier
        voting_ensemble = VotingClassifier(estimators=rf_models, voting='soft')
        voting_ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        y_val_pred = voting_ensemble.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_approval_rate = np.mean(y_val_pred)
        
        # Business validation
        business_validation = validate_model_business_logic(
            voting_ensemble, X_val, y_val, config, logger, "RF_Voting_Ensemble"
        )
        
        if business_validation['passes_business_logic']:
            ensemble_results['rf_voting_ensemble'] = {
                'model': voting_ensemble,
                'val_f1': val_f1,
                'val_acc': val_acc,
                'optimization_time': 0,  # Ensemble creation time
                'business_validation': business_validation,
                'params': {'ensemble_type': 'rf_voting', 'n_models': len(rf_models)}
            }
            logger.info(f"    RF Voting Ensemble: F1={val_f1:.4f}, Approval={val_approval_rate:.1%}")
        else:
            logger.warning(f"    RF Voting Ensemble failed business validation")
        
        # Ensemble 2: Multi-Algorithm Ensemble (if enabled)
        logger.info("  Creating multi-algorithm ensemble...")
        
        # Create pipeline for logistic regression (needs scaling)
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=config.random_state, class_weight='balanced')),
            ('gb', GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=config.random_state)),
            ('et', ExtraTreesClassifier(n_estimators=200, max_depth=20, random_state=config.random_state, class_weight='balanced')),
            ('lr_pipeline', Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(random_state=config.random_state, max_iter=1000, class_weight='balanced'))
            ]))
        ]
        
        multi_algo_ensemble = VotingClassifier(estimators=base_models, voting='soft')
        multi_algo_ensemble.fit(X_train, y_train)
        
        # Evaluate
        y_val_pred = multi_algo_ensemble.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_approval_rate = np.mean(y_val_pred)
        
        business_validation = validate_model_business_logic(
            multi_algo_ensemble, X_val, y_val, config, logger, "Multi_Algorithm_Ensemble"
        )
        
        if business_validation['passes_business_logic']:
            ensemble_results['multi_algorithm_ensemble'] = {
                'model': multi_algo_ensemble,
                'val_f1': val_f1,
                'val_acc': val_acc,
                'optimization_time': 0,
                'business_validation': business_validation,
                'params': {'ensemble_type': 'multi_algorithm', 'algorithms': ['RF', 'GB', 'ET', 'LR']}
            }
            logger.info(f"    Multi-Algorithm Ensemble: F1={val_f1:.4f}, Approval={val_approval_rate:.1%}")
        else:
            logger.warning(f"    Multi-Algorithm Ensemble failed business validation")
        
        # Ensemble 3: Stacked Ensemble (if enabled)
        if config.use_stacking:
            logger.info("  Creating stacked ensemble...")
            
            from sklearn.ensemble import StackingClassifier
            
            # Level 1 models (base learners)
            level1_models = [
                ('rf1', RandomForestClassifier(n_estimators=150, max_depth=15, random_state=config.random_state)),
                ('rf2', RandomForestClassifier(n_estimators=200, max_depth=20, random_state=config.random_state+1, class_weight='balanced')),
                ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=config.random_state)),
                ('et', ExtraTreesClassifier(n_estimators=150, max_depth=18, random_state=config.random_state))
            ]
            
            # Level 2 model (meta-learner)
            meta_learner = LogisticRegression(random_state=config.random_state, max_iter=1000, class_weight='balanced')
            
            # Create stacking classifier
            stacking_ensemble = StackingClassifier(
                estimators=level1_models,
                final_estimator=meta_learner,
                cv=3,  # Use 3-fold CV for meta-features
                stack_method='predict_proba'
            )
            
            stacking_ensemble.fit(X_train, y_train)
            
            # Evaluate
            y_val_pred = stacking_ensemble.predict(X_val)
            val_f1 = f1_score(y_val, y_val_pred)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_approval_rate = np.mean(y_val_pred)
            
            business_validation = validate_model_business_logic(
                stacking_ensemble, X_val, y_val, config, logger, "Stacking_Ensemble"
            )
            
            if business_validation['passes_business_logic']:
                ensemble_results['stacking_ensemble'] = {
                    'model': stacking_ensemble,
                    'val_f1': val_f1,
                    'val_acc': val_acc,
                    'optimization_time': 0,
                    'business_validation': business_validation,
                    'params': {'ensemble_type': 'stacking', 'meta_learner': 'LogisticRegression'}
                }
                logger.info(f"    Stacking Ensemble: F1={val_f1:.4f}, Approval={val_approval_rate:.1%}")
            else:
                logger.warning(f"    Stacking Ensemble failed business validation")
        
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
                    
                    # Create probability wrapper and log model
                    try:
                        probability_model = create_mlflow_probability_model(result['model'])
                        import mlflow.models
                        signature = mlflow.models.infer_signature(X_train, y_train)
                        
                        # Use pyfunc to log the probability wrapper
                        import mlflow.pyfunc
                        mlflow.pyfunc.log_model(
                            artifact_path=f"ensemble_{ensemble_name}",
                            python_model=probability_model,
                            signature=signature
                        )
                        logger.info(f"    Logged {ensemble_name} with probability wrapper to MLflow")
                    except Exception as e:
                        logger.warning(f"    Failed to log {ensemble_name} with probability wrapper: {e}")
                        # Fallback to regular sklearn logging
                        mlflow.sklearn.log_model(result['model'], f"ensemble_{ensemble_name}")
        
        logger.info(f"Advanced ensemble creation completed: {len(ensemble_results)} valid ensembles")
        return ensemble_results
        
    except Exception as e:
        logger.error(f"Advanced ensemble creation failed: {e}")
        return {}

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
        self.n_iter_optuna = int(os.getenv('N_ITER_OPTUNA', '200'))
        self.grid_search_iter = int(os.getenv('GRID_SEARCH_ITER', '200'))
        
        # Target improvement
        self.target_improvement = 0.025
        
        # BUSINESS LOGIC SETTINGS - FROM ENVIRONMENT OR DEFAULTS
        self.min_approval_rate = float(os.getenv('MIN_APPROVAL_RATE', '0.3'))
        self.max_approval_rate = float(os.getenv('MAX_APPROVAL_RATE', '0.8'))
        self.target_approval_rate = float(os.getenv('TARGET_APPROVAL_RATE', '0.6'))
        
        # ADVANCED OPTIMIZATION SETTINGS - NEW
        self.use_advanced_features = True      # Enable advanced feature engineering
        self.use_feature_selection = True      # Enable feature selection
        self.use_ensemble_methods = True       # Enable ensemble optimization
        self.use_stacking = True               # Enable stacking ensemble
        self.feature_selection_k = 20          # Number of features to select
        self.ensemble_models = 5               # Number of models in ensemble
        
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
    """Enhanced MLflow setup dengan DagsHub integration"""
    try:
        import mlflow
        import mlflow.sklearn
        import mlflow.models
        import mlflow.pyfunc
        
        logger.info("Setting up MLflow with DagsHub integration...")
        
        # Validate credentials
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
        
        # Setup DagsHub MLflow tracking
        try:
            # Set tracking URI to DagsHub MLflow
            dagshub_mlflow_uri = f"{config.dagshub_url}.mlflow"
            mlflow.set_tracking_uri(dagshub_mlflow_uri)
            
            # Set authentication
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
        
        # Set or create experiment
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
        
        # Test connection
        try:
            # Try to start and end a test run
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
        
        # Log to MLflow if available
        if mlflow:
            with mlflow.start_run(run_name="Baseline_Performance", nested=True):
                mlflow.log_param("model_type", best_baseline['name'])
                mlflow.log_param("class_weight", str(best_baseline['class_weight']))
                mlflow.log_metric("baseline_test_f1", best_baseline['f1'])
                mlflow.log_metric("baseline_test_accuracy", best_baseline['accuracy'])
                mlflow.log_metric("baseline_cv_f1", best_baseline['cv_f1'])
                mlflow.log_metric("baseline_approval_rate", best_baseline['business_validation']['approval_rate'])
                
                # Log model with probability wrapper
                try:
                    probability_model = create_mlflow_probability_model(best_baseline['model'])
                    signature = mlflow.models.infer_signature(data['X_train'], data['y_train'])
                    mlflow.pyfunc.log_model(
                        artifact_path="baseline_model",
                        python_model=probability_model,
                        signature=signature
                    )
                    logger.info("    Baseline model logged with probability wrapper")
                except Exception as e:
                    logger.warning(f"    Failed to log baseline with probability wrapper: {e}")
                    # Fallback to regular sklearn logging
                    mlflow.sklearn.log_model(best_baseline['model'], "baseline_model")
        
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
        
        # Log to MLflow
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
                        # Create probability wrapper and log model
                        probability_model = create_mlflow_probability_model(best_rf)
                        signature = mlflow.models.infer_signature(data['X_train_opt'], data['y_train_opt'])
                        mlflow.pyfunc.log_model(
                            artifact_path="optimized_random_forest",
                            python_model=probability_model,
                            signature=signature
                        )
                        logger.info("    RandomizedSearch RF logged with probability wrapper")
                    except Exception as e:
                        logger.warning(f"    Failed to log RF with probability wrapper: {e}")
                        # Fallback to regular sklearn logging
                        mlflow.sklearn.log_model(best_rf, "optimized_random_forest")
        
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
            
            # Log to MLflow
            if mlflow:
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
                        try:
                            # Create probability wrapper and log model
                            probability_model = create_mlflow_probability_model(best_rf_optuna)
                            signature = mlflow.models.infer_signature(data['X_train_opt'], data['y_train_opt'])
                            mlflow.pyfunc.log_model(
                                artifact_path="optuna_random_forest",
                                python_model=probability_model,
                                signature=signature
                            )
                            logger.info("    Optuna RF logged with probability wrapper")
                        except Exception as e:
                            logger.warning(f"    Failed to log Optuna RF with probability wrapper: {e}")
                            # Fallback to regular sklearn logging
                            mlflow.sklearn.log_model(best_rf_optuna, "optuna_random_forest")
        
        except Exception as e:
            logger.error(f"Optuna optimization failed: {e}")
    
    return results

# ============================================================================
# 5. MULTI-MODEL OPTIMIZATION WITH BUSINESS CONSTRAINTS
# ============================================================================

def optimize_multiple_models(data, config, logger, mlflow=None):
    """Optimize multiple model types dengan business constraints dan custom parameters"""
    logger.info("Starting multi-model optimization with business constraints...")
    
    models_config = {
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=config.random_state),
            'param_grid': {
                'n_estimators': [100, 200],           # Reduced range
                'learning_rate': [0.05, 0.1, 0.15],   # Conservative learning rates
                'max_depth': [3, 5, 7],               # Prevent overfitting
                'min_samples_split': [5, 10],         # Conservative
                'min_samples_leaf': [2, 4],           # Conservative
                'subsample': [0.8, 0.9]               # Regularization
            },
            'scale_features': False
        },
        'Extra Trees': {
            'model': ExtraTreesClassifier(random_state=config.random_state),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],            # Controlled depth
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4],
                'max_features': ['sqrt', 'log2'],
                'class_weight': [None, 'balanced']
            },
            'scale_features': False
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=config.random_state, max_iter=1000),
            'param_grid': {
                'C': [0.1, 1, 10],                   # Moderate regularization
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'class_weight': [None, 'balanced'],
                'alpha': [config.alpha],             # Use alpha from command line
                'l1_ratio': [config.l1_ratio]        # Use l1_ratio from command line
            },
            'scale_features': True
        }
    }
    
    # Add optimal class weights if available
    if data.get('class_weights'):
        for model_config in models_config.values():
            if 'class_weight' in model_config['param_grid']:
                model_config['param_grid']['class_weight'].append(data['class_weights'])
    
    results = {}
    
    for model_name, config_model in models_config.items():
        try:
            logger.info(f"  Optimizing {model_name}...")
            
            # Prepare data (scale if needed)
            if config_model['scale_features']:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(data['X_train_opt'])
                X_val_scaled = scaler.transform(data['X_val'])
            else:
                X_train_scaled = data['X_train_opt']
                X_val_scaled = data['X_val']
                scaler = None
            
            # Business-aware scoring
            def business_aware_scorer(estimator, X, y):
                if config_model['scale_features'] and scaler:
                    X_scaled = scaler.transform(X)
                    y_pred = estimator.predict(X_scaled)
                else:
                    y_pred = estimator.predict(X)
                
                approval_rate = np.mean(y_pred)
                f1 = f1_score(y, y_pred)
                
                # Apply business constraints
                if approval_rate < config.min_approval_rate or approval_rate > config.max_approval_rate:
                    return f1 * 0.5
                elif approval_rate < 0.4 or approval_rate > 0.75:
                    return f1 * 0.8
                else:
                    return f1
            
            # Randomized search
            search = RandomizedSearchCV(
                config_model['model'],
                config_model['param_grid'],
                n_iter=config.n_iter_random,
                cv=config.cv_folds,
                scoring=business_aware_scorer,
                n_jobs=-1,
                random_state=config.random_state,
                verbose=0
            )
            
            start_time = time.time()
            search.fit(X_train_scaled, data['y_train_opt'])
            optimization_time = time.time() - start_time
            
            # Evaluate
            best_model = search.best_estimator_
            y_val_pred = best_model.predict(X_val_scaled)
            val_f1 = f1_score(data['y_val'], y_val_pred)
            val_acc = accuracy_score(data['y_val'], y_val_pred)
            
            # Business logic validation
            business_validation = validate_model_business_logic(
                best_model, X_val_scaled, data['y_val'], config, logger, model_name
            )
            
            if business_validation['passes_business_logic']:
                results[model_name] = {
                    'model': best_model,
                    'scaler': scaler,
                    'params': search.best_params_,
                    'cv_f1': search.best_score_,
                    'val_f1': val_f1,
                    'val_acc': val_acc,
                    'optimization_time': optimization_time,
                    'scale_features': config_model['scale_features'],
                    'business_validation': business_validation
                }
                
                logger.info(f"    CV F1: {search.best_score_:.4f}")
                logger.info(f"    Val F1: {val_f1:.4f}")
                logger.info(f"    Approval Rate: {business_validation['approval_rate']:.1%}")
            else:
                logger.warning(f"    {model_name} failed business validation, skipping...")
            
            # Log to MLflow
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
                        # Create probability wrapper and log model
                        probability_model = create_mlflow_probability_model(best_model, scaler)
                        signature = mlflow.models.infer_signature(X_train_scaled, data['y_train_opt'])
                        mlflow.pyfunc.log_model(
                            artifact_path="optimized_model",
                            python_model=probability_model,
                            signature=signature
                        )
                        logger.info(f"    {model_name} logged with probability wrapper")
                    except Exception as e:
                        logger.warning(f"    Failed to log {model_name} with probability wrapper: {e}")
                        # Fallback to regular sklearn logging
                        mlflow.sklearn.log_model(best_model, "optimized_model")
                        if scaler:
                            mlflow.sklearn.log_model(scaler, "feature_scaler")
        
        except Exception as e:
            logger.error(f"Optimization failed for {model_name}: {e}")
    
    return results

# ============================================================================
# 6. MODEL EVALUATION & SELECTION WITH BUSINESS VALIDATION
# ============================================================================

def evaluate_all_models(rf_results, multi_results, baseline, data, config, logger):
    """Evaluate all optimized models dan select best dengan business validation"""
    logger.info("Evaluating all optimized models with business validation...")
    
    # Collect all models that passed business validation
    all_models = {}
    
    # Add RF results
    for strategy, result in rf_results.items():
        if result.get('business_validation', {}).get('passes_business_logic', False):
            all_models[f'RF_{strategy}'] = result
    
    # Add multi-model results
    for model_name, result in multi_results.items():
        if result.get('business_validation', {}).get('passes_business_logic', False):
            all_models[model_name] = result
    
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
            
            # Prepare test data
            if result.get('scaler'):
                X_test_processed = result['scaler'].transform(data['X_test'])
            else:
                X_test_processed = data['X_test']
            
            # Predict
            y_pred = model.predict(X_test_processed)
            
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
# 7. ROBUST MODEL SAVING
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
        
        # Save scaler jika ada
        scaler_filename = None
        if best_model_result.get('scaler'):
            scaler_filename = f"{config.output_dir}/scaler_{config.model_name}_{timestamp}.pkl"
            joblib.dump(best_model_result['scaler'], scaler_filename)
            logger.info(f"Scaler saved: {scaler_filename}")
        
        # Save comprehensive model info
        info_filename = f"{config.output_dir}/model_info_{timestamp}.txt"
        with open(info_filename, 'w') as f:
            f.write(f"Credit Approval Model Information\n")
            f.write(f"=================================\n\n")
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
        
        # Log to MLflow jika available
        if mlflow:
            with mlflow.start_run(run_name=f"PRODUCTION_MODEL_{timestamp}", nested=True):
                mlflow.log_param("model_name", best_model_result['name'])
                mlflow.log_param("production_ready", True)
                mlflow.log_param("business_logic_validated", best_model_result.get('test_business_validation', {}).get('passes_business_logic', False))
                mlflow.log_metric("final_test_f1", best_model_result['test_f1'])
                mlflow.log_metric("final_test_accuracy", best_model_result['test_acc'])
                mlflow.log_metric("final_approval_rate", best_model_result['test_approval_rate'])
                mlflow.log_metric("final_improvement", best_model_result['f1_improvement'])
                
                # Log model with probability wrapper
                try:
                    if best_model_result.get('scaler'):
                        X_signature = best_model_result['scaler'].transform(data['X_train'])
                    else:
                        X_signature = data['X_train']
                    
                    # Create probability wrapper for production model
                    probability_model = create_mlflow_probability_model(
                        best_model_result['model'], 
                        best_model_result.get('scaler')
                    )
                    
                    signature = mlflow.models.infer_signature(X_signature, data['y_train'])
                    mlflow.pyfunc.log_model(
                        artifact_path="production_model",
                        python_model=probability_model,
                        signature=signature
                    )
                    logger.info("Production model logged with probability wrapper")
                    
                except Exception as e:
                    logger.warning(f"Failed to log production model with probability wrapper: {e}")
                    # Fallback to regular sklearn logging
                    mlflow.sklearn.log_model(best_model_result['model'], "production_model")
                    if best_model_result.get('scaler'):
                        mlflow.sklearn.log_model(best_model_result['scaler'], "production_scaler")
                
                # Log files
                mlflow.log_artifact(model_filename)
                mlflow.log_artifact(info_filename)
                if scaler_filename:
                    mlflow.log_artifact(scaler_filename)
        
        return model_filename
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return None

# ============================================================================
# 8. MAIN EXECUTION
# ============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Advanced Credit Approval Model Tuning with Business Validation'
    )
    
    # Regularization parameters
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
    
    # Optional parameters
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
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup with arguments
    config = Config(args)
    logger = setup_logging()
    mlflow, mlflow_available = setup_mlflow_dagshub(config, logger)
    optuna, optuna_available = setup_optuna_optional(logger)
    
    # Log configuration
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
    
    # Check for dry run
    if args.dry_run:
        logger.info("DRY RUN MODE - Validation only")
        # Load data for validation
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
            mlflow.log_param("n_iter_optuna", config.n_iter_optuna)
            mlflow.log_param("grid_search_iter", config.grid_search_iter)
            mlflow.log_param("total_samples", len(data['X_train']))
            mlflow.log_param("features", data['X_train'].shape[1])
            mlflow.log_param("train_approval_rate", data['train_approval_rate'])
            mlflow.log_param("dagshub_integration", True)
            mlflow.log_param("dagshub_repo", config.dagshub_url)
            mlflow.log_param("probability_wrapper_enabled", True)
            
            # Run optimizations
            rf_results = optimize_random_forest(data, config, logger, mlflow, optuna)
            multi_results = optimize_multiple_models(data, config, logger, mlflow)
            
            # Evaluate and select best
            best_model, all_results = evaluate_all_models(rf_results, multi_results, baseline, data, config, logger)
            
            if best_model:
                # Save model
                model_file = save_best_model(best_model, data, config, logger, mlflow)
                
                # Final logging
                mlflow.log_metric("optimization_success", 1)
                mlflow.log_param("best_model_name", best_model['name'])
                mlflow.log_metric("final_improvement", best_model['f1_improvement'])
                mlflow.log_metric("final_approval_rate", best_model['test_approval_rate'])
                mlflow.log_param("model_file_saved", model_file is not None)
                mlflow.log_metric("alpha_used", config.alpha)
                mlflow.log_metric("l1_ratio_used", config.l1_ratio)
            else:
                logger.error("No valid models found")
                if mlflow_available:
                    mlflow.log_metric("optimization_success", 0)
    else:
        # Run without MLflow
        rf_results = optimize_random_forest(data, config, logger)
        multi_results = optimize_multiple_models(data, config, logger)
        best_model, all_results = evaluate_all_models(rf_results, multi_results, baseline, data, config, logger)
        
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
        
        target_met = best_model['f1_improvement'] >= config.target_improvement
        business_valid = best_model.get('test_business_validation', {}).get('passes_business_logic', False)
        
        logger.info(f"\nVALIDATION RESULTS:")
        logger.info(f"  F1 Target (+{config.target_improvement:.3f}): {'ACHIEVED' if target_met else 'NOT MET'} ({best_model['f1_improvement']:+.4f})")
        logger.info(f"  Business Logic: {'VALID' if business_valid else 'INVALID'}")
        logger.info(f"  Approval Rate: {'HEALTHY' if config.min_approval_rate <= best_model['test_approval_rate'] <= config.max_approval_rate else 'OUT OF RANGE'} ({best_model['test_approval_rate']:.1%})")
        logger.info(f"  Probability Wrapper: ENABLED (MLflow will return probabilities)")
        
        if mlflow_available:
            logger.info(f"\nDAGSHUB INTEGRATION:")
            logger.info(f"  Repository: {config.dagshub_url}")
            logger.info(f"  Experiment tracked: {config.experiment_name}")
            logger.info(f"  Models logged with probability wrappers")
        
        if target_met and business_valid:
            logger.info(f"\nOPTIMIZATION FULLY SUCCESSFUL!")
            logger.info(f"   Model ready for production deployment")
            logger.info(f"   Check DagsHub for detailed experiment tracking")
            logger.info(f"   Used Alpha: {config.alpha}, L1 Ratio: {config.l1_ratio}")
            logger.info(f"   MLflow models return probabilities (not class predictions)")
        elif business_valid:
            logger.info(f"\nBUSINESS VALIDATION PASSED")
            logger.info(f"   Model safe for production despite F1 target not fully met")
            logger.info(f"   Consider adjusting Alpha ({config.alpha}) or L1 Ratio ({config.l1_ratio})")
            logger.info(f"   MLflow models return probabilities (not class predictions)")
        else:
            logger.warning(f"\nBUSINESS VALIDATION FAILED")
            logger.warning(f"   Model may need further tuning before production")
            logger.warning(f"   Try different Alpha/L1 Ratio values or adjust business constraints")
            logger.warning(f"   MLflow models return probabilities (not class predictions)")
    else:
        logger.error("Optimization failed - no valid models produced")
        logger.error("Consider:")
        logger.error(f"  1. Adjusting regularization parameters (Alpha: {config.alpha}, L1 Ratio: {config.l1_ratio})")
        logger.error("  2. Adjusting business constraints")
        logger.error("  3. Using different dataset")
        logger.error("  4. Manual hyperparameter tuning")
        logger.error("  5. Checking DagsHub logs for detailed analysis")

if __name__ == "__main__":
    main()