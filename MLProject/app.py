#!/usr/bin/env python3
"""
app_fixed.py - Enhanced Flask application with multi-format model loading and threshold support
"""
from flask import Flask, request, jsonify
import pickle
import joblib
import pandas as pd
import numpy as np
import json
import os
import logging
import gzip
import traceback
import sys  # FIXED: Added missing import
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class EnhancedModelLoader:
    """Enhanced model loader with support for multiple formats and threshold configuration"""
    
    def __init__(self, model_path='output/models/best_model.pkl', threshold=0.5):
        self.model_path = model_path
        self.model = None
        self.feature_names = []
        self.model_type = None
        self.loaded_at = None
        self.loading_method = None
        self.threshold = threshold  # Added threshold parameter
        
    def set_threshold(self, threshold):
        """Set prediction threshold for binary classification"""
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.threshold = threshold
        logger.info(f"Threshold updated to: {threshold}")
        
    def detect_file_format(self):
        """Detect the file format"""
        try:
            with open(self.model_path, 'rb') as f:
                header = f.read(16)
            
            if header.startswith(b'\x80'):
                return 'pickle_binary'
            elif header.startswith(b']\x8c'):
                return 'pickle_text'
            elif header.startswith(b'\x1f\x8b'):
                return 'gzip'
            elif b'sklearn' in header or b'joblib' in header:
                return 'joblib'
            elif header.startswith(b'PK'):
                return 'zip'
            else:
                return 'unknown'
                
        except Exception as e:
            logger.error(f"Error detecting file format: {e}")
            return 'unknown'
    
    def try_pickle_load(self):
        """Try loading with pickle"""
        try:
            logger.info("Trying pickle.load()...")
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info("Successfully loaded with pickle")
            
            # FIXED: Validate model has predict method
            if not hasattr(model, 'predict'):
                logger.warning("Loaded object doesn't have predict method!")
                logger.info(f"Object type: {type(model)}")
                logger.info(f"Available methods: {[m for m in dir(model) if not m.startswith('_')]}")
                # Don't return None, let caller decide
            
            return model, 'pickle'
        except Exception as e:
            logger.warning(f"Pickle loading failed: {e}")
            logger.debug(traceback.format_exc())  # Menambahkan detail error
            return None, None
    
    def try_joblib_load(self):
        """Try loading with joblib"""
        try:
            logger.info("Trying joblib.load()...")
            model = joblib.load(self.model_path)
            logger.info("Successfully loaded with joblib")
            
            # FIXED: Validate model has predict method
            if not hasattr(model, 'predict'):
                logger.warning("Loaded object doesn't have predict method!")
                logger.info(f"Object type: {type(model)}")
                logger.info(f"Available methods: {[m for m in dir(model) if not m.startswith('_')]}")
            
            return model, 'joblib'
        except Exception as e:
            logger.warning(f"Joblib loading failed: {e}")
            logger.debug(traceback.format_exc())
            return None, None
    
    def try_gzip_pickle_load(self):
        """Try loading gzipped pickle"""
        try:
            logger.info("Trying gzip + pickle...")
            with gzip.open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info("Successfully loaded with gzip + pickle")
            return model, 'gzip_pickle'
        except Exception as e:
            logger.warning(f"Gzip pickle loading failed: {e}")
            return None, None
    
    def try_different_pickle_protocols(self):
        """Try different pickle protocols"""
        protocols = [0, 1, 2, 3, 4, 5]
        
        for protocol in protocols:
            try:
                logger.info(f"Trying pickle protocol {protocol}...")
                with open(self.model_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"Successfully loaded with pickle protocol {protocol}")
                return model, f'pickle_protocol_{protocol}'
            except Exception as e:
                continue
        
        return None, None
    
    def try_mlflow_load(self):
        """Try loading MLflow model"""
        try:
            import mlflow.sklearn
            
            # Check if it's an MLflow model directory
            model_dir = Path(self.model_path).parent
            if (model_dir / 'MLmodel').exists():
                logger.info("Trying MLflow sklearn format...")
                model = mlflow.sklearn.load_model(str(model_dir))
                logger.info("Successfully loaded with MLflow")
                return model, 'mlflow'
                
        except ImportError:
            logger.warning("MLflow not available")
        except Exception as e:
            logger.warning(f"MLflow loading failed: {e}")
        
        return None, None
    
    def extract_feature_info(self):
        """Extract feature information from model"""
        try:
            # Try to get feature names from model
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = list(self.model.feature_names_in_)
                logger.info(f"Found feature names in model: {len(self.feature_names)} features")
            elif hasattr(self.model, 'feature_importances_'):
                # For tree-based models, create generic feature names
                n_features = len(self.model.feature_importances_)
                self.feature_names = [f'feature_{i}' for i in range(n_features)]
                logger.info(f"Generated feature names from importances: {n_features} features")
            else:
                # Try to infer from test prediction
                logger.info("Attempting to detect feature count...")
                for n_features in [1, 6, 8, 10, 12, 15, 20, 50, 100]:
                    try:
                        test_data = np.random.rand(1, n_features)
                        self.model.predict(test_data)
                        self.feature_names = [f'feature_{i}' for i in range(n_features)]
                        logger.info(f"Detected {n_features} features from test prediction")
                        break
                    except:
                        continue
                else:
                    logger.warning("Could not determine feature count")
                    self.feature_names = []
            
        except Exception as e:
            logger.error(f"Error extracting feature info: {e}")
            self.feature_names = []
    
    def load_model(self):
        """Load model using multiple methods"""
        logger.info(f"Loading model from {self.model_path}...")
        
        # FIXED: Better file existence check
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            
            # FIXED: Try to find alternative files in the directory
            parent_dir = os.path.dirname(self.model_path) or '.'
            if os.path.exists(parent_dir):
                files = os.listdir(parent_dir)
                pkl_files = [f for f in files if f.endswith(('.pkl', '.joblib', '.pickle'))]
                logger.info(f"Available model files in {parent_dir}: {pkl_files}")
                
                if pkl_files:
                    # Try first available pkl file
                    alternative_path = os.path.join(parent_dir, pkl_files[0])
                    logger.info(f"Trying alternative file: {alternative_path}")
                    self.model_path = alternative_path
                else:
                    raise FileNotFoundError(f"No model files found in {parent_dir}")
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Check file size
        file_size = os.path.getsize(self.model_path)
        logger.info(f"Model file size: {file_size / (1024*1024):.2f} MB")
        
        if file_size == 0:
            raise ValueError("Model file is empty")
        
        # FIXED: Add file readability check
        if not os.access(self.model_path, os.R_OK):
            raise PermissionError(f"Cannot read model file: {self.model_path}")
        
        # Detect file format
        format_detected = self.detect_file_format()
        logger.info(f"Detected file format: {format_detected}")
        
        # Try different loading methods
        loading_methods = [
            self.try_pickle_load,
            self.try_joblib_load,
            self.try_gzip_pickle_load,
            self.try_different_pickle_protocols,
            self.try_mlflow_load
        ]
        
        model_loaded = False
        
        for method in loading_methods:
            try:
                model, loading_method = method()
                if model is not None:
                    self.model = model
                    self.loading_method = loading_method
                    self.model_type = type(model).__name__
                    self.loaded_at = datetime.now().isoformat()
                    
                    logger.info(f"Model loaded successfully using: {loading_method}")
                    logger.info(f"   Model type: {self.model_type}")
                    
                    # Extract feature information
                    self.extract_feature_info()
                    
                    # Test basic functionality
                    if not hasattr(self.model, 'predict'):
                        logger.error("Model missing predict() method")
                        # FIXED: Don't raise error immediately, continue trying other methods
                        self.model = None
                        continue
                    
                    # FIXED: Test prediction to ensure model works
                    try:
                        if self.feature_names:
                            test_data = np.random.rand(1, len(self.feature_names))
                        else:
                            # Try common feature counts
                            test_data = np.random.rand(1, 6)  # Default to 6 features
                        
                        test_pred = self.model.predict(test_data)
                        logger.info(f"Model prediction test successful: {test_pred}")
                        model_loaded = True
                        break
                        
                    except Exception as e:
                        logger.warning(f"Model prediction test failed: {e}")
                        # Don't break, this might still be a valid model
                        model_loaded = True
                        break
                        
            except Exception as e:
                logger.error(f"Method {method.__name__} failed: {e}")
                continue
        
        # FIXED: Ensure we return the correct status
        if not model_loaded or self.model is None:
            logger.error("Could not load model with any available method")
            # FIXED: Print more detailed error info
            logger.error(f"Final status - Model: {self.model}, Type: {self.model_type}")
            raise RuntimeError("Could not load model with any available method")
        
        return True
    
    def get_model_info(self):
        """Get comprehensive model information including threshold"""
        info = {
            "model_loaded": self.model is not None,
            "model_type": self.model_type,
            "model_path": self.model_path,
            "loading_method": self.loading_method,
            "loaded_at": self.loaded_at,
            "feature_names": self.feature_names,
            "n_features": len(self.feature_names),
            "threshold": self.threshold,  # Include threshold in info
            "has_predict": hasattr(self.model, 'predict') if self.model else False,
            "has_predict_proba": hasattr(self.model, 'predict_proba') if self.model else False,
            "has_feature_importances": hasattr(self.model, 'feature_importances_') if self.model else False
        }
        
        # Add model-specific info
        if self.model:
            if hasattr(self.model, 'get_params'):
                try:
                    info["model_params"] = self.model.get_params()
                except:
                    info["model_params"] = "Could not retrieve parameters"
            
            if hasattr(self.model, 'feature_importances_'):
                try:
                    importances = self.model.feature_importances_
                    if len(self.feature_names) == len(importances):
                        feature_importance = dict(zip(self.feature_names, importances.tolist()))
                        # Sort by importance
                        info["feature_importances"] = dict(sorted(feature_importance.items(), 
                                                                 key=lambda x: x[1], reverse=True))
                except:
                    pass
        
        return info
    
    def validate_input(self, data):
        """Validate and preprocess input data"""
        try:
            # Handle different input formats
            if isinstance(data, dict):
                if 'instances' in data:
                    # MLflow style
                    instances = data['instances']
                    if not isinstance(instances, list):
                        raise ValueError("'instances' must be a list")
                    df = pd.DataFrame(instances)
                else:
                    # Single instance
                    df = pd.DataFrame([data])
            elif isinstance(data, list):
                # List of instances
                df = pd.DataFrame(data)
            else:
                raise ValueError("Input must be dict or list")
            
            # Handle feature alignment
            if self.feature_names:
                # Check if all required features are present
                missing_features = set(self.feature_names) - set(df.columns)
                extra_features = set(df.columns) - set(self.feature_names)
                
                if missing_features:
                    logger.warning(f"Missing features (will fill with 0): {missing_features}")
                    for feature in missing_features:
                        df[feature] = 0
                
                if extra_features:
                    logger.info(f"Extra features (will be ignored): {extra_features}")
                
                # Select and order features correctly
                df = df[self.feature_names]
            else:
                # No feature names available, use all columns
                logger.warning("No feature names available, using all input columns")
            
            return df
            
        except Exception as e:
            raise ValueError(f"Input validation failed: {str(e)}")
    
    def predict(self, data):
        """Make predictions with configurable threshold"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Validate input
        df = self.validate_input(data)
        
        # Convert to numpy array
        X = df.values
        
        # Get probabilities if available, otherwise use default predict
        if hasattr(self.model, 'predict_proba'):
            # Use probabilities with custom threshold
            probabilities = self.model.predict_proba(X)
            
            # For binary classification
            if probabilities.shape[1] == 2:
                # Apply custom threshold
                predictions = (probabilities[:, 1] >= self.threshold).astype(int)
            else:
                # Multi-class: use argmax (threshold not applicable)
                predictions = np.argmax(probabilities, axis=1)
                
            result = {
                "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist(),
                "threshold_used": self.threshold,
                "n_instances": len(predictions)
            }
            
            # Add individual class probabilities for binary classification
            if probabilities.shape[1] == 2:
                result["probability_class_0"] = probabilities[:, 0].tolist()
                result["probability_class_1"] = probabilities[:, 1].tolist()
                result["predictions_default"] = self.model.predict(X).tolist()  # Original predictions
                
        else:
            # Fallback to default predict method
            predictions = self.model.predict(X)
            result = {
                "predictions": predictions.tolist(),
                "threshold_used": "model_default",
                "n_instances": len(predictions),
                "note": "Model doesn't support probabilities, using default prediction"
            }
        
        return result

# Global model loader with default threshold
model_loader = EnhancedModelLoader(threshold=0.5)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    status = {
        "status": "healthy" if model_loader.model else "unhealthy",
        "model_loaded": model_loader.model is not None,
        "loading_method": model_loader.loading_method,
        "threshold": model_loader.threshold,
        "timestamp": datetime.now().isoformat(),
        "model_file": model_loader.model_path,
        "file_exists": os.path.exists(model_loader.model_path)
    }
    
    status_code = 200 if model_loader.model else 503
    return jsonify(status), status_code

@app.route('/info', methods=['GET'])
def info():
    """Get model information"""
    return jsonify(model_loader.get_model_info())

@app.route('/debug', methods=['GET'])
def debug():
    """Debug endpoint for troubleshooting"""
    debug_info = {
        "model_path": model_loader.model_path,
        "file_exists": os.path.exists(model_loader.model_path),
        "loading_method": model_loader.loading_method,
        "model_type": model_loader.model_type,
        "feature_names": model_loader.feature_names,
        "threshold": model_loader.threshold,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    
    if os.path.exists(model_loader.model_path):
        debug_info["file_size"] = os.path.getsize(model_loader.model_path)
        debug_info["file_readable"] = os.access(model_loader.model_path, os.R_OK)
        
        # Read file header
        try:
            with open(model_loader.model_path, 'rb') as f:
                header = f.read(16)
            debug_info["file_header"] = header.hex()
        except:
            debug_info["file_header"] = "Could not read"
    
    # FIXED: Add directory listing for debugging
    try:
        parent_dir = os.path.dirname(model_loader.model_path) or '.'
        if os.path.exists(parent_dir):
            files = os.listdir(parent_dir)
            debug_info["directory_files"] = files
            debug_info["pkl_files"] = [f for f in files if f.endswith(('.pkl', '.joblib', '.pickle'))]
    except Exception as e:
        debug_info["directory_error"] = str(e)
    
    return jsonify(debug_info)

@app.route('/threshold', methods=['GET'])
def get_threshold():
    """Get current threshold setting"""
    return jsonify({
        "threshold": model_loader.threshold,
        "model_supports_probabilities": hasattr(model_loader.model, 'predict_proba') if model_loader.model else False,
        "model_type": model_loader.model_type
    })

@app.route('/threshold', methods=['POST'])
def set_threshold():
    """Set prediction threshold"""
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No JSON data provided"}), 400
        
        if 'threshold' not in data:
            return jsonify({"error": "Missing 'threshold' field"}), 400
        
        threshold = data['threshold']
        
        # Validate threshold
        if not isinstance(threshold, (int, float)):
            return jsonify({"error": "Threshold must be a number"}), 400
        
        if not 0 <= threshold <= 1:
            return jsonify({"error": "Threshold must be between 0 and 1"}), 400
        
        # Set threshold
        model_loader.set_threshold(threshold)
        
        return jsonify({
            "status": "success",
            "message": f"Threshold updated to {threshold}",
            "threshold": model_loader.threshold,
            "updated_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Set threshold error: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to set threshold: {str(e)}"
        }), 500

@app.route('/reload', methods=['POST'])
def reload_model():
    """Reload model from file"""
    try:
        # FIXED: Reset model state before reloading
        model_loader.model = None
        model_loader.model_type = None
        model_loader.loading_method = None
        
        if model_loader.load_model():
            return jsonify({
                "status": "success",
                "message": "Model reloaded successfully",
                "loading_method": model_loader.loading_method,
                "model_type": model_loader.model_type,
                "threshold": model_loader.threshold,
                "reloaded_at": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "status": "error", 
                "message": "Failed to reload model"
            }), 500
    except Exception as e:
        logger.error(f"Reload error: {e}")
        return jsonify({
            "status": "error",
            "message": f"Reload failed: {str(e)}"
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions with current threshold"""
    try:
        if model_loader.model is None:
            return jsonify({"error": "Model not loaded"}), 503
        
        # Get input data
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Make prediction
        result = model_loader.predict(data)
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({"error": f"Input validation error: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/predict_with_threshold', methods=['POST'])
def predict_with_threshold():
    """Make predictions with optional custom threshold for this request only"""
    try:
        if model_loader.model is None:
            return jsonify({"error": "Model not loaded"}), 503
        
        # Get input data
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Extract threshold if provided
        custom_threshold = data.pop('threshold', None)
        original_threshold = model_loader.threshold
        
        try:
            # Temporarily set custom threshold if provided
            if custom_threshold is not None:
                if not 0 <= custom_threshold <= 1:
                    return jsonify({"error": "Threshold must be between 0 and 1"}), 400
                model_loader.set_threshold(custom_threshold)
            
            # Make prediction
            result = model_loader.predict(data)
            
            return jsonify(result)
            
        finally:
            # Restore original threshold
            if custom_threshold is not None:
                model_loader.set_threshold(original_threshold)
        
    except ValueError as e:
        return jsonify({"error": f"Input validation error: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Prediction with threshold error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/threshold_analysis', methods=['POST'])
def threshold_analysis():
    """Analyze predictions across multiple threshold values"""
    try:
        if model_loader.model is None:
            return jsonify({"error": "Model not loaded"}), 503
        
        if not hasattr(model_loader.model, 'predict_proba'):
            return jsonify({"error": "Model doesn't support probability predictions"}), 400
        
        # Get input data
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Extract threshold range
        thresholds = data.pop('thresholds', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        
        # Validate input
        df = model_loader.validate_input(data)
        X = df.values
        
        # Get probabilities
        probabilities = model_loader.model.predict_proba(X)
        
        if probabilities.shape[1] != 2:
            return jsonify({"error": "Threshold analysis only available for binary classification"}), 400
        
        # Analyze across thresholds
        results = []
        for threshold in thresholds:
            predictions = (probabilities[:, 1] >= threshold).astype(int)
            positive_count = int(predictions.sum())
            negative_count = len(predictions) - positive_count
            
            results.append({
                "threshold": threshold,
                "predictions": predictions.tolist(),
                "positive_predictions": positive_count,
                "negative_predictions": negative_count,
                "positive_rate": positive_count / len(predictions)
            })
        
        return jsonify({
            "probabilities": probabilities.tolist(),
            "threshold_analysis": results,
            "n_instances": len(probabilities)
        })
        
    except Exception as e:
        logger.error(f"Threshold analysis error: {e}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/features', methods=['GET'])
def get_features():
    """Get feature information and examples"""
    feature_info = {
        "feature_names": model_loader.feature_names,
        "n_features": len(model_loader.feature_names),
        "model_type": model_loader.model_type,
        "loading_method": model_loader.loading_method,
        "threshold": model_loader.threshold
    }
    
    # Add examples if we have feature names
    if model_loader.feature_names:
        # Create example with realistic values
        if len(model_loader.feature_names) >= 6:
            # Assume loan prediction features
            example_single = {
                model_loader.feature_names[0]: 50000,    # income
                model_loader.feature_names[1]: 700,      # credit_score  
                model_loader.feature_names[2]: 0.3,      # debt_ratio
                model_loader.feature_names[3]: 5,        # employment_years
                model_loader.feature_names[4]: 25000,    # loan_amount
                model_loader.feature_names[5]: 35        # age
            }
            
            # Fill remaining features with default values
            for i, feature in enumerate(model_loader.feature_names[6:], 6):
                example_single[feature] = float(i)
        else:
            # Generic example
            example_single = {
                feature: float(i) for i, feature in enumerate(model_loader.feature_names)
            }
        
        feature_info["examples"] = {
            "single_prediction": example_single,
            "batch_prediction": [example_single, example_single],
            "mlflow_style": {
                "instances": [example_single, example_single]
            },
            "with_custom_threshold": {
                **example_single,
                "threshold": 0.7
            }
        }
    
    return jsonify(feature_info)

@app.route('/test', methods=['GET'])
def test_prediction():
    """Test prediction with sample data"""
    try:
        if model_loader.model is None:
            return jsonify({"error": "Model not loaded"}), 503
        
        if not model_loader.feature_names:
            return jsonify({"error": "No feature names available for testing"}), 400
        
        # Create test data
        test_data = {}
        for i, feature in enumerate(model_loader.feature_names):
            if 'income' in feature.lower():
                test_data[feature] = 50000
            elif 'credit' in feature.lower() or 'score' in feature.lower():
                test_data[feature] = 700
            elif 'ratio' in feature.lower() or 'debt' in feature.lower():
                test_data[feature] = 0.3
            elif 'year' in feature.lower() or 'employment' in feature.lower():
                test_data[feature] = 5
            elif 'amount' in feature.lower() or 'loan' in feature.lower():
                test_data[feature] = 25000
            elif 'age' in feature.lower():
                test_data[feature] = 35
            else:
                test_data[feature] = float(i)
        
        # Make prediction
        result = model_loader.predict(test_data)
        
        return jsonify({
            "test_input": test_data,
            "prediction_result": result,
            "current_threshold": model_loader.threshold,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Test prediction error: {e}")
        return jsonify({
            "error": f"Test prediction failed: {str(e)}",
            "status": "error"
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "GET  /health - Health check",
            "GET  /info - Model information", 
            "GET  /debug - Debug information",
            "GET  /features - Feature information and examples",
            "GET  /test - Test prediction with sample data",
            "GET  /threshold - Get current threshold",
            "POST /threshold - Set threshold",
            "POST /predict - Make predictions with current threshold",
            "POST /predict_with_threshold - Predict with custom threshold",
            "POST /threshold_analysis - Analyze multiple thresholds",
            "POST /reload - Reload model from file"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "Please check server logs for details"
    }), 500

def initialize_app(model_path='output/models/best_model.pkl', threshold=0.5):
    """Initialize the application with threshold configuration"""
    global model_loader
    
    logger.info("Initializing Enhanced Flask app with threshold support...")
    
    # Set model path and threshold
    model_loader.model_path = model_path
    model_loader.threshold = threshold
    
    # FIXED: Better error handling and logging
    try:
        logger.info(f"Attempting to load model from: {model_path}")
        logger.info(f"Default threshold: {threshold}")
        
        # Check if path exists, if not try to find alternatives
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}")
            parent_dir = os.path.dirname(model_path) or '.'
            if os.path.exists(parent_dir):
                files = os.listdir(parent_dir)
                pkl_files = [f for f in files if f.endswith(('.pkl', '.joblib', '.pickle'))]
                logger.info(f"Available model files: {pkl_files}")
        
        success = model_loader.load_model()
        
        if success and model_loader.model is not None:
            logger.info("Application initialized successfully!")
            logger.info(f"   Model type: {model_loader.model_type}")
            logger.info(f"   Loading method: {model_loader.loading_method}")
            logger.info(f"   Features: {len(model_loader.feature_names)}")
            logger.info(f"   Threshold: {model_loader.threshold}")
            logger.info(f"   Supports probabilities: {hasattr(model_loader.model, 'predict_proba')}")
            return True
        else:
            logger.error("Application initialization failed - model is None")
            logger.error(f"   Model loaded: {model_loader.model is not None}")
            logger.error(f"   Model type: {model_loader.model_type}")
            logger.error(f"   Loading method: {model_loader.loading_method}")
            return False
            
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return False

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Loan Model API Server with Threshold Support')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5002, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--model', default='output/models/best_model.pkl', help='Path to model file')
    parser.add_argument('--threshold', type=float, default=0.2, help='Default prediction threshold (0-1)')
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0 <= args.threshold <= 1:
        logger.error("Threshold must be between 0 and 1")
        sys.exit(1)
    
    # FIXED: Initialize app and check result
    init_success = initialize_app(args.model, args.threshold)
    
    if init_success:
        logger.info(f"Starting server at http://{args.host}:{args.port}")
        logger.info(f"   Model loaded using: {model_loader.loading_method}")
        logger.info(f"   Model type: {model_loader.model_type}")
        logger.info(f"   Default threshold: {model_loader.threshold}")
    else:
        logger.warning("Starting server with model loading issues...")
        logger.warning("   Check /debug endpoint for more information")
    
    # Run Flask app
    app.run(host=args.host, port=args.port, debug=args.debug)