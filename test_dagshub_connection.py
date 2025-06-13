#!/usr/bin/env python3
"""
DagsHub Connection Test Script
Test your DagsHub and MLflow connection before running main modeling scripts
"""

import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_dagshub_connection():
    """Test DagsHub connection comprehensively"""
    print("=" * 60)
    print("DAGSHUB CONNECTION TEST")
    print("=" * 60)
    
    dagshub_success = False
    local_fallback_success = False
    
    # Step 1: Check environment variables
    print("\n1. Checking environment variables...")
    
    dagshub_url = os.getenv('DAGSHUB_REPO_URL')
    dagshub_username = os.getenv('DAGSHUB_USERNAME')
    dagshub_token = os.getenv('DAGSHUB_TOKEN')
    
    print(f"   DAGSHUB_REPO_URL: {'✓' if dagshub_url else '❌'} {dagshub_url}")
    print(f"   DAGSHUB_USERNAME: {'✓' if dagshub_username else '❌'}")
    print(f"   DAGSHUB_TOKEN: {'✓' if dagshub_token else '❌'}")
    
    if not all([dagshub_url, dagshub_username, dagshub_token]):
        print("❌ Missing DagsHub credentials in .env file")
        print("\nCreate a .env file with:")
        print("   DAGSHUB_REPO_URL=https://dagshub.com/agusprasetyo811/kredit_pinjaman_1")
        print("   DAGSHUB_USERNAME=your_username")
        print("   DAGSHUB_TOKEN=your_token")
        return False, False
    
    # Step 2: Test imports
    print("\n2. Testing imports...")
    try:
        import dagshub
        print("   ✓ dagshub imported successfully")
    except ImportError:
        print("   ❌ dagshub not installed. Install with: pip install dagshub")
        return False, False
    
    try:
        import mlflow
        import mlflow.sklearn
        print("   ✓ mlflow imported successfully")
    except ImportError:
        print("   ❌ mlflow not installed. Install with: pip install mlflow")
        return False, False
    
    # Step 3: Initialize DagsHub
    print("\n3. Initializing DagsHub...")
    try:
        dagshub.init(
            repo_owner='agusprasetyo811',
            repo_name='kredit_pinjaman_1',
            mlflow=True
        )
        print("   ✓ DagsHub initialized successfully")
    except Exception as e:
        print(f"   ⚠️ DagsHub initialization warning: {e}")
        print("   Continuing with manual setup...")
    
    # Step 4: Setup MLflow
    print("\n4. Setting up MLflow...")
    try:
        # Set environment variables
        os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
        
        # Configure tracking URI
        tracking_uri = dagshub_url + ".mlflow"
        mlflow.set_tracking_uri(tracking_uri)
        print(f"   ✓ Tracking URI set: {tracking_uri}")
        
    except Exception as e:
        print(f"   ❌ MLflow setup failed: {e}")
        return False, False
    
    # Step 5: Test connection with timeout
    print("\n5. Testing DagsHub connection...")
    try:
        # Set timeout
        import socket
        socket.setdefaulttimeout(15)
        
        # Test experiment operations
        experiment_name = "connection_test"
        
        try:
            # Try to get or create experiment
            exp = mlflow.get_experiment_by_name(experiment_name)
            if exp and exp.lifecycle_stage != 'deleted':
                mlflow.set_experiment(experiment_name)
                print("   ✓ Using existing test experiment")
            else:
                mlflow.create_experiment(experiment_name)
                print("   ✓ Created new test experiment")
        except Exception as e:
            print(f"   ⚠️ Experiment setup issue: {e}")
            print("   Trying with default experiment...")
            mlflow.set_experiment("Default")
        
        # Test run creation
        with mlflow.start_run(run_name=f"test_run_{datetime.now().strftime('%H%M%S')}"):
            mlflow.log_param("test_param", "connection_successful")
            mlflow.log_metric("test_metric", 1.0)
            mlflow.log_metric("timestamp", time.time())
            
            run_id = mlflow.active_run().info.run_id
            print(f"   ✓ Test run created successfully: {run_id[:8]}...")
        
        socket.setdefaulttimeout(None)
        print("   ✓ Connection test completed successfully")
        dagshub_success = True
        
    except Exception as e:
        socket.setdefaulttimeout(None)
        print(f"   ❌ Connection test failed: {e}")
        print("\n   This might be due to:")
        print("     - Network connectivity issues")
        print("     - DagsHub server issues")
        print("     - Incorrect credentials")
        print("     - Repository not found")
        dagshub_success = False
    
    # Step 6: Test local fallback
    print("\n6. Testing local MLflow fallback...")
    try:
        # Create local MLflow setup
        local_mlruns_dir = "./mlruns_test"
        os.makedirs(local_mlruns_dir, exist_ok=True)
        
        local_uri = f"file://{os.path.abspath(local_mlruns_dir)}"
        mlflow.set_tracking_uri(local_uri)
        
        # Create local experiment first
        try:
            local_exp_id = mlflow.create_experiment("local_test_experiment")
            mlflow.set_experiment("local_test_experiment")
            print("   ✓ Local experiment created")
        except Exception as e:
            # If experiment exists or other issue, use default
            try:
                mlflow.set_experiment("Default")
                print("   ✓ Using default experiment")
            except:
                print(f"   ⚠️ Experiment setup warning: {e}")
        
        # Test local run
        with mlflow.start_run(run_name="local_test"):
            mlflow.log_param("local_test", True)
            mlflow.log_metric("local_metric", 2.0)
            run_id = mlflow.active_run().info.run_id
            print(f"   ✓ Local test run created: {run_id[:8]}...")
        
        print("   ✓ Local MLflow fallback works")
        local_fallback_success = True
        
        # Cleanup
        import shutil
        shutil.rmtree(local_mlruns_dir, ignore_errors=True)
        
    except Exception as e:
        print(f"   ❌ Local fallback failed: {e}")
        print("   This is not critical - DagsHub connection is working")
        # Don't return False for local fallback failure since DagsHub works
        print("   ✓ DagsHub is primary tracking method")
        local_fallback_success = False
    
    return dagshub_success, local_fallback_success

def main():
    """Main test function"""
    dagshub_success, local_fallback_success = test_dagshub_connection()
    
    print("\n" + "=" * 60)
    if dagshub_success:
        print("✅ DAGSHUB CONNECTION SUCCESSFUL!")
        print("Your DagsHub connection is working correctly.")
        print("\nYou can now run:")
        print("   python modelling_fixed.py")
        print("   python modelling_tuning_fixed.py")
        
        if not local_fallback_success:
            print("\n⚠️ Note: Local fallback test had issues, but this is not critical")
            print("   since DagsHub is working as primary tracking method.")
    else:
        print("❌ DAGSHUB CONNECTION FAILED!")
        print("Please check your credentials and network connection.")
        print("\nThe scripts will fallback to local MLflow automatically.")
        
        if local_fallback_success:
            print("✅ Local MLflow fallback is working as backup.")
        else:
            print("❌ Local MLflow fallback also failed.")
            print("Scripts will run without MLflow tracking.")
    
    print("=" * 60)
    
    return dagshub_success or local_fallback_success

if __name__ == "__main__":
    main()