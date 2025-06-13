# ============================================================================
# SCRIPT UNTUK TROUBLESHOOT DAGSHUB CONNECTION
# ============================================================================

import os
import requests
import mlflow
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_dagshub_connection():
    """Comprehensive DagsHub connection checker"""
    
    print("🔍 DagsHub Connection Diagnostic")
    print("=" * 50)
    
    # Step 1: Check environment variables
    print("\n1. Environment Variables Check:")
    repo_url = os.getenv('DAGSHUB_REPO_URL')
    username = os.getenv('DAGSHUB_USERNAME') 
    token = os.getenv('DAGSHUB_TOKEN')
    
    print(f"   DAGSHUB_REPO_URL: {'✅ Set' if repo_url else '❌ Missing'}")
    print(f"   DAGSHUB_USERNAME: {'✅ Set' if username else '❌ Missing'}")
    print(f"   DAGSHUB_TOKEN: {'✅ Set' if token else '❌ Missing'}")
    
    if not all([repo_url, username, token]):
        print("\n❌ Missing required environment variables!")
        print("   Set them using:")
        print("   export DAGSHUB_REPO_URL='https://dagshub.com/agusprasetyo811/kredit_pinjaman'")
        print("   export DAGSHUB_USERNAME='agusprasetyo811'")
        print("   export DAGSHUB_TOKEN='your_token_here'")
        return False
    
    # Step 2: Check repository exists
    print(f"\n2. Repository Accessibility Check:")
    try:
        # Test if repository page is accessible
        response = requests.get(repo_url, timeout=10)
        if response.status_code == 200:
            print(f"   ✅ Repository accessible: {repo_url}")
        elif response.status_code == 404:
            print(f"   ❌ Repository not found: {repo_url}")
            print(f"   Create repository at: https://dagshub.com/new")
            return False
        else:
            print(f"   ⚠️  Repository response code: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Repository check failed: {e}")
        return False
    
    # Step 3: Check MLflow endpoint
    print(f"\n3. MLflow Endpoint Check:")
    mlflow_uri = repo_url + '.mlflow'
    try:
        # Test MLflow API endpoint
        auth = (username, token)
        mlflow_response = requests.get(f"{mlflow_uri}/api/2.0/mlflow/experiments/list", 
                                     auth=auth, timeout=15)
        
        if mlflow_response.status_code == 200:
            print(f"   ✅ MLflow API accessible: {mlflow_uri}")
        elif mlflow_response.status_code == 401:
            print(f"   ❌ Authentication failed - check username/token")
            return False
        elif mlflow_response.status_code == 404:
            print(f"   ❌ MLflow not enabled for repository")
            print(f"   Enable MLflow in repository settings")
            return False
        else:
            print(f"   ⚠️  MLflow API response: {mlflow_response.status_code}")
            
    except Exception as e:
        print(f"   ❌ MLflow endpoint check failed: {e}")
        return False
    
    # Step 4: Test MLflow Python client
    print(f"\n4. MLflow Python Client Test:")
    try:
        # Setup MLflow
        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = token
        mlflow.set_tracking_uri(mlflow_uri)
        
        print(f"   ✅ MLflow client configured")
        
        # Test experiment creation
        test_exp_name = f"connection_test_{int(__import__('time').time())}"
        exp_id = mlflow.create_experiment(test_exp_name, exist_ok=True)
        print(f"   ✅ Test experiment created: {exp_id}")
        
        # Clean up test experiment
        try:
            mlflow.delete_experiment(exp_id)
            print(f"   ✅ Test experiment cleaned up")
        except:
            pass
            
        return True
        
    except Exception as e:
        print(f"   ❌ MLflow client test failed: {e}")
        return False

def setup_dagshub_alternative():
    """Alternative setup using dagshub library"""
    print("\n🔄 Alternative Setup with DagsHub Library:")
    
    try:
        import dagshub
        
        # Method 1: Direct initialization
        dagshub.init(repo_owner='agusprasetyo811', 
                     repo_name='kredit_pinjaman', 
                     mlflow=True)
        
        print("   ✅ DagsHub library setup successful")
        
        # Test MLflow
        current_uri = mlflow.get_tracking_uri()
        print(f"   ✅ Current tracking URI: {current_uri}")
        
        return True
        
    except ImportError:
        print("   ❌ DagsHub library not installed")
        print("   Install with: pip install dagshub")
        return False
    except Exception as e:
        print(f"   ❌ DagsHub library setup failed: {e}")
        return False

def create_env_file():
    """Create .env file template"""
    env_content = """# DagsHub Configuration
DAGSHUB_REPO_URL=https://dagshub.com/agusprasetyo811/kredit_pinjaman
DAGSHUB_USERNAME=agusprasetyo811
DAGSHUB_TOKEN=your_actual_token_here

# MLflow Configuration  
MODEL_NAME=credit_approval_model
"""
    
    with open('.env.template', 'w') as f:
        f.write(env_content)
    
    print("📝 Created .env.template file")
    print("   1. Copy to .env: cp .env.template .env")
    print("   2. Edit .env and replace 'your_actual_token_here' with actual token")

if __name__ == "__main__":
    print("🚀 DagsHub Connection Troubleshooter")
    print("=" * 60)
    
    # Run diagnostics
    success = check_dagshub_connection()
    
    if not success:
        print("\n🔄 Trying alternative setup...")
        alt_success = setup_dagshub_alternative()
        
        if not alt_success:
            print("\n📝 Creating environment template...")
            create_env_file()
            
            print("\n❌ CONNECTION FAILED")
            print("\nTROUBLESHOOTING STEPS:")
            print("1. Ensure repository exists: https://dagshub.com/agusprasetyo811/kredit_pinjaman")
            print("2. Get valid token: https://dagshub.com/user/settings/tokens")
            print("3. Enable MLflow in repository settings")
            print("4. Check internet connection and DagsHub server status")
            print("5. Try using local MLflow: export MLFLOW_TRACKING_URI='file:./mlruns'")
    else:
        print("\n✅ DAGSHUB CONNECTION SUCCESSFUL!")
        print("You can now run your MLflow experiments with DagsHub tracking.")