#!/usr/bin/env python3
"""
Quick check script untuk masalah MLflow DagsHub
"""

import requests
import time

def quick_check():
    """Quick check untuk MLflow DagsHub connectivity"""
    
    # Data dari error message Anda
    username = "agusprasetyo811"
    repo_name = "kredit_pinjaman"
    run_id = "63de925f560c4e5291977bb11050094e"
    
    base_url = f"https://dagshub.com/{username}/{repo_name}.mlflow"
    api_url = f"{base_url}/api/2.0/mlflow"
    
    print("🔍 Quick MLflow DagsHub Check")
    print("-" * 40)
    
    # 1. Check DagsHub main site
    try:
        print("1. Checking DagsHub main site...", end=" ")
        response = requests.get("https://dagshub.com", timeout=5)
        print("✅ OK" if response.status_code == 200 else f"❌ {response.status_code}")
    except:
        print("❌ FAILED")
    
    # 2. Check MLflow UI
    try:
        print("2. Checking MLflow UI...", end=" ")
        response = requests.get(f"{base_url}/", timeout=10)
        print("✅ OK" if response.status_code == 200 else f"❌ {response.status_code}")
    except:
        print("❌ FAILED")
    
    # 3. Check MLflow API
    try:
        print("3. Checking MLflow API...", end=" ")
        response = requests.get(f"{api_url}/experiments/list", timeout=10)
        if response.status_code == 200:
            print("✅ OK")
        elif response.status_code == 500:
            print("❌ SERVER ERROR (500) - Same as your issue!")
        else:
            print(f"❌ {response.status_code}")
    except Exception as e:
        print(f"❌ FAILED: {e}")
    
    # 4. Check specific run
    try:
        print("4. Checking your run...", end=" ")
        params = {'run_uuid': run_id, 'run_id': run_id}
        response = requests.get(f"{api_url}/runs/get", params=params, timeout=10)
        if response.status_code == 200:
            print("✅ FOUND")
        elif response.status_code == 500:
            print("❌ SERVER ERROR (500)")
        elif response.status_code == 404:
            print("❌ NOT FOUND")
        else:
            print(f"❌ {response.status_code}")
    except Exception as e:
        print(f"❌ FAILED: {e}")
    
    # 5. Test with retry
    print("5. Testing with retry...", end=" ")
    success = False
    for i in range(3):
        try:
            response = requests.get(f"{api_url}/experiments/list", timeout=5)
            if response.status_code == 200:
                success = True
                break
            time.sleep(1)
        except:
            time.sleep(1)
    
    print("✅ OK" if success else "❌ FAILED")
    
    print("\n💡 CONCLUSION:")
    if success:
        print("   MLflow API is working. Your issue might be temporary.")
        print("   Try running your code again.")
    else:
        print("   MLflow API is having issues (confirms your error).")
        print("   This is a server-side problem with DagsHub.")
        print("   Recommendations:")
        print("   - Wait and try again later")
        print("   - Add retry logic to your MLflow code")
        print("   - Contact DagsHub support if issue persists")

if __name__ == "__main__":
    quick_check()