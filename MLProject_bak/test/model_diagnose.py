import requests
import json
import numpy as np

API_URL = "http://localhost:5002"

def diagnose_model_issue():
    """Diagnose why model rejects everything"""
    
    print("URGENT MODEL DIAGNOSTIC")
    print("Issue: Model rejects ALL applications including excellent profiles")
    print("=" * 70)
    
    # Test API connection
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code != 200:
            print("API not healthy")
            return
        print("API connection OK")
    except:
        print("Cannot connect to API")
        return
    
    # Get detailed model info
    try:
        info_response = requests.get(f"{API_URL}/info")
        if info_response.status_code == 200:
            info = info_response.json()
            print(f"\n📊 MODEL INFO:")
            print(f"   Model Type: {info.get('model_type', 'Unknown')}")
            print(f"   Loading Method: {info.get('loading_method', 'Unknown')}")
            print(f"   Features: {info.get('n_features', 0)}")
            print(f"   Has Predict: {info.get('has_predict', False)}")
            print(f"   Has Predict Proba: {info.get('has_predict_proba', False)}")
        else:
            print("Cannot get model info")
    except Exception as e:
        print(f"Error getting model info: {e}")
    
    # Test excellent profile with detailed probability analysis
    excellent_profile = {
        "name": "VP Bank - Should be STRONG APPROVE",
        "data": {
            "umur": 38.0,
            "pendapatan": 28000000.0,  # 28M/month - excellent income
            "skor_kredit": 800.0,      # Perfect credit score
            "jumlah_pinjaman": 120000000.0,  # 120M loan
            "rasio_pinjaman_pendapatan": 4.29,  # 4.3x ratio - reasonable
            "pekerjaan_Freelance": False,
            "pekerjaan_Kontrak": False,
            "pekerjaan_Tetap": True,   # Permanent job
            "kategori_umur_Dewasa": True,
            "kategori_umur_Muda": False,
            "kategori_umur_Senior": False,
            "kategori_skor_kredit_Excellent": True,  # Excellent credit
            "kategori_skor_kredit_Fair": False,
            "kategori_skor_kredit_Good": False,
            "kategori_skor_kredit_Poor": False,
            "kategori_pendapatan_Rendah": False,
            "kategori_pendapatan_Sedang": False,
            "kategori_pendapatan_Tinggi": True   # High income
        }
    }
    
    print(f"\n TESTING EXCELLENT PROFILE:")
    print(f"   Profile: VP Bank, 38y, Rp 28M/month, credit 800")
    print(f"   THIS SHOULD BE APPROVED WITH HIGH CONFIDENCE")
    print("-" * 50)
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=excellent_profile['data'],
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            prediction = result['predictions'][0]
            
            print(f"🎯 RESULT:")
            decision = " APPROVED" if prediction == 1 else " REJECTED"
            print(f"   Decision: {decision}")
            
            if 'probabilities' in result:
                prob_reject = result['probabilities'][0][0]
                prob_approve = result['probabilities'][0][1]
                
                print(f"   Reject Probability: {prob_reject:.4f} ({prob_reject*100:.2f}%)")
                print(f"   Approve Probability: {prob_approve:.4f} ({prob_approve*100:.2f}%)")
                
                # CRITICAL ANALYSIS
                print(f"\n🔍 CRITICAL ANALYSIS:")
                
                if prob_approve < 0.1:
                    print(f" SEVERE ISSUE: Approval prob {prob_approve:.3f} is extremely low!")
                    print(f"   Model thinks excellent profile has <10% approval chance")
                    print(f"   This indicates MAJOR model calibration problem")
                elif prob_approve < 0.3:
                    print(f" MAJOR ISSUE: Approval prob {prob_approve:.3f} is very low")
                    print(f"   Excellent profile should have >70% approval probability")
                elif prob_approve < 0.5:
                    print(f"THRESHOLD ISSUE: Approval prob {prob_approve:.3f} < 0.5")
                    print(f"   Model probabilities seem reasonable but threshold wrong")
                    print(f"   SOLUTION: Adjust threshold to ~0.3-0.4")
                else:
                    print(f"PREDICTION LOGIC ISSUE: Approval prob {prob_approve:.3f} > 0.5")
                    print(f"   But final prediction is REJECT - check prediction logic")
                
                # Test different thresholds
                print(f"\nTHRESHOLD ANALYSIS:")
                thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
                for threshold in thresholds:
                    would_approve = prob_approve >= threshold
                    status = " APPROVE" if would_approve else " REJECT"
                    print(f"   Threshold {threshold:.1f}: {status}")
                
                # Recommend solution
                print(f"\n💡 RECOMMENDED SOLUTION:")
                if prob_approve < 0.1:
                    print(f"   URGENT: Model needs complete retraining")
                    print(f"   - Current model is fundamentally broken")
                    print(f"   - Use previous working model as backup")
                    print(f"   - Investigate training data issues")
                elif prob_approve < 0.3:
                    print(f"   Model retraining recommended")
                    print(f"   - Class imbalance too extreme")
                    print(f"   - Adjust class weights in training")
                elif prob_approve < 0.5:
                    print(f"   Adjust prediction threshold")
                    optimal_threshold = max(0.1, prob_approve - 0.1)
                    print(f"   - Use threshold ~{optimal_threshold:.2f} instead of 0.5")
                    print(f"   - Quick fix without retraining")
                else:
                    print(f"   Check prediction logic in API")
                    
            else:
                print(" No probability information available")
                print("   Cannot diagnose threshold vs model issue")
                
        else:
            print(f" Prediction failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f" Error: {e}")
    
    # Test multiple profiles to confirm pattern
    print(f"\nTESTING MULTIPLE PROFILES FOR PATTERN CONFIRMATION:")
    
    test_profiles = [
        {
            "name": "Low Risk",
            "income": 15000000,
            "credit": 750,
            "debt_ratio": 3.0,
            "expected_approve_prob": ">0.8"
        },
        {
            "name": "Medium Risk", 
            "income": 8000000,
            "credit": 650,
            "debt_ratio": 4.0,
            "expected_approve_prob": "0.5-0.7"
        },
        {
            "name": "High Risk",
            "income": 4000000,
            "credit": 550,
            "debt_ratio": 7.0,
            "expected_approve_prob": "<0.3"
        }
    ]
    
    approval_probs = []
    
    for profile in test_profiles:
        data = {
            "umur": 35.0,
            "pendapatan": float(profile['income']),
            "skor_kredit": float(profile['credit']),
            "jumlah_pinjaman": float(profile['income'] * profile['debt_ratio']),
            "rasio_pinjaman_pendapatan": profile['debt_ratio'],
            "pekerjaan_Tetap": True, "pekerjaan_Kontrak": False, "pekerjaan_Freelance": False,
            "kategori_umur_Dewasa": True, "kategori_umur_Muda": False, "kategori_umur_Senior": False,
            "kategori_skor_kredit_Good": profile['credit'] >= 650,
            "kategori_skor_kredit_Fair": 550 <= profile['credit'] < 650,
            "kategori_skor_kredit_Poor": profile['credit'] < 550,
            "kategori_skor_kredit_Excellent": profile['credit'] >= 750,
            "kategori_pendapatan_Tinggi": profile['income'] >= 15000000,
            "kategori_pendapatan_Sedang": 7000000 <= profile['income'] < 15000000,
            "kategori_pendapatan_Rendah": profile['income'] < 7000000
        }
        
        try:
            response = requests.post(f"{API_URL}/predict", json=data)
            if response.status_code == 200:
                result = response.json()
                if 'probabilities' in result:
                    prob_approve = result['probabilities'][0][1]
                    approval_probs.append(prob_approve)
                    print(f"   {profile['name']}: {prob_approve:.3f} (expected {profile['expected_approve_prob']})")
                else:
                    print(f"   {profile['name']}: No probabilities available")
            else:
                print(f"   {profile['name']}: Error {response.status_code}")
        except:
            print(f"   {profile['name']}: Connection error")
    
    # Final diagnosis
    print(f"\nFINAL DIAGNOSIS:")
    if approval_probs:
        max_prob = max(approval_probs)
        min_prob = min(approval_probs)
        avg_prob = sum(approval_probs) / len(approval_probs)
        
        print(f"   Approval probabilities range: {min_prob:.3f} - {max_prob:.3f}")
        print(f"   Average approval probability: {avg_prob:.3f}")
        
        if max_prob < 0.2:
            print(f"   CRITICAL: Model is completely broken")
            print(f"   Even best profiles have <20% approval probability")
            print(f"   IMMEDIATE ACTION: Revert to previous model")
        elif max_prob < 0.5:
            print(f"   MAJOR ISSUE: Model too conservative")
            print(f"   SOLUTION: Adjust threshold to {max_prob - 0.1:.2f}")
        else:
            print(f"   THRESHOLD ISSUE: Probabilities seem OK")
            print(f"   SOLUTION: Check prediction logic in API")
    
    print(f"\nIMMEDIATE ACTIONS REQUIRED:")
    print(f"1. If probabilities all <20%: Revert to backup model")
    print(f"2. If probabilities 20-50%: Adjust threshold")
    print(f"3. If probabilities >50%: Fix API prediction logic")
    print(f"4. Test solution immediately before production use")

def quick_threshold_test():
    """Quick test different thresholds"""
    print(f"\n⚡ QUICK THRESHOLD ADJUSTMENT TEST")
    print("=" * 40)
    
    # Test excellent profile
    data = {
        "umur": 38.0, "pendapatan": 28000000.0, "skor_kredit": 800.0,
        "jumlah_pinjaman": 120000000.0, "rasio_pinjaman_pendapatan": 4.29,
        "pekerjaan_Tetap": True, "pekerjaan_Kontrak": False, "pekerjaan_Freelance": False,
        "kategori_umur_Dewasa": True, "kategori_umur_Muda": False, "kategori_umur_Senior": False,
        "kategori_skor_kredit_Excellent": True, "kategori_skor_kredit_Fair": False, 
        "kategori_skor_kredit_Good": False, "kategori_skor_kredit_Poor": False,
        "kategori_pendapatan_Tinggi": True, "kategori_pendapatan_Sedang": False, "kategori_pendapatan_Rendah": False
    }
    
    try:
        response = requests.post(f"{API_URL}/predict", json=data)
        if response.status_code == 200:
            result = response.json()
            if 'probabilities' in result:
                prob_approve = result['probabilities'][0][1]
                
                print(f"VP Bank profile approval probability: {prob_approve:.3f}")
                print(f"\nWith different thresholds:")
                
                optimal_threshold = None
                for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    would_approve = prob_approve >= threshold
                    status = " APPROVE" if would_approve else " REJECT"
                    print(f"  Threshold {threshold}: {status}")
                    
                    if would_approve and optimal_threshold is None:
                        optimal_threshold = threshold
                
                if optimal_threshold:
                    print(f"\nSUGGESTED THRESHOLD: {optimal_threshold}")
                    print(f"   This would approve excellent profiles")
                else:
                    print(f"\nNO VIABLE THRESHOLD FOUND")
                    print(f"   Model probabilities too low - needs retraining")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    diagnose_model_issue()
    quick_threshold_test()