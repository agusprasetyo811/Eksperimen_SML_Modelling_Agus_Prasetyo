import requests
import json
import pandas as pd
from datetime import datetime

API_URL = "http://localhost:5002"

def transform_to_model_format(raw_data):
    """Transform natural data to model format"""
    rasio_pinjaman_pendapatan = raw_data['jumlah_pinjaman'] / raw_data['pendapatan']
    
    # One-hot encoding pekerjaan
    pekerjaan_freelance = 1 if raw_data['pekerjaan'] == 'Freelance' else 0
    pekerjaan_kontrak = 1 if raw_data['pekerjaan'] == 'Kontrak' else 0
    pekerjaan_tetap = 1 if raw_data['pekerjaan'] == 'Tetap' else 0
    
    # Kategori umur
    umur = raw_data['umur']
    kategori_umur_muda = 1 if umur < 30 else 0
    kategori_umur_dewasa = 1 if 30 <= umur < 50 else 0
    kategori_umur_senior = 1 if umur >= 50 else 0
    
    # Kategori skor kredit
    skor = raw_data['skor_kredit']
    kategori_skor_poor = 1 if skor < 600 else 0
    kategori_skor_fair = 1 if 600 <= skor < 700 else 0
    kategori_skor_good = 1 if 700 <= skor < 800 else 0
    kategori_skor_excellent = 1 if skor >= 800 else 0
    
    # Kategori pendapatan (dalam juta rupiah)
    pendapatan_juta = raw_data['pendapatan'] / 1000000
    kategori_pendapatan_rendah = 1 if pendapatan_juta < 7 else 0
    kategori_pendapatan_sedang = 1 if 7 <= pendapatan_juta < 15 else 0
    kategori_pendapatan_tinggi = 1 if pendapatan_juta >= 15 else 0
    
    return {
        "umur": float(raw_data['umur']),
        "pendapatan": float(raw_data['pendapatan']),
        "skor_kredit": float(raw_data['skor_kredit']),
        "jumlah_pinjaman": float(raw_data['jumlah_pinjaman']),
        "rasio_pinjaman_pendapatan": float(rasio_pinjaman_pendapatan),
        "pekerjaan_Freelance": bool(pekerjaan_freelance),
        "pekerjaan_Kontrak": bool(pekerjaan_kontrak),
        "pekerjaan_Tetap": bool(pekerjaan_tetap),
        "kategori_umur_Dewasa": bool(kategori_umur_dewasa),
        "kategori_umur_Muda": bool(kategori_umur_muda),
        "kategori_umur_Senior": bool(kategori_umur_senior),
        "kategori_skor_kredit_Excellent": bool(kategori_skor_excellent),
        "kategori_skor_kredit_Fair": bool(kategori_skor_fair),
        "kategori_skor_kredit_Good": bool(kategori_skor_good),
        "kategori_skor_kredit_Poor": bool(kategori_skor_poor),
        "kategori_pendapatan_Rendah": bool(kategori_pendapatan_rendah),
        "kategori_pendapatan_Sedang": bool(kategori_pendapatan_sedang),
        "kategori_pendapatan_Tinggi": bool(kategori_pendapatan_tinggi)
    }

# All test scenarios for batch processing
BATCH_SCENARIOS = [
    # Strong Positive Cases
    {
        "name": "PNS Senior",
        "category": "🟢 STRONG POSITIVE",
        "data": {"umur": 45, "pekerjaan": "Tetap", "pendapatan": 15000000, "skor_kredit": 750, "jumlah_pinjaman": 60000000},
        "expected": "APPROVED"
    },
    {
        "name": "Dokter Spesialis",
        "category": "🟢 STRONG POSITIVE", 
        "data": {"umur": 32, "pekerjaan": "Tetap", "pendapatan": 22000000, "skor_kredit": 780, "jumlah_pinjaman": 85000000},
        "expected": "APPROVED"
    },
    {
        "name": "VP Bank",
        "category": "🟢 STRONG POSITIVE",
        "data": {"umur": 38, "pekerjaan": "Tetap", "pendapatan": 28000000, "skor_kredit": 800, "jumlah_pinjaman": 120000000},
        "expected": "APPROVED"
    },
    
    # Moderate Positive Cases
    {
        "name": "Guru Bersertifikat",
        "category": "🟡 MODERATE POSITIVE",
        "data": {"umur": 35, "pekerjaan": "Tetap", "pendapatan": 8500000, "skor_kredit": 720, "jumlah_pinjaman": 30000000},
        "expected": "APPROVED"
    },
    {
        "name": "Software Engineer",
        "category": "🟡 MODERATE POSITIVE",
        "data": {"umur": 28, "pekerjaan": "Tetap", "pendapatan": 12000000, "skor_kredit": 690, "jumlah_pinjaman": 45000000},
        "expected": "APPROVED"
    },
    
    # Border Cases
    {
        "name": "Freelance Designer",
        "category": "🟡 BORDER CASE",
        "data": {"umur": 30, "pekerjaan": "Freelance", "pendapatan": 11000000, "skor_kredit": 680, "jumlah_pinjaman": 42000000},
        "expected": "UNCERTAIN"
    },
    {
        "name": "Kontrak MNC",
        "category": "🟡 BORDER CASE",
        "data": {"umur": 26, "pekerjaan": "Kontrak", "pendapatan": 9500000, "skor_kredit": 650, "jumlah_pinjaman": 35000000},
        "expected": "UNCERTAIN"
    },
    {
        "name": "UMKM Owner",
        "category": "🟡 BORDER CASE",
        "data": {"umur": 42, "pekerjaan": "Freelance", "pendapatan": 13000000, "skor_kredit": 670, "jumlah_pinjaman": 65000000},
        "expected": "UNCERTAIN"
    },
    
    # Moderate Negative Cases
    {
        "name": "Fresh Grad Overambitious",
        "category": "🟠 MODERATE NEGATIVE",
        "data": {"umur": 23, "pekerjaan": "Kontrak", "pendapatan": 5500000, "skor_kredit": 620, "jumlah_pinjaman": 35000000},
        "expected": "REJECTED"
    },
    {
        "name": "Driver Online",
        "category": "🟠 MODERATE NEGATIVE",
        "data": {"umur": 35, "pekerjaan": "Freelance", "pendapatan": 6500000, "skor_kredit": 590, "jumlah_pinjaman": 40000000},
        "expected": "REJECTED"
    },
    
    # Strong Negative Cases
    {
        "name": "Pengangguran",
        "category": "🔴 STRONG NEGATIVE",
        "data": {"umur": 28, "pekerjaan": "Kontrak", "pendapatan": 3000000, "skor_kredit": 520, "jumlah_pinjaman": 25000000},
        "expected": "REJECTED"
    },
    {
        "name": "Mahasiswa",
        "category": "🔴 STRONG NEGATIVE",
        "data": {"umur": 22, "pekerjaan": "Kontrak", "pendapatan": 2500000, "skor_kredit": 480, "jumlah_pinjaman": 20000000},
        "expected": "REJECTED"
    },
    {
        "name": "Pensiunan Dini",
        "category": "🔴 STRONG NEGATIVE",
        "data": {"umur": 52, "pekerjaan": "Freelance", "pendapatan": 4000000, "skor_kredit": 540, "jumlah_pinjaman": 50000000},
        "expected": "REJECTED"
    },
    
    # Edge Cases
    {
        "name": "Content Creator",
        "category": "🔵 EDGE CASE",
        "data": {"umur": 25, "pekerjaan": "Freelance", "pendapatan": 18000000, "skor_kredit": 710, "jumlah_pinjaman": 75000000},
        "expected": "UNCERTAIN"
    },
    {
        "name": "Professional Athlete",
        "category": "🔵 EDGE CASE",
        "data": {"umur": 27, "pekerjaan": "Freelance", "pendapatan": 15000000, "skor_kredit": 690, "jumlah_pinjaman": 55000000},
        "expected": "UNCERTAIN"
    },
    {
        "name": "Ex-Official Reformed",
        "category": "🔵 EDGE CASE",
        "data": {"umur": 48, "pekerjaan": "Freelance", "pendapatan": 12000000, "skor_kredit": 450, "jumlah_pinjaman": 30000000},
        "expected": "REJECTED"
    }
]

def run_batch_test():
    """Run batch prediction test"""
    print(f"BATCH CREDIT APPROVAL TEST")
    print(f"Total Scenarios: {len(BATCH_SCENARIOS)}")
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*80}")
    
    # Test API connection
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code != 200:
            print("API is not healthy")
            return
        print("API connection successful\n")
    except:
        print("Cannot connect to API")
        return
    
    # Prepare batch data
    batch_data = []
    scenario_info = []
    
    for scenario in BATCH_SCENARIOS:
        transformed = transform_to_model_format(scenario['data'])
        batch_data.append(transformed)
        scenario_info.append({
            'name': scenario['name'],
            'category': scenario['category'],
            'expected': scenario['expected'],
            'debt_ratio': scenario['data']['jumlah_pinjaman'] / scenario['data']['pendapatan'],
            'income': scenario['data']['pendapatan'],
            'credit_score': scenario['data']['skor_kredit'],
            'job_type': scenario['data']['pekerjaan']
        })
    
    # Make batch prediction
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=batch_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            predictions = result['predictions']
            probabilities = result.get('probabilities', [[0,0]] * len(predictions))
            
            # Create results dataframe for analysis
            results = []
            for i, (scenario, prediction, prob) in enumerate(zip(scenario_info, predictions, probabilities)):
                results.append({
                    'Name': scenario['name'],
                    'Category': scenario['category'], 
                    'Expected': scenario['expected'],
                    'Actual': 'APPROVED' if prediction == 1 else 'REJECTED',
                    'Prob_Approve': prob[1] if len(prob) > 1 else 0,
                    'Confidence': max(prob) if len(prob) > 1 else 0,
                    'Income_M': scenario['income'] / 1000000,
                    'Credit_Score': scenario['credit_score'],
                    'Debt_Ratio': scenario['debt_ratio'],
                    'Job_Type': scenario['job_type'],
                    'Match_Expected': (
                        (scenario['expected'] == 'APPROVED' and prediction == 1) or
                        (scenario['expected'] == 'REJECTED' and prediction == 0) or
                        (scenario['expected'] == 'UNCERTAIN')
                    )
                })
            
            # Display results
            print(f"BATCH PREDICTION RESULTS")
            print(f"{'='*80}")
            print(f"{'Name':<20} {'Category':<18} {'Expected':<10} {'Actual':<10} {'Confidence':<12} {'Match':<6}")
            print(f"{'-'*80}")
            
            for r in results:
                match_icon = "✅" if r['Match_Expected'] else "❌"
                confidence_str = f"{r['Confidence']:.3f}"
                print(f"{r['Name']:<20} {r['Category']:<18} {r['Expected']:<10} {r['Actual']:<10} {confidence_str:<12} {match_icon}")
            
            print(f"\nSUMMARY STATISTICS")
            print(f"{'='*50}")
            
            # Overall stats
            total_approved = sum(1 for r in results if r['Actual'] == 'APPROVED')
            total_rejected = len(results) - total_approved
            
            print(f"Overall Results:")
            print(f"   Total Cases: {len(results)}")
            print(f"   Approved: {total_approved} ({total_approved/len(results)*100:.1f}%)")
            print(f"   Rejected: {total_rejected} ({total_rejected/len(results)*100:.1f}%)")
            
            # Category analysis
            categories = {}
            for r in results:
                cat = r['Category']
                if cat not in categories:
                    categories[cat] = {'total': 0, 'approved': 0, 'high_conf': 0}
                categories[cat]['total'] += 1
                if r['Actual'] == 'APPROVED':
                    categories[cat]['approved'] += 1
                if r['Confidence'] > 0.8:
                    categories[cat]['high_conf'] += 1
            
            print(f"\nCategory Analysis:")
            for cat, stats in categories.items():
                approval_rate = stats['approved'] / stats['total'] * 100
                high_conf_rate = stats['high_conf'] / stats['total'] * 100
                print(f"   {cat}:")
                print(f"      Approval Rate: {stats['approved']}/{stats['total']} ({approval_rate:.1f}%)")
                print(f"      High Confidence: {stats['high_conf']}/{stats['total']} ({high_conf_rate:.1f}%)")
            
            # Risk analysis
            print(f"\nRisk Factor Analysis:")
            
            # High debt ratio cases
            high_debt = [r for r in results if r['Debt_Ratio'] > 4.0]
            high_debt_approved = sum(1 for r in high_debt if r['Actual'] == 'APPROVED')
            if high_debt:
                print(f"   High Debt Ratio (>4x): {high_debt_approved}/{len(high_debt)} approved ({high_debt_approved/len(high_debt)*100:.1f}%)")
            
            # Low credit score cases  
            low_credit = [r for r in results if r['Credit_Score'] < 600]
            low_credit_approved = sum(1 for r in low_credit if r['Actual'] == 'APPROVED')
            if low_credit:
                print(f"   Poor Credit (<600): {low_credit_approved}/{len(low_credit)} approved ({low_credit_approved/len(low_credit)*100:.1f}%)")
            
            # Freelance job cases
            freelance = [r for r in results if r['Job_Type'] == 'Freelance']
            freelance_approved = sum(1 for r in freelance if r['Actual'] == 'APPROVED')
            if freelance:
                print(f"   Freelance Jobs: {freelance_approved}/{len(freelance)} approved ({freelance_approved/len(freelance)*100:.1f}%)")
            
            # High confidence predictions
            high_conf = [r for r in results if r['Confidence'] > 0.8]
            print(f"\nModel Confidence:")
            print(f"   High Confidence (>80%): {len(high_conf)}/{len(results)} cases ({len(high_conf)/len(results)*100:.1f}%)")
            print(f"   Average Confidence: {sum(r['Confidence'] for r in results)/len(results):.3f}")
            
            # Most/least confident predictions
            most_confident = max(results, key=lambda x: x['Confidence'])
            least_confident = min(results, key=lambda x: x['Confidence'])
            
            print(f"\nConfidence Extremes:")
            print(f"   Most Confident: {most_confident['Name']} - {most_confident['Actual']} ({most_confident['Confidence']:.3f})")
            print(f"   Least Confident: {least_confident['Name']} - {least_confident['Actual']} ({least_confident['Confidence']:.3f})")
            
            # Export results to CSV
            try:
                df = pd.DataFrame(results)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"batch_test_results_{timestamp}.csv"
                df.to_csv(filename, index=False)
                print(f"\nResults exported to: {filename}")
            except:
                print(f"\nCould not export results to CSV")
            
        else:
            print(f"Batch prediction failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    print(f"\nTest completed at: {datetime.now().strftime('%H:%M:%S')}")

def run_category_comparison():
    """Compare model performance across different risk categories"""
    print(f"CATEGORY PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    # Group scenarios by category
    categories = {}
    for scenario in BATCH_SCENARIOS:
        cat = scenario['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(scenario)
    
    for category, scenarios in categories.items():
        print(f"\n{category} ({len(scenarios)} cases)")
        print(f"{'-'*40}")
        
        batch_data = [transform_to_model_format(s['data']) for s in scenarios]
        
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=batch_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                predictions = result['predictions']
                probabilities = result.get('probabilities', [[0,0]] * len(predictions))
                
                approved_count = sum(predictions)
                avg_confidence = sum(max(prob) for prob in probabilities) / len(probabilities)
                
                print(f"   Approval Rate: {approved_count}/{len(scenarios)} ({approved_count/len(scenarios)*100:.1f}%)")
                print(f"   Avg Confidence: {avg_confidence:.3f}")
                
                # Show individual results
                for i, (scenario, pred, prob) in enumerate(zip(scenarios, predictions, probabilities)):
                    decision = "APPROVED" if pred == 1 else "REJECTED"
                    confidence = max(prob)
                    print(f"     {scenario['name']:<25} {decision} ({confidence:.3f})")
                    
        except Exception as e:
            print(f"   Error testing category: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "category":
        run_category_comparison()
    else:
        run_batch_test()