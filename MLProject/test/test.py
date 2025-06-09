#!/usr/bin/env python3
"""
Quick Start Guide - Credit Approval API Testing
Contoh cepat untuk test API dengan data realistic Indonesia

Usage: python quick_start_guide.py
"""

import requests
import json

# ============================================================================
# QUICK TEST EXAMPLES - Siap Pakai!
# ============================================================================

API_URL = "http://localhost:5002"

def quick_test():
    """Test cepat dengan 5 contoh realistic"""
    
    print("🚀 QUICK TEST - Credit Approval API")
    print("=" * 50)
    
    # Test koneksi
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print("✅ API connection successful")
        else:
            print("❌ API not healthy")
            return
    except:
        print("❌ Cannot connect to API")
        print("   Make sure server is running: python app.py --port 5002")
        return
    
    # 5 Test cases - siap pakai
    test_cases = [
        {
            "name": "✅ SHOULD APPROVE - PNS dengan gaji bagus",
            "data": {
                "umur": 35.0,
                "pendapatan": 12000000.0,  # 12 juta/bulan
                "skor_kredit": 720.0,      # Credit score bagus
                "jumlah_pinjaman": 40000000.0,  # 40 juta
                "rasio_pinjaman_pendapatan": 3.33,
                "pekerjaan_Freelance": False,
                "pekerjaan_Kontrak": False,
                "pekerjaan_Tetap": True,    # PNS/Pegawai tetap
                "kategori_umur_Dewasa": True,
                "kategori_umur_Muda": False,
                "kategori_umur_Senior": False,
                "kategori_skor_kredit_Excellent": False,
                "kategori_skor_kredit_Fair": False,
                "kategori_skor_kredit_Good": True,
                "kategori_skor_kredit_Poor": False,
                "kategori_pendapatan_Rendah": False,
                "kategori_pendapatan_Sedang": True,
                "kategori_pendapatan_Tinggi": False
            }
        },
        
        {
            "name": "🤔 UNCERTAIN - Freelancer dengan income tinggi tapi tidak stabil",
            "data": {
                "umur": 28.0,
                "pendapatan": 15000000.0,  # 15 juta (freelancer)
                "skor_kredit": 680.0,      # Credit score sedang
                "jumlah_pinjaman": 60000000.0,  # 60 juta
                "rasio_pinjaman_pendapatan": 4.0,
                "pekerjaan_Freelance": True,   # Freelancer (riskier)
                "pekerjaan_Kontrak": False,
                "pekerjaan_Tetap": False,
                "kategori_umur_Dewasa": False,
                "kategori_umur_Muda": True,
                "kategori_umur_Senior": False,
                "kategori_skor_kredit_Excellent": False,
                "kategori_skor_kredit_Fair": True,
                "kategori_skor_kredit_Good": False,
                "kategori_skor_kredit_Poor": False,
                "kategori_pendapatan_Rendah": False,
                "kategori_pendapatan_Sedang": False,
                "kategori_pendapatan_Tinggi": True
            }
        },
        
        {
            "name": "❌ SHOULD REJECT - Fresh grad dengan pinjaman terlalu besar",
            "data": {
                "umur": 23.0,
                "pendapatan": 5000000.0,   # 5 juta/bulan
                "skor_kredit": 620.0,      # Credit score rendah
                "jumlah_pinjaman": 35000000.0,  # 35 juta (terlalu besar)
                "rasio_pinjaman_pendapatan": 7.0,
                "pekerjaan_Freelance": False,
                "pekerjaan_Kontrak": True,    # Kontrak (kurang stabil)
                "pekerjaan_Tetap": False,
                "kategori_umur_Dewasa": False,
                "kategori_umur_Muda": True,
                "kategori_umur_Senior": False,
                "kategori_skor_kredit_Excellent": False,
                "kategori_skor_kredit_Fair": True,
                "kategori_skor_kredit_Good": False,
                "kategori_skor_kredit_Poor": False,
                "kategori_pendapatan_Rendah": True,
                "kategori_pendapatan_Sedang": False,
                "kategori_pendapatan_Tinggi": False
            }
        },
        
        {
            "name": "✅ SHOULD APPROVE - Manager senior dengan track record bagus",
            "data": {
                "umur": 42.0,
                "pendapatan": 20000000.0,  # 20 juta/bulan
                "skor_kredit": 780.0,      # Credit score excellent
                "jumlah_pinjaman": 80000000.0,  # 80 juta
                "rasio_pinjaman_pendapatan": 4.0,
                "pekerjaan_Freelance": False,
                "pekerjaan_Kontrak": False,
                "pekerjaan_Tetap": True,
                "kategori_umur_Dewasa": True,
                "kategori_umur_Muda": False,
                "kategori_umur_Senior": False,
                "kategori_skor_kredit_Excellent": False,
                "kategori_skor_kredit_Fair": False,
                "kategori_skor_kredit_Good": True,
                "kategori_skor_kredit_Poor": False,
                "kategori_pendapatan_Rendah": False,
                "kategori_pendapatan_Sedang": False,
                "kategori_pendapatan_Tinggi": True
            }
        },
        
        {
            "name": "❌ SHOULD REJECT - Credit score buruk dengan ratio tinggi",
            "data": {
                "umur": 30.0,
                "pendapatan": 6000000.0,   # 6 juta/bulan
                "skor_kredit": 520.0,      # Credit score buruk
                "jumlah_pinjaman": 40000000.0,  # 40 juta
                "rasio_pinjaman_pendapatan": 6.67,
                "pekerjaan_Freelance": True,
                "pekerjaan_Kontrak": False,
                "pekerjaan_Tetap": False,
                "kategori_umur_Dewasa": True,
                "kategori_umur_Muda": False,
                "kategori_umur_Senior": False,
                "kategori_skor_kredit_Excellent": False,
                "kategori_skor_kredit_Fair": False,
                "kategori_skor_kredit_Good": False,
                "kategori_skor_kredit_Poor": True,
                "kategori_pendapatan_Rendah": True,
                "kategori_pendapatan_Sedang": False,
                "kategori_pendapatan_Tinggi": False
            }
        }
    ]
    
    # Test setiap case
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/5] {test_case['name']}")
        print("-" * 60)
        
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=test_case['data'],
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = result['predictions'][0]
                
                # Show result
                decision = "✅ APPROVED" if prediction == 1 else "❌ REJECTED"
                print(f"🎯 Result: {decision}")
                
                if 'probabilities' in result:
                    prob_approve = result['probabilities'][0][1]
                    confidence = max(result['probabilities'][0])
                    print(f"📊 Approval Probability: {prob_approve:.3f} ({prob_approve*100:.1f}%)")
                    print(f"🎲 Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
                    
                    # Simple interpretation
                    if confidence > 0.8:
                        print(f"💪 Model is very confident in this decision")
                    elif confidence > 0.65:
                        print(f"👍 Model is confident in this decision")
                    else:
                        print(f"🤔 Model is uncertain about this decision")
                
            else:
                print(f"❌ Prediction failed: {response.status_code}")
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        if i < len(test_cases):
            input("\nPress Enter to continue...")
    
    print(f"\n{'='*60}")
    print("✅ Quick test completed!")
    print("📝 For more detailed testing, use:")
    print("   python mixed_test_scenarios.py")
    print("   python batch_comparison_test.py")
    print(f"{'='*60}")

def custom_test():
    """Test dengan input custom dari user"""
    print("\n🎯 CUSTOM INPUT TEST")
    print("=" * 40)
    
    try:
        # Input natural dari user
        print("Masukkan data pemohon kredit:")
        umur = int(input("Umur (tahun): "))
        
        print("\nPilih jenis pekerjaan:")
        print("1. Pegawai Tetap (PNS/Swasta)")
        print("2. Kontrak")
        print("3. Freelance/Wiraswasta")
        pekerjaan_choice = int(input("Pilih (1-3): "))
        
        pendapatan = int(input("Pendapatan per bulan (Rp): "))
        skor_kredit = int(input("Skor Kredit (300-850): "))
        jumlah_pinjaman = int(input("Jumlah Pinjaman yang diminta (Rp): "))
        
        # Transform ke format model
        rasio = jumlah_pinjaman / pendapatan
        
        # Job encoding
        pekerjaan_tetap = pekerjaan_choice == 1
        pekerjaan_kontrak = pekerjaan_choice == 2
        pekerjaan_freelance = pekerjaan_choice == 3
        
        # Age category
        kategori_muda = umur < 30
        kategori_dewasa = 30 <= umur < 50
        kategori_senior = umur >= 50
        
        # Credit category
        kategori_poor = skor_kredit < 600
        kategori_fair = 600 <= skor_kredit < 700
        kategori_good = 700 <= skor_kredit < 800
        kategori_excellent = skor_kredit >= 800
        
        # Income category (in millions)
        pendapatan_juta = pendapatan / 1000000
        kategori_rendah = pendapatan_juta < 7
        kategori_sedang = 7 <= pendapatan_juta < 15
        kategori_tinggi = pendapatan_juta >= 15
        
        # Prepare data
        data = {
            "umur": float(umur),
            "pendapatan": float(pendapatan),
            "skor_kredit": float(skor_kredit),
            "jumlah_pinjaman": float(jumlah_pinjaman),
            "rasio_pinjaman_pendapatan": rasio,
            "pekerjaan_Freelance": pekerjaan_freelance,
            "pekerjaan_Kontrak": pekerjaan_kontrak,
            "pekerjaan_Tetap": pekerjaan_tetap,
            "kategori_umur_Dewasa": kategori_dewasa,
            "kategori_umur_Muda": kategori_muda,
            "kategori_umur_Senior": kategori_senior,
            "kategori_skor_kredit_Excellent": kategori_excellent,
            "kategori_skor_kredit_Fair": kategori_fair,
            "kategori_skor_kredit_Good": kategori_good,
            "kategori_skor_kredit_Poor": kategori_poor,
            "kategori_pendapatan_Rendah": kategori_rendah,
            "kategori_pendapatan_Sedang": kategori_sedang,
            "kategori_pendapatan_Tinggi": kategori_tinggi
        }
        
        # Show summary
        print(f"\n📊 Data Summary:")
        print(f"   Umur: {umur} tahun")
        print(f"   Pekerjaan: {'Tetap' if pekerjaan_tetap else 'Kontrak' if pekerjaan_kontrak else 'Freelance'}")
        print(f"   Pendapatan: Rp {pendapatan:,}/bulan")
        print(f"   Skor Kredit: {skor_kredit}")
        print(f"   Pinjaman: Rp {jumlah_pinjaman:,}")
        print(f"   Debt Ratio: {rasio:.2f}x")
        
        # Make prediction
        response = requests.post(
            f"{API_URL}/predict",
            json=data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            prediction = result['predictions'][0]
            
            decision = "✅ APPROVED" if prediction == 1 else "❌ REJECTED"
            print(f"\n🎯 HASIL PREDIKSI: {decision}")
            
            if 'probabilities' in result:
                prob_approve = result['probabilities'][0][1]
                prob_reject = result['probabilities'][0][0]
                
                print(f"📊 Probabilitas:")
                print(f"   Ditolak: {prob_reject:.3f} ({prob_reject*100:.1f}%)")
                print(f"   Disetujui: {prob_approve:.3f} ({prob_approve*100:.1f}%)")
                
                # Interpretation
                if prediction == 1:
                    if prob_approve > 0.8:
                        print(f"💪 Sangat mungkin disetujui!")
                    elif prob_approve > 0.65:
                        print(f"👍 Kemungkinan besar disetujui")
                    else:
                        print(f"🤔 Mungkin disetujui, tapi perlu review")
                else:
                    if prob_reject > 0.8:
                        print(f"🚫 Sangat mungkin ditolak")
                    elif prob_reject > 0.65:
                        print(f"👎 Kemungkinan besar ditolak")
                    else:
                        print(f"⚠️ Mungkin ditolak, borderline case")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except ValueError:
        print("❌ Input tidak valid. Pastikan angka dimasukkan dengan benar.")
    except Exception as e:
        print(f"❌ Error: {e}")

def show_help():
    """Show help and available commands"""
    print("""
🔧 AVAILABLE TESTING OPTIONS:

1. QUICK TEST (Recommended)
   python quick_start_guide.py
   - 5 realistic test cases
   - Fast and easy to understand
   - Good for validating API works

2. CUSTOM INPUT TEST  
   python quick_start_guide.py custom
   - Input your own data
   - Interactive testing
   - Good for specific scenarios

3. COMPREHENSIVE TESTING
   python mixed_test_scenarios.py
   - 18 detailed scenarios
   - Complete analysis
   - Best for thorough validation

4. BATCH TESTING
   python batch_comparison_test.py  
   - All scenarios at once
   - Statistical analysis
   - Best for performance metrics

5. API ENDPOINTS
   GET  /health    - Check API status
   GET  /info      - Model information
   GET  /features  - Feature requirements
   GET  /test      - Sample prediction
   POST /predict   - Make predictions

📋 EXAMPLE API CALLS:

# Check if API is running
curl http://localhost:5002/health

# Get model info
curl http://localhost:5002/info

# Test sample prediction
curl http://localhost:5002/test

🚀 START HERE: Run 'python quick_start_guide.py' for immediate testing!
    """)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "custom":
            custom_test()
        elif sys.argv[1] == "help":
            show_help()
        else:
            print("Unknown option. Use 'custom' or 'help'")
    else:
        quick_test()