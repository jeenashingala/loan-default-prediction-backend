"""
Comprehensive Backend Test Suite:
Validates startup, ML artifact consistency, /health, /model, /predict, DB insertion,
/predictions, /predictions/{id}, DELETE, /analytics/dashboard, and /analytics/metrics.
"""
import sys
from pathlib import Path

# Add backend directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient
from app.main import app

def run_tests():
    print("=== STARTING BACKEND TEST SUITE ===")
    
    with TestClient(app) as client:
        # 1. Root & Health Check
        print("\n1. Testing GET / and GET /health...")
        res_root = client.get("/")
        assert res_root.status_code == 200, f"Root failed: {res_root.text}"
        print("Root response:", res_root.json())

        res_health = client.get("/health")
        assert res_health.status_code == 200, f"Health check failed: {res_health.text}"
        health_data = res_health.json()
        print("Health response:", health_data)
        assert health_data["status"] == "healthy"
        assert health_data["model_loaded"] is True
        assert health_data["model"] == "Balanced Logistic Regression"

        # 2. Model Info
        print("\n2. Testing GET /model...")
        res_model = client.get("/model")
        assert res_model.status_code == 200, f"Model info failed: {res_model.text}"
        model_data = res_model.json()
        print(f"Model: {model_data['name']}, Features: {model_data['features']}")
        assert model_data["features"] == 24
        assert len(model_data["training_features"]) == 24

        # 3. Predict Endpoint (Low Risk Sample)
        print("\n3. Testing POST /predict with prime sample...")
        prime_payload = {
            "applicantName": "Vikram Sengupta",
            "Age": 42,
            "Income": 145000,
            "LoanAmount": 20000,
            "CreditScore": 780,
            "MonthsEmployed": 72,
            "NumCreditLines": 3,
            "InterestRate": 8.2,
            "LoanTerm": 36,
            "DTIRatio": 0.22,
            "Education": "Master's",
            "EmploymentType": "Full-time",
            "MaritalStatus": "Married",
            "HasMortgage": "Yes",
            "HasDependents": "No",
            "LoanPurpose": "Home",
            "HasCoSigner": "Yes"
        }
        res_pred1 = client.post("/predict", json=prime_payload)
        assert res_pred1.status_code == 200, f"Predict prime failed: {res_pred1.text}"
        pred1_data = res_pred1.json()
        print("Prime prediction output:", pred1_data)
        assert pred1_data["prediction"] in (0, 1)
        assert "default_probability" in pred1_data
        assert "no_default_probability" in pred1_data
        assert pred1_data["model"] == "Balanced Logistic Regression"
        record1_id = pred1_data["id"]

        # 4. Predict Endpoint (High Risk Sample)
        print("\n4. Testing POST /predict with high risk sample...")
        high_risk_payload = {
            "applicantName": "Aman Verma",
            "Age": 22,
            "Income": 18000,
            "LoanAmount": 90000,
            "CreditScore": 420,
            "MonthsEmployed": 3,
            "NumCreditLines": 8,
            "InterestRate": 25.0,
            "LoanTerm": 60,
            "DTIRatio": 0.85,
            "Education": "High School",
            "EmploymentType": "Unemployed",
            "MaritalStatus": "Single",
            "HasMortgage": "No",
            "HasDependents": "Yes",
            "LoanPurpose": "Business",
            "HasCoSigner": "No"
        }
        res_pred2 = client.post("/predict", json=high_risk_payload)
        assert res_pred2.status_code == 200, f"Predict high-risk failed: {res_pred2.text}"
        pred2_data = res_pred2.json()
        print("High risk prediction output:", pred2_data)
        assert pred2_data["prediction"] == 1
        assert pred2_data["risk_level"] == "High"
        record2_id = pred2_data["id"]

        # 5. Invalid categorical validation (422 test)
        print("\n5. Testing 422 validation on invalid categorical...")
        invalid_payload = dict(prime_payload)
        invalid_payload["Education"] = "Kindergarten"
        res_inv = client.post("/predict", json=invalid_payload)
        assert res_inv.status_code == 422, f"Expected 422, got {res_inv.status_code}"
        print("422 Validation correctly returned:", res_inv.json()["detail"][0]["msg"])

        # 6. Prediction History
        print("\n6. Testing GET /predictions...")
        res_hist = client.get("/predictions")
        assert res_hist.status_code == 200, f"History failed: {res_hist.text}"
        hist_data = res_hist.json()
        print(f"Total history records in DB: {hist_data['total']}")
        assert hist_data["total"] >= 2
        assert len(hist_data["items"]) >= 2

        # 7. Single Prediction Details
        print(f"\n7. Testing GET /predictions/{record1_id}...")
        res_single = client.get(f"/predictions/{record1_id}")
        assert res_single.status_code == 200, f"Single prediction failed: {res_single.text}"
        assert res_single.json()["id"] == record1_id

        # 8. Single Prediction 404 test
        print("\n8. Testing GET /predictions/NON_EXISTENT...")
        res_404 = client.get("/predictions/APP-999999")
        assert res_404.status_code == 404, f"Expected 404, got {res_404.status_code}"

        # 9. Dashboard Analytics
        print("\n9. Testing GET /analytics/dashboard...")
        res_dash = client.get("/analytics/dashboard")
        assert res_dash.status_code == 200, f"Dashboard analytics failed: {res_dash.text}"
        dash_data = res_dash.json()
        print("Dashboard stats summary:", dash_data["stats"])
        assert dash_data["total_applications"] >= 2
        assert len(dash_data["distribution"]) == 2
        assert len(dash_data["applicationsOverTime"]) == 7

        # 10. Metrics Analytics
        print("\n10. Testing GET /analytics/metrics...")
        res_metrics = client.get("/analytics/metrics")
        assert res_metrics.status_code == 200, f"Analytics metrics failed: {res_metrics.text}"
        metrics_data = res_metrics.json()
        print("Cohort breakdown count (employment):", len(metrics_data["byEmployment"]))
        print("Scatter data count:", len(metrics_data["scatterData"]))

        # 11. Delete Prediction
        print(f"\n11. Testing DELETE /predictions/{record2_id}...")
        res_del = client.delete(f"/predictions/{record2_id}")
        assert res_del.status_code == 200, f"Delete failed: {res_del.text}"
        assert res_del.json()["success"] is True

        # Verify it's deleted
        res_del_verify = client.get(f"/predictions/{record2_id}")
        assert res_del_verify.status_code == 404, "Deleted record still exists!"
        print("Delete verified successfully!")

    print("\n=== ALL BACKEND TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_tests()
