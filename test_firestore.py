from firebase_client import save_detection
from datetime import datetime

print("Program started")

save_detection({
    "filename": "python_test.jpg",
    "prediction": "Pothole",
    "confidence": 0.93,
    "severity": "High",
    "location": "Test Location",
    "lat": 28.61,
    "lon": 77.21,
    "timestamp": datetime.now().isoformat()
})

print("Saved ✅ Check Firestore now")