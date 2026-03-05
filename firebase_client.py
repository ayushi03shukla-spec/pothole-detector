import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

SERVICE_ACCOUNT_PATH = "serviceAccountKey.json"

def get_db():
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        firebase_admin.initialize_app(cred)
    return firestore.client()

def save_detection(record: dict):
    db = get_db()
    if "timestamp" not in record:
        record["timestamp"] = datetime.now().isoformat()
    db.collection("detections").add(record)

def fetch_detections(limit=200):
    db = get_db()
    docs = (
        db.collection("detections")
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
    )

    data = []
    for doc in docs:
        item = doc.to_dict()
        item["id"] = doc.id
        data.append(item)
    return data