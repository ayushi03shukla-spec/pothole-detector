import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

def get_db():
    # Initialize only once
    if not firebase_admin._apps:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
    return firestore.client()

def save_detection(record: dict):
    db = get_db()
    # collection name: detections
    db.collection("detections").add(record)