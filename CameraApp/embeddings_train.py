from deepface import DeepFace
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_embeddings(model_name="Facenet512", db_dir="Deepface DB"):
    db_path = os.path.join(BASE_DIR, db_dir)
    output_filename = os.path.join(BASE_DIR, "model.joblib")
    embeddings_db = {}

    for person in os.listdir(db_path):
        person_path = os.path.join(db_path, person)
        if not os.path.isdir(person_path):
            continue
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            embedding = DeepFace.represent(
                img_path=img_path,
                model_name=model_name,
                enforce_detection=False
            )[0]["embedding"]

            if person not in embeddings_db:
                embeddings_db[person] = []
            embeddings_db[person].append(embedding)

    joblib.dump(embeddings_db, output_filename)
    return output_filename
