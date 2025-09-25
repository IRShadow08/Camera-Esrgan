from deepface import DeepFace
import os
import joblib

def generate_embeddings(model_name="Facenet512", db_dir="Deepface DB"):
    output_filename = "model.joblib"
    embeddings_db = {}

    for person in os.listdir(db_dir):
        person_path = os.path.join(db_dir, person)
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
