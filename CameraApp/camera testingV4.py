from deepface import DeepFace
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import os
import joblib
from scipy.spatial.distance import cosine
import datetime as dt
import requests
import json
import io
import re
from embeddings_train import generate_embeddings

#ESRGAN new
import time
from datetime import datetime
import subprocess
import threading
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DeepFace Camera Testings")

        self.recognition_logs = []
        self.last_seen = {}
        self.logged_names = set()

        self.tabs = ttk.Notebook(root)
        self.tabs.pack(expand=1, fill="both")
        self.image_refs = {}

        # Camera Tab
        self.camera_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.camera_tab, text="Add to Database")

        self.add_image_label = tk.Label(self.camera_tab)
        self.add_image_label.pack()

        self.name_var = tk.StringVar()
        ttk.Label(self.camera_tab, text="Enter name:").pack()
        ttk.Entry(self.camera_tab, textvariable=self.name_var).pack()

        ttk.Button(self.camera_tab, text="Take Picture", command=self.take_picture).pack()

        ttk.Label(self.camera_tab, text="Select Course to Upload Model To:").pack(pady=(10, 2))
        self.selected_course = tk.StringVar()
        self.upload_course_combo = ttk.Combobox(self.camera_tab, textvariable=self.selected_course, state="readonly")
        self.upload_course_combo.pack(pady=2)

        self.train_button = ttk.Button(self.camera_tab, text="Train Faces", command=self.train_faces)
        self.train_button.pack(pady=5)

        # Recognition Tab
        self.fr_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.fr_tab, text="Live Recognition")

        self.live_image_label = tk.Label(self.fr_tab)
        self.live_image_label.pack()

        # self.live_name_var = tk.StringVar(value="Recognized: Unknown")
        # ttk.Label(self.fr_tab, textvariable=self.live_name_var, font=("Arial", 14)).pack(pady=10)

        self.cap = cv2.VideoCapture(0)

        # ESRGAN auto-save setup
        self.last_esrgan_capture = 0
        self.esrgan_interval = 60  # seconds
        self.esrgan_enabled = False

        self.esrgan_input_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "ESRGAN", "input"))
        self.esrgan_output_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "ESRGAN", "output"))
        self.last_checked_file = None

        threading.Thread(target=self.watch_esrgan_output, daemon=True).start()
        self.last_checked_output = 0
        self.output_check_interval = 5  # check every 5 seconds for new ESRGAN result

        self.frame = None
        os.makedirs(os.path.join(BASE_DIR, "Deepface DB"), exist_ok=True)

        self.db_integration = DatabaseIntegration()

        self.course_var = tk.StringVar()
        self.course_combo = ttk.Combobox(self.fr_tab, textvariable=self.course_var, state="readonly")
        self.course_combo.pack(pady=5)

        self.root.after(5000, self.load_courses)
        ttk.Button(self.fr_tab, text="Retry Load Courses", command=self.load_courses).pack(pady=5)
        ttk.Button(self.fr_tab, text="Select Course", command=self.select_course).pack(pady=5)

        # Delete Images Tab
        self.delete_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.delete_tab, text="Delete Images")
        ttk.Button(self.delete_tab, text="Load Courses", command=self.load_courses).pack(pady=5)
        ttk.Button(self.delete_tab, text="Delete Images", command=self.delete_images_from_tab).pack(pady=5)

        os.makedirs(os.path.join(BASE_DIR, "Deepface DB Record"), exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, "Deepface DB Frame"), exist_ok=True)

        self.update_frame()

    def take_picture(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("Missing name", "Please enter a name.")
            return
        folder_path = os.path.join(BASE_DIR, "Deepface DB", name)
        os.makedirs(folder_path, exist_ok=True)
        filename = f"img{len(os.listdir(folder_path))+1:03}.jpg"
        filepath = os.path.join(folder_path, filename)
        cv2.imwrite(filepath, self.frame)
        messagebox.showinfo("Saved", f"Picture saved to {filepath}")

    def update_frame(self): #Deleted old Recognizer Live
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.live_image_label.imgtk = imgtk
            self.live_image_label.configure(image=imgtk)
            self.add_image_label.imgtk = imgtk
            self.add_image_label.configure(image=imgtk)

            # Save one frame to ESRGAN input folder every 30 seconds after course selection
            if self.esrgan_enabled:
                now = time.time()
                if now - self.last_esrgan_capture >= self.esrgan_interval:
                    self.last_esrgan_capture = now
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(self.esrgan_input_dir, f"capture_{timestamp}.png")
                    cv2.imwrite(save_path, self.frame)
                    print(f"[ESRGAN] Saved frame to {save_path}")

                    # Automatically run ESRGAN Main.py after saving frame
                    try:
                        esrgan_main_path = os.path.abspath(os.path.join(BASE_DIR, "..", "ESRGAN", "Main.py"))
                        print(f"[ESRGAN] Running ESRGAN script: {esrgan_main_path}")

                        # ✅ Run ESRGAN in a background thread so it doesn’t freeze the UI
                        threading.Thread(target=self.run_esrgan, args=(esrgan_main_path,), daemon=True).start()

                    except Exception as e:
                        print(f"[ESRGAN] Unexpected error: {e}")

        self.root.after(20, self.update_frame)

    def watch_esrgan_output(self):
        print(f"[WATCHER] Monitoring ESRGAN output folder: {self.esrgan_output_dir}")
        while True:
            try:
                # Get latest file in output folder
                files = [f for f in os.listdir(self.esrgan_output_dir) if f.lower().endswith(".png")]
                if not files:
                    time.sleep(2)
                    continue

                latest_file = max(
                    [os.path.join(self.esrgan_output_dir, f) for f in files],
                    key=os.path.getmtime
                )

                # If it's new, process it
                if latest_file != self.last_checked_file:
                    self.last_checked_file = latest_file
                    print(f"[WATCHER] New ESRGAN output detected: {latest_file}")
                    self.recognize_from_image(latest_file)

                time.sleep(2)

            except Exception as e:
                print(f"[WATCHER] Error: {e}")
                time.sleep(3)

    def recognize_from_image(self, image_path):  #Updated Recognizer ESRGAN
        print(f"[RECOGNITION] Starting recognition for {image_path}")
        try:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_img = img[y:y+h, x:x+w]
                temp_path = os.path.join(BASE_DIR, "temp_recognition.jpg")
                cv2.imwrite(temp_path, face_img)

                emb = DeepFace.represent(img_path=temp_path, model_name="Facenet512", enforce_detection=False)[0]["embedding"]

                db = self.db_integration.embeddings_db
                if not db:
                    print("[RECOGNITION] No embeddings database found.")
                    return

                min_dist = float("inf")
                name = "unknown"
                for person, refs in db.items():
                    for ref in refs:
                        dist = cosine(emb, ref)
                        if dist < min_dist:
                            min_dist = dist
                            name = person

                if name != "unknown" and min_dist < 0.18:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    confidence = (1 - min_dist) * 100
                    print(f"[RECOGNIZED] {name} | Confidence: {confidence:.2f}% | Dist: {min_dist:.4f}")
                    self.db_integration.submit_attendance(name, confidence, min_dist, timestamp)
                else:
                    print(f"[RECOGNITION] Unknown face detected in {os.path.basename(image_path)}")

        except Exception as e:
            print(f"[RECOGNITION] Error: {e}")


    def run_esrgan(self, esrgan_main_path):
        process = subprocess.Popen(
            ["python", esrgan_main_path],
            cwd=os.path.dirname(esrgan_main_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Stream ESRGAN's internal prints live
        for line in process.stdout:
            print("[ESRGAN]", line.strip())

        process.wait()
        print("[ESRGAN] Enhancement completed successfully.")

        # ✅ Start the watcher after ESRGAN completes
        print("[WATCHER] Starting ESRGAN output monitoring thread...")
        threading.Thread(target=self.watch_esrgan_output, daemon=True).start()

    def check_esrgan_output(self):
        now = time.time()
        if now - self.last_checked_output >= self.output_check_interval:
            self.last_checked_output = now
            # Get newest file in ESRGAN output folder
            output_files = sorted(
                [f for f in os.listdir(self.esrgan_output_dir) if f.lower().endswith(('.jpg', '.png'))],
                key=lambda x: os.path.getmtime(os.path.join(self.esrgan_output_dir, x)),
                reverse=True
            )

            if output_files:
                latest_file = os.path.join(self.esrgan_output_dir, output_files[0])
                print(f"[ESRGAN] Detected new output file: {latest_file}")
                self.recognize_from_file(latest_file)

        self.root.after(1000, self.check_esrgan_output)  # loop every second


    def recognize_from_file(self, image_path): #Checks Input IN ESRGAN output
        try:
            emb = DeepFace.represent(img_path=image_path, model_name="Facenet512", enforce_detection=False)[0]["embedding"]
            db = self.db_integration.embeddings_db
            if not db:
                print("[Recognition] Database empty.")
                return

            min_dist = float("inf")
            name = "unknown"
            for person, refs in db.items():
                for ref in refs:
                    dist = cosine(emb, ref)
                    if dist < min_dist:
                        min_dist = dist
                        name = person

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if name != "unknown" and min_dist < 0.18:
                confidence = (1 - min_dist) * 100
                print(f"[ESRGAN Recognition] {name} | Confidence: {confidence:.2f}% | Distance: {min_dist:.4f}")
                self.db_integration.submit_attendance(name, confidence, min_dist, timestamp)
            else:
                print("[ESRGAN Recognition] Unknown or low confidence")

        except Exception as e:
            print(f"[Error] {e}")


    def train_faces(self):
        selected = self.selected_course.get()
        if not selected:
            messagebox.showwarning("No Course", "Please select a course to upload the trained model to.")
            return

        course_id = selected.split("ID: ")[1].rstrip(")")
        course = next((c for c in self.db_integration.courses if str(c['id']) == course_id), None)

        if not course:
            messagebox.showerror("Error", "Selected course not found.")
            return

        if course.get('model_pickle'):
            if not messagebox.askyesno("Overwrite Confirmation", f"A model already exists for {course['name']}. Overwrite it?"):
                return

        try:
            model_path = generate_embeddings()

            success = self.db_integration.upload_model_to_course(course_id, model_path)
            if success:
                messagebox.showinfo("Success", "Model uploaded successfully.")
            else:
                messagebox.showerror("Upload Failed", "Model could not be uploaded.")
        except Exception as e:
            messagebox.showerror("Error", f"Training failed:\n{str(e)}")

    def delete_images_from_tab(self):
        db_folder = os.path.join(BASE_DIR, "Deepface DB")
        if not os.path.exists(db_folder):
            messagebox.showerror("Error", "Deepface DB folder not found.")
            return

        delete_window = tk.Toplevel(self.root)
        delete_window.title("Select Images to Delete")

        var_dict = {}

        for student in os.listdir(db_folder):
            student_folder = os.path.join(db_folder, student)
            if not os.path.isdir(student_folder):
                continue

            for img_file in os.listdir(student_folder):
                full_path = os.path.join(student_folder, img_file)
                var = tk.BooleanVar()
                chk = tk.Checkbutton(delete_window, text=f"{student}/{img_file}", variable=var)
                chk.pack(anchor='w')
                var_dict[full_path] = var

        def confirm_delete():
            deleted = 0
            for path, var in var_dict.items():
                if var.get():
                    os.remove(path)
                    deleted += 1
            messagebox.showinfo("Delete Complete", f"{deleted} images deleted.")
            delete_window.destroy()

        ttk.Button(delete_window, text="Delete Selected", command=confirm_delete).pack(pady=10)

    def load_courses(self):
        courses = self.db_integration.fetch_courses_with_models()
        print(f"[DEBUG] Loaded {len(courses)} courses")
        for c in courses:
            print(f"[DEBUG] Course: {c['name']} | ID: {c['id']} | Model: {'Yes' if c['model_pickle'] else 'No'}")
            
        self.db_integration.courses = self.db_integration.fetch_courses_with_models()
        if not self.db_integration.courses:
            messagebox.showerror("Error", "No courses found or failed to fetch.")
            return

        upload_values = [
            f"{c['name']} by {c['teacher_name']} (ID: {c['id']})"
            for c in self.db_integration.courses
        ]
        self.upload_course_combo['values'] = upload_values

        recognition_values = [
            f"{c['name']} by {c['teacher_name']} (ID: {c['id']})"
            for c in self.db_integration.courses
            if c.get('model_pickle')
        ]
        self.course_combo['values'] = recognition_values

    def select_course(self):
        selected = self.course_var.get()
        self.logged_names = set()
        if selected:
            self.esrgan_enabled = True  # start 30s ESRGAN image saving
            self.last_esrgan_capture = time.time()
            course_id = selected.split("ID: ")[1].rstrip(")")
            self.db_integration.selected_course_id = course_id
            course = next((c for c in self.db_integration.courses if str(c['id']) == course_id), None)
            if course and course.get('model_pickle'):
                model = self.db_integration.load_model_from_bytea(course['model_pickle'])
                if model:
                    self.db_integration.embeddings_db = model
                    messagebox.showinfo("Course Selected", f"Model loaded for course ID: {course_id}")
                else:
                    messagebox.showerror("Model Error", "Failed to load model for selected course.")
            else:
                messagebox.showwarning("Missing Model", "No model available for this course.")


class DatabaseIntegration:
    def __init__(self, api_base_url="https://sams-backend-u79d.onrender.com"):
        self.api_base_url = api_base_url
        self.selected_course_id = None
        self.embeddings_db = None
        self.courses = []

    def upload_model_to_server(self, model_bytes):
        if not self.selected_course_id:
            return False
        res = requests.post(
            f"{self.api_base_url}/api/uploadModel.php",
            files={"model": ("model.joblib", model_bytes)},
            data={"course_id": self.selected_course_id}
        )
        return res.status_code == 200 and res.json().get("success", False)

    def fetch_courses_with_models(self):
        res = requests.get(f"{self.api_base_url}/api/getCourseModel.php")
        if res.status_code == 200:
            data = res.json()
            if data.get("success"):
                self.courses = data["courses"]
                return self.courses
        return []

    def upload_model_to_course(self, course_id, model_path):
        try:
            with open(model_path, 'rb') as f:
                file_data = f.read()
                print(f"[DEBUG] Model file size: {len(file_data)} bytes")
                files = {'model': ('model.joblib', io.BytesIO(file_data))}
                data = {'course_id': course_id}
                
                print(f"[DEBUG] Uploading model for course ID: {course_id}")
                response = requests.post(f"{self.api_base_url}/api/uploadModel.php", files=files, data=data)
            
            print(f"[DEBUG] Server responded with {response.status_code}: {response.text}")

            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print(f"Model successfully uploaded to course {course_id}")
                    return True
                else:
                    print(f"Server error: {result.get('error')}")
            else:
                print(f"[!] HTTP error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"[!] Exception during upload: {str(e)}")

        return False

    def load_model_from_bytea(self, bytea_data):
        if isinstance(bytea_data, str):
            first_pass = bytes.fromhex(bytea_data)
            if first_pass.startswith(b'\\x'):
                hex_str = first_pass.decode()[2:]
                model_bytes = bytes.fromhex(hex_str)
            else:
                model_bytes = first_pass
            return joblib.load(io.BytesIO(model_bytes))
        return None

    def submit_attendance(self, student_name, confidence=None, distance=None, timestamp=None):
        if not self.selected_course_id:
            return False

        date_part, time_part = timestamp.split(" ")
        clean_date = re.sub(r'\D', '', date_part)
        clean_time = re.sub(r'\D', '', time_part)
        image_path = os.path.join(BASE_DIR, "Deepface DB Record", f"{student_name}_{clean_date}_{clean_time}_face.jpg")

        if not os.path.exists(image_path):
            return False

        with open(image_path, 'rb') as img_file:
            files = {'picture': (os.path.basename(image_path), img_file, 'image/jpeg')}
            data = {
                "course_id": self.selected_course_id,
                "student_name": student_name,
                "timestamp": timestamp,
                "confidence": confidence,
                "distance": distance
            }
            res = requests.post(f"{self.api_base_url}/api/submitCameraAttendance.php", data=data, files=files)
            return res.status_code == 200 and res.json().get("success", False)

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
