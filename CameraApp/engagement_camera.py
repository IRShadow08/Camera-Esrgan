import cv2
import dlib
import numpy as np
import requests
import json
from datetime import datetime
import threading
import time
import os

class EngagementCameraApp:
    def __init__(self):
        # Initialize dlib face detector and predictor (more reliable than OpenFace package)
        self.face_detector = dlib.get_frontal_face_detector()
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "models", "shape_predictor_68_face_landmarks.dat")
        
        self.face_predictor = dlib.shape_predictor(model_path)
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.running = False
        
        # Engagement tracking
        self.current_course = None
        self.engagement_data = []
        
        # Backend URL
        self.backend_url = "https://sams-backend-u79d.onrender.com/api/submitEngagementData.php"
        
    def detect_emotions(self, frame, landmarks):
        """Detect emotions using facial landmarks"""
        # This is a simplified emotion detection
        # You would typically use a trained model here
        
        # Calculate facial features ratios
        mouth_width = np.linalg.norm(landmarks[54] - landmarks[48])
        mouth_height = np.linalg.norm(landmarks[57] - landmarks[51])
        eye_aspect_ratio = self.calculate_ear(landmarks)
        
        emotions = {
            'happy': 0,
            'sad': 0,
            'surprised': 0,
            'focused': 0,
            'distracted': 0
        }
        
        # Simple rules-based emotion detection
        if mouth_width / mouth_height > 3:
            emotions['happy'] = 0.8
        elif eye_aspect_ratio < 0.2:
            emotions['distracted'] = 0.7
        else:
            emotions['focused'] = 0.6
            
        return emotions
    
    def calculate_ear(self, landmarks):
        """Calculate Eye Aspect Ratio for attention detection"""
        # Left eye landmarks
        left_eye = landmarks[36:42]
        # Right eye landmarks  
        right_eye = landmarks[42:48]
        
        # Calculate EAR for both eyes
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        
        return (left_ear + right_ear) / 2.0
    
    def eye_aspect_ratio(self, eye):
        """Calculate aspect ratio for single eye"""
        # Vertical distances
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        # Horizontal distance
        C = np.linalg.norm(eye[0] - eye[3])
        
        return (A + B) / (2.0 * C)
    
    def get_head_pose(self, landmarks):
        """Calculate head pose angles"""
        # Simplified head pose calculation
        nose_tip = landmarks[30]
        chin = landmarks[8]
        
        # Calculate approximate angles
        pitch = np.arctan2(nose_tip[1] - chin[1], nose_tip[0] - chin[0])
        
        return {
            'pitch': float(pitch),
            'yaw': 0.0,  # Would need more complex calculation
            'roll': 0.0
        }
    
    def calculate_engagement_score(self, emotions, attention_score, head_pose):
        """Calculate overall engagement score"""
        # Weighted combination of factors
        emotion_factor = emotions.get('focused', 0) + emotions.get('happy', 0) * 0.5
        attention_factor = attention_score
        pose_factor = 1.0 - abs(head_pose['pitch']) / 3.14  # Penalize extreme head positions
        
        engagement = (emotion_factor * 0.4 + attention_factor * 0.4 + pose_factor * 0.2)
        return min(max(engagement, 0), 1)  # Clamp between 0 and 1
    
    def process_frame(self, frame, student_name=None):
        """Process single frame for engagement detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        
        engagement_results = []
        
        for face in faces:
            landmarks = self.face_predictor(gray, face)
            if landmarks is not None:
                # Convert landmarks to numpy array
                landmarks_np = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
                
                # Detect emotions and engagement
                emotions = self.detect_emotions(frame, landmarks_np)
                attention_score = self.calculate_ear(landmarks_np)
                head_pose = self.get_head_pose(landmarks_np)
                engagement_score = self.calculate_engagement_score(emotions, attention_score, head_pose)
                
                # Draw visualization on frame
                self.draw_engagement_overlay(frame, face, emotions, engagement_score, attention_score)
                
                # Store result
                result = {
                    'student_name': student_name or 'Unknown',
                    'timestamp': datetime.now().isoformat(),
                    'emotions': emotions,
                    'attention_score': float(attention_score),
                    'head_pose': head_pose,
                    'engagement_level': float(engagement_score),
                    'course_name': self.current_course
                }
                
                engagement_results.append(result)
        
        return frame, engagement_results
    
    def draw_engagement_overlay(self, frame, face, emotions, engagement_score, attention_score):
        """Draw engagement information on frame"""
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        
        # Draw face rectangle
        color = (0, 255, 0) if engagement_score > 0.7 else (0, 165, 255) if engagement_score > 0.4 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw engagement info
        info_text = [
            f"Engagement: {engagement_score:.2f}",
            f"Attention: {attention_score:.2f}",
            f"Emotion: {max(emotions, key=emotions.get)}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (x, y - 40 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def start_engagement_monitoring(self, course_name):
        """Start the engagement monitoring session"""
        self.current_course = course_name
        self.running = True
        
        print(f"Starting engagement monitoring for course: {course_name}")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process frame for engagement
            processed_frame, engagement_data = self.process_frame(frame)
            
            # Send engagement data to backend (every 5 seconds to avoid spam)
            current_time = time.time()
            if hasattr(self, 'last_upload') and current_time - self.last_upload < 5:
                pass  # Skip upload
            else:
                if engagement_data:
                    self.upload_engagement_data(engagement_data[0])  # Upload first detected face
                    self.last_upload = current_time
            
            # Display frame
            cv2.imshow('Engagement Monitoring', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def upload_engagement_data(self, data):
        """Upload engagement data to backend"""
        try:
            # For testing purposes - you'll need to replace these with actual values
            headers = {
                'Content-Type': 'application/json',
                'Provider': 'google',
                'Token': 'ya29.A0AS3H6Nx-p3PHwhOxM8CyZM1WjnUYTh-wXv64jBz0dgyh04qFYgeo5t947O91N67eTOnm1UhPOvDrIRRv9LhpgrpIiX2w5xfIdgyGrL3gKEtKNZIrOEZssS22c7E2urjIvZW73EagU2umNg7L5i1v8zIxgnOLooHLIKLkBgkZuL-sXOBd9N1NUbIxbh72DbyqMn-bT361fgaCgYKAZoSARUSFQHGX2MiWoyysMlOGDhU-IDoUjSF5Q0209'  # Get from your browser session
            }
            
            response = requests.post(self.backend_url, json=data, headers=headers)
            if response.status_code == 200:
                print(f"Engagement data uploaded: {data['engagement_level']:.2f}")
            else:
                print(f"Failed to upload engagement data: {response.status_code}")
        except Exception as e:
            print(f"Error uploading engagement data: {e}")
    
    def stop_monitoring(self):
        """Stop the engagement monitoring"""
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    app = EngagementCameraApp()
    
    try:
        app.start_engagement_monitoring("Computer Science 101")
    except KeyboardInterrupt:
        print("Stopping engagement monitoring...")
    finally:
        app.stop_monitoring()
