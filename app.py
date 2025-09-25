import os
import cv2
import numpy as np
import sqlite3
import datetime
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN
from keras.models import Sequential
from keras.layers import Input, Lambda
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
import atexit
import pytz

# --- Configuration & Initialization ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = Flask(__name__)
CORS(app)

# Use Indian Standard Time
IST = pytz.timezone('Asia/Kolkata')

# --- Core Face Attendance System Class ---
class FaceAttendanceSystem:
    """
    Manages face detection, recognition, database interactions, and attendance logic.
    """
    def __init__(self):
        print("Initializing system...")
        try:
            self.face_detector = MTCNN()
            # This model is pre-trained for creating face embeddings
            self.face_recognizer = Sequential([
                Input(shape=(160, 160, 3)),
                Lambda(preprocess_input),
                InceptionResNetV2(include_top=False, weights='imagenet', pooling='avg')
            ])
            print("TensorFlow and MTCNN models loaded successfully.")
        except Exception as e:
            print(f"FATAL: Could not load ML models. Error: {e}")
            exit()
            
        self.input_size = (160, 160)
        # Path for the persistent volume on Railway
        self.db_path = '/data/attendance_system.db'
        self.conn = self._create_db_connection()
        self._create_db_tables()
        
        self.known_face_embeddings = []
        self.known_face_ids = []
        self.known_face_names = []
        self.load_known_faces()
        
        # Using camera index 0 (default webcam)
        self.video_capture = cv2.VideoCapture(0)
        self.is_recognition_active = False
        # Cooldown to prevent spamming the database for the same person
        self.last_attendance_time = {} 
        
        # Cache for today's attendance status to optimize lookups
        self.attendance_cache = {}
        self.last_cache_update = None

    def _create_db_connection(self):
        try:
            # FIX: Ensure the directory for the database file exists before connecting.
            # This is crucial for environments like Railway where volumes are mounted.
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            # `check_same_thread=False` is needed for Flask's multi-threaded environment
            return sqlite3.connect(self.db_path, check_same_thread=False)
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            return None

    def _create_db_tables(self):
        """Creates the necessary SQLite tables if they don't exist."""
        try:
            cursor = self.conn.cursor()
            # Students table stores permanent records
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS students (
                    id TEXT PRIMARY KEY, 
                    name TEXT NOT NULL, 
                    photo_path TEXT NOT NULL
                )
            ''')
            # Attendance table stores daily records
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    id TEXT, 
                    date TEXT, 
                    time TEXT, 
                    status TEXT,
                    FOREIGN KEY(id) REFERENCES students(id) ON DELETE CASCADE,
                    UNIQUE(id, date)
                )
            ''')
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Database table creation error: {e}")

    def get_face_embedding(self, face_image):
        """Converts a face image into a 1D vector embedding."""
        try:
            face_pixels = cv2.resize(face_image, self.input_size)
            face_pixels = np.expand_dims(face_pixels, axis=0)
            # The model predicts the embedding
            return self.face_recognizer.predict(face_pixels, verbose=0)[0]
        except Exception as e:
            print(f"Could not generate embedding: {e}")
            return None

    def load_known_faces(self):
        """Loads all student face embeddings from photos into memory for fast comparison."""
        print("Loading known faces from database...")
        cursor = self.conn.cursor() 
        cursor.execute("SELECT id, name, photo_path FROM students")
        records = cursor.fetchall()
        
        # Clear existing lists before reloading
        self.known_face_embeddings, self.known_face_ids, self.known_face_names = [], [], []
        
        for student_id, name, photo_path in records:
            if photo_path and os.path.exists(photo_path):
                try:
                    img = cv2.imread(photo_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = self.face_detector.detect_faces(img_rgb)
                    if results:
                        x, y, w, h = results[0]['box']
                        face_img = img_rgb[max(0, y):y+h, max(0, x):x+w]
                        embedding = self.get_face_embedding(face_img)
                        if embedding is not None:
                            self.known_face_embeddings.append(embedding)
                            self.known_face_ids.append(student_id)
                            self.known_face_names.append(name)
                except Exception as e:
                    print(f"Error processing photo for {name} ({student_id}): {e}")
        print(f"Finished loading. {len(self.known_face_ids)} faces loaded.")

    def update_attendance_cache(self):
        """Updates the in-memory cache with attendance statuses for today."""
        current_date = datetime.datetime.now(IST).strftime("%Y-%m-%d")
        
        if self.last_cache_update != current_date or not self.attendance_cache:
            self.attendance_cache = {}
            try:
                cursor = self.conn.cursor()
                # Join students and attendance to get status for everyone for today's date
                cursor.execute("""
                    SELECT s.id, COALESCE(a.status, 'Absent') as status 
                    FROM students s 
                    LEFT JOIN attendance a ON s.id = a.id AND a.date = ?
                """, (current_date,))
                
                for student_id, status in cursor.fetchall():
                    self.attendance_cache[student_id] = status
                
                self.last_cache_update = current_date
                print("Attendance cache updated for date:", current_date)
            except Exception as e:
                print(f"Error updating attendance cache: {e}")

    def process_live_frame(self, frame):
        """Detects and recognizes faces in a single video frame and updates attendance."""
        if not self.is_recognition_active or frame is None:
            return frame
        
        self.update_attendance_cache()
        
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detector.detect_faces(frame_rgb)

            for result in results:
                x, y, w, h = result['box']
                face_img = frame_rgb[max(0, y):y+h, max(0, x):x+w]
                embedding = self.get_face_embedding(face_img)
                
                if embedding is not None and self.known_face_embeddings:
                    similarities = cosine_similarity([embedding], self.known_face_embeddings)[0]
                    match_index = np.argmax(similarities)
                    
                    name_to_display = "Unknown"
                    color = (0, 0, 255)  # Red for Unknown
                    status_indicator = ""

                    if similarities[match_index] > 0.75:
                        student_id = self.known_face_ids[match_index]
                        student_name = self.known_face_names[match_index]
                        name_to_display = student_name
                        
                        current_time = datetime.datetime.now(IST)
                        cooldown_seconds = 2
                        
                        last_seen = self.last_attendance_time.get(student_id, datetime.datetime.min.replace(tzinfo=IST))
                        
                        if (current_time - last_seen).total_seconds() > cooldown_seconds:
                            # Live recognition always uses the current date
                            self.record_attendance(student_id, status="Present")
                            self.last_attendance_time[student_id] = current_time
                        
                        current_status = self.attendance_cache.get(student_id, "Absent")
                        if current_status == "Present":
                            color = (0, 255, 0)  # Green
                            status_indicator = " (Present)"
                        else:
                            color = (0, 165, 255) # Orange
                            status_indicator = " (Absent)"
                    
                    display_text = f"{name_to_display}{status_indicator}"
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        except Exception as e:
            print(f"Error in process_live_frame: {e}")
        return frame

    def record_attendance(self, student_id, status="Present", date_str=None):
        """Saves or updates an attendance record for a specific date."""
        now = datetime.datetime.now(IST)
        
        # If no date is provided, use today's date.
        if date_str is None:
            date_str = now.strftime("%Y-%m-%d")

        # Time is only recorded if the status is 'Present'.
        time_str = now.strftime("%H:%M:%S") if status == "Present" else None
        
        try:
            cursor = self.conn.cursor()
            # Use the provided or determined date_str for the operation
            cursor.execute("DELETE FROM attendance WHERE id=? AND date=?", (student_id, date_str))
            cursor.execute(
                "INSERT INTO attendance (id, date, time, status) VALUES (?, ?, ?, ?)", 
                (student_id, date_str, time_str, status)
            )
            self.conn.commit()
            
            # Update cache only if the operation was for today's date
            if date_str == now.strftime("%Y-%m-%d"):
                self.attendance_cache[student_id] = status
            
            print(f"ATTENDANCE: Marked {student_id} as {status} on {date_str} at {time_str or 'N/A'} IST.")
        except sqlite3.Error as e:
            print(f"Database error on recording attendance: {e}")

    def shutdown(self):
        """Gracefully releases resources on application exit."""
        if self.video_capture.isOpened():
            self.video_capture.release()
        if self.conn:
            self.conn.close()
        print("System shut down gracefully.")

# --- Initialize Singleton System ---
face_system = FaceAttendanceSystem()

# --- Flask Web Routes ---
@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    """Generator function to stream video frames to the browser."""
    while True:
        success, frame = face_system.video_capture.read()
        if not success:
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Camera Error", (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_frame)
            frame_bytes = buffer.tobytes()
        else:
            processed_frame = face_system.process_live_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/recognition/start', methods=['POST'])
def start_recognition():
    face_system.is_recognition_active = True
    return jsonify({"success": True, "message": "Recognition started."})

@app.route('/api/recognition/stop', methods=['POST'])
def stop_recognition():
    face_system.is_recognition_active = False
    return jsonify({"success": True, "message": "Recognition stopped."})

@app.route('/api/students', methods=['GET', 'POST'])
def handle_students():
    if request.method == 'GET':
        cursor = face_system.conn.cursor()
        cursor.execute("SELECT id, name FROM students ORDER BY name")
        students = [{"id": r[0], "name": r[1]} for r in cursor.fetchall()]
        return jsonify(students)
    
    if request.method == 'POST':
        student_id = request.form.get('id')
        name = request.form.get('name')
        photo = request.files.get('photo')
        
        if not all([student_id, name, photo]):
            return jsonify({"success": False, "message": "Student ID, Name, and Photo are required."}), 400
        
        uploads_dir = os.path.join('static', 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        photo_path = os.path.join(uploads_dir, f"{student_id}.jpg")
        photo.save(photo_path)
        
        try:
            db_path = photo_path.replace("\\", "/")
            cursor = face_system.conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO students (id, name, photo_path) VALUES (?, ?, ?)", (student_id, name, db_path))
            face_system.conn.commit()
            face_system.load_known_faces() 
            return jsonify({"success": True, "message": "Student registered successfully."})
        except sqlite3.Error as e:
            return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/students/<student_id>', methods=['DELETE'])
def delete_student(student_id):
    try:
        cursor = face_system.conn.cursor()
        cursor.execute("SELECT photo_path FROM students WHERE id=?", (student_id,))
        result = cursor.fetchone()
        
        cursor.execute("DELETE FROM students WHERE id=?", (student_id,))
        face_system.conn.commit()
        
        if result and os.path.exists(result[0]):
            os.remove(result[0])
            
        face_system.load_known_faces() 
        return jsonify({"success": True, "message": "Student deleted."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    date_str = request.args.get('date', default=datetime.datetime.now(IST).strftime("%Y-%m-%d"))
    try:
        cursor = face_system.conn.cursor()
        query = """
            SELECT s.id, s.name, a.time,
                   COALESCE(a.status, 'Absent') as status
            FROM students s
            LEFT JOIN attendance a ON s.id = a.id AND a.date = ?
            ORDER BY s.name
        """
        cursor.execute(query, (date_str,))
        records = [{"id": r[0], "name": r[1], "time": r[2] or 'N/A', "status": r[3]} for r in cursor.fetchall()]
        return jsonify(records)
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/attendance/manual', methods=['POST'])
def manual_attendance():
    """API endpoint to manually mark a student present or absent."""
    data = request.get_json()
    student_id, date, status = data.get('student_id'), data.get('date'), data.get('status')

    if not all([student_id, date, status]):
        return jsonify({"success": False, "message": "Missing required fields."}), 400
    
    try:
        # Pass the specific date from the request to the recording function
        face_system.record_attendance(student_id, status, date_str=date)

        # Clear the cooldown timer for this student
        if student_id in face_system.last_attendance_time:
            del face_system.last_attendance_time[student_id]
        
        print(f"Manual attendance updated: {student_id} marked as {status} on {date}")
        return jsonify({"success": True, "message": f"Attendance for {student_id} updated."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# --- Main Execution ---
if __name__ == '__main__':
    atexit.register(face_system.shutdown)
    print("\n--- System is ready. Starting web server... ---")
    print("--- Access at: http://127.0.0.1:5000 ---")
    app.run(host='0.0.0.0', port=5000, debug=False)
