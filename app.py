"""
DRIVER DROWSINESS DETECTION - HYBRID
Mata: EAR (Geometric)
Mulut: CNN (SavedModel)
DEPLOY READY - RENDER.COM
"""

import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import base64
import time
import warnings
import threading
from collections import deque
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

# ==================================================
# KONFIGURASI UNTUK RENDER
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# Set environment untuk mengurangi log TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==================================================
# IMPOR TENSORFLOW (dengan error handling)
# ==================================================
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    print("✅ TensorFlow loaded successfully")
except Exception as e:
    TF_AVAILABLE = False
    print(f"⚠️ TensorFlow error: {e}")

warnings.filterwarnings('ignore')

# ==================================================
# FLASK APP
# ==================================================
app = Flask(__name__, 
            template_folder=TEMPLATE_DIR,
            static_folder=STATIC_DIR)
CORS(app)

# ==================================================
# KONFIGURASI DETEKSI
# ==================================================
EAR_THRESHOLD = 0.29
MAR_THRESHOLD = 0.65
REQUIRED_DURATION = 1.0
WINDOW_SIZE = 5

# ==================================================
# KONFIGURASI CNN - HANYA UNTUK MULUT
# ==================================================
CNN_CONF_THRESHOLD = 0.80

# ==================================================
# LOAD MEDIAPIPE
# ==================================================
print("="*60)
print("LOADING MEDIAPIPE...")
print("="*60)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==================================================
# INDEKS LANDMARK
# ==================================================
LEFT_EYE_EAR_INDICES = [33, 133, 160, 159, 158, 144]
RIGHT_EYE_EAR_INDICES = [362, 263, 387, 386, 385, 380]
MOUTH_MAR_INDICES = [61, 291, 13, 14, 78, 308]

print(f"✅ Mata Kiri (EAR): {LEFT_EYE_EAR_INDICES}")
print(f"✅ Mata Kanan (EAR): {RIGHT_EYE_EAR_INDICES}")
print(f"✅ Mulut (MAR): {MOUTH_MAR_INDICES}")

# ==================================================
# LOAD CNN MODEL - HANYA UNTUK MULUT
# ==================================================
print("="*60)
print("LOADING CNN MODEL (HANYA UNTUK MULUT)...")
print("="*60)

cnn_predict_fn = None
CLASS_NAMES = ['Closed_Eyes', 'No_yawn', 'Open_Eyes', 'Yawn']
CNN_LOADED = False

def load_cnn_model():
    global cnn_predict_fn, CNN_LOADED
    
    saved_model_path = os.path.join(BASE_DIR, "drowsiness_saved_model")
    
    if os.path.exists(saved_model_path):
        try:
            loaded = tf.saved_model.load(saved_model_path)
            print(f"✅ Loaded SavedModel from {saved_model_path}")
            
            if 'serving_default' in loaded.signatures:
                cnn_predict_fn = loaded.signatures['serving_default']
                print("✅ Using 'serving_default' signature")
            else:
                sig_key = list(loaded.signatures.keys())[0]
                cnn_predict_fn = loaded.signatures[sig_key]
                print(f"✅ Using '{sig_key}' signature")
            
            # Test prediction
            dummy = np.random.rand(1, 128, 128, 3).astype(np.float32)
            tf_input = tf.constant(dummy, dtype=tf.float32)
            test_output = cnn_predict_fn(tf_input)
            print(f"✅ CNN Test OK!")
            CNN_LOADED = True
            return True
        except Exception as e:
            print(f"❌ Failed to load SavedModel: {e}")
            CNN_LOADED = False
            return False
    else:
        print(f"⚠️ SavedModel not found at {saved_model_path}")
        CNN_LOADED = False
        return False

# Load CNN jika TensorFlow tersedia
if TF_AVAILABLE:
    CNN_LOADED = load_cnn_model()
else:
    print("⚠️ TensorFlow not available, CNN disabled")

if CNN_LOADED:
    print("✅ CNN Model READY! (HANYA UNTUK MULUT)")
    print(f"   CNN Confidence Threshold: {CNN_CONF_THRESHOLD*100:.0f}%")
else:
    print("⚠️ CNN Model NOT LOADED - Using MAR only")

print("="*60)
print("SERVER SIAP!")
print("="*60)

# ==================================================
# FUNGSI BANTUAN
# ==================================================

def get_landmarks_from_mediapipe(results):
    if not results.multi_face_landmarks:
        return None
    landmarks = []
    for lm in results.multi_face_landmarks[0].landmark:
        landmarks.append([lm.x, lm.y, lm.z])
    return np.array(landmarks)

def eye_aspect_ratio_mediapipe(landmarks, eye_indices, frame_shape):
    h, w = frame_shape[:2]
    points = []
    
    for idx in eye_indices:
        if idx < len(landmarks):
            x = landmarks[idx][0] * w
            y = landmarks[idx][1] * h
            points.append([x, y])
    
    if len(points) < 6:
        return 0.0
    
    pts = np.array(points)
    
    p1 = pts[0]
    p2 = pts[1]
    p3 = pts[2]
    p4 = pts[3]
    p5 = pts[4]
    p6 = pts[5]
    
    vertical1 = np.linalg.norm(p3 - p5)
    vertical2 = np.linalg.norm(p4 - p6)
    horizontal = np.linalg.norm(p1 - p2)
    
    ear = (vertical1 + vertical2) / (2.0 * horizontal + 1e-6)
    return ear

def mouth_aspect_ratio_mediapipe(landmarks, mouth_indices, frame_shape):
    h, w = frame_shape[:2]
    points = []
    
    for idx in mouth_indices:
        if idx < len(landmarks):
            x = landmarks[idx][0] * w
            y = landmarks[idx][1] * h
            points.append([x, y])
    
    if len(points) < 6:
        return 0.0
    
    pts = np.array(points)
    
    p1 = pts[0]
    p2 = pts[1]
    p3 = pts[2]
    p4 = pts[3]
    p5 = pts[4]
    p6 = pts[5]
    
    vertical1 = np.linalg.norm(p3 - p4)
    vertical2 = np.linalg.norm(p5 - p6)
    horizontal = np.linalg.norm(p1 - p2)
    
    mar = (vertical1 + vertical2) / (2.0 * horizontal + 1e-6)
    return mar

def get_mouth_roi(frame, landmarks, mouth_indices):
    h, w = frame.shape[:2]
    points = []
    
    for idx in mouth_indices:
        if idx < len(landmarks):
            x = int(landmarks[idx][0] * w)
            y = int(landmarks[idx][1] * h)
            points.append([x, y])
    
    if len(points) < 6:
        return None
    
    pts = np.array(points)
    x_min = max(0, int(min(pts[:, 0]) - 15))
    x_max = min(w, int(max(pts[:, 0]) + 15))
    y_min = max(0, int(min(pts[:, 1]) - 10))
    y_max = min(h, int(max(pts[:, 1]) + 10))
    
    roi = frame[y_min:y_max, x_min:x_max]
    return roi

def cnn_predict(roi):
    """Fungsi untuk prediksi CNN"""
    global cnn_predict_fn
    if cnn_predict_fn is None or roi is None or roi.size == 0:
        return None, 0.0
    
    try:
        resized = cv2.resize(roi, (128, 128))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb / 255.0
        input_tensor = np.expand_dims(normalized, axis=0).astype(np.float32)
        tf_input = tf.constant(input_tensor, dtype=tf.float32)
        
        result = cnn_predict_fn(tf_input)
        
        if isinstance(result, dict):
            output = list(result.values())[0].numpy()
        else:
            output = result.numpy()
        
        if len(output.shape) > 1:
            idx = np.argmax(output[0])
            confidence = float(output[0][idx])
        else:
            idx = np.argmax(output)
            confidence = float(output[idx])
        
        class_name = CLASS_NAMES[idx]
        return class_name, confidence
        
    except Exception as e:
        print(f"CNN predict error: {e}")
        return None, 0.0

def draw_alert_on_frame(frame, result):
    """Gambar hanya 'Wajah tidak terdeteksi' di frame - TANPA ALERT"""
    display_frame = frame.copy()
    
    # ========== HANYA WAJAH TIDAK TERDETEKSI ==========
    if not result.get('face_detected', False) or result.get('landmarks') is None:
        h, w = display_frame.shape[:2]
        cv2.putText(display_frame, "Wajah tidak terdeteksi", (w//2 - 110, h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
    
    return display_frame

# ==================================================
# DROWSINESS DETECTOR CLASS
# ==================================================

class DrowsinessDetector:
    def __init__(self):
        self.latest_result = {
            'face_detected': False,
            'eye_state': 'Unknown',
            'mouth_state': 'Unknown',
            'ear': 0.0,
            'mar': 0.0,
            'eye_closed_duration': 0.0,
            'yawn_duration': 0.0,
            'status': 'SAFE',
            'message': 'Inisialisasi...',
            'color': '#00ff00',
            'alert': False,
            'alert_type': None,
            'cnn_mouth_pred': None,
            'cnn_mouth_conf': 0.0,
            'landmarks': None,
            'face_box': None,
            'timestamp': 0
        }
        
        self.lock = threading.Lock()
        
        self.eye_closed_start_time = None
        self.yawn_start_time = None
        self.eye_alert_triggered = False
        self.yawn_alert_triggered = False
        self.ear_history = deque(maxlen=WINDOW_SIZE)
        self.mar_history = deque(maxlen=WINDOW_SIZE)
        
        self.frame_queue = deque(maxlen=2)
        self.processing = False
        self.running = True
        
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
    
    def _detection_loop(self):
        while self.running:
            if len(self.frame_queue) == 0:
                time.sleep(0.005)
                continue
            
            frame = self.frame_queue[-1].copy()
            self._process_detection(frame)
            time.sleep(0.005)
    
    def _process_detection(self, frame):
        self.processing = True
        current_time = time.time()
        
        ear = 0.0
        mar = 0.0
        ear_left = 0.0
        ear_right = 0.0
        eye_closed = False
        yawning = False
        eye_duration = 0.0
        yawn_duration = 0.0
        landmarks = None
        face_box = None
        cnn_mouth_pred = None
        cnn_mouth_conf = 0.0
        status = "UNKNOWN"
        message = "Menunggu deteksi..."
        color = "#666666"
        alert = False
        alert_type = None
        
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                mediapipe_landmarks = get_landmarks_from_mediapipe(results)
                if mediapipe_landmarks is not None:
                    h, w = frame.shape[:2]
                    landmarks = mediapipe_landmarks.copy()
                    
                    x_coords = [lm[0] * w for lm in landmarks]
                    y_coords = [lm[1] * h for lm in landmarks]
                    
                    x_min = int(max(0, min(x_coords) - 20))
                    x_max = int(min(w, max(x_coords) + 20))
                    y_min = int(max(0, min(y_coords) - 20))
                    y_max = int(min(h, max(y_coords) + 20))
                    face_box = (x_min, y_min, x_max, y_max)
                    
                    # ========== MATA: EAR ==========
                    ear_left = eye_aspect_ratio_mediapipe(landmarks, LEFT_EYE_EAR_INDICES, frame.shape)
                    ear_right = eye_aspect_ratio_mediapipe(landmarks, RIGHT_EYE_EAR_INDICES, frame.shape)
                    ear = (ear_left + ear_right) / 2.0
                    
                    self.ear_history.append(ear)
                    ear_smooth = np.mean(self.ear_history) if self.ear_history else ear
                    
                    eye_closed = ear_smooth < EAR_THRESHOLD
                    
                    # ========== MULUT: CNN + MAR ==========
                    mar = mouth_aspect_ratio_mediapipe(landmarks, MOUTH_MAR_INDICES, frame.shape)
                    self.mar_history.append(mar)
                    mar_smooth = np.mean(self.mar_history) if self.mar_history else mar
                    
                    yawning = mar_smooth > MAR_THRESHOLD
                    
                    # CNN Prediction untuk mulut (jika tersedia)
                    if CNN_LOADED and landmarks is not None:
                        try:
                            mouth_roi = get_mouth_roi(frame, landmarks, MOUTH_MAR_INDICES)
                            
                            if mouth_roi is not None and mouth_roi.size > 0:
                                pred_mouth, conf_mouth = cnn_predict(mouth_roi)
                                if pred_mouth in ['Yawn', 'No_yawn']:
                                    cnn_mouth_pred = pred_mouth
                                    cnn_mouth_conf = conf_mouth
                            
                            # Gunakan CNN jika confidence tinggi
                            if cnn_mouth_pred == 'Yawn' and cnn_mouth_conf > CNN_CONF_THRESHOLD:
                                yawning = True
                            elif cnn_mouth_pred == 'No_yawn' and cnn_mouth_conf > CNN_CONF_THRESHOLD:
                                yawning = False
                            else:
                                yawning = mar_smooth > MAR_THRESHOLD
                                
                        except Exception as e:
                            print(f"CNN error: {e}")
                    
                    # Debug output (akan muncul di log Render)
                    if np.random.random() < 0.01:  # Hanya 1% frame untuk mengurangi log
                        print(f"🔍 EAR: {ear:.3f} | MAR: {mar:.3f}")
                        print(f"   → Mata: {'TUTUP' if eye_closed else 'BUKA'} | Mulut: {'MENGUAP' if yawning else 'NORMAL'}")
            
            # ========== HITUNG DURASI ==========
            if eye_closed:
                if self.eye_closed_start_time is None:
                    self.eye_closed_start_time = current_time
                eye_duration = current_time - self.eye_closed_start_time
            else:
                self.eye_closed_start_time = None
                self.eye_alert_triggered = False
            
            if yawning:
                if self.yawn_start_time is None:
                    self.yawn_start_time = current_time
                yawn_duration = current_time - self.yawn_start_time
            else:
                self.yawn_start_time = None
                self.yawn_alert_triggered = False
            
            # ========== TENTUKAN STATUS ==========
            if face_box is not None:
                if eye_closed and yawning:
                    both_duration = min(eye_duration, yawn_duration)
                    if both_duration >= REQUIRED_DURATION:
                        status = "VERY_DANGEROUS"
                        message = "💀 SANGAT BERBAHAYA - Mata Tertutup + Menguap"
                        color = "#8B0000"
                        alert = True
                        alert_type = "very_dangerous"
                    else:
                        remaining = max(0, REQUIRED_DURATION - both_duration)
                        status = "COUNTDOWN"
                        message = f"⏰ Keduanya! {remaining:.1f}d"
                        color = "#8B0000"
                elif eye_closed and eye_duration >= REQUIRED_DURATION:
                    status = "DANGER"
                    message = "🔴 BAHAYA - Tutup Mata"
                    color = "#ff0000"
                    alert = True
                    alert_type = "danger"
                elif yawning and yawn_duration >= REQUIRED_DURATION:
                    status = "WARNING"
                    message = "🟡 PERINGATAN - Menguap"
                    color = "#ffa500"
                    alert = True
                    alert_type = "warning"
                elif eye_closed or yawning:
                    if eye_closed:
                        remaining = max(0, REQUIRED_DURATION - eye_duration)
                        status = "COUNTDOWN"
                        message = f"⏰ Mata! {remaining:.1f}d"
                        color = "#ff6600"
                    else:
                        remaining = max(0, REQUIRED_DURATION - yawn_duration)
                        status = "COUNTDOWN"
                        message = f"⏰ Menguap! {remaining:.1f}d"
                        color = "#ff6600"
                else:
                    status = "SAFE"
                    message = "✅ AMAN"
                    color = "#00ff00"
            else:
                status = "UNKNOWN"
                message = "❌ Wajah tidak terdeteksi"
                color = "#666666"
            
            with self.lock:
                self.latest_result = {
                    'face_detected': face_box is not None,
                    'eye_state': "Closed_Eyes" if eye_closed else "Open_Eyes",
                    'mouth_state': "Yawn" if yawning else "No_yawn",
                    'ear': round(ear, 4),
                    'mar': round(mar, 4),
                    'eye_closed_duration': round(eye_duration, 1),
                    'yawn_duration': round(yawn_duration, 1),
                    'status': status,
                    'message': message,
                    'color': color,
                    'alert': alert,
                    'alert_type': alert_type,
                    'cnn_mouth_pred': cnn_mouth_pred,
                    'cnn_mouth_conf': round(cnn_mouth_conf * 100, 1),
                    'landmarks': landmarks,
                    'face_box': face_box,
                    'timestamp': current_time
                }
        
        except Exception as e:
            print(f"Detection error: {e}")
            import traceback
            traceback.print_exc()
        
        self.processing = False
    
    def update_frame(self, frame):
        self.frame_queue.append(frame)
    
    def get_display_frame(self, original_frame, pip_mode=False):
        result = self.get_result()
        
        # ========== HANYA WAJAH TIDAK TERDETEKSI ==========
        display_frame = draw_alert_on_frame(original_frame, result)
        
        return display_frame, result
    
    def get_result(self):
        with self.lock:
            return self.latest_result.copy()
    
    def reset(self):
        with self.lock:
            self.ear_history.clear()
            self.mar_history.clear()
            self.eye_closed_start_time = None
            self.yawn_start_time = None
            self.eye_alert_triggered = False
            self.yawn_alert_triggered = False
    
    def stop(self):
        self.running = False

# ==================================================
# INISIALISASI DETECTOR
# ==================================================
detector_instance = DrowsinessDetector()

# ==================================================
# ROUTES
# ==================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files seperti CSS, JS, dan audio"""
    try:
        return send_from_directory('static', filename)
    except Exception as e:
        print(f"Error serving static file {filename}: {e}")
        return jsonify({'error': 'File not found'}), 404

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        facing_mode = data.get('facing_mode', 'user')
        pip_mode = data.get('pip_mode', False)
        
        if facing_mode == 'user':
            frame = cv2.flip(frame, 1)
        
        detector_instance.update_frame(frame)
        display_frame, result = detector_instance.get_display_frame(frame, pip_mode=pip_mode)
        
        _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        processed_image = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'processed_image': processed_image,
            'eye_state': result['eye_state'],
            'mouth_state': result['mouth_state'],
            'status': result['status'],
            'message': result['message'],
            'ear': result['ear'],
            'mar': result['mar'],
            'eye_closed_duration': result['eye_closed_duration'],
            'yawn_duration': result['yawn_duration'],
            'color': result['color'],
            'alert': result['alert'],
            'alert_type': result['alert_type'],
            'cnn_mouth_pred': result['cnn_mouth_pred'],
            'cnn_mouth_conf': result['cnn_mouth_conf']
        })
        
    except Exception as e:
        print(f"❌ ERROR in process_frame: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/process_frame_video', methods=['POST'])
def process_frame_video():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        pip_mode = data.get('pip_mode', False)
        
        detector_instance.update_frame(frame)
        display_frame, result = detector_instance.get_display_frame(frame, pip_mode=pip_mode)
        
        _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        processed_image = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'processed_image': processed_image,
            'eye_state': result['eye_state'],
            'mouth_state': result['mouth_state'],
            'status': result['status'],
            'message': result['message'],
            'ear': result['ear'],
            'mar': result['mar'],
            'eye_closed_duration': result['eye_closed_duration'],
            'yawn_duration': result['yawn_duration'],
            'color': result['color'],
            'alert': result['alert'],
            'alert_type': result['alert_type'],
            'cnn_mouth_pred': result['cnn_mouth_pred'],
            'cnn_mouth_conf': result['cnn_mouth_conf']
        })
        
    except Exception as e:
        print(f"Error in video processing: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    detector_instance.reset()
    return jsonify({'success': True})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint untuk Render"""
    return jsonify({
        'status': 'ok',
        'mode': 'Hybrid (EAR untuk Mata, CNN untuk Mulut)',
        'cnn_loaded': CNN_LOADED,
        'cnn_conf_threshold': CNN_CONF_THRESHOLD,
        'tensorflow_available': TF_AVAILABLE,
        'config': {
            'EAR_THRESHOLD': EAR_THRESHOLD,
            'MAR_THRESHOLD': MAR_THRESHOLD,
            'REQUIRED_DURATION': REQUIRED_DURATION,
            'WINDOW_SIZE': WINDOW_SIZE
        }
    })

@app.route('/pip_status', methods=['POST'])
def pip_status():
    try:
        data = request.json
        print(f"📱 PiP Status: {data}")
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================================================
# ERROR HANDLERS
# ==================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ==================================================
# MAIN
# ==================================================

if __name__ == '__main__':
    import socket
    
    # Cek apakah di environment Railway atau Render
    is_railway = os.environ.get('RAILWAY_ENVIRONMENT', '').lower() == 'production'
    is_render = os.environ.get('RENDER', 'false').lower() == 'true'
    
    if is_railway:
        # Di Railway
        port = int(os.environ.get('PORT', 5000))
        print("\n" + "="*60)
        print("🚀 RUNNING ON RAILWAY")
        print("="*60)
        print(f"📊 KONFIGURASI:")
        print(f"   EAR_THRESHOLD: {EAR_THRESHOLD}")
        print(f"   MAR_THRESHOLD: {MAR_THRESHOLD}")
        print(f"   REQUIRED_DURATION: {REQUIRED_DURATION}s")
        print(f"   WINDOW_SIZE: {WINDOW_SIZE}")
        print(f"   TensorFlow: {'✅' if TF_AVAILABLE else '❌'}")
        print(f"   CNN Loaded: {'✅' if CNN_LOADED else '❌'}")
        print(f"   CNN Confidence Threshold: {CNN_CONF_THRESHOLD*100:.0f}%")
        print("="*60)
        print(f"✅ Server running on port {port}")
        
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
        
    elif is_render:
        # Di Render
        print("\n" + "="*60)
        print("🚀 RUNNING ON RENDER")
        print("="*60)
        print(f"📊 KONFIGURASI:")
        print(f"   EAR_THRESHOLD: {EAR_THRESHOLD}")
        print(f"   MAR_THRESHOLD: {MAR_THRESHOLD}")
        print(f"   REQUIRED_DURATION: {REQUIRED_DURATION}s")
        print(f"   WINDOW_SIZE: {WINDOW_SIZE}")
        print(f"   TensorFlow: {'✅' if TF_AVAILABLE else '❌'}")
        print(f"   CNN Loaded: {'✅' if CNN_LOADED else '❌'}")
        print("="*60)
        print("✅ Server running on Render")
    else:
        # Local development
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        port = int(os.environ.get('PORT', 5000))
        
        print("\n" + "="*60)
        print("🌐 AKSES URL:")
        print("="*60)
        print(f"📍 Local access:    http://localhost:{port}")
        print(f"📍 Local network:   http://{local_ip}:{port}")
        print("="*60)
        print(f"\n📊 KONFIGURASI:")
        print(f"   EAR_THRESHOLD: {EAR_THRESHOLD}")
        print(f"   MAR_THRESHOLD: {MAR_THRESHOLD}")
        print(f"   REQUIRED_DURATION: {REQUIRED_DURATION}s")
        print(f"   WINDOW_SIZE: {WINDOW_SIZE}")
        print(f"   TensorFlow: {'✅' if TF_AVAILABLE else '❌'}")
        print(f"   CNN Loaded: {'✅' if CNN_LOADED else '❌'}")
        print(f"   CNN Confidence Threshold: {CNN_CONF_THRESHOLD*100:.0f}%")
        print("="*60)
        
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)