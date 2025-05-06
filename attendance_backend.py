# AI-Based Automated Attendance System - Backend Script

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import cv2
import torch
import numpy as np
import os
import sys
import glob
import pandas as pd
# Ensure facenet_pytorch and mtcnn are installed
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
except ImportError:
    print("ERROR: facenet-pytorch or mtcnn-pytorch not found. Please install them (`pip install facenet-pytorch mtcnn-pytorch`)")
    sys.exit(1) # Exit if core dependencies are missing
from PIL import Image
try:
    import pytesseract
except ImportError:
    print("Warning: pytesseract not found. ID Card processing will be disabled.")
    pytesseract = None # Set to None if not available
from sklearn.cluster import KMeans
from tqdm import tqdm # Use standard tqdm for scripts
import re
import time
import json
from torchvision import transforms
# import matplotlib.pyplot as plt # Not needed for backend script display
import traceback # For detailed error printing
import argparse # For command-line arguments
import sys # To access command-line arguments if needed directly

print("Backend Script: Imports loaded.")

# -----------------------------------------------------------------------------
# Configuration (Defaults - Can be overridden by command-line args)
# -----------------------------------------------------------------------------
# Default paths (adjust if necessary or rely on command-line args)
FACE_VIDEO_PATH = r"D:\sp1\dataset\student_face_videos"
UNIFORM_VIDEO_PATH = r"D:\sp1\dataset\student_360_view_video"
IDCARD_VIDEO_PATH = r"D:\sp1\dataset\id_card_videos"
FRAMES_PATH = r"D:\sp1\frames"
MODELS_PATH = r"D:\sp1\models"
ATTENDANCE_PATH = r"D:\sp1\attendance"

# Default processing parameters
FRAME_SKIP_RATE = 5
FACE_CONFIDENCE_THRESHOLD = 0.90
FACE_REC_THRESHOLD = 0.9
UNIFORM_COLOR_TOLERANCE = (15, 80, 80)
MIN_UNIFORM_AREA_RATIO = 0.1
PROCESS_ID_CARDS = False # Default, overridden by --process_ids arg

# --- Derived Paths ---
FACE_EMBEDDINGS_FILE = os.path.join(MODELS_PATH, 'face_embeddings.pt')
STUDENT_DATA_FILE = os.path.join(MODELS_PATH, 'student_data.json')

# --- Tesseract OCR Configuration (Optional) ---
# Optional: Set Tesseract path if not in system PATH and PROCESS_ID_CARDS is True
# TESSERACT_CMD_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Example
TESSERACT_CMD_PATH = None # Set to None by default
if PROCESS_ID_CARDS and TESSERACT_CMD_PATH and pytesseract:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH
    try:
        pytesseract.get_tesseract_version()
        print(f"Backend Script: Tesseract path set to: {TESSERACT_CMD_PATH}")
    except Exception as tess_ex:
        print(f"Backend Script Warning: Tesseract path set but failed test: {tess_ex}")
elif PROCESS_ID_CARDS and not pytesseract:
     print("Backend Script Warning: PROCESS_ID_CARDS is True, but pytesseract library is not installed.")
     PROCESS_ID_CARDS = False # Disable if library missing
elif PROCESS_ID_CARDS:
     # Check if tesseract is findable without explicit path
     try:
          pytesseract.get_tesseract_version()
          print("Backend Script: Tesseract found in system PATH.")
     except Exception:
          print("Backend Script Warning: PROCESS_ID_CARDS is True, but Tesseract executable not found in PATH and TESSERACT_CMD_PATH not set.")
          # PROCESS_ID_CARDS = False # Optionally disable if not found

print("Backend Script: Default configuration loaded.")

# -----------------------------------------------------------------------------
# Model Initialization (Global Scope)
# -----------------------------------------------------------------------------
print("Backend Script: Initializing models...")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Backend Script: Running on device: {device}')

mtcnn = None
facenet_resnet = None
try:
    # It's often better to initialize MTCNN with select_largest=False if you expect multiple faces
    # and handle the selection later based on confidence or size if needed.
    # keep_all=True ensures all detected faces are returned.
    mtcnn = MTCNN(
        keep_all=True, min_face_size=40, thresholds=[0.6, 0.7, 0.7],
        factor=0.709, post_process=True, device=device, select_largest=False
    )
    facenet_resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
    print("Backend Script: Models initialized successfully.")
except Exception as model_init_error:
    print(f"Backend Script ERROR initializing models: {model_init_error}")
    print("Please ensure facenet-pytorch and mtcnn-pytorch are installed correctly.")
    # Exit if models are essential and failed to load
    sys.exit(1)
print("-" * 20)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def parse_video_filename(filepath):
    """Extracts enrollment/name/ID from 'enrollment_student.mp4'."""
    basename = os.path.basename(filepath)
    name_part = os.path.splitext(basename)[0]
    parts = name_part.split('_', 1)
    if len(parts) == 2: return parts[0], parts[1], name_part
    else: print(f"Warning: Could not parse filename '{basename}'. Skipping."); return None, None, None

def extract_frames(video_path, output_base_folder, student_id, file_type, skip_rate=5):
    """Extracts frames, skips if already done."""
    if not os.path.exists(video_path): return
    specific_output_folder = os.path.join(output_base_folder, student_id, file_type.lower())
    if os.path.exists(specific_output_folder) and len(os.listdir(specific_output_folder)) > 0: return
    os.makedirs(specific_output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"Error opening video: {video_path}"); return
    frame_count, saved_count = 0, 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Use standard tqdm for scripts, output to stdout
    pbar = tqdm(total=total_frames, desc=f"Extracting {student_id} [{file_type}]", leave=False, file=sys.stdout, ascii=True, unit='frame')
    while True:
        ret, frame = cap.read()
        if not ret: break
        current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        update_val = max(0, current_frame_pos - frame_count)
        pbar.update(update_val)
        frame_count = current_frame_pos
        if frame_count >= 1 and frame_count % skip_rate == 0:
            frame_filename = os.path.join(specific_output_folder, f"frame_{saved_count:05d}.png")
            try: cv2.imwrite(frame_filename, frame); saved_count += 1
            except Exception as e: print(f"Error writing frame {frame_filename}: {e}")
    pbar.close()
    cap.release()

def get_dominant_color(image, k=1):
    """Finds dominant color using K-Means."""
    try:
        if image is None or len(image.shape) < 3 or image.shape[2] != 3: return None
        pixel_count = image.shape[0] * image.shape[1]; actual_k = min(k, pixel_count)
        if actual_k < 1: return None
        if pixel_count < k: return np.uint8(np.average(np.average(image, axis=0), axis=0))
        pixels = image.reshape(-1, 3); pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, actual_k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers); _, counts = np.unique(labels, return_counts=True)
        return centers[np.argmax(counts)]
    except cv2.error: # Fallback
        if image is not None and image.size > 0 : return np.uint8(np.average(np.average(image, axis=0), axis=0))
        else: return None
    except Exception as e: print(f"Error finding dominant color: {e}"); return None

def analyze_uniform_color(student_id):
    """Analyzes 360 view frames for dominant uniform color."""
    uniform_frames_path = os.path.join(FRAMES_PATH, student_id, '360')
    if not os.path.exists(uniform_frames_path): return None, None
    print(f"Analyzing uniform color for {student_id}...")
    dominant_colors_hsv = []
    frame_files = sorted(glob.glob(os.path.join(uniform_frames_path, '*.png')))
    if not frame_files: return None, None
    # Use standard tqdm
    for frame_file in tqdm(frame_files, desc="Analyzing Uniform Frames", leave=False, file=sys.stdout, ascii=True):
        frame = cv2.imread(frame_file)
        if frame is None: continue
        h, w, channels = frame.shape;
        if channels != 3: continue
        roi_h_start, roi_h_end = int(h * 0.25), int(h * 0.75)
        roi_w_start, roi_w_end = int(w * 0.25), int(w * 0.75)
        roi = frame[roi_h_start:roi_h_end, roi_w_start:roi_w_end]
        if roi.size == 0: continue
        dom_color_bgr = get_dominant_color(roi, k=1)
        if dom_color_bgr is not None:
            dom_color_hsv = cv2.cvtColor(np.uint8([[dom_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            dominant_colors_hsv.append(dom_color_hsv)
    if not dominant_colors_hsv: print(f"Warning: Could not determine dominant color for {student_id}"); return None, None
    dominant_colors_hsv = np.array(dominant_colors_hsv)
    median_hsv = np.median(dominant_colors_hsv, axis=0).astype(int)
    print(f"Determined median uniform HSV for {student_id}: {median_hsv}")
    lower_h = max(0, median_hsv[0] - UNIFORM_COLOR_TOLERANCE[0]); upper_h = min(179, median_hsv[0] + UNIFORM_COLOR_TOLERANCE[0])
    lower_s = max(0, median_hsv[1] - UNIFORM_COLOR_TOLERANCE[1]); upper_s = min(255, median_hsv[1] + UNIFORM_COLOR_TOLERANCE[1])
    lower_v = max(0, median_hsv[2] - UNIFORM_COLOR_TOLERANCE[2]); upper_v = min(255, median_hsv[2] + UNIFORM_COLOR_TOLERANCE[2])
    lower_bound = [int(lower_h), int(lower_s), int(lower_v)]; upper_bound = [int(upper_h), int(upper_s), int(upper_v)]
    if lower_h > upper_h: print(f"Warning: Hue range wraps around for {student_id}.")
    print(f"Uniform color range (HSV): Lower={lower_bound}, Upper={upper_bound}")
    return lower_bound, upper_bound

def preprocess_for_ocr(image):
    """Applies preprocessing steps for OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

def process_id_card(student_id):
    """Processes ID card frames using OCR."""
    if not pytesseract: return None, None, False # Skip if library missing
    id_frames_path = os.path.join(FRAMES_PATH, student_id, 'idcard')
    if not os.path.exists(id_frames_path): return None, None, False
    print(f"Processing ID card for {student_id}...")
    enrollment_found, name_found = None, None; ocr_successful = False
    frame_files = sorted(glob.glob(os.path.join(id_frames_path, '*.png')))
    if not frame_files: return None, None, False
    try: expected_enrollment, expected_name = student_id.split('_', 1)
    except ValueError: expected_enrollment, expected_name = None, None
    # Use standard tqdm
    for frame_file in tqdm(frame_files, desc="Processing ID Frames", leave=False, file=sys.stdout, ascii=True):
        frame = cv2.imread(frame_file)
        if frame is None: continue
        processed_frame = preprocess_for_ocr(frame)
        try:
            custom_config = r'--oem 3 --psm 11'
            ocr_text = pytesseract.image_to_string(processed_frame, config=custom_config)
            # !!! ADJUST Regex for your EXACT enrollment number format !!!
            enrollment_pattern = r'\b([A-Za-z0-9]{4,})\b'
            enroll_match = re.search(enrollment_pattern, ocr_text)
            name_match = False
            if expected_name and re.search(r'\b' + re.escape(expected_name) + r'\b', ocr_text, re.IGNORECASE): name_match = True
            if enroll_match and not enrollment_found:
                potential_enrollment = enroll_match.group(1)
                if expected_enrollment and potential_enrollment == expected_enrollment: enrollment_found = potential_enrollment
                elif not expected_enrollment: enrollment_found = potential_enrollment
            if name_match and not name_found: name_found = expected_name
            if expected_enrollment and expected_name:
                if enrollment_found == expected_enrollment and name_found == expected_name: ocr_successful = True; break
            elif enrollment_found and name_found: ocr_successful = True; break
        except pytesseract.TesseractNotFoundError: print("\nERROR: Tesseract not installed/found."); return None, None, False
        except Exception as e: print(f"OCR Error on {os.path.basename(frame_file)}: {e}")
    final_enrollment = enrollment_found if enrollment_found else (expected_enrollment if expected_enrollment else None)
    final_name = name_found if name_found else (expected_name if expected_name else None)
    if ocr_successful: print(f"  --> OCR Result: Enroll='{final_enrollment}', Name='{final_name}', Success=True")
    else: print(f"Warning: OCR failed/unverified for {student_id}")
    return final_enrollment, final_name, ocr_successful

def generate_face_embeddings(student_id):
    """Generates FaceNet embeddings for a student's face frames."""
    face_frames_path = os.path.join(FRAMES_PATH, student_id, 'face')
    if not os.path.exists(face_frames_path): return None
    print(f"Generating face embeddings for {student_id}...")
    embeddings = []
    frame_files = sorted(glob.glob(os.path.join(face_frames_path, '*.png')))
    if not frame_files: return None
    if mtcnn is None or facenet_resnet is None: print("ERROR: Models not initialized."); return None
    # Use standard tqdm
    for frame_file in tqdm(frame_files, desc="Processing Face Frames", leave=False, file=sys.stdout, ascii=True):
        try:
            img = Image.open(frame_file).convert('RGB')
            # Detect all faces, get boxes and probs
            boxes, probs = mtcnn.detect(img)
            if boxes is not None:
                # Process the face with the highest probability (or first one above threshold)
                best_box = None
                max_prob = FACE_CONFIDENCE_THRESHOLD # Use threshold as minimum
                if probs is not None:
                     valid_indices = [idx for idx, p in enumerate(probs) if p >= FACE_CONFIDENCE_THRESHOLD]
                     if valid_indices:
                          best_idx = valid_indices[np.argmax(probs[valid_indices])] # Index of highest prob face above threshold
                          best_box = boxes[best_idx]
                          max_prob = probs[best_idx] # Store max prob for info if needed

                if best_box is not None:
                    box = best_box
                    # Check box validity
                    if box[0] >= box[2] or box[1] >= box[3] or any(c < 0 for c in box): continue
                    face_img = img.crop(box)
                    if face_img.width == 0 or face_img.height == 0: continue
                    face_img_resized = face_img.resize((160, 160))
                    face_tensor = transforms.ToTensor()(face_img_resized)
                    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    face_tensor = normalize(face_tensor).unsqueeze(0).to(device)
                    with torch.no_grad(): embedding = facenet_resnet(face_tensor)
                    embeddings.append(embedding.squeeze().cpu().numpy())
                    # Only process one face (the best one) per frame for embedding generation
                    # break # Removed break: process all faces above threshold? No, average embedding needs one face per image. Stick to best face.
        except FileNotFoundError: print(f"Error: Frame file not found: {frame_file}")
        except Exception as e: print(f"Error processing face frame {os.path.basename(frame_file)}: {e}"); traceback.print_exc()
    if not embeddings: print(f"Warning: Could not generate any embeddings for {student_id}"); return None
    mean_embedding = np.mean(np.array(embeddings), axis=0)
    print(f"Generated mean embedding for {student_id}.")
    return mean_embedding

print("Backend Script: Helper functions defined.")
print("-" * 20)

# -----------------------------------------------------------------------------
# Core Logic Functions
# -----------------------------------------------------------------------------

def verify_uniform(frame, bbox, lower_hsv, upper_hsv):
    """Checks uniform color below face bbox."""
    if lower_hsv is None or upper_hsv is None: return False
    lower_bound, upper_bound = tuple(lower_hsv), tuple(upper_hsv)
    x1, y1, x2, y2 = map(int, bbox); h, w, _ = frame.shape; face_h = y2 - y1
    roi_y1, roi_y2 = max(0, y2), min(h, y2 + int(face_h * 1.5))
    roi_x1, roi_x2 = max(0, x1), min(w, x2)
    if roi_y1 >= roi_y2 or roi_x1 >= roi_x2: return False
    torso_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    if torso_roi.size == 0: return False
    hsv_roi = cv2.cvtColor(torso_roi, cv2.COLOR_BGR2HSV)
    mask = None
    if lower_bound[0] > upper_bound[0]: # Hue wrap
        mask1 = cv2.inRange(hsv_roi, (lower_bound[0], lower_bound[1], lower_bound[2]), (179, upper_bound[1], upper_bound[2]))
        mask2 = cv2.inRange(hsv_roi, (0, lower_bound[1], lower_bound[2]), (upper_bound[0], upper_bound[1], upper_bound[2]))
        mask = cv2.bitwise_or(mask1, mask2)
    else: mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)
    match_ratio = np.count_nonzero(mask) / (mask.size + 1e-6)
    return match_ratio > MIN_UNIFORM_AREA_RATIO

def recognize_face(face_embedding, known_embeddings_dict):
    """Finds best match using Euclidean distance."""
    if not known_embeddings_dict: return None, float('inf')
    min_dist = float('inf'); best_match_id = None
    current_embedding_np = np.array(face_embedding)
    for student_id, known_emb_np in known_embeddings_dict.items():
        dist = np.linalg.norm(current_embedding_np - known_emb_np)
        if dist < min_dist: min_dist = dist; best_match_id = student_id
    return best_match_id, min_dist

def run_preprocessing_and_training(face_video_dir, uniform_video_dir, id_card_video_dir, process_ids_flag):
    """Runs preprocessing pipeline."""
    print("--- Starting Preprocessing ---")
    if mtcnn is None or facenet_resnet is None: print("ERROR: Models not initialized."); return
    os.makedirs(FRAMES_PATH, exist_ok=True); os.makedirs(MODELS_PATH, exist_ok=True)
    all_embeddings = {}; all_student_data = {}
    print(f"Scanning for students in: {face_video_dir}")
    face_video_files = glob.glob(os.path.join(face_video_dir, '*.mp4'))
    if not face_video_files: print(f"Error: No face videos found in {face_video_dir}."); return
    unique_student_ids, student_name_map, enrollment_map = set(), {}, {}
    for video_path in face_video_files:
        enrollment, student_name, student_id = parse_video_filename(video_path)
        if student_id: unique_student_ids.add(student_id); student_name_map[student_id] = student_name; enrollment_map[student_id] = enrollment
    if not unique_student_ids: print("No valid student IDs found."); return
    print(f"Found {len(unique_student_ids)} unique student identifiers: {sorted(list(unique_student_ids))}")
    print("\n--- Step 2: Extracting Frames ---")
    # Use standard tqdm
    for student_id in tqdm(sorted(list(unique_student_ids)), desc="Extracting Frames", file=sys.stdout, ascii=True):
        face_vid_path = os.path.join(face_video_dir, f"{student_id}.mp4")
        uniform_vid_path = os.path.join(uniform_video_dir, f"{student_id}.mp4")
        id_vid_path = os.path.join(id_card_video_dir, f"{student_id}.mp4")
        if os.path.exists(face_vid_path): extract_frames(face_vid_path, FRAMES_PATH, student_id, 'face', FRAME_SKIP_RATE)
        if os.path.exists(uniform_vid_path): extract_frames(uniform_vid_path, FRAMES_PATH, student_id, '360', FRAME_SKIP_RATE)
        if process_ids_flag and os.path.exists(id_vid_path): extract_frames(id_vid_path, FRAMES_PATH, student_id, 'idcard', FRAME_SKIP_RATE)
    print(f"\n--- Step 3: Processing Extracted Frames ---")
    valid_students = 0
    # Use standard tqdm
    for student_id in tqdm(sorted(list(unique_student_ids)), desc="Processing Student Data", file=sys.stdout, ascii=True):
        print(f"\n--- Processing data for: {student_id} ---")
        expected_enrollment = enrollment_map.get(student_id, None); expected_name = student_name_map.get(student_id, None)
        lower_hsv, upper_hsv = analyze_uniform_color(student_id)
        if process_ids_flag:
            id_enrollment, id_name, id_verified = process_id_card(student_id)
            if not id_verified: id_enrollment = expected_enrollment; id_name = expected_name
        else: print(f"Skipping ID Card processing for {student_id}."); id_enrollment = expected_enrollment; id_name = expected_name; id_verified = False
        embedding = generate_face_embeddings(student_id)
        if embedding is not None:
            all_embeddings[student_id] = embedding
            all_student_data[student_id] = {'enrollment': id_enrollment if id_enrollment else 'Unknown', 'name': id_name if id_name else 'Unknown', 'uniform_lower_hsv': lower_hsv, 'uniform_upper_hsv': upper_hsv, 'id_verified': id_verified }
            valid_students += 1
        else: print(f"@@@ Final Skip: {student_id} skipped due to missing face embeddings. Check logs above. @@@")
    print("\n--- Step 4: Saving Models and Data ---")
    if all_embeddings:
        print(f"Attempting to save data for {valid_students} students with embeddings.")
        try: torch.save(all_embeddings, FACE_EMBEDDINGS_FILE); print(f"Saved face embeddings to {FACE_EMBEDDINGS_FILE}")
        except Exception as e: print(f"Error saving embeddings file: {e}")
        try:
            with open(STUDENT_DATA_FILE, 'w') as f: json.dump(all_student_data, f, indent=4)
            print(f"Saved student data to {STUDENT_DATA_FILE}")
        except Exception as e: print(f"Error saving student data JSON: {e}")
    else: print("No valid embeddings generated. Nothing saved.")
    print(f"--- Preprocessing Complete --- Processed {valid_students} / {len(unique_student_ids)} students successfully.")
    print("-" * 20)

def mark_attendance(mode, video_path=None):
    """Marks attendance (real-time or video). Generates CSV of ALL students."""
    print(f"--- Starting Attendance Marking (Mode: {mode}) ---")
    if mtcnn is None or facenet_resnet is None: print("ERROR: Models not initialized."); return
    os.makedirs(ATTENDANCE_PATH, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    attendance_file = os.path.join(ATTENDANCE_PATH, f"attendance_{timestamp}.csv")
    output_video_file = os.path.join(ATTENDANCE_PATH, f"attendance_output_{timestamp}.mp4")
    if not os.path.exists(FACE_EMBEDDINGS_FILE) or not os.path.exists(STUDENT_DATA_FILE): print(f"Error: {FACE_EMBEDDINGS_FILE} or {STUDENT_DATA_FILE} not found. Run 'preprocess'."); return
    try: known_embeddings = torch.load(FACE_EMBEDDINGS_FILE);
    except Exception as e: print(f"Error loading embeddings: {e}"); return
    try:
        with open(STUDENT_DATA_FILE, 'r') as f: student_data = json.load(f)
    except Exception as e: print(f"Error loading student data: {e}"); return
    if not known_embeddings or not student_data: print("Error: No student data/embeddings loaded."); return
    print(f"Loaded data for {len(student_data)} registered students.")
    full_attendance_log = {}
    print("Initializing attendance sheet...")
    for student_id, data in student_data.items():
        if student_id not in known_embeddings: print(f"Warning: Student '{data.get('name', student_id)}' lacks embedding. Skipping."); continue
        full_attendance_log[student_id] = {'Enrollment': data.get('enrollment', 'N/A'), 'Name': data.get('name', 'N/A'), 'Face_Recognized': 'No', 'Uniform_Verified': 'No', 'ID_Card_Verified': 'Yes' if data.get('id_verified', False) else 'No', 'Present': 'No'}
    if not full_attendance_log: print("Error: No students with embeddings found."); return
    print(f"Attendance sheet initialized for {len(full_attendance_log)} students.")
    cap, video_writer = None, None
    if mode == 'realtime':
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): print("Error: Cannot open webcam."); return
        print("Starting real-time attendance... Press 'q' in OpenCV window to quit.")
    elif mode == 'video':
        if not video_path or not os.path.exists(video_path): print(f"Error: Video path invalid: '{video_path}'"); return
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): print(f"Error: Cannot open video: {video_path}"); return
        print(f"Starting attendance from video: {video_path}...")
        try:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_in = cap.get(cv2.CAP_PROP_FPS); fps_in = max(fps_in, 1)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_file, fourcc, fps_in, (frame_width, frame_height))
            if not video_writer.isOpened(): print(f"Error: Could not open VideoWriter."); video_writer = None
            else: print(f"Output video -> {output_video_file}")
        except Exception as e: print(f"Error setting up video writer: {e}"); video_writer = None
    frame_idx, skip_counter = 0, 0
    pbar = None
    # Use standard tqdm
    if mode == 'video': pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing Video", file=sys.stdout, ascii=True, unit='frame')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if mode == 'video' and pbar: pbar.update(1) # Update progress bar per frame read
        skip_counter += 1
        if skip_counter % FRAME_SKIP_RATE != 0: continue # Skip frame processing
        display_frame = frame.copy(); start_time = time.time()
        try:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Detect faces using global MTCNN model
            boxes, confidences = mtcnn.detect(img_pil)
            if boxes is not None:
                for i, box in enumerate(boxes):
                    confidence = confidences[i] if confidences is not None and i<len(confidences) else 1.0
                    if confidence < FACE_CONFIDENCE_THRESHOLD: continue
                    if box[0] >= box[2] or box[1] >= box[3] or any(c < 0 for c in box): continue
                    face_img = img_pil.crop(box)
                    if face_img.width == 0 or face_img.height == 0: continue
                    face_img_resized = face_img.resize((160, 160))
                    face_tensor = transforms.ToTensor()(face_img_resized)
                    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    face_tensor = normalize(face_tensor).unsqueeze(0).to(device)
                    # Generate embedding using global FaceNet model
                    with torch.no_grad(): current_embedding = facenet_resnet(face_tensor).squeeze().cpu().numpy()
                    # Recognize face
                    best_match_id, distance = recognize_face(current_embedding, known_embeddings)
                    face_recognized, uniform_verified = False, False
                    label, color = "Unknown", (0, 0, 255) # Red
                    if best_match_id and distance < FACE_REC_THRESHOLD and best_match_id in full_attendance_log:
                        face_recognized = True
                        student_record = full_attendance_log[best_match_id]
                        s_name = student_record.get('Name', 'N/A')
                        label = f"{s_name} ({distance:.2f})"
                        color = (0, 255, 0) # Green
                        lower_hsv = student_data[best_match_id].get('uniform_lower_hsv')
                        upper_hsv = student_data[best_match_id].get('uniform_upper_hsv')
                        if lower_hsv and upper_hsv:
                            uniform_verified = verify_uniform(frame, box, lower_hsv, upper_hsv)
                            label += " (U:Ok)" if uniform_verified else " (U:?)"
                            if uniform_verified: color = (0, 255, 255) # Yellow
                        else: label += " (U:N/A)"
                        if face_recognized: # Update Log
                            student_record['Face_Recognized'] = 'Yes'
                            student_record['Uniform_Verified'] = 'Yes' if uniform_verified else 'No'
                            if student_record['Present'] == 'No': print(f"  -> Marking {s_name} present.")
                            student_record['Present'] = 'Yes'
                    elif best_match_id and distance < FACE_REC_THRESHOLD:
                         print(f"Warning: Recog ID '{best_match_id}' not in log.")
                         label = f"Log Err ({distance:.2f})"
                    # Draw BBox
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    text_y = y1 - 10 if y1 > 20 else y1 + 15
                    cv2.putText(display_frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception as e: print(f"\nError during frame processing: {e}"); traceback.print_exc()
        if mode == 'realtime':
            fps = 1 / (time.time() - start_time + 1e-6)
            cv2.putText(display_frame, f"FPS:{fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,0),3)
            cv2.putText(display_frame, f"FPS:{fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)
            cv2.imshow('Attendance System (Press Q to Quit)', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): print("Quitting..."); break
        elif video_writer:
            try: video_writer.write(display_frame)
            except Exception as write_error: print(f"Error writing frame: {write_error}")
        frame_idx += 1
    if pbar: pbar.close()
    if cap: cap.release()
    if mode == 'realtime': cv2.destroyAllWindows()
    if video_writer: video_writer.release(); print(f"\nFinished writing output video.")
    if full_attendance_log:
        print("\nSaving final attendance sheet...")
        attendance_df = pd.DataFrame(list(full_attendance_log.values()))
        cols = ['Enrollment', 'Name', 'Face_Recognized', 'Uniform_Verified', 'ID_Card_Verified', 'Present']
        for col in cols:
            if col not in attendance_df.columns: attendance_df[col] = 'No'
        attendance_df = attendance_df[cols]
        try: attendance_df.to_csv(attendance_file, index=False); print(f"Attendance sheet saved to: {attendance_file}")
        except Exception as e: print(f"Error saving attendance CSV: {e}")
    else: print("Attendance log empty. Nothing saved.")
    print("--- Attendance Marking Complete ---")
    print("-" * 20)

print("Backend Script: Core logic functions defined.")
print("-" * 20)

# -----------------------------------------------------------------------------
# Main Execution Trigger (using Argparse)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Backend Script: Parsing command-line arguments...")
    parser = argparse.ArgumentParser(description="AI Based Automated Attendance System - Backend")
    parser.add_argument('--mode', type=str, required=True,
                        choices=['preprocess', 'attend_realtime', 'attend_video'],
                        help="Operation mode")
    parser.add_argument('--video_path', type=str, default=None,
                        help="Path to the class video file (for attend_video mode)")
    # Add arguments for input paths
    parser.add_argument('--face_dir', type=str, default=None, help="Path to face videos directory (overrides default)")
    parser.add_argument('--uniform_dir', type=str, default=None, help="Path to 360/uniform videos directory (overrides default)")
    parser.add_argument('--id_dir', type=str, default=None, help="Path to ID card videos directory (overrides default)")
    # Argument to enable ID processing (default is False based on global var)
    # store_true means if the flag is present, set to True, otherwise use default (False)
    parser.add_argument('--process_ids', action='store_true', help="Enable ID card processing")

    args = parser.parse_args()

    # Determine effective paths and flags, prioritizing command-line args
    # (This assumes global variables hold the defaults set earlier)
    face_video_path_eff = args.face_dir if args.face_dir else FACE_VIDEO_PATH
    uniform_video_path_eff = args.uniform_dir if args.uniform_dir else UNIFORM_VIDEO_PATH
    id_card_video_path_eff = args.id_dir if args.id_dir else IDCARD_VIDEO_PATH
    process_ids_eff = args.process_ids # If flag present, True, else False (correctly uses default False)

    print(f"Backend Script: Effective Args: Mode={args.mode}, VideoPath={args.video_path}, ProcessIDs={process_ids_eff}")
    print(f"Backend Script: Effective Face Dir: {face_video_path_eff}")
    print(f"Backend Script: Effective Uniform Dir: {uniform_video_path_eff}")
    print(f"Backend Script: Effective ID Dir: {id_card_video_path_eff}")


    # Ensure models are loaded before running attendance
    if args.mode != 'preprocess' and (mtcnn is None or facenet_resnet is None):
         print("ERROR: Models not initialized. Cannot run attendance.")
         sys.exit(1) # Exit if models failed initialization

    # --- Execute based on parsed arguments ---
    print(f"\n--- Backend Executing Mode: {args.mode} ---")
    if args.mode == 'preprocess':
        # Pass effective paths and flag to the function
        run_preprocessing_and_training(
            face_video_dir=face_video_path_eff,
            uniform_video_dir=uniform_video_path_eff,
            id_card_video_dir=id_card_video_path_eff,
            process_ids_flag=process_ids_eff # Pass the effective flag value
        )
    elif args.mode == 'attend_realtime':
        mark_attendance(mode='realtime')
    elif args.mode == 'attend_video':
        if not args.video_path:
            print("Error: --video_path is required for attend_video mode.")
        else:
            mark_attendance(mode='video', video_path=args.video_path)
    else:
         # This case should not be reachable due to 'choices' in argparse
         print(f"Error: Unknown mode '{args.mode}' received by backend.")

    print("\n--- Backend Script Execution Finished ---")

