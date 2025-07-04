# AI-Based Automated Attendance System

This project is a comprehensive, AI-powered backend system for automating student attendance. It uses a multi-factor verification process including facial recognition, uniform color detection, and optional ID card OCR to ensure a high degree of accuracy and security. The system is designed to run from the command line and has two main operational modes: **preprocessing** new student data and **marking attendance** (either in real-time via a webcam or from a pre-recorded video).


## ✨ Features

-   **👨‍🎓 Advanced Facial Recognition**: Utilizes `facenet-pytorch`, a high-accuracy deep learning model (InceptionResnetV1 pretrained on VGGFace2), for robust face detection and recognition.
-   **👕 Uniform Verification**: The system analyzes a 360-degree video of each student to learn their specific uniform color. During attendance, it verifies that the person recognized is also wearing a uniform of the correct color.
-   **💳 Optional ID Card OCR**: Can process videos of student ID cards using `pytesseract` to cross-verify enrollment numbers and names, adding another layer of identity confirmation.
-   **💻 Two-Phase Architecture**:
    1.  **Preprocessing & Training (`preprocess`)**: Processes student videos (face, uniform, ID card), extracts frames, generates face embeddings, determines uniform color profiles, and saves this data for later use.
    2.  **Attendance Marking (`attend_realtime` / `attend_video`)**: Loads the preprocessed data to recognize students, verify their uniform, and mark them as present in a final CSV log.
-   **📹 Flexible Modes**: Can run a live attendance session using a webcam or process an existing video file of a class session.
-   **✅ Detailed Logging**: Generates a timestamped CSV attendance sheet with columns for Enrollment, Name, and the status of each verification check (Face, Uniform, ID Card, Present).
-   **🚀 Command-Line Interface**: Managed entirely through command-line arguments for easy integration into automated workflows and scripts.

---

## ⚙️ System Workflow

The system operates in a logical sequence to ensure data is prepared before attendance is taken.

### 1. Preprocessing (`--mode preprocess`)

This is the initial setup step that must be run for all new students.

1.  **Video Ingestion**: The script scans specified directories for student videos. Each student should have videos for their face, a 360-degree view (for uniform), and optionally, their ID card. Videos must be named in the format `EnrollmentID_StudentName.mp4`.
2.  **Frame Extraction**: Keyframes are extracted from each video at a defined skip rate to create a dataset of images.
3.  **Uniform Analysis**: The 360-degree view frames are analyzed using K-Means clustering to determine the dominant color of the student's uniform. A permissible color range (in HSV) is calculated and saved.
4.  **ID Card Processing (Optional)**: If enabled (`--process_ids`), the ID card frames are processed using Tesseract OCR to extract the enrollment number and name for verification.
5.  **Face Embedding Generation**: The MTCNN model detects faces in the face frames, and the FaceNet ResNet model generates a 512-dimension vector embedding for each face. A mean embedding is calculated to represent the student's unique facial identity.
6.  **Data Storage**: The final data—a dictionary of all student embeddings and a JSON file containing student metadata (name, enrollment, uniform color range, ID verification status)—is saved to the `models` directory.

### 2. Attendance Marking (`--mode attend_...`)

Once preprocessing is complete, this mode can be run.

1.  **Load Models**: The system loads the pre-trained student embeddings and metadata from the `models` directory.
2.  **Initialize Attendance Sheet**: A full attendance log for all registered students is created in memory, with everyone initially marked as 'No' for 'Present'.
3.  **Capture and Process Frames**: The system captures frames from the webcam or a video file.
4.  **Detect and Recognize**: For each detected face in a frame:
    -   An embedding is generated.
    -   This embedding is compared against all known student embeddings using Euclidean distance.
    -   If the distance is below a set threshold, the student is considered **recognized**.
5.  **Multi-Factor Verification**:
    -   If a student is recognized, the system then checks the area below their face for the correct **uniform color**.
    -   The system logs whether the pre-verified ID card check was successful.
6.  **Mark Present**: If a student is successfully recognized, their status is updated to 'Yes' for `Face_Recognized`, `Uniform_Verified` (if applicable), and `Present`.
7.  **Generate Output**:
    -   A final CSV attendance report is saved in the `attendance` directory.
    -   If processing a video, a new video file with bounding boxes and labels drawn on it is also saved.

---

## 📂 Directory Structure

For the script to work correctly, you must organize your files and folders as follows. You can change these paths using command-line arguments.

```
D:/sp1/
├── dataset/
│   ├── student_face_videos/
│   │   └── 21A91A0501_JohnDoe.mp4
│   │   └── 21A91A0502_JaneSmith.mp4
│   │
│   ├── student_360_view_video/
│   │   └── 21A91A0501_JohnDoe.mp4
│   │   └── 21A91A0502_JaneSmith.mp4
│   │
│   └── id_card_videos/
│       └── 21A91A0501_JohnDoe.mp4
│       └── 21A91A0502_JaneSmith.mp4
│
├── frames/      (Generated by script)
├── models/      (Generated by script)
└── attendance/  (Generated by script)
```

---

## 🛠️ Setup and Installation

### Prerequisites

-   **Python 3.8+**: Ensure Python is installed and accessible from your terminal.
-   **Tesseract OCR**: If you plan to use ID card verification, you must install Tesseract OCR from [Google's Tesseract repository](https://github.com/tesseract-ocr/tesseract) and ensure its executable is in your system's PATH.

### Installation Steps

1.  **Clone the Repository**
    ```bash
    git clone <your-repo-link>
    cd <your-repo-folder>
    ```

2.  **Create a Virtual Environment** (Recommended)
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Python Dependencies**
    Create a file named `requirements.txt` with the contents provided below and run:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Usage

The script is controlled via the command line.

### 1. Preprocess Student Data

This is the first step. Run this command to process all the videos in your dataset folders. To enable ID card scanning, add the `--process_ids` flag.

```bash
python your_script_name.py --mode preprocess
```
*or with ID card processing:*
```bash
python your_script_name.py --mode preprocess --process_ids
```

You can also override the default directory paths:
```bash
python your_script_name.py --mode preprocess --face_dir "path/to/your/face_videos" --uniform_dir "path/to/your/uniform_videos"
```

### 2. Mark Attendance (Real-Time)

To start a live attendance session using your default webcam (usually camera index 0):

```bash
python your_script_name.py --mode attend_realtime
```
An OpenCV window will appear. Press the **'q'** key on your keyboard while the window is active to stop the session and save the attendance log.

### 3. Mark Attendance (From Video)

To process a pre-recorded class video and generate an attendance sheet:

```bash
python your_script_name.py --mode attend_video --video_path "D:/path/to/your/class_session.mp4"
```

The script will process the entire video and save the attendance CSV and a new annotated video file to the `attendance` directory.

---
