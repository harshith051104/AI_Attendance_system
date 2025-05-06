import streamlit as st
import subprocess
import os
import threading
import queue
import time
import tempfile
import glob
import pandas as pd
import sys

# --- Configuration ---
# !! IMPORTANT: Set this to the actual name of your main backend Python script !!
BACKEND_SCRIPT_NAME = "attendance_backend.py"
# Path where the backend script saves attendance CSVs (must match backend config)
# Get this path from the backend script's configuration (Cell 2)
ATTENDANCE_PATH = r"D:\sp1\attendance" # <--- EDIT THIS PATH if different

# --- Helper Function to Read Process Output ---
def enqueue_output(pipe, q, pipe_name):
    """Reads lines from a process pipe and puts them into a queue."""
    try:
        with pipe:
            for line in iter(pipe.readline, b''):
                q.put((pipe_name, line.decode('utf-8', errors='ignore')))
    except Exception as e:
        st.error(f"Error reading {pipe_name}: {e}")
    finally:
        q.put((pipe_name, None)) # Signal completion for this pipe

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="AI Attendance System")
st.title("🎓 AI Based Automated Attendance System")

# --- Initialize Session State ---
if 'running' not in st.session_state:
    st.session_state.running = False
    st.session_state.process = None
    st.session_state.output_log = ""
    st.session_state.output_queue = queue.Queue()
    st.session_state.mode = 'preprocess' # Default mode
    st.session_state.uploaded_file_path = None
    st.session_state.last_run_mode = None # Track mode for result display

# --- Backend Script Path ---
# Assumes backend script is in the same directory as this Streamlit app
backend_script_path = os.path.join(os.path.dirname(__file__), BACKEND_SCRIPT_NAME)
if not os.path.exists(backend_script_path):
    st.error(f"Backend script '{BACKEND_SCRIPT_NAME}' not found at '{backend_script_path}'. Please ensure it's in the same directory as this app.")
    st.stop() # Stop execution if backend script is missing

# --- Mode Selection ---
st.sidebar.header("⚙️ Select Mode")
mode_options = ['preprocess', 'attend_realtime', 'attend_video']
st.session_state.mode = st.sidebar.radio(
    "Choose the operation mode:",
    mode_options,
    index=mode_options.index(st.session_state.mode), # Keep selection sticky
    key="mode_select",
    disabled=st.session_state.running # Disable while running
)

# --- Video Upload (Conditional) ---
uploaded_file = None
if st.session_state.mode == 'attend_video':
    st.sidebar.header("🎬 Upload Class Video")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a video file (.mp4, .avi, etc.)",
        type=['mp4', 'avi', 'mov', 'mkv'], # Add other relevant video types
        key="video_uploader",
        disabled=st.session_state.running
    )
    if uploaded_file and not st.session_state.running:
        # Save uploaded file temporarily only when starting
        pass # We'll handle saving inside the button click

# --- Control Buttons ---
col1, col2 = st.columns(2)
with col1:
    start_button_pressed = st.button("🚀 Start Process", disabled=st.session_state.running)

with col2:
    stop_button_pressed = st.button("🛑 Stop Process", disabled=not st.session_state.running)

# --- Process Execution Logic ---
if start_button_pressed and not st.session_state.running:
    st.session_state.output_log = "" # Clear previous log
    st.session_state.last_run_mode = st.session_state.mode # Store mode for result display

    # --- Argument Validation ---
    if st.session_state.mode == 'attend_video' and uploaded_file is None:
        st.warning("Please upload a video file for 'attend_video' mode.")
    else:
        st.session_state.running = True
        st.session_state.uploaded_file_path = None # Reset temp file path

        # --- Prepare Command ---
        # Use sys.executable to ensure using the same Python interpreter
        command = [sys.executable, backend_script_path, "--mode", st.session_state.mode]

        if st.session_state.mode == 'attend_video' and uploaded_file is not None:
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                st.session_state.uploaded_file_path = tmp_file.name
                command.extend(["--video_path", st.session_state.uploaded_file_path])
            st.info(f"Using temporary video file: {st.session_state.uploaded_file_path}")

        # --- Launch Subprocess ---
        try:
            st.info(f"Starting backend script in '{st.session_state.mode}' mode...")
            # Use PIPE for stdout/stderr to capture output
            st.session_state.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1, # Line buffered
                universal_newlines=False # Read as bytes for enqueue_output
            )

            # --- Start Output Reading Threads ---
            st.session_state.output_queue = queue.Queue()
            stdout_thread = threading.Thread(target=enqueue_output, args=(st.session_state.process.stdout, st.session_state.output_queue, 'stdout'), daemon=True)
            stderr_thread = threading.Thread(target=enqueue_output, args=(st.session_state.process.stderr, st.session_state.output_queue, 'stderr'), daemon=True)
            stdout_thread.start()
            stderr_thread.start()

            st.rerun() # Rerun to update button states and start showing logs

        except Exception as e:
            st.error(f"Failed to start backend script: {e}")
            st.session_state.running = False
            if st.session_state.uploaded_file_path and os.path.exists(st.session_state.uploaded_file_path):
                os.unlink(st.session_state.uploaded_file_path) # Clean up temp file on error

if stop_button_pressed and st.session_state.running:
    st.warning("Attempting to stop the backend script...")
    if st.session_state.process:
        try:
            st.session_state.process.terminate() # Try graceful termination first
            time.sleep(1) # Give it a moment
            if st.session_state.process.poll() is None: # Check if still running
                 st.warning("Process did not terminate gracefully, attempting to kill...")
                 st.session_state.process.kill() # Force kill
            st.session_state.running = False # Assume stopped (poll check later)
            st.info("Stop signal sent.")
            # Let the output threads finish reading remaining output
        except Exception as e:
            st.error(f"Error stopping process: {e}")
            st.session_state.running = False # Force state update
    else:
        st.session_state.running = False # Process was already gone

    # Clean up temporary file if it exists
    if st.session_state.uploaded_file_path and os.path.exists(st.session_state.uploaded_file_path):
        try:
            os.unlink(st.session_state.uploaded_file_path)
            st.session_state.uploaded_file_path = None
        except Exception as e:
            st.error(f"Error deleting temporary file: {e}")

    st.rerun() # Update UI


# --- Display Log Output ---
st.header("📜 Script Output Log")
log_placeholder = st.empty() # Placeholder for dynamic updates

if st.session_state.running:
    log_placeholder.info("Process is running... Logs will appear below.")
    # Continuously read from queue and update log
    new_output = ""
    while not st.session_state.output_queue.empty():
        pipe_name, line = st.session_state.output_queue.get_nowait()
        if line is None: # Signal that a pipe finished
            # Could add logic here if needed when stdout/stderr closes
            pass
        else:
            new_output += line
            st.session_state.output_log += line

    # Update the text area content
    with log_placeholder.container():
        st.text_area("Live Log:", value=st.session_state.output_log, height=400, key="log_area")

    # Check if process finished while reading logs
    if st.session_state.process and st.session_state.process.poll() is not None:
        st.session_state.running = False
        st.info(f"Process finished with return code: {st.session_state.process.returncode}")
        # Clean up temporary file
        if st.session_state.uploaded_file_path and os.path.exists(st.session_state.uploaded_file_path):
            try:
                os.unlink(st.session_state.uploaded_file_path)
                st.session_state.uploaded_file_path = None
            except Exception as e:
                st.error(f"Error deleting temporary file after run: {e}")
        st.rerun() # Rerun to display final results

    # Add a small delay and rerun to keep checking for output/process end
    time.sleep(0.5)
    st.rerun()

elif not st.session_state.running and st.session_state.output_log:
    # Display final log if process is not running but log exists
    with log_placeholder.container():
        st.text_area("Final Log:", value=st.session_state.output_log, height=400, key="log_area_final")

    # --- Display Attendance Results (Optional) ---
    if st.session_state.last_run_mode in ['attend_realtime', 'attend_video']:
        st.divider()
        st.header("📊 Attendance Results")
        try:
            # Find the most recent CSV file in the attendance directory
            list_of_files = glob.glob(os.path.join(ATTENDANCE_PATH, 'attendance_*.csv'))
            if list_of_files:
                latest_file = max(list_of_files, key=os.path.getctime)
                st.info(f"Displaying latest attendance file: {os.path.basename(latest_file)}")
                try:
                    df = pd.read_csv(latest_file)
                    st.dataframe(df)
                except Exception as read_e:
                    st.error(f"Error reading attendance CSV '{latest_file}': {read_e}")
            else:
                st.warning(f"No attendance CSV files found in '{ATTENDANCE_PATH}'.")
        except Exception as find_e:
            st.error(f"Error accessing attendance directory '{ATTENDANCE_PATH}': {find_e}")
            st.warning("Please ensure the ATTENDANCE_PATH is correctly set and accessible.")

else:
    log_placeholder.info("Select a mode and click 'Start Process'.")

