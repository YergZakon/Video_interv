# Multimodal Response Analysis System

## 1. Project Description

This application provides a real-time multimodal analysis of user responses. It captures video and audio, processes them to extract various features (like facial expressions, voice characteristics, and transcribed speech), and integrates these modalities to provide an overall assessment or summary. The primary use case is for analyzing responses during interviews or interactive sessions, offering insights into the user's engagement, emotional state, and content of their answers.

The system is built using Streamlit for the web interface, OpenCV for video analysis, PyAudio and Whisper for audio processing and speech-to-text, and various other Python libraries for data handling and visualization.

## 2. Prerequisites

### 2.1. Python
*   Python 3.8 or newer is recommended. (The application has been tested with Python 3.9 and 3.10).

### 2.2. System Dependencies

*   **PortAudio (for PyAudio)**:
    *   **Debian/Ubuntu**: `sudo apt-get install portaudio19-dev`
    *   **macOS**: `brew install portaudio`
    *   **Windows**: Wheels for PyAudio often bundle this. If installation fails, you might need to install PortAudio manually from the official website or via a package manager like Chocolatey (`choco install portaudio`).

*   **FFmpeg (Recommended for Whisper and OpenCV)**:
    Whisper (for audio transcription) and potentially OpenCV (for broader video format support) can benefit from having FFmpeg installed on the system.
    *   **Debian/Ubuntu**: `sudo apt-get install ffmpeg`
    *   **macOS**: `brew install ffmpeg`
    *   **Windows**: Download FFmpeg from the official website and add it to your system's PATH.

*   **OpenCV Dependencies (`opencv-python-headless`)**:
    The `opencv-python-headless` package used in `requirements.txt` is designed to be self-contained. However, on some minimal Linux systems, certain underlying libraries might be missing. If you encounter OpenCV import errors after Python package installation, you might need system libraries like:
    *   `sudo apt-get install libgl1-mesa-glx libglib2.0-0` (or similar, depending on your Linux distribution).

## 3. Setup Instructions

### 3.1. Clone the Repository
```bash
git clone <your-repository-url>
cd <repository-name>
```
(Replace `<your-repository-url>` and `<repository-name>` with the actual URL and directory name)

### 3.2. Create and Activate a Virtual Environment (Recommended)
Using a virtual environment helps manage dependencies and avoid conflicts.
```bash
python -m venv .venv
```
*   **Linux/macOS (bash/zsh)**:
    ```bash
    source .venv/bin/activate
    ```
*   **Windows (Command Prompt)**:
    ```bash
    .venv\Scripts\activate.bat
    ```
*   **Windows (PowerShell)**:
    ```bash
    .venv\Scripts\Activate.ps1
    ```

### 3.3. Install Python Packages
Ensure your virtual environment is activated, then run:
```bash
pip install -r requirements.txt
```

## 4. Configuration

### 4.1. `config.yaml`
This file controls various settings for the audio and video analysis modules. If the file is not found, the application will use default values and log an error.

```yaml
audio:
  whisper_model: "tiny"  # Options: "tiny", "base", "small", "medium", "large"
                         # Affects accuracy, resource usage, and download size.
  whisper_language: "ru" # Language for speech recognition (e.g., "en", "ru").
                         # Set to empty string "" or comment out for auto-detection by Whisper.

# video: # Placeholder for future video-specific configurations
  # face_detection_confidence: 0.5
```

*   `audio.whisper_model`: Specifies the OpenAI Whisper model. Smaller models (`tiny`, `base`) are faster and require fewer resources but are less accurate. Larger models (`small`, `medium`, `large`) are more accurate but demand more CPU/GPU, RAM, and have larger download sizes.
*   `audio.whisper_language`: Sets the language for transcription. For multilingual Whisper models, setting this can improve accuracy if the language is known.

### 4.2. `data/questions.csv`
This CSV file stores the questions that the application can present to the user. It must have the following columns:

*   `id`: A unique identifier for each question (integer).
*   `text`: The full text of the question (string).
*   `category`: A category for the question (string, e.g., "technical", "behavioral", "general").

If this file is not found or is improperly formatted, the Question Manager will fail to initialize, and an error will be displayed in the UI and logs. A default `questions.csv` is created if one is not found during the first run of `init_analyzers`.

**Example `data/questions.csv`:**
```csv
id,text,category
1,"Tell me about yourself.","general"
2,"Describe a challenging project you worked on.","behavioral"
3,"What are your strengths and weaknesses?","general"
```
You can edit this file to add, remove, or modify questions. Ensure the CSV format is valid.

## 5. Running the Application

1.  Ensure your virtual environment is activated.
2.  Navigate to the root directory of the project.
3.  Run the following command in your terminal:

    ```bash
    streamlit run app.py
    ```

This will start the Streamlit development server, and the application should automatically open in your default web browser. If not, the console will display a URL (usually `http://localhost:8501`) to access the application.

## 6. Troubleshooting

*   **Microphone/Camera Access Permissions**:
    *   Your operating system must grant permission for microphone and camera access.
    *   **OS Level**: Check your system's privacy settings (e.g., "Privacy & Security" on macOS, "Privacy" settings on Windows).
    *   **Browser Level**: Your web browser will likely ask for permission to access the camera and microphone when the application attempts to use them. Ensure you allow this for the application's URL.

*   **PyAudio Installation Issues**:
    *   Most `PyAudio` installation problems stem from a missing `portaudio` system library. Ensure it's installed as per **Section 2.2**.
    *   On Linux, you might also need `python3-dev` (or `python-dev` for Python 2, though this project uses Python 3): `sudo apt-get install python3-dev`.

*   **OpenCV (`cv2`) Issues**:
    *   `opencv-python-headless` (as used in `requirements.txt`) is generally easier to install as it doesn't bundle GUI components.
    *   If `cv2` fails to import, it might be due to missing system libraries on minimal Linux installations (see Section 2.2).

*   **Error Messages in the UI**:
    *   The application is designed to show error messages directly in the Streamlit interface if critical components fail (e.g., "OpenCV (cv2) не найден", "PyAudio не найден", or errors related to analyzer initialization). These messages provide immediate feedback on what might be wrong.

*   **Console Logs for Detailed Diagnostics**:
    *   The application uses Python's `logging` module. Detailed logs are printed to the console/terminal where you executed `streamlit run app.py`.
    *   These logs show the sequence of operations, warnings, errors, and full tracebacks for exceptions, which are invaluable for debugging.
    *   The default log level is `INFO`. You can change this to `logging.DEBUG` by modifying the line `logger.setLevel(logging.INFO)` near the top of `app.py` for more verbose output, though `INFO` level already provides good detail for most troubleshooting.

*   **Whisper Model Download**:
    *   The first time a specific Whisper model is used, it will be downloaded by the `AudioAnalyzer` (which uses `whisper.load_model()`). This can take some time depending on model size and internet speed. Check the console logs for download progress. Subsequent runs use the cached model.

*   **`config.yaml` or `questions.csv` Not Found/Malformed**:
    *   The application logs errors if these files are missing or cannot be parsed. For `config.yaml`, defaults will be used. For `questions.csv`, question functionality will be impaired. Ensure these files are correctly placed and formatted.

## 7. Project Structure

```
.
├── .venv/                  # Virtual environment directory
├── config.yaml             # Configuration: Whisper model, language, etc.
├── data/
│   └── questions.csv       # Questions for the interview session
├── modules/
│   ├── __init__.py
│   ├── audio_analyzer.py   # Audio capture, processing, STT (Whisper)
│   ├── data_manager.py     # Manages loading and serving questions
│   ├── video_analyzer.py   # Video capture, face/feature analysis (conceptual)
│   └── multimodal_integrator.py # Combines and reasons about multimodal data
├── utils/
│   ├── __init__.py
│   └── visualization.py    # Plotting utilities for UI
├── app.py                  # Main Streamlit application script
├── requirements.txt        # Python dependencies for pip
├── README.md               # This file: project overview and instructions
└── LICENSE                 # Project license information (e.g., MIT)
```

*   **`app.py`**: Main entry point for the Streamlit web application. Handles UI, threads, and session state.
*   **`config.yaml`**: For configuring application parameters like AI models.
*   **`data/`**: Contains data files like `questions.csv`.
*   **`modules/`**: Core backend modules.
    *   `audio_analyzer.py`: Manages audio input, uses Whisper for transcription, and extracts voice features.
    *   `video_analyzer.py`: Manages video input, performs facial detection and (planned) feature extraction.
    *   `data_manager.py`: Loads and provides questions from `questions.csv`.
    *   `multimodal_integrator.py`: (Potentially) Central module for fusing data from audio and video analyzers.
*   **`utils/`**: Utility functions, especially for creating visualizations displayed in the UI.
*   **`requirements.txt`**: Defines all necessary Python packages.

## 8. License

This project is licensed under the terms specified in the `LICENSE` file. Please refer to it for details.