# 🎓 Smart Attendance & Behavior Analysis System

A Python-based AI-powered system that uses **face recognition** to automatically mark student attendance and analyzes **attention/behavior** in real-time using webcam input.

---

## 📁 Folder Structure

```
smart_attendance/
│
├── app.py                            # Flask web server (dashboard)
├── capture_faces.py                  # Register new students via webcam
├── attendance_runner.py              # Real-time detection + attendance marking
├── requirements.txt                  # All Python dependencies
├── README.md                         # This file
│
├── database/
│   ├── __init__.py
│   └── db_manager.py                 # SQLite setup, all DB queries
│
├── face_recognition_module/
│   ├── __init__.py
│   ├── encoder.py                    # Capture + generate face encodings
│   └── recognizer.py                 # Match live faces to known encodings
│
├── behavior_analysis/
│   ├── __init__.py
│   ├── eye_detector.py               # Open/closed eye detection (EAR method)
│   ├── head_pose.py                  # Head direction estimation (solvePnP)
│   └── attention_classifier.py       # Combine signals → Attentive/Distracted
│
├── utils/
│   ├── __init__.py
│   ├── csv_exporter.py               # Export attendance to CSV
│   └── helpers.py                    # Shared utility functions
│
├── reports/                          # Auto-generated CSV files saved here
│
├── static/
│   ├── css/dashboard.css             # Dashboard dark theme styles
│   └── js/dashboard.js               # Live clock, auto-refresh, table filter
│
└── templates/
    ├── index.html                     # Main dashboard
    ├── students.html                  # Student registry
    └── report.html                    # Full attendance history
```

---

## ⚙️ Installation — Step by Step

### Step 1 — Prerequisites

Make sure you have **Python 3.8 or higher** installed:
```bash
python --version
# Should print: Python 3.8.x or higher
```

Install **CMake** (required to build dlib):

- **Windows**: Download from https://cmake.org/download/ and tick "Add to PATH"
- **macOS**: `brew install cmake`
- **Linux/Ubuntu**: `sudo apt update && sudo apt install cmake build-essential`

---

### Step 2 — Clone or Download the Project

```bash
git clone https://github.com/yourusername/smart-attendance-system.git
cd smart-attendance-system
```

Or download and extract the ZIP, then open the folder in terminal.

---

### Step 3 — Create a Virtual Environment

A virtual environment keeps project dependencies isolated.

```bash
# Create the environment
python -m venv venv

# Activate it:
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

---

### Step 4 — Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ **dlib takes a few minutes to compile** — this is normal. Do not cancel it.

If dlib fails on Windows, try the pre-built wheel:
```bash
pip install https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.99-cp310-cp310-win_amd64.whl
```
(Replace `cp310` with your Python version, e.g. `cp38`, `cp39`, `cp311`)

---

### Step 5 — Download the Facial Landmark Model

The behavior analysis module needs dlib's 68-point shape predictor file.

**Option A — wget (Linux/macOS)**:
```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

**Option B — Manual (Windows)**:
1. Open: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
2. Download and extract the `.bz2` file (use 7-Zip or WinRAR)
3. Place `shape_predictor_68_face_landmarks.dat` in the project root folder

---

## 🚀 Running the Project — Step by Step

### Step 1 — Initialize the Database

Run this **once** to create the SQLite database and tables:

```bash
python -c "from database.db_manager import init_db; init_db()"
```

You will see: `[DB] Database initialized successfully.`
A file `attendance.db` is created in the project root.

---

### Step 2 — Register Students

Register each student by capturing their face:

```bash
python capture_faces.py
```

- Enter the student's **name** and **ID** when prompted
- The webcam opens — have the student look at the camera
- Position their face inside the yellow rectangle
- Press **`S`** to capture and save
- Press **`Q`** to quit

To register multiple students in one session:
```bash
python capture_faces.py --multi
```
Type `done` as the name when finished.

---

### Step 3 — Start Real-Time Attendance

```bash
python attendance_runner.py
```

What happens:
- Webcam opens and begins scanning faces
- Recognized students are **automatically marked Present**
- Eye and head pose analysis runs on each frame
- Attention score (0–100) is computed per student
- Press **`Q`** to end the session

The HUD shows: FPS, number of faces detected, session timer.

---

### Step 4 — View the Web Dashboard

In a **new terminal** (keep attendance_runner.py running or run it separately):

```bash
python app.py
```

Open your browser: **http://localhost:5000**

Pages available:
- `/` — Today's summary dashboard with attendance table
- `/students` — All registered students
- `/report` — Full attendance history (all dates)
- `/api/export/today` — Download today's CSV
- `/api/export/all` — Download all-time CSV

The dashboard **auto-refreshes every 15 seconds**.

---

### Step 5 — Export CSV Report Manually

```bash
# Today only
python -c "from utils.csv_exporter import export_today; export_today()"

# All records
python -c "from utils.csv_exporter import export_all; export_all()"
```

CSV files are saved to the `reports/` folder.

---

## 🧠 How It Works

| Component | Method | What it does |
|-----------|--------|-------------|
| Face Detection | OpenCV HOG | Finds face bounding boxes in frame |
| Face Recognition | face_recognition (128-d encoding) | Identifies which student it is |
| Attendance | SQLite UNIQUE constraint | Marks present once per day, no duplicates |
| Eye Detection | dlib 68-landmarks + EAR formula | Detects if eyes are open or closed |
| Head Pose | OpenCV solvePnP | Estimates yaw/pitch to detect looking away |
| Behavior Score | Rolling average (30 frames) | Attentive ≥ 70%, else Distracted |
| Dashboard | Flask + Jinja2 | Renders live data from SQLite |
| CSV Export | Python csv module | Downloads attendance as spreadsheet |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| Python 3.8+ | Core language |
| OpenCV | Camera, image processing, drawing |
| face_recognition | Face encoding and matching |
| dlib | 68-point facial landmarks |
| Flask | Web dashboard server |
| SQLite | Zero-config embedded database |
| NumPy | Array math |
| Pandas | CSV/data handling |

---

## 📂 Key Files Explained

| File | Role |
|------|------|
| `capture_faces.py` | Run once per student to register their face |
| `attendance_runner.py` | The main detection loop — run during class |
| `app.py` | Flask server — run to view dashboard |
| `database/db_manager.py` | All database logic in one place |
| `face_recognition_module/encoder.py` | Generate + load face encodings |
| `face_recognition_module/recognizer.py` | Match faces in live frames |
| `behavior_analysis/eye_detector.py` | Eye Aspect Ratio calculation |
| `behavior_analysis/head_pose.py` | 3D head orientation estimation |
| `behavior_analysis/attention_classifier.py` | Combine signals into a score |
| `utils/csv_exporter.py` | Export attendance data to CSV |

---

## ⚠️ Troubleshooting

**Webcam not opening**
```bash
# Try index 1 instead of 0 in attendance_runner.py:
CAMERA_INDEX = 1
```

**dlib install fails**
- Make sure CMake is installed and in PATH
- Try: `pip install dlib --verbose` to see the error

**shape_predictor_68_face_landmarks.dat not found**
- Place the `.dat` file in the project root (same folder as `app.py`)

**Face not recognized**
- Ensure good lighting when registering
- Try lowering `RECOGNITION_TOLERANCE` in `recognizer.py` (e.g., 0.45)

**Behavior analysis disabled**
- This means the `.dat` landmark file is missing
- Attendance still works; only eye/head analysis is skipped

---

## 🤝 Contributing

Pull requests welcome! Open an issue first for large changes.

## 📄 License

MIT License — free to use, modify, and distribute.
