# Soccer Video Analysis Platform

## Project Development

**Original Developer:** Andrew Brandenburg  
**Current Development:** Meena Kasthuri  
**Organization:** Piedra Alta Sistemas

This repository is a continuation of the Soccer Video Analysis Platform originally developed by Andrew Brandenburg. Current development focuses on performance benchmarking and optimization, structured data export, event analytics, documentation, and continued development of the inherited project backlog.

# About This Project:

This sports analysis platform takes a short soccer video clip and uses Ultralytics YOLO and ByteTrack to detect objects like players, referees, and the ball. It provides analysis features like convex hulls (Polygons showing team shapes and formations), heatmaps, trajectories, and more analytics both on a 2D pitch view and an overlay on the original video. It also collects and exports JSON data and stores information in SQL Lite and SQL Server databases.

This platform was built to provide professional analysis from soccer match clips and answer questions that many coaches and soccer clubs often ask like formations, player histories, player trajectories, ball control, team zones, and more which can be used to provide analysis and answers to professional questions.

## Recent Development

Recent work includes:

- Performance profiling of the soccer video analytics pipeline
- Separate YOLO detection and tracking timing measurements
- Resolution benchmarking at 640, 960, and 1280
- Experimental true YOLO batch inference
- Full-video CPU performance validation
- Camera-motion processing optimization
- CSV export for player, referee, and ball tracking data
- Event data model supporting touch and possession-change events
- JSON export of structured event data
- Updated project backlog and benchmark documentation

Performance benchmark details are available in:

`docs/benchmarks/`

The maintained project backlog is available in:

`docs/backlog/PROJECT_BACKLOG.md`

## Getting Started

### Prerequisites

- Python 3.14.0+
- Ultralytics YOLO v11.0
- Supervision 0.28.0+

### Installation
1. **Clone the repository:**
```bash
   git clone https://github.com/MeenaKasthuri/SoccerVideoAnalysisDemo.git
   ```
2. **Navigate into the project folder:**
   ```bash
   cd SoccerVideoAnalysisDemo
   ```

3. **Install all required packages:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: If you are using PyCharm, you can simply open the project folder and click the "Install requirements" popup bar).*

4. **Run the program**

    *For Streamlit Usage:*
    ```bash
   streamlit run streamlit_app.py
   ```
   
    *For Running the Main:*
    ```bash
   python main.py
   ```
