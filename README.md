# LearnPulse-Camera-Based-Student-Engagement-and-Understanding-Index-System

**A camera-based system for analysing behavioural engagement in classroom sessions.**

LearnPulse was developed as part of an MSc dissertation project for **COMP5200M** in the School of Computing at the **University of Leeds**.

The system analyses classroom video, either from a live webcam or from a previously recorded session, and provides lecturers with an overview of behavioural engagement across the class.

Rather than focusing on individual students, LearnPulse reports engagement at class level. The results include an overall engagement index over time, a breakdown of different seating areas, periods where engagement appears to fall, teaching suggestions based on the observed patterns, a recap quiz based on the lecture material, and a downloadable PDF report.

---

## What LearnPulse measures

LearnPulse is designed to estimate **behavioural engagement** from signals that can be observed in video. These include factors such as head orientation, posture, movement or stillness, and visible device use.

It is important to distinguish behavioural engagement from actual understanding. LearnPulse does not attempt to determine whether a student understands the material, is interested in the lecture, or is experiencing a particular emotion.

For example, a student may look away from the screen while thinking about a difficult problem, while another student may appear attentive without fully understanding the topic. The results should therefore be treated as supporting information for lecturers rather than as a direct measurement of learning.

The system also deliberately avoids producing individual student scores. The database contains no student entity and no named student records are created. Results are instead aggregated across the classroom.

This design decision was also influenced by the fairness evaluation carried out during the project. The evaluation showed that prediction accuracy was not equally consistent across every individual in the dataset. For this reason, LearnPulse is intended to support class-level reflection rather than assessment of individual students.

---

## Requirements

To run LearnPulse, you will need:

* Python 3.10 or later

  * The project was developed using Python 3.13.
* Approximately 4 GB of available disk space.
* A webcam if you want to use live classroom analysis.
* An internet connection during the first setup so that the required YOLO model weights can be downloaded.
* A dedicated graphics card is not required.

The application can run entirely on a CPU, although processing speed will depend on the computer being used.

---

## Installation

First, clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/learnpulse.git
cd learnpulse
```

It is recommended to create a virtual environment before installing the dependencies.

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS or Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The project uses YOLO models for object and pose detection. The required model weights, including `yolov8n.pt` and `yolov8n-pose.pt`, are downloaded automatically the first time they are needed.

Together, these files are approximately 12 MB.

---

## Running the application

Run the server from the main project directory rather than from inside the `Webapp` folder:

```bash
python Webapp/server.py
```

Once the server has started, open the following address in a browser:

**http://localhost:8000**

You can then create an account and start a classroom analysis session.

The first launch may take around twenty to thirty seconds because the computer-vision models need to be loaded into memory.

If the trained engagement classifier is available, the terminal will display:

```text
Trained engagement model loaded.
```

If `engagement_model.joblib` cannot be found, LearnPulse will continue to run using its rule-based engagement estimation instead.

---

## Using LearnPulse

### 1. Sign in

Create an account or sign in to an existing account.

Passwords must contain at least eight characters and include:

* an uppercase letter;
* a lowercase letter;
* a number; and
* a symbol.

Passwords are not stored as plain text. They are stored using **PBKDF2-SHA256 hashing with an individual salt for each user**.

### 2. Enter the class information

Enter the subject and the class or group name.

A lecture plan can also be provided, although this is optional. Topics can be entered one per line, for example:

```text
12: Recursion
25: Trees
40: Graph Search
```

This information can help relate changes in engagement to different parts of the lecture.

### 3. Upload the lecture slides

Lecture slides are required when using the video analysis modes.

LearnPulse accepts:

* `.pptx`
* `.pdf`

The slide content is used to provide context for the engagement timeline and to generate the recap material.

### 4. Choose an analysis mode

LearnPulse provides three main modes.

**Live mode**

The system opens the connected webcam and analyses the classroom while the lecture is taking place. A running timer is displayed and the session continues until the lecturer presses stop.

**Recorded mode**

A previously recorded classroom video can be uploaded and analysed retrospectively.

**Recap from slides**

Slides can also be uploaded without a classroom video. In this mode, LearnPulse focuses on generating a content recap and quiz rather than measuring engagement.

### 5. Review the results

After processing is complete, the results are presented on a continuous results page.

The page contains:

* the engagement analysis;
* engagement changes over time;
* seating-zone information;
* flagged periods of lower attention;
* a class-level summary;
* teaching recommendations;
* a recap of the lecture content; and
* a recap quiz.

The intention is to give the lecturer both information about the observed classroom session and practical material that can be used when reviewing or planning subsequent teaching.

### 6. Download the report

A PDF report can be generated for each completed session.

The report brings together the main engagement results, summaries, recommendations and supporting visualisations in a format that can be stored or reviewed later.

---

## Command-line use

The core analysis engine can also be used without starting the web interface.

To analyse a video:

```bash
python engine/analyze_fast.py data/your_video.mp4 --headless
```

To generate a report from an existing engagement log:

```bash
python engine/generate_report.py engagement_log_your_video.csv
```

This can be useful for testing the analysis components separately from the web application.

---

## Optional AI-based content recap

LearnPulse can optionally use a local language model to produce a more detailed recap of the uploaded lecture slides.

For local processing, install **Ollama** and download a model:

```bash
ollama pull llama3.2
```

When Ollama is used, the model runs on the local computer. This means the extracted slide content does not need to be sent to an external AI service.

A hosted OpenAI-compatible API can also be used as a fallback by setting an `OPENAI_API_KEY`. When this option is used, the text extracted from the slides is sent to the configured external service for processing.

The following environment variables are supported:

```text
OLLAMA_URL      default http://localhost:11434
OLLAMA_MODEL    default llama3.2
OPENAI_API_KEY  enables the hosted fallback
OPENAI_BASE     default https://api.openai.com/v1
OPENAI_MODEL    default gpt-4o-mini
```

On a CPU-only computer, generating the AI recap may take around one to two minutes depending on the hardware and model being used.

---

## Reproducing the trained engagement model

The repository includes `engagement_model.joblib`, so retraining the engagement classifier is not required simply to run LearnPulse.

The model was trained using the **DAiSEE dataset**.

DAiSEE is not included in this repository because it is distributed under a research-only licence and cannot be redistributed as part of this project.

Researchers who want to reproduce the training process must request access to the dataset from the DAiSEE authors.

After obtaining the dataset, place it in:

```text
data/DAiSEE/
```

The training pipeline can then be reproduced using:

```bash
python engine/extract_features_full.py
python engine/train_classifier_full.py
python engine/bias_audit.py
```

`extract_features_full.py` performs feature extraction and can take several hours when running only on a CPU. The process is resumable so that previously processed data does not need to be extracted again.

`train_classifier_full.py` trains the classifier using the official dataset splits.

`bias_audit.py` evaluates how consistently the trained model performs across different individuals in the dataset.

---

## Repository structure

```text
engine/
  analyze_fast.py            Command-line analyser
  analyze_video_hybrid.py    Hybrid model and rule-based analyser
  extract_features_full.py   Resumable feature extraction
  train_classifier_full.py   Training using official dataset splits
  train_classifier_final.py  Oversampled training variant
  bias_audit.py              Fairness evaluation
  generate_report.py         PDF report generation

Webapp/
  server.py                  FastAPI application
  dashboard.html             Main browser interface
  learnpulse.db              SQLite database generated at runtime
  uploads/                   Temporary uploaded files
  logs/                      Analysis logs

engagement_model.joblib      Trained engagement classifier
requirements.txt             Python dependencies
```

The `Webapp/server.py` file contains the main application logic and connects the user interface with the analysis engine, database, slide processing, recap generation and report functions.

---

## API

The web application exposes the following main API routes:

| Method | Route                       | Purpose                                                     |
| ------ | --------------------------- | ----------------------------------------------------------- |
| GET    | `/`                         | Opens the main interface                                    |
| POST   | `/api/signup`               | Creates a user account                                      |
| POST   | `/api/login`                | Signs a user in                                             |
| POST   | `/api/logout`               | Signs the current user out                                  |
| GET    | `/api/me`                   | Returns the current account                                 |
| GET    | `/api/sessions`             | Returns sessions belonging to the signed-in user            |
| GET    | `/api/sessions/{id}`        | Returns one session, including its summary, advice and quiz |
| POST   | `/api/analyze`              | Analyses an uploaded classroom recording                    |
| GET    | `/api/progress`             | Returns analysis progress information                       |
| POST   | `/api/live/start`           | Starts live classroom analysis                              |
| POST   | `/api/live/stop`            | Stops live classroom analysis                               |
| GET    | `/api/live/status`          | Returns the current live engagement information             |
| POST   | `/api/slides/parse`         | Extracts information from uploaded lecture slides           |
| POST   | `/api/slides/ai`            | Generates an AI-supported slide recap                       |
| GET    | `/api/llm/status`           | Reports which language model is currently available         |
| POST   | `/api/sessions/{id}/recap`  | Generates a recap for an existing session                   |
| GET    | `/api/sessions/{id}/report` | Generates and downloads the session PDF report              |

During live analysis, `/api/live/status` is normally polled approximately once per second so that the interface can display updated engagement information.

---

## Configuration

Several important processing settings are defined near the top of `Webapp/server.py`:

```python
SAMPLES_PER_SEC = 1.5
INFER_WIDTH     = 512
LIVE_SAMPLES_PS = 2.0
GRID_ROWS, GRID_COLS = 3, 4
```

`SAMPLES_PER_SEC` controls how many frames are sampled from an uploaded recording each second.

`INFER_WIDTH` controls the resolution used during model inference.

`LIVE_SAMPLES_PS` controls the sampling rate used during live analysis.

`GRID_ROWS` and `GRID_COLS` determine how the classroom image is divided into seating zones.

The default values were selected as a compromise between processing speed and the amount of temporal and spatial information available to the system.

For example, using approximately 1.5 sampled frames per second and a 512-pixel inference width allows a thirty-six-minute classroom recording to be processed in roughly one to two minutes on a typical laptop CPU.

Increasing the sampling rate provides more temporal information, while increasing the inference resolution can improve the amount of visible spatial detail. Both changes also increase processing time.

---

## Known limitations

LearnPulse is a research prototype and has several important limitations.

* The engagement classifier was trained using **DAiSEE**, which mainly contains single-person e-learning footage. A physical classroom is a considerably different environment, so the trained model does not transfer perfectly to classroom footage. LearnPulse combines the classifier with rule-based signals to reduce this problem, but it cannot remove it completely.

* Teacher identification currently relies partly on the assumption that the teacher is standing. This can become unreliable if the lecturer is seated among the students or if several people are standing.

* Students who are far away from the camera, partially hidden, or heavily occluded are more difficult to analyse reliably.

* The current slide-to-video alignment assumes that uploaded slides were presented in their original order and for approximately equal periods of time. Real lectures do not always follow this pattern.

* Requiring lecture slides makes the current system less suitable for teaching formats based mainly on whiteboards, laboratory activities, demonstrations or informal seminars.

* Email delivery of generated reports is not implemented in the current version. Although an email-related option may appear in the interface, there is currently no `/api/sessions/{id}/email` endpoint. Reports therefore need to be downloaded manually.

* The current version does not include a complete automated test suite.

These limitations should be considered when interpreting the system's results.

---

## Privacy

Privacy was an important design consideration during the development of LearnPulse.

Uploaded classroom videos are used only while the analysis is running. The uploaded file is deleted in a `finally` block when analysis finishes, including cases where the analysis fails because of an error.

Uploaded slides are similarly parsed for their content and then removed.

LearnPulse does not store video frames, facial identities, names, or individual student engagement histories.

The database contains class-level session information rather than student profiles.

The system also distinguishes between low engagement and missing measurements. If no student can be detected during a particular period, that period is treated as a lack of measurement coverage rather than automatically being recorded as zero engagement.

This distinction is important because failure to detect a student does not mean that the student was disengaged.

---

## Interpreting the results

LearnPulse is intended as a **teaching-support tool**, not an automated judgement system.

A low engagement score should not automatically be interpreted as poor teaching or poor student behaviour. Instead, lecturers can use the results to identify parts of a session that may be worth reviewing.

For example, a noticeable drop in class-level engagement could encourage a lecturer to consider whether:

* a topic required more explanation;
* the pace of the lecture changed;
* students needed an activity or short break;
* a slide contained too much information;
* a difficult concept should be revisited; or
* the observation may simply have resulted from limitations in the camera view.

The engagement analysis is therefore most useful when considered alongside the lecturer's own knowledge of the class and the teaching session.

---

## Licence and data

The source code in this repository was developed for the LearnPulse MSc dissertation project.

The **DAiSEE dataset is not included**. Anyone wishing to reproduce the model training must obtain the dataset separately from its authors and follow the conditions of its research licence.

The project also makes use of YOLO model weights supplied by **Ultralytics**. These weights are downloaded at runtime and remain subject to the licence terms provided by Ultralytics.

Users who deploy or redistribute LearnPulse should review the licences of the project dependencies and external model weights separately.
