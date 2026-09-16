# Design Documentation

All diagrams below are written in Mermaid (GitHub renders them inline). The sources live in
`docs/diagrams/*.mmd` and rendered SVGs sit beside them for use in the project report.

---

## 1. System architecture

Five layers, each depending only on the layer beneath it. The CLI never touches OpenCV
directly and the vision layer never touches the database, which is what makes each piece
independently testable.

```mermaid
flowchart TB
    subgraph Input["Input Layer"]
        CAM["Webcam<br/>CameraSource"]
        VID["Video file<br/>VideoFileSource"]
        IMG["Image folder<br/>ImageFolderSource"]
    end

    subgraph CLI["Presentation Layer"]
        MAIN["main.py / cli.py<br/>argparse sub-commands"]
        REP["reporting.py<br/>console tables + CSV"]
    end

    subgraph Vision["Computer Vision Layer"]
        DET["detector.py<br/>Haar cascade / passthrough"]
        PRE["preprocessing.py<br/>gray, align, denoise, CLAHE, resize"]
        REC["recognizer.py<br/>LBPH / Eigenfaces / Fisherfaces"]
    end

    subgraph Domain["Application Layer"]
        SVC["attendance_service.py<br/>temporal voting, marking rules"]
        DS["dataset.py<br/>enrolment + splits"]
        EVAL["evaluation.py<br/>metrics, calibration"]
    end

    subgraph Storage["Persistence Layer"]
        DB[("SQLite<br/>attendance.db")]
        MODEL[["face_model.yml<br/>labels.json"]]
        FILES[["data/dataset/<br/>face chips"]]
        LOGS[["logs/attendance.log"]]
    end

    CAM --> MAIN
    VID --> MAIN
    IMG --> MAIN
    MAIN --> SVC
    MAIN --> DS
    MAIN --> EVAL
    SVC --> DET --> PRE --> REC
    DS --> DET
    DS --> FILES
    REC --> MODEL
    SVC --> DB
    EVAL --> REC
    MAIN --> REP --> DB
    SVC --> LOGS
```

---

## 2. Process flow — an attendance session

The two decision diamonds that matter are the threshold check (reject unknown faces) and
the confirmation count (temporal voting). Everything else is plumbing.

```mermaid
flowchart TD
    A([Start session]) --> B["Load trained model<br/>+ label map"]
    B --> C{"Model found?"}
    C -- No --> C1["Error: run 'train' first"] --> Z([Exit])
    C -- Yes --> D["Open frame source<br/>camera / video / folder"]
    D --> E["Read next frame"]
    E --> F{"Frame available?"}
    F -- No --> Y["Close session<br/>print report + CSV"] --> Z
    F -- Yes --> G["Detect faces<br/>Haar cascade"]
    G --> H{"Face found?"}
    H -- No --> E
    H -- Yes --> I["Pre-process crop<br/>align, denoise, CLAHE, resize"]
    I --> J["Predict label + distance<br/>LBPH"]
    J --> K{"distance &lt;= threshold?"}
    K -- No --> L["Log REJECTED<br/>display 'Unknown'"] --> E
    K -- Yes --> M{"Already marked<br/>this session?"}
    M -- Yes --> E
    M -- No --> N["Increment confirmation count"]
    N --> O{"hits &gt;= required?"}
    O -- No --> E
    O -- Yes --> P["Insert attendance row<br/>PRESENT or LATE"] --> E
```

---

## 3. Use case diagram

```mermaid
flowchart LR
    ADMIN(["Teacher / Admin"])
    STUDENT(["Student"])
    EVAL(["Evaluator"])

    subgraph System["Smart Attendance System"]
        UC1(["Enrol student faces"])
        UC2(["Train recognition model"])
        UC3(["Run attendance session"])
        UC4(["View / export attendance report"])
        UC5(["Evaluate model accuracy"])
        UC6(["Manage students"])
        UC7(["Calibrate rejection threshold"])
        UC8(["Appear before the camera"])
    end

    ADMIN --> UC1
    ADMIN --> UC2
    ADMIN --> UC3
    ADMIN --> UC4
    ADMIN --> UC6
    EVAL --> UC5
    EVAL --> UC4
    STUDENT --> UC8
    UC8 -.->|"is recognised by"| UC3
    UC2 -.->|"includes"| UC7
    UC3 -.->|"includes"| UC4
```

---

## 4. Sequence diagram — `recognize`

```mermaid
sequenceDiagram
    actor Teacher
    participant CLI as cli.py
    participant Src as VideoSource
    participant Svc as AttendanceService
    participant Det as FaceDetector
    participant Pre as FacePreprocessor
    participant Rec as FaceRecognizer
    participant DB as AttendanceDatabase

    Teacher->>CLI: python main.py recognize --source 0
    CLI->>Rec: load(model_dir)
    Rec-->>CLI: trained model + label map
    CLI->>Src: open_source("0")
    CLI->>Svc: run_session(source, session_id)
    Svc->>DB: start_session()

    loop for every frame
        Src-->>Svc: frame
        Svc->>Det: detect(frame)
        Det-->>Svc: bounding boxes
        Svc->>Pre: process(frame, box)
        Pre-->>Svc: 200x200 normalised chip
        Svc->>Rec: predict(chip)
        Rec-->>Svc: (student_id, distance, accepted)
        alt distance > threshold
            Svc->>DB: log_recognition(REJECTED)
        else confirmed N times
            Svc->>DB: mark_attendance(session, student)
            DB-->>Svc: inserted / already present
        end
    end

    Svc->>DB: end_session()
    Svc-->>CLI: SessionStats
    CLI->>DB: session_attendance()
    DB-->>CLI: rows
    CLI-->>Teacher: report table + CSV
```

---

## 5. Class / component diagram

```mermaid
classDiagram
    class AppConfig {
        +PathConfig paths
        +DetectionConfig detection
        +PreprocessConfig preprocess
        +RecognitionConfig recognition
        +AttendanceConfig attendance
        +load(path) AppConfig
        +validate()
    }

    class FaceDetector {
        <<interface>>
        +detect(image) List~Box~
    }
    class HaarCascadeDetector {
        -CascadeClassifier cascade
        +detect(image) List~Box~
    }
    class PassthroughDetector {
        +detect(image) List~Box~
    }

    class FacePreprocessor {
        -PreprocessConfig config
        +crop_chip(image, box) ndarray
        +normalize(chip) ndarray
        +process(image, box) ndarray
        +is_blurry(chip) bool
    }

    class FaceRecognizer {
        -model
        -label_map
        +train(DatasetSplit)
        +predict(chip) Prediction
        +save(dir)
        +load(dir, config) FaceRecognizer
    }
    class Prediction {
        +str student_id
        +float distance
        +bool accepted
        +confidence() float
    }

    class DatasetManager {
        +enroll_from_images(id, name, dir) int
        +enroll_from_camera(id, name, src, n) int
        +load(min_samples) DatasetSplit
        +counts() dict
    }
    class DatasetSplit {
        +List~ndarray~ images
        +List~int~ labels
        +dict label_map
    }

    class AttendanceService {
        +process_frame(frame, session, idx, stats)
        +run_session(source, session) SessionStats
    }
    class AttendanceDatabase {
        +add_student(id, name)
        +mark_attendance(session, student, conf) bool
        +session_attendance(session) List~AttendanceRecord~
        +absentees(session) List~Student~
    }
    class BaseSource {
        <<interface>>
        +frames() Iterator
        +release()
    }

    FaceDetector <|.. HaarCascadeDetector
    FaceDetector <|.. PassthroughDetector
    BaseSource <|.. CameraSource
    BaseSource <|.. VideoFileSource
    BaseSource <|.. ImageFolderSource
    AttendanceService --> FaceDetector
    AttendanceService --> FacePreprocessor
    AttendanceService --> FaceRecognizer
    AttendanceService --> AttendanceDatabase
    AttendanceService --> BaseSource
    FaceRecognizer --> Prediction
    FaceRecognizer --> DatasetSplit
    DatasetManager --> DatasetSplit
    DatasetManager --> FaceDetector
    DatasetManager --> FacePreprocessor
    AttendanceService --> AppConfig
```

---

## 6. ER diagram and schema

```mermaid
erDiagram
    STUDENTS ||--o{ ATTENDANCE : "is recorded in"
    SESSIONS ||--o{ ATTENDANCE : "contains"
    SESSIONS ||--o{ RECOGNITION_LOG : "produces"
    STUDENTS ||--o{ RECOGNITION_LOG : "may be matched in"

    STUDENTS {
        TEXT student_id PK
        TEXT name
        TEXT email
        INTEGER sample_count
        TEXT enrolled_at
    }
    SESSIONS {
        TEXT session_id PK
        TEXT course_code
        TEXT started_at
        TEXT ended_at
        TEXT source
    }
    ATTENDANCE {
        INTEGER id PK
        TEXT session_id FK
        TEXT student_id FK
        TEXT marked_at
        REAL confidence
        TEXT status
    }
    RECOGNITION_LOG {
        INTEGER id PK
        TEXT session_id FK
        TEXT student_id FK
        REAL distance
        TEXT decision
        INTEGER frame_index
        TEXT logged_at
    }
```

Key constraints:

- `attendance` carries `UNIQUE(session_id, student_id)` — duplicate marking is impossible at
  the storage level, not merely discouraged in application code.
- Foreign keys are enforced (`PRAGMA foreign_keys = ON`), so attendance cannot reference a
  student who was never enrolled, and deleting a student cascades to their records.
- `recognition_log` stores **every** decision including rejections, which is what allows the
  threshold to be re-tuned after the fact from real session data.

---

## 7. Design decisions

| Decision | Alternatives considered | Why this one |
|---|---|---|
| LBPH as the default recogniser | Eigenfaces, Fisherfaces, deep embeddings (FaceNet/dlib) | LBP codes depend on *relative* pixel intensity, so they tolerate illumination change far better than PCA on raw intensity. Trains in under a second on CPU, supports incremental `update()`, and needs no pre-trained weights or network access. Eigenfaces scored higher on the synthetic benchmark, but that set has near-constant framing — exactly the condition PCA flatters. |
| Haar cascade for detection | HOG + SVM, DNN face detector (SSD/ResNet) | Ships with OpenCV, no model download, real-time on CPU. Its frontal-pose limitation is acceptable for a classroom camera facing the room. |
| Distance threshold with explicit rejection | Always take the nearest match | A nearest-neighbour classifier with no reject option will mark *someone* present for every face it sees, including a stranger. The threshold is the difference between a demo and a usable system. |
| Threshold calibrated from data | Hard-coded constant | LBPH distance scale depends on chip size, grid layout and camera. `mean + 3σ` of the genuine-match distribution transfers across datasets; a constant does not. |
| Temporal voting before marking | Mark on first match | A single frame's classification is noisy. Requiring N confirmations trades a fraction of a second for a large drop in false marks. |
| Idempotency enforced in SQL | Check-then-insert in Python | A `UNIQUE` constraint cannot be bypassed by a logic bug or a concurrent run. |
| Split preprocessing into `crop_chip` / `normalize` | One `process()` call | Stored samples keep geometric normalisation only; photometric normalisation is applied identically at training and inference time. Without this split, samples loaded from disk would be CLAHE'd twice and live frames once — a subtle train/test skew. |
| Detection as a strategy interface | `if backend == "haar"` branches | Pre-cropped benchmark datasets and unit tests need a passthrough path. The interface keeps that out of the pipeline code. |
| SQLite | CSV files, PostgreSQL | Relational constraints and transactions with zero setup; the database is a single file the evaluator can inspect. |
| Synthetic dataset generator | Ship real face images, or download ORL at runtime | Distributing real faces in a public repo is a privacy problem; downloading needs network access an evaluator may not have. Procedural faces with realistic nuisance variation exercise the same code path. |

---

## 8. Non-functional requirements

| # | Requirement | Implementation | Where |
|---|---|---|---|
| 1 | **Performance** | ~15–40 ms per frame on CPU at 200×200; `--stride` skips frames on long videos; detections capped per frame; largest faces prioritised | `detector.py`, `video_source.py` |
| 2 | **Security & privacy** | All processing local, no network calls; parameterised SQL everywhere; only grayscale chips and IDs stored; `purge_older_than()` retention helper | `database.py` |
| 3 | **Reliability** | Typed exception hierarchy; transactional writes with rollback; `KeyboardInterrupt` saves partial sessions; blur rejection screens bad samples | `exceptions.py`, `database.py`, `attendance_service.py` |
| 4 | **Usability** | Single entry point, eight sub-commands, `--help` on each; actionable error messages that name the next command to run; head-less by default | `cli.py` |
| 5 | **Maintainability** | Twelve single-responsibility modules, dataclass config with validation, 58 unit tests, docstrings throughout | project-wide |
| 6 | **Observability** | Rotating file log plus console; every accepted *and* rejected match written to `recognition_log` with its distance | `logger.py`, `database.py` |
| 7 | **Scalability** | Indexed foreign keys; incremental LBPH `update()` avoids full retraining as students are added; per-frame cost independent of cohort size except for the linear model comparison | `database.py`, `recognizer.py` |
| 8 | **Portability** | Pure CPU, two required dependencies, works on Linux/macOS/Windows, no GUI required | `requirements.txt` |

---

## 9. Testing approach

| Layer | What is tested | Examples |
|---|---|---|
| Pre-processing | Shape, dtype and invariants of each stage | fixed output size, idempotent normalisation, blur separation, crop clamping |
| Detection | Both strategies and the factory | passthrough returns the full frame, unknown backend raises, missing cascade raises |
| Dataset | Folder round-trip, loading, splitting | stratified split keeps every class in both halves, too-few-samples raises |
| Recognition | Real accuracy, not just "no exception" | ≥80 % on held-out synthetic samples, threshold rejection produces `Unknown`, model survives save/load |
| Service | Business rules | temporal voting delays marking, unknown face never marked, no duplicate rows |
| Persistence | Constraints | idempotent marking, foreign keys reject unknown students, cascade delete, retention purge |
| Reporting | Output format | table alignment, present/absent counts, CSV header and row count |
| Config & CLI | Validation and wiring | unknown key rejected, YAML round-trip, `init` and `students` exit 0 |

All 58 tests run offline in about three seconds; fixtures generate their own faces, so there
is no dependency on a camera, a network, or a pre-existing dataset.

---

## 10. Known limitations

- **No liveness detection.** A printed photograph or a phone screen held to the camera will
  be recognised. Defeating that needs blink/texture analysis or depth sensing, both out of
  scope here.
- **Frontal pose only.** Haar cascades degrade beyond roughly ±30° yaw; a student looking
  sideways will not be detected at all.
- **Accuracy figures come from synthetic data.** They characterise the benchmark, not a real
  classroom. Real faces vary far more in expression and pose; measure your own cohort with
  `evaluate` before trusting any number.
- **Each frame is classified independently.** There is no identity tracking across frames,
  which is why temporal voting is needed to compensate.
- **Demographic performance is uneven.** Classical recognisers are documented to perform
  differently across skin tones and age groups. This is a property of the method, not a bug
  to be configured away, and it is a reason to audit before deployment.
