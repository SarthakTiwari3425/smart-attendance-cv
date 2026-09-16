# Problem Statement

## 1. The problem

Attendance in most classrooms is still taken by roll call or a signed sheet. For a class of
sixty students a roll call consumes five to ten minutes of every lecture — roughly 10 % of
contact time across a semester — and the record it produces is unreliable: proxy attendance
("buddy punching") is easy, sheets are lost, and the data arrives in a spreadsheet days
later, too late to act on. Card-swipe and fingerprint systems fix some of this but
introduce hardware cost, queues at the door, shared-surface contact, and a card that can
simply be handed to a friend.

A camera at the front of the room already sees who is present. The gap is software that can
turn those frames into a trustworthy attendance record — and that does so without a GPU, a
cloud service, or uploading students' faces to a third party.

## 2. Proposed solution

A command-line attendance system built on classical computer vision:

- **Detect** faces in each frame using a Viola-Jones Haar cascade.
- **Normalise** every face through a fixed pipeline — grayscale, eye-based alignment,
  denoising, CLAHE contrast equalisation, resize to 200×200 — so that lighting and pose
  variation is reduced before comparison.
- **Identify** each face with Local Binary Pattern Histograms, comparing the probe against
  a model trained on enrolled students. Eigenfaces and Fisherfaces are switchable
  alternatives for comparison.
- **Decide** using an explicit distance threshold, so an unenrolled face is reported as
  `Unknown` rather than forced onto the nearest match, and a student is marked only after
  several independent confirmations.
- **Record** the outcome in a SQLite database that structurally prevents duplicate rows,
  and produce console and CSV reports.

The choice of classical CV over a deep-learning face embedder is deliberate: it runs on any
CPU, trains on 15 images per student in under a second, needs no pre-trained weights or
network access, and every stage is inspectable and explainable — which is exactly what a
computer-vision course project should demonstrate, and what an institution deploying
biometrics should be able to audit.

## 3. Scope

**In scope**

- Enrolment of students from photographs, video files, or a live webcam
- Face detection, alignment, normalisation and recognition on CPU
- Rejection of unknown faces via a calibrated distance threshold
- Session-based attendance marking with temporal voting and `PRESENT` / `LATE` status
- Persistent storage of students, sessions, attendance and a full recognition audit log
- Console and CSV reports, including cumulative per-student attendance percentages
- Offline model evaluation: accuracy, per-class metrics, confusion matrix, threshold sweep
- A self-contained synthetic dataset so the system can be demonstrated without a camera
- Full command-line operation with no GUI dependency

**Out of scope**

- Deep-learning face embeddings (FaceNet, ArcFace, dlib ResNet) and GPU training
- Presentation-attack detection (liveness): a printed photograph held to the camera is not
  distinguished from a real face
- Web or mobile front-end, multi-user authentication, institutional LMS/ERP integration
- Recognition of faces at extreme pose (beyond roughly ±30° yaw), heavy occlusion, or in
  very low light — known limitations of frontal Haar cascades and LBPH
- Real-time tracking of identity across frames (each frame is classified independently)

## 4. Target users

| User | Needs | How the system serves them |
|---|---|---|
| **Teacher / lecturer** | Take attendance without losing lecture time; export records | One command starts a session; reports print to the console and export to CSV |
| **Department administrator** | Reliable cumulative attendance for eligibility decisions | `report --summary` gives per-student percentages across all sessions |
| **Student** | Attendance recorded fairly, without queuing or handing over a card | Presence is recorded passively; the audit log makes disputes checkable |
| **Evaluator / instructor of this course** | Verify the project runs and the CV concepts are correctly applied | `demo` runs the whole pipeline offline; `evaluate` reports measured accuracy |
| **Developer extending the system** | Modify or reuse components | Twelve focused modules, strategy-based detection, 58 unit tests |

## 5. High-level features

1. **Enrolment and dataset management** — multi-source capture, blur rejection, per-student
   sample folders, sample counting, student removal.
2. **Detection and pre-processing** — Haar cascade with configurable sensitivity, a
   passthrough strategy for pre-cropped datasets, eye-based alignment, denoising, CLAHE,
   fixed-size normalisation.
3. **Recognition** — LBPH, Eigenfaces and Fisherfaces behind one interface; model
   persistence with its label map; explicit unknown-face rejection.
4. **Attendance management** — sessions, temporal voting, idempotent marking enforced by a
   database constraint, `LATE` status after a configurable grace period.
5. **Reporting** — per-session present/absent tables, CSV export, cumulative attendance
   percentages, session listing.
6. **Evaluation and calibration** — stratified splitting, rank-1 accuracy, per-class
   precision/recall/F1, confusion matrix (CSV and PNG), threshold sweep, automatic
   threshold calibration from the genuine-match distribution.
7. **Operations** — YAML configuration with CLI overrides, rotating log file, full
   recognition audit trail, retention purge, structured error handling with actionable
   messages.

## 6. Success criteria

| Criterion | Target | Status |
|---|---|---|
| Rank-1 accuracy on a held-out split | ≥ 90 % | 95.8 % (LBPH, synthetic benchmark) |
| Unenrolled faces rejected | Not marked present | Verified in the demo and unit tests |
| Duplicate attendance rows | Impossible | Enforced by a `UNIQUE(session_id, student_id)` constraint |
| Runs without GPU, camera or network | Yes | `python main.py demo` |
| Per-frame latency on CPU | < 100 ms at 200×200 | ~15–40 ms measured |
| Automated tests | Comprehensive, offline | 58 pytest tests, ~3 s |
