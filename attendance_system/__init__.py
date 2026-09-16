"""Smart Attendance System - a classical computer-vision attendance pipeline.

Modules
-------
config              configuration objects and YAML loading
logger              logging setup
preprocessing       grayscale / CLAHE / alignment / resize pipeline
detector            Haar-cascade face detection (strategy pattern)
dataset             enrolment and dataset loading
recognizer          LBPH / Eigenfaces / Fisherfaces wrapper
database            SQLite persistence layer (repository pattern)
attendance_service  session orchestration and marking rules
evaluation          offline accuracy / threshold analysis
reporting           CSV + console attendance reports
cli                 command-line interface
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
