# AI-Powered Visual Data Analytics and Business Intelligence Platform

## Project Overview

This project is an AI-powered visual data analytics platform developed using Python, YOLOv8 and Streamlit.

The project contains two main modules:

### 1. YOLOv8 Object Detection

The YOLOv8 model is used to detect objects in images using bounding boxes and confidence scores.

### 2. PPE Detection

A trained YOLOv8 model is used to detect Personal Protective Equipment (PPE) from construction worker images.

The detected objects are displayed with bounding boxes and confidence scores.

## Technologies Used

- Python
- YOLOv8
- Ultralytics
- Streamlit
- NumPy
- Pillow

## Project Files

- `app.py` – Streamlit application for PPE detection
- `milestone1.py` – Project milestone code
- `best.pt` – Trained PPE detection model
- `yolov8n.pt` – YOLOv8 nano model
- `.gitignore` – Specifies files and folders excluded from Git

## How to Run

Install the required packages:

```bash
pip install streamlit ultralytics pillow numpy