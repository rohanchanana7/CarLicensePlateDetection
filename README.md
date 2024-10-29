# Car License Plate Detection

## Overview
This project implements a car license plate detection system using the YOLOv8 architecture, which is a state-of-the-art object detection model. The goal is to automatically detect and localize license plates in images, which can be useful for applications such as automatic number plate recognition (ANPR), parking management, and traffic monitoring.

### Dataset
Due to the large size of the dataset, you can download it from Kaggle:
- [Car Plate Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)

### Key Features
- **Dataset Preparation**: The project includes a script to convert annotations from XML format to a format suitable for YOLO training. It organizes the dataset into training, validation, and testing subsets.
- **Model Training**: Utilizes the YOLOv8 model to train on the prepared dataset, enabling the detection of license plates with high accuracy.
- **Inference**: After training, the model can be used to make predictions on new images, highlighting detected license plates with bounding boxes.
- **Visualization**: Provides visual feedback by displaying images with detected license plates and their confidence scores.

### Scripts
The project includes several Python scripts:
- `load_data.py`             # Loads and prepares data
- `split_data.py`            # Splits data into train, val, and test sets
- `convert_annotations.py`    # Converts annotations to YOLO format
- `train.py`                 # Trains the YOLO model
- `evaluate.py`              # Evaluates and visualizes model predictions
- `helpers.py`               # Helper functions
- `main.py`                  # Main script to run the pipeline
- `datasets.yaml`            # YOLO configuration file

This project aims to provide a comprehensive solution for license plate detection, combining advanced machine learning techniques with practical applications in the real world.
