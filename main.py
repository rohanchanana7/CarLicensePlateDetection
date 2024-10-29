# Setup Instructions
# 1. Place your dataset "car-plate-detection" in the same directory as the Python files.
# 2. Run this script (main.py). This will create a "datasets" folder, which contains the 'train', 'val', and 'test' folders.
# 3. Use the path of the "datasets" folder to configure the datasets.yaml file.

import os
import pandas as pd
from load_data import load_data
from split_data import split_data
from convert_annotations import make_split_folder_in_yolo_format
from train import train_model
from evaluate import predict_and_plot
from ultralytics import YOLO

# Define paths and configuration variables
dataset_path = "car-plate-detection"       # Path to the dataset folder
model_path = 'yolov8n.pt'                  # Pretrained YOLO model path
data_yaml = 'datasets.yaml'                # YAML file for YOLO configuration

def main():
    # Step 1: Load and prepare data
    print("Loading data...")
    alldata = load_data(dataset_path)
    print(f"Loaded {len(alldata)} images with annotations.")

    # Step 2: Split data into train, validation, and test sets
    print("Splitting data into train, validation, and test sets...")
    train_df, val_df, test_df = split_data(alldata)
    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")

    # Step 3: Convert annotations to YOLO format and save data splits
    print("Converting annotations and saving data splits in YOLO format...")
    make_split_folder_in_yolo_format("train", train_df)
    make_split_folder_in_yolo_format("val", val_df)
    make_split_folder_in_yolo_format("test", test_df)
    print("Data preparation complete.")

    # Step 4: Train the YOLO model
    print("Training the YOLO model...")
    model = train_model(model_path, data_yaml, epochs=100, batch_size=16, img_size=320)
    print("Model training complete.")

    # Step 5: Evaluate the model
    print("Evaluating the model on test images...")
    test_image_path = test_df.iloc[0].img_path  # Taking the first image in the test set for demonstration
    predict_and_plot(model, test_image_path)

if __name__ == "__main__":
    main()
