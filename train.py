from ultralytics import YOLO

def train_model(model_path, data_yaml, epochs=100, batch_size=16, img_size=320):
    # Load the pretrained YOLO model
    model = YOLO(model_path)

    # Train the model with specified parameters
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        cache=True
    )

    # Save the trained model to a specified file
    best_model_path = 'best_license_plate_model.pt'
    model.save(best_model_path)
    print(f"Model saved to {best_model_path}")

    return model  
