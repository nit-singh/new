import os
from pathlib import Path
from ultralytics import YOLO

# Set paths
DATASET_DIR = Path("dataset")
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
CLASSES_FILE = TRAIN_DIR / "classes.txt"

def read_classes():
    """Read class names from classes.txt"""
    with open(CLASSES_FILE, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    return classes

def create_yaml_config():
    """Create YOLOv8 dataset configuration file"""
    classes = read_classes()
    num_classes = len(classes)
    
    # Get absolute paths
    train_images = str(Path(TRAIN_DIR / "images").absolute())
    train_labels = str(Path(TRAIN_DIR / "labels").absolute())
    
    yaml_content = f"""# YOLOv8 Dataset Configuration
path: {str(DATASET_DIR.absolute())}
train: {train_images}
val: {train_images}  # Using train for validation (you can create a separate val folder later)

# Classes
nc: {num_classes}
names:
"""
    for idx, class_name in enumerate(classes):
        yaml_content += f"  {idx}: {class_name}\n"
    
    yaml_path = DATASET_DIR / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✓ Created dataset configuration: {yaml_path}")
    return yaml_path, classes

def train_model(yaml_path, epochs=30, imgsz=480):
    """Train YOLOv8 model"""
    print("\n🚀 Starting YOLOv8 training...")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {imgsz}")
    
    # Initialize YOLOv8 model (using nano version for faster training, change to 'yolov8s.pt', 'yolov8m.pt', etc. for better accuracy)
    model = YOLO('yolov8n.pt')  # You can change to yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    
    # Train the model
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=16,  # Adjust based on your GPU memory
        name='car_detection',
        project='runs/detect'
    )
    
    print("\n✓ Training completed!")
    return model

def test_model(model_path, test_dir):
    """Test the trained model on test images"""
    print(f"\n🧪 Testing model on test images from: {test_dir}")
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Get all test images
    test_images = list(Path(test_dir).glob("*.jpg")) + list(Path(test_dir).glob("*.png"))
    
    if not test_images:
        print(f"⚠ No test images found in {test_dir}")
        return
    
    print(f"   Found {len(test_images)} test images")
    
    # Run inference on test images
    results = model.predict(
        source=str(test_dir),
        save=True,
        save_txt=True,
        conf=0.25,  # Confidence threshold
        project='runs/detect',
        name='test_predictions'
    )
    
    print(f"\n✓ Testing completed! Results saved in runs/detect/test_predictions/")
    
    # Print summary
    for i, result in enumerate(results):
        if result.boxes is not None:
            num_detections = len(result.boxes)
            print(f"   Image {i+1}: {num_detections} detections")
        else:
            print(f"   Image {i+1}: No detections")

def main():
    """Main function"""
    print("=" * 60)
    print("YOLOv8 Custom Dataset Training & Testing")
    print("=" * 60)
    
    # Check if dataset exists
    if not DATASET_DIR.exists():
        print(f"❌ Error: Dataset folder '{DATASET_DIR}' not found!")
        return
    
    if not TRAIN_DIR.exists():
        print(f"❌ Error: Train folder '{TRAIN_DIR}' not found!")
        return
    
    if not TEST_DIR.exists():
        print(f"❌ Error: Test folder '{TEST_DIR}' not found!")
        return
    
    # Step 1: Create YAML configuration
    print("\n[Step 1/3] Creating dataset configuration...")
    yaml_path, classes = create_yaml_config()
    print(f"   Found {len(classes)} classes")
    
    # Step 2: Train the model
    print("\n[Step 2/3] Training model...")
    model = train_model(yaml_path, epochs=50, imgsz=640)
    
    # Step 3: Test the model
    print("\n[Step 3/3] Testing model...")
    # Find the best model from training
    best_model_path = Path("runs/detect/car_detection/weights/best.pt")
    if best_model_path.exists():
        test_model(str(best_model_path), TEST_DIR)
    else:
        print(f"⚠ Best model not found at {best_model_path}")
        print("   You can manually test using: model.predict(source='dataset/test')")
    
    print("\n" + "=" * 60)
    print("✅ All done!")
    print("=" * 60)
    print(f"\nTrained model saved at: runs/detect/car_detection/weights/best.pt")
    print(f"Test results saved at: runs/detect/test_predictions/")

if __name__ == "__main__":
    main()

