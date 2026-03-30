"""
Train a YOLOv11 model to detect FUEL balls for FRC REBUILT shot counting.

Usage:
    python scripts/train_yolo.py [options]

Options:
    --data PATH         Path to dataset YAML file (default: dataset/data.yaml)
    --model MODEL       Base YOLO model to fine-tune (default: yolo11n.pt)
    --epochs N          Number of training epochs (default: 100)
    --imgsz N           Image size for training (default: 640)
    --batch N           Batch size (default: 16, use -1 for auto)
    --device DEVICE     Device: 'mps', 'cuda', 'cpu', or device ID (default: auto)
    --name NAME         Run name for experiment tracking (default: fuel_detector)
    --resume            Resume training from last checkpoint

Prerequisites:
    1. Annotate FUEL balls using Roboflow or another tool.
    2. Export dataset in YOLOv8 format to the `dataset/` directory.
    3. Ensure `dataset/data.yaml` exists and points to train/val image directories.

Expected dataset structure:
    dataset/
    ├── data.yaml          # Class names + paths to train/val splits
    ├── train/
    │   ├── images/        # Training images
    │   └── labels/        # YOLO-format .txt label files
    └── valid/
        ├── images/        # Validation images
        └── labels/        # YOLO-format .txt label files

The trained model weights will be saved to:
    models/fuel_detector/weights/best.pt
"""

import argparse
import os
import sys

# Fix for macOS OpenMP library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# CRITICAL: Import ultralytics/torch BEFORE cv2 on macOS to avoid NSException/Segfaults
from ultralytics import YOLO
import torch


def check_dataset(data_yaml_path):
    """Verify the dataset YAML file exists and is readable."""
    if not os.path.isfile(data_yaml_path):
        print(f"ERROR: Dataset YAML not found at: {data_yaml_path}")
        print()
        print("To create a dataset:")
        print("  1. Run: python scripts/extract_frames.py assets/")
        print("  2. Upload frames to Roboflow and annotate FUEL balls")
        print("  3. Export in YOLOv8 format to the 'dataset/' directory")
        print("  4. Ensure 'dataset/data.yaml' exists")
        sys.exit(1)

    import yaml  # bundled with ultralytics
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    required_keys = ["train", "val", "nc", "names"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        print(f"ERROR: data.yaml is missing required keys: {missing}")
        print(f"  Expected keys: {required_keys}")
        sys.exit(1)

    print(f"Dataset config: {data_yaml_path}")
    print(f"  Classes ({data['nc']}): {data['names']}")
    print(f"  Train path: {data['train']}")
    print(f"  Val path:   {data['val']}")
    print()

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 to detect FUEL balls for FRC REBUILT."
    )
    parser.add_argument("--data", default="dataset/data.yaml", help="Dataset YAML path")
    parser.add_argument("--model", default="yolo11n.pt", help="Base YOLO model")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (-1 for auto)")
    parser.add_argument("--device", default=None, help="Device (mps, cuda, cpu)")
    parser.add_argument("--name", default="fuel_detector", help="Run name")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    # Check dataset before importing heavy dependencies
    check_dataset(args.data)

    # Import ultralytics (triggers PyTorch import)
    print("Loading YOLO...")

    # Auto-detect device
    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            print("Using Apple Silicon MPS")
        else:
            device = "cpu"
            print("Using CPU (training will be slower)")
    print()

    # Load model
    if args.resume:
        # Resume from last checkpoint
        checkpoint = os.path.join("models", args.name, "weights", "last.pt")
        if not os.path.isfile(checkpoint):
            print(f"ERROR: No checkpoint found at {checkpoint}")
            sys.exit(1)
        print(f"Resuming training from: {checkpoint}")
        model = YOLO(checkpoint)
    else:
        print(f"Fine-tuning from pre-trained model: {args.model}")
        model = YOLO(args.model)

    # Train
    print(f"\nStarting training...")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch size: {args.batch}")
    print(f"  Device:     {device}")
    print()

    results = model.train(
        data=os.path.abspath(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name=args.name,
        project="models",
        exist_ok=True,
        # Data augmentation tuned for small ball detection
        mosaic=1.0,         # mosaic augmentation
        mixup=0.1,          # light mixup
        scale=0.5,          # random scale
        fliplr=0.5,         # horizontal flip
        flipud=0.0,         # no vertical flip (balls don't appear upside down)
        hsv_h=0.015,        # hue augmentation (small — balls are consistently yellow)
        hsv_s=0.4,          # saturation augmentation (handle lighting variance)
        hsv_v=0.4,          # value augmentation (handle exposure variance)
        degrees=0.0,        # no rotation
        translate=0.1,      # small translation
        # Training hyperparameters
        patience=20,         # early stopping patience
        save_period=10,      # save checkpoint every 10 epochs
        val=True,            # validate during training
        plots=True,          # generate training plots
        verbose=True,
    )

    # Summary
    best_weights = os.path.join("models", args.name, "weights", "best.pt")
    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"  Best weights: {os.path.abspath(best_weights)}")
    print(f"  Results dir:  {os.path.abspath(os.path.join('models', args.name))}")
    print()
    print("Next steps:")
    print(f"  python scripts/shot_counter_ml.py --model {best_weights} assets/<video>")


if __name__ == "__main__":
    main()
