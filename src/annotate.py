from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch

ROOT = Path("/kaggle/input/datasets/niharika41298/yoga-poses-dataset/DATASET")
CLASSES = ["downdog", "goddess", "plank", "tree", "warrior2"]

teacher = YOLO("yolo26n-pose.pt")

def create_annotations(split):
    records = []

    for class_id, class_name in enumerate(CLASSES):
        folder = ROOT / split / class_name

        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            try:
                with Image.open(path) as img:
                    image = np.asarray(img.convert("RGB"))

                h, w = image.shape[:2]

                results = teacher.predict(source=image, imgsz=640, conf=0.25, verbose=False)

                if not results:
                    continue

                result = results[0]

                if result.keypoints is None or result.boxes is None:
                    continue

                if len(result.boxes) == 0:
                    continue

                boxes = result.boxes.xyxy.cpu().numpy()
                areas = ((boxes[:, 2] - boxes[:, 0]) *  (boxes[:, 3] - boxes[:, 1]))
                idx = int(np.argmax(areas))

                kp = result.keypoints.data[idx].cpu().numpy()

                if kp.shape != (17, 3):
                    continue

                # Image-normalised coordinates
                xy = kp[:, :2].copy()
                xy[:, 0] /= w
                xy[:, 1] /= h
                xy = np.clip(xy, 0, 1)

                # Mask unreliable keypoints
                mask = (kp[:, 2] >= 0.3).astype(np.float32)

                records.append({"path": str(path), "label": class_id, "keypoints": xy.astype(np.float32), "mask": mask})

            except Exception as e:
                print("Skipped:", path, e)

    print(split, "usable images:", len(records))
    return records

train_records = create_annotations("TRAIN")
test_records = create_annotations("TEST")

torch.save(train_records, "/kaggle/working/train_annotations.pt")
torch.save(test_records, "/kaggle/working/test_annotations.pt")