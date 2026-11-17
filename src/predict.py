import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Standard COCO skeleton connections (0-based)
SKELETON = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (3, 5), (4, 6)
]

def predict_multitask(image_path):
    image = Image.open(image_path).convert("RGB")

    # Same transform used during training
    x = transform(image).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        output = model(x)

        keypoints = output["keypoints"][0].cpu().numpy()
        probabilities = torch.softmax(output["classes"][0], dim=0)

    class_id = probabilities.argmax().item()
    confidence = probabilities[class_id].item()

    # Draw prediction on original image
    width, height = image.size

    points = keypoints.copy()
    points[:, 0] *= width
    points[:, 1] *= height

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(image)

    for a, b in SKELETON:
        ax.plot(
            [points[a, 0], points[b, 0]],
            [points[a, 1], points[b, 1]],
            linewidth=2
        )

    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=30
    )

    ax.set_title(f"{CLASSES[class_id]} ({confidence:.1%})")
    ax.axis("off")
    plt.show()

    return {"class": CLASSES[class_id], "confidence": confidence, "keypoints": points}

prediction = predict_multitask("/kaggle/input/datasets/niharika41298/yoga-poses-dataset/DATASET/TEST/tree/00000001.jpg")