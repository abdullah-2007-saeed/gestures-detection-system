import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
DATA_FILES = sorted(DATA_DIR.glob("*.json"))

gesture_data = {}
for path in DATA_FILES:
    with path.open("r", encoding="utf-8") as file:
        gesture_data[path.stem] = json.load(file)

fig, axes = plt.subplots(4, 3, figsize=(15, 20))

for row, (gesture_name, records) in enumerate(sorted(gesture_data.items())):
    sample_ids = list(records.keys())[:3]
    
    for col in range(3):
        axis = axes[row, col]
        if col < len(sample_ids):
            sample_id = sample_ids[col]
            record = records[sample_id]
            
            landmarks = np.asarray(record["landmarks"], dtype=float)
            boxes = np.asarray(record["bboxes"], dtype=float)
            
            for hand_landmarks, bbox, label in zip(landmarks, boxes, record["labels"]):
                axis.scatter(hand_landmarks[:, 0], hand_landmarks[:, 1], s=18, label=label)
                x, y, width, height = bbox
                axis.add_patch(
                    plt.Rectangle(
                        (x, y), width, height,
                        fill=False, linewidth=2, edgecolor="black"
                    )
                )
            
            axis.set_title(f"{gesture_name} | Sample {col+1}\nLabels: {', '.join(record['labels'])}")
        
        axis.set_xlim(0, 1)
        axis.set_ylim(1, 0)
        axis.set_aspect("equal")
        if col == 0:
            axis.set_ylabel(f"{gesture_name}\nNormalized y")
        if row == 3:
            axis.set_xlabel("Normalized x")

plt.suptitle("First 3 samples from each gesture file", y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig("gesture_samples.png")
print("Success! Visualization saved as 'gesture_samples.png'. Please open this file to see the images.")
