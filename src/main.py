import time
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, AutoConfig

def load_detectors():
    mtcnn = MTCNN(
        image_size=160,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        keep_all=False,
        device='cpu'
    )
    extractor = AutoFeatureExtractor.from_pretrained("trpakov/vit-face-expression")
    model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression")
    return {"mtcnn": mtcnn, "extractor": extractor, "model": model}

def detect_faces(frame, detectors):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = detectors["mtcnn"].detect(rgb_frame)
    if boxes is not None:
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    return frame

def detect_emotion(frame, detectors):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    timestamp = time.strftime("%H:%M:%S")

    boxes, _ = detectors["mtcnn"].detect(rgb_frame)
    if boxes is not None:
        x_min, y_min, x_max, y_max = map(int, boxes[0])
        h, w, _ = frame.shape
        x_min, x_max = max(0, x_min), min(w, x_max)
        y_min, y_max = max(0, y_min), min(h, y_max)
        if x_max > x_min and y_max > y_min:
            face = rgb_frame[y_min:y_max, x_min:x_max]
            inputs = detectors["extractor"](images=face, return_tensors="pt")
            with torch.no_grad():
                outputs = detectors["model"](**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

            id2label = AutoConfig.from_pretrained("trpakov/vit-face-expression").id2label
            probabilities = probabilities.detach().numpy()[0]
            max_idx = np.argmax(probabilities)
            emotion = id2label[max_idx]

    return emotion, timestamp, frame


def process_video(video_path: str) -> None:
    detectors = load_detectors()

    frame = cv2.imread(video_path)

    frame = detect_faces(frame, detectors)
    emotion, timestamp, frame = detect_emotion(frame, detectors)
    cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Face Emotion Detection', frame)
    print(f"{timestamp}: {emotion}")
    cv2.waitKey(0)

if __name__ == "__main__":
    process_video("./data/angry_face.png")