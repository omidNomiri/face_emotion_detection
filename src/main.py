import time
from typing import Any, Dict, Optional, Tuple
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from transformers import (
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    AutoConfig,
    PreTrainedModel,
    PretrainedConfig,
    FeatureExtractionMixin)


def load_detectors() -> Dict[str, Any]:
    mtcnn: MTCNN = MTCNN(
        image_size=160,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        keep_all=False,
        device="cpu")

    extractor: FeatureExtractionMixin = AutoFeatureExtractor.from_pretrained("trpakov/vit-face-expression")
    model: PreTrainedModel = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression")

    return {"mtcnn": mtcnn, "extractor": extractor, "model": model}


def detect_faces(frame: np.ndarray, detectors: Dict[str, Any]) -> np.ndarray:
    rgb_frame: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = detectors["mtcnn"].detect(rgb_frame)

    if boxes is not None:
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    return frame


def detect_emotion(frame: np.ndarray, detectors: Dict[str, Any]) -> Tuple[Optional[str], str, np.ndarray]:
    rgb_frame: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    timestamp: str = time.strftime("%H:%M:%S")

    boxes, _ = detectors["mtcnn"].detect(rgb_frame)
    emotion: Optional[str] = None

    if boxes is not None:
        x_min, y_min, x_max, y_max = map(int, boxes[0])
        h, w, _ = frame.shape

        x_min, x_max = max(0, x_min), min(w, x_max)
        y_min, y_max = max(0, y_min), min(h, y_max)

        if x_max > x_min and y_max > y_min:
            face: np.ndarray = rgb_frame[y_min:y_max, x_min:x_max]
            inputs = detectors["extractor"](images=face, return_tensors="pt")

            with torch.no_grad():
                outputs = detectors["model"](**inputs)

            probabilities: torch.Tensor = torch.nn.functional.softmax(outputs.logits, dim=-1)

            config: PretrainedConfig = AutoConfig.from_pretrained("trpakov/vit-face-expression")
            id2label = config.id2label

            probs_np: np.ndarray = probabilities.detach().cpu().numpy()[0]
            max_idx: int = int(np.argmax(probs_np))
            emotion = id2label[max_idx]

    return emotion, timestamp, frame


def process_video(video_path: str) -> None:
    detectors: Dict[str, Any] = load_detectors()
    frame: Optional[np.ndarray] = cv2.imread(video_path)

    if frame is None:
        raise FileNotFoundError(f"Could not read image/video at {video_path}")

    frame = detect_faces(frame, detectors)
    emotion, timestamp, frame = detect_emotion(frame, detectors)

    if emotion:
        cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),2)
        print(f"{timestamp}: {emotion}")
    else:
        print(f"{timestamp}: No face detected")

    cv2.imshow("Face Emotion Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_video("./data/image.png")