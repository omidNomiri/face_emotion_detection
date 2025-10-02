import cv2
import mediapipe as mp

def load_detectors():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return {"face_mesh": face_mesh}


def detect_faces(frame, detectors):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detectors["face_mesh"].process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = [(int(p.x * w), int(p.y * h))
                         for p in face_landmarks.landmark]

            x_min = min([p[0] for p in landmarks])
            x_max = max([p[0] for p in landmarks])
            y_min = min([p[1] for p in landmarks])
            y_max = max([p[1] for p in landmarks])
            cv2.rectangle(frame, (x_min, y_min),
                          (x_max, y_max), (0, 255, 0), 2)

            # Left eye: indices 33, 160, 158, 133, 153, 144
            # Right eye: indices 362, 385, 387, 263, 373, 380
            left_eye = [
                landmarks[33],
                landmarks[160],
                landmarks[158],
                landmarks[133],
                landmarks[153],
                landmarks[144]
            ]
            right_eye = [
                landmarks[362],
                landmarks[385],
                landmarks[387],
                landmarks[263],
                landmarks[373],
                landmarks[380]
            ]

            for (x, y) in left_eye + right_eye:
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    return frame


def process_video(video_path: str) -> None:
    detectors = load_detectors()
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_faces(frame, detectors)
        cv2.imshow('Improved Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detectors["face_mesh"].close()


if __name__ == "__main__":
    process_video("./data/sample_video.mp4")
