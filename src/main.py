import cv2
import os
import torch
from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
INPUT_VIDEO_PATH = "data/input/ch5.mp4"
OUTPUT_VIDEO_PATH = "data/output/output_video.mp4"
ROI_PERSON_PATH = "data/output/roi_pessoas/"
ROI_CAR_PATH = "data/output/roi_carros/"

os.makedirs(ROI_PERSON_PATH, exist_ok=True)
os.makedirs(ROI_CAR_PATH, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

tracked_objects = {"pessoa": {}, "veiculo": {}}

ALERT_LOG_PATH = "logs/alertas.log"
os.makedirs(os.path.dirname(ALERT_LOG_PATH), exist_ok=True)

with open(ALERT_LOG_PATH, "w") as f:
    f.write("Log de Alertas - Pessoas sem Capacete\n")

CLASS_COLORS = {
    "capacete": (0, 255, 0),
    "pessoa": (0, 0, 255),
    "veiculo": (255, 0, 0)
}

start_frame = 0

def resize_frame(frame, max_width=800, max_height=600):
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(frame, (new_width, new_height))

def set_start_frame(val):
    global start_frame
    start_frame = val

def select_start_time(video_path):
    global start_frame

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps

    cv2.namedWindow("Selecione o ponto inicial")
    cv2.createTrackbar("Tempo (s)", "Selecione o ponto inicial", 0, int(duration), set_start_frame)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame * fps)
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(
            frame,
            f"Tempo: {start_frame}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        resized_frame = resize_frame(frame)
        cv2.imshow("Selecione o ponto inicial", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return start_frame

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def save_roi_without_helmet(frame, detected_objects, frame_idx):
    global tracked_objects

    persons = [obj for obj in detected_objects if obj["label"] == "pessoa"]
    helmets = [obj for obj in detected_objects if obj["label"] == "capacete"]

    for person in persons:
        box_person = person["box"]
        is_wearing_helmet = False

        person_head_region = [
            box_person[0],
            box_person[1],
            box_person[2],
            box_person[1] + (box_person[3] - box_person[1]) // 3,
        ]

        for helmet in helmets:
            box_helmet = helmet["box"]

            if (
                box_helmet[0] >= person_head_region[0]
                and box_helmet[1] >= person_head_region[1]
                and box_helmet[2] <= person_head_region[2]
                and box_helmet[3] <= person_head_region[3]
            ):
                is_wearing_helmet = True
                break

        if not is_wearing_helmet:
            is_new_object = True
            for tracked_box in tracked_objects["pessoa"].values():
                if calculate_iou(box_person, tracked_box) > 0.5:
                    is_new_object = False
                    break

            if is_new_object:
                object_id = len(tracked_objects["pessoa"])
                tracked_objects["pessoa"][object_id] = box_person

                roi_person = frame[box_person[1]:box_person[3], box_person[0]:box_person[2]]
                roi_path = os.path.join(ROI_PERSON_PATH, f"frame_{frame_idx}_person_{object_id}.jpg")
                cv2.imwrite(roi_path, roi_person)
                
                log_message = f"[ALERTA] Pessoa SEM capacete detectada no frame {frame_idx}. ROI salva em {roi_path}\n"
                print(log_message)
                with open(ALERT_LOG_PATH, "a") as f:
                    f.write(log_message)

def process_frame(model, frame, frame_idx):
    global tracked_objects
    results = model(frame, device=device)[0]
    detected_objects = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detected_objects.append({
            "label": label,
            "box": [x1, y1, x2, y2],
            "confidence": conf
        })

    save_roi_without_helmet(frame, detected_objects, frame_idx)

    for obj in detected_objects:
        label = obj["label"]
        box = obj["box"]

        if label in tracked_objects:
            is_new_object = True
            for tracked_box in tracked_objects[label].values():
                if calculate_iou(box, tracked_box) > 0.5:
                    is_new_object = False
                    break

            if is_new_object:
                object_id = len(tracked_objects[label])
                tracked_objects[label][object_id] = box

                if label == "veiculo":
                    roi = frame[box[1]:box[3], box[0]:box[2]]
                    roi_path = os.path.join(ROI_CAR_PATH, f"frame_{frame_idx}_car_{object_id}.jpg")
                    cv2.imwrite(roi_path, roi)
                    print(f"[INFO] ROI de veículo salva em {roi_path}")

        color = CLASS_COLORS.get(label, (255, 255, 255))
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(
            frame,
            f"{label} {obj['confidence']:.2f}",
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return frame

def main():
    print("Carregando o modelo...")
    model = YOLO(MODEL_PATH).to(device)

    print("Selecionando o ponto inicial do vídeo...")
    start_time = select_start_time(INPUT_VIDEO_PATH)

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, codec, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

    print("Processando o vídeo...")
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(model, frame, frame_idx)
        out.write(processed_frame)

        resized_frame = resize_frame(processed_frame)
        cv2.imshow("Detecção em Tempo Real", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processamento concluído.")

if __name__ == "__main__":
    main()
