from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
import cv2
import math
import cvzone
import os
import gc
from ultralytics import YOLO

app = FastAPI()

# Load YOLO model once
model = YOLO("Weights/best.pt")

# Class names based on your model
classNames = ['0', 'c', 'garbage', 'garbage_bag', 'sampah-detection', 'trash']


@app.post("/process-video/")
async def process_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    # Define input/output paths
    input_path = f"Media/{file.filename}"
    output_path = f"Media/processed_{file.filename}"

    # Save uploaded video in chunks to reduce memory usage
    os.makedirs("Media", exist_ok=True)
    with open(input_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            f.write(chunk)

    # Read input video
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Frame-by-frame processing
    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                if conf > 0.1:
                    cvzone.cornerRect(img, (x1, y1, w, h), t=2)
                    cvzone.putTextRect(
                        img,
                        f'{classNames[cls]} {conf}',
                        (max(0, x1), max(35, y1)),
                        scale=1,
                        thickness=1
                    )

        out.write(img)
        del img
        gc.collect()

    cap.release()
    out.release()

    # Cleanup files after sending
    if background_tasks:
        background_tasks.add_task(os.remove, input_path)
        background_tasks.add_task(os.remove, output_path)

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="processed.mp4"
    )
