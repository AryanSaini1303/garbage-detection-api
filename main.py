from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import uvicorn
import cv2
import math
import cvzone
import os
from ultralytics import YOLO

app = FastAPI()

# Load YOLO model
model = YOLO("Weights/best.pt")

# Define class names (update as per your model)
classNames = ['0', 'c', 'garbage', 'garbage_bag', 'sampah-detection', 'trash']

@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    input_path = f"Media/{file.filename}"
    output_path = "output.mp4"

    # Save uploaded file
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Open video
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Prepare output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                if conf > 0.1:
                    cvzone.cornerRect(img, (x1, y1, w, h), t=2)
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                       (max(0, x1), max(35, y1)),
                                       scale=1, thickness=1)

        out.write(img)

    cap.release()
    out.release()

    return FileResponse(output_path, media_type="video/mp4", filename="processed.mp4")
