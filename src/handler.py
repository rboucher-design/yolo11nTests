import runpod
import io
from PIL import Image as ImagePIL
import base64
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

# Load YOLOv11n model once at startup
model_path = "yolo11n.pt"
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.3,
    device="cuda:0",  # or 'cuda:0'
)

def predict(event):
        # Input parsing
    input_data = event['input']
    image_base64 = input_data.get("image_base64")
    if not image_base64:
        return {"error": "Missing image_base64", "status_code": 400}
    try:
        image_data = base64.b64decode(image_base64)
        image = ImagePIL.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        return {"error": f"Invalid image data: {str(e)}", "status_code": 400}   

 # Run SAHI sliced prediction with YOLOv11n
    try:
        result = get_sliced_prediction(
            image=image,
            detection_model=detection_model,
            slice_height=1024,
            slice_width=1024,
            overlap_height_ratio=0.05,
            overlap_width_ratio=0.05,
            verbose = 2
        )

        # Convert results to JSON
        detections = []
        for det in result.object_prediction_list:
            detections.append({
                "class": det.category.name,
                "score": float(det.score.value),
                "bbox": det.bbox.to_xywh(),   # x, y, w, h
            })

        return {"detections": detections, "status_code": 200}

    except Exception as e:
        return {"error": f"Inference failed: {str(e)}", "status_code": 500}



runpod.serverless.start({"handler": predict})