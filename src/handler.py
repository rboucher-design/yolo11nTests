import runpod
import sys
import io
from contextlib import redirect_stderr, redirect_stdout
from PIL import Image as ImagePIL, ImageDraw, ImageFont
import base64
from sahi.predict import get_sliced_prediction
from ultralyticsRetina  import UltralyticsDetectionModel

# Load YOLOv11n model once at startup
model_path = "yolo11n.pt"
detection_model = UltralyticsDetectionModel(
    model_path=model_path,
    confidence_threshold=0.26,
    device="cuda:0",  # or 'cuda:0'
)

def draw_boxes(image, coco_predictions):
    """Draw bounding boxes and labels on the image."""
    draw = ImageDraw.Draw(image)
    # Optionally, use a font if you want to customize text
    # font = ImageFont.load_default()  # or use a custom font

    for pred in coco_predictions:
        x, y, w, h = pred['bbox']
        label = f"{pred['category_name']} {pred['score']:.2f}"
        # Draw rectangle
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
        # Draw label background
        draw.rectangle([x, y, x + len(label)*8, y + 16], fill="red")
        # Draw label text
        draw.text((x, y), label, fill="white")

    return image

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
         # Redirect stdout and stderr to capture verbose output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            result = get_sliced_prediction(
                image=image,
                detection_model=detection_model,
                slice_height=2048,
                slice_width=2048,
                overlap_height_ratio=0.25,
                overlap_width_ratio=0.25,
                postprocess_type="NMM",
                postprocess_match_threshold=0.15,
                verbose = 2
            )
         # Get the captured output
        verbose_output = stdout_capture.getvalue() + stderr_capture.getvalue()

        coco_predictions = result.to_coco_predictions(image_id=1)

         # Draw boxes on the image
        annotated_image = draw_boxes(image, coco_predictions)

         # Optionally, return the annotated image as base64
        buffered = io.BytesIO()
        annotated_image.save(buffered, format="JPEG")
        annotated_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")


        return {"detections": coco_predictions, "verbose_output": verbose_output,"annotated_image_base64": annotated_image_base64,"status_code": 200}

    except Exception as e:
        return {"error": f"Inference failed: {str(e)}", "status_code": 500}



runpod.serverless.start({"handler": predict})