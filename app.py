import os
import io
import cv2
import numpy as np
import torch
import logging
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
from simple_lama_inpainting import SimpleLama
from PIL import Image

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Decide device based on CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# ----------------------------
# Global model initializations
# ----------------------------

# For watermark removal
WATERMARK_MODEL_PATH = "yolo_olx_chota.pt"  # Update with your watermark detection model
watermark_yolo_model = YOLO(WATERMARK_MODEL_PATH)
watermark_yolo_model.to(device)
simple_lama = SimpleLama(device=device)


# For number-plate removal
NUMPLATE_MODEL_PATH = "number_plate_blur.pt"  # Update with your number-plate detection model
LOGO_PATH = "logo.png"                         # Update with your logo image path
numplate_yolo_model = YOLO(NUMPLATE_MODEL_PATH)
numplate_yolo_model.to(device)

# ----------------------------
# Helper Functions
# ----------------------------

def order_points(pts):
    """
    Order the four points in the order: top-left, top-right, bottom-right, bottom-left.
    """
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most
    right_most = right_most[np.argsort(right_most[:, 1]), :]
    (tr, br) = right_most
    return np.array([tl, tr, br, bl], dtype=np.float32)

def create_mask(image, model):
    """
    Create a binary mask for watermark removal by drawing filled rectangles 
    for each detection made by the provided YOLO model.
    """
    # Run detection on the PIL image using the provided YOLO model
    results = model(image, conf=0.8, device=device)
    width, height = image.size
    mask = np.zeros((height, width), dtype=np.uint8)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    
    # Ensure binary mask (0 or 255)
    mask = (mask > 0).astype(np.uint8) * 255
    return Image.fromarray(mask).convert('L')

# ----------------------------
# Flask App & Endpoints
# ----------------------------

app = Flask(__name__)
CORS(app)

@app.route("/watermark-removal", methods=["POST"])
def remove_watermark():
    """
    Remove watermark from an image using YOLO for detection and SimpleLama for inpainting.
    Expects an image file under the "image" key in the form data.
    """
    if "image" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    image_file = request.files["image"]
    original_filename = image_file.filename
    if not original_filename:
        return jsonify({"error": "No selected file"}), 400

    allowed_extensions = {'jpg', 'jpeg', 'png'}
    if '.' not in original_filename or original_filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({"error": "Invalid file type"}), 400

    try:
        # Read image as PIL Image
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Create a mask for watermark region(s)
        mask = create_mask(image, watermark_yolo_model)
        mask = mask.resize(image.size, Image.NEAREST)

        # Perform inpainting using SimpleLama
        inpainted_image = simple_lama(image, mask)
    except Exception as e:
        logger.error(f"Error during watermark removal: {e}")
        return jsonify({"error": f"Error during inpainting: {str(e)}"}), 500

    output_stream = io.BytesIO()
    inpainted_image.save(output_stream, format="JPEG")
    output_stream.seek(0)

    return send_file(output_stream, mimetype="image/jpeg", as_attachment=True,
                     download_name=original_filename)




@app.route("/number-plate-removal", methods=["POST"])
def number_plate_removal():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    original_filename = file.filename

    # Read the uploaded image from the request into a NumPy array
    file_bytes = file.read()
    np_img = np.frombuffer(file_bytes, np.uint8)
    original_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if original_image is None:
        return jsonify({"error": "Invalid image"}), 400

    try:
        # Predict quadrilateral points using your model (ensure model is defined elsewhere)
        # The model is expected to return a structure where you can access .obb.xyxyxyxy[0].numpy()
        tensor_data = numplate_yolo_model.predict(source=original_image, conf=0.05)[0].obb.xyxyxyxy[0].numpy()
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

    # Convert to NumPy array with float32 type and reorder points
    quad_points = np.array(tensor_data, dtype=np.float32)
    ordered_quad = order_points(quad_points)

    # Compute bounding box of the ordered quadrilateral
    x_coords = ordered_quad[:, 0]
    y_coords = ordered_quad[:, 1]
    w = int(np.max(x_coords) - np.min(x_coords))
    h = int(np.max(y_coords) - np.min(y_coords))

    # Load the logo image with alpha channel from a fixed path
    logo_path = "logo.png"
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    if logo is None:
        return jsonify({"error": "Logo image not found"}), 500

    # Resize logo to fit within the bounding box while maintaining aspect ratio
    logo_height, logo_width = logo.shape[:2]
    scale = min(w / logo_width, h / logo_height)
    new_width = int(logo_width * scale)
    new_height = int(logo_height * scale)
    resized_logo = cv2.resize(logo, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Ensure the logo image has an alpha channel
    if resized_logo.shape[2] == 3:
        resized_logo = cv2.cvtColor(resized_logo, cv2.COLOR_BGR2BGRA)

    # Create a white background with the same size as the bounding box (RGBA)
    white_image = np.zeros((h, w, 4), dtype=np.uint8)
    white_image[:, :] = [255, 255, 255, 255]

    # Center the resized logo on the white background
    x_center = (w - new_width) // 2
    y_center = (h - new_height) // 2
    white_image[y_center:y_center+new_height, x_center:x_center+new_width] = resized_logo

    # Define source points (corners of the white image)
    src_pts = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)

    # Compute the perspective transform matrix from the white image to the quadrilateral
    M = cv2.getPerspectiveTransform(src_pts, ordered_quad)
    warped_logo = cv2.warpPerspective(white_image, M, (original_image.shape[1], original_image.shape[0]))

    # Blend the warped logo onto the original image
    warped_alpha = warped_logo[:, :, 3]
    original_image_rgba = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
    mask = warped_alpha > 0
    original_image_rgba[mask] = warped_logo[mask]
    result_image = cv2.cvtColor(original_image_rgba, cv2.COLOR_BGRA2BGR)

    # Encode the result image to JPEG in memory
    success, encoded_image = cv2.imencode('.jpg', result_image)
    if not success:
        return jsonify({"error": "Image encoding failed"}), 500
    io_buf = io.BytesIO(encoded_image.tobytes())
    io_buf.seek(0)

    return send_file(io_buf, mimetype="image/jpeg", as_attachment=True,
                     download_name=f"processed_{original_filename}")


if __name__ == "__main__":
        import os
        port = int(os.getenv("PORT", 8080))
        app.run(host="0.0.0.0", port=port)
