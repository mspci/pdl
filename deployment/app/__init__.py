import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import sys

sys.path.append("../")

import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize

# Initialize Flask application
app = Flask(__name__)

# Set the path where uploaded images will be stored temporarily
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure the Mask R-CNN model
# CLASS_NAMES = ["BG", "kangaroo"]
CLASS_NAMES = [
    "BG",
    "airplane",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "vehicle",
]


class SimpleConfig(mrcnn.config.Config):
    NAME = "coco_inference"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = len(CLASS_NAMES)


# Initialize the Mask R-CNN model for inference
model = mrcnn.model.MaskRCNN(
    mode="inference", config=SimpleConfig(), model_dir=os.getcwd()
)

# Load the weights into the model
# model.load_weights(filepath="../Kangaro_mask_rcnn_trained.h5", by_name=True)
# model.load_weights(filepath="../remote_sensing_mask_rcnn_trained_nay.h5", by_name=True)
model.load_weights(
    filepath="../sat_mask_rcnn_trained20.h5",
    by_name=True,
)


# Route to serve the index.html template
@app.route("/")
def index():
    return render_template("index.html", input_image=None, output_image=None)


# Route to handle image uploads and perform object detection
@app.route("/upload", methods=["POST"])
def upload():
    # Check if a file was uploaded
    if "fileInput" not in request.files:
        return redirect(request.url)

    file = request.files["fileInput"]

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == "":
        return redirect(request.url)

    # If the file is valid
    if file:
        # Save the uploaded file to the UPLOAD_FOLDER directory
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Read the uploaded image
        image = cv2.imread(filepath)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform object detection using the Mask R-CNN model
        results = model.detect([image_rgb], verbose=0)

        # Get the results for the first image
        r = results[0]

        # Check if any objects were detected
        # if r["rois"].shape[0] == 0:
        #     return "No objects detected in the image. Please try again with a different image."

        # Render the detected objects on the image
        output_image = mrcnn.visualize.display_instances(
            image=image_rgb,
            boxes=r["rois"],
            masks=r["masks"],
            class_ids=r["class_ids"],
            class_names=CLASS_NAMES,
            scores=r["scores"],
        )

        # Save the output image with detected objects
        output_filename = "output_" + filename
        output_filepath = os.path.join(UPLOAD_FOLDER, output_filename)
        # cv2.imwrite(output_filepath, output_image)

        # Pass the paths of the input and output images to the frontend
        input_image_path = url_for("static", filename="uploads/" + filename)
        output_image_path = url_for("static", filename="uploads/" + output_filename)

        print(
            "Input Image Path:", input_image_path
        )  # Check the path to the input image

        return render_template(
            "index.html", input_image=input_image_path, output_image=output_image_path
        )


# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)
