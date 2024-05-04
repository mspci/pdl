import os
import xml.etree
from numpy import zeros, asarray

import mrcnn.utils
import mrcnn.config
import mrcnn.model


class RemoteSensingDataset(mrcnn.utils.Dataset):

    def load_dataset(self, dataset_dir, is_train=True):
        # Adds information (image ID, image path, and annotation file path) about each image in a dictionary.
        self.add_class("dataset", 1, "airplane")
        self.add_class("dataset", 2, "ship")
        self.add_class("dataset", 3, "storage tank")
        self.add_class("dataset", 4, "baseball diamond")
        self.add_class("dataset", 5, "tennis court")
        self.add_class("dataset", 6, "basketball court")
        self.add_class("dataset", 7, "ground track field")
        self.add_class("dataset", 8, "harbor")
        self.add_class("dataset", 9, "bridge")
        self.add_class("dataset", 10, "vehicle")

        images_dir = dataset_dir + "/images/"
        annotations_dir = dataset_dir + "/annots/"

        for filename in os.listdir(images_dir):
            image_id = filename[:-4]

            # Split all_images: 800 * 0.7 = 560
            # Split positive_images only: 650 * 0.7 = 455
            if is_train and int(image_id) >= 456:
                continue

            if not is_train and int(image_id) < 456:
                continue

            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + ".xml"

            self.add_image(
                "dataset",
                image_id=image_id,
                path=img_path,
                annotation=ann_path,
                class_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            )

    # Loads the binary masks for an image.
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info["annotation"]
        boxes, w, h = self.extract_boxes(path)
        masks = zeros([h, w, len(boxes)], dtype="uint8")

        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]

            # box[4] has the name of the class
            if box[4] == "airplane":
                masks[row_s:row_e, col_s:col_e, i] = 1
                class_ids.append(self.class_names.index("airplane"))
            elif box[4] == "ship":
                masks[row_s:row_e, col_s:col_e, i] = 2
                class_ids.append(self.class_names.index("ship"))
            elif box[4] == "storage tank":
                masks[row_s:row_e, col_s:col_e, i] = 3
                class_ids.append(self.class_names.index("storage tank"))
            elif box[4] == "baseball diamond":
                masks[row_s:row_e, col_s:col_e, i] = 4
                class_ids.append(self.class_names.index("baseball diamond"))
            elif box[4] == "tennis court":
                masks[row_s:row_e, col_s:col_e, i] = 5
                class_ids.append(self.class_names.index("tennis court"))
            elif box[4] == "basketball court":
                masks[row_s:row_e, col_s:col_e, i] = 6
                class_ids.append(self.class_names.index("basketball court"))
            elif box[4] == "ground track field":
                masks[row_s:row_e, col_s:col_e, i] = 7
                class_ids.append(self.class_names.index("ground track field"))
            elif box[4] == "harbor":
                masks[row_s:row_e, col_s:col_e, i] = 8
                class_ids.append(self.class_names.index("harbor"))
            elif box[4] == "bridge":
                masks[row_s:row_e, col_s:col_e, i] = 9
                class_ids.append(self.class_names.index("bridge"))
            elif box[4] == "vehicle":
                masks[row_s:row_e, col_s:col_e, i] = 10
                class_ids.append(self.class_names.index("vehicle"))

        return masks, asarray(class_ids, dtype="int32")

    # A helper method to extract the bounding boxes from the annotation file
    def extract_boxes(self, filename):
        tree = xml.etree.ElementTree.parse(filename)

        root = tree.getroot()

        boxes = list()
        for box in root.findall(".//object"):
            name = box.find("name").text
            xmin = int(box.find("./bndbox/xmin").text)
            ymin = int(box.find("./bndbox/ymin").text)
            xmax = int(box.find("./bndbox/xmax").text)
            ymax = int(box.find("./bndbox/ymax").text)
            coors = [xmin, ymin, xmax, ymax, name]
            boxes.append(coors)

        width = int(root.find(".//size/width").text)
        height = int(root.find(".//size/height").text)
        return boxes, width, height


class RemoteSensingConfig(mrcnn.config.Config):
    NAME = "remote_sensing_cfg"
    BACKBONE = "resnet50"

    # GPU_COUNT = 1
    # IMAGES_PER_GPU = 1

    # 10 classes + background
    NUM_CLASSES = 11
    # number of training steps per epoch
    STEPS_PER_EPOCH = 100


# Train
train_dataset = RemoteSensingDataset()
train_dataset.load_dataset(dataset_dir="NWPU VHR-10 dataset", is_train=True)
train_dataset.prepare()
print("Train: %d" % len(train_dataset.image_ids))

# Validation
validation_dataset = RemoteSensingDataset()
validation_dataset.load_dataset(dataset_dir="NWPU VHR-10 dataset", is_train=False)
validation_dataset.prepare()
print("Test: %d" % len(validation_dataset.image_ids))

# Model Configuration
remote_sensing_config = RemoteSensingConfig()
remote_sensing_config.display()

# ROOT_DIR = os.path.abspath("./")
# DEFAULT_LOGS_DIRS = os.path.join(ROOT_DIR, "logs")

# Build the Mask R-CNN Model Architecture
model = mrcnn.model.MaskRCNN(
    mode="training", model_dir="./", config=remote_sensing_config
)

model.load_weights(
    filepath="mask_rcnn_coco.h5",
    by_name=True,
    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"],
)

model.train(
    train_dataset=train_dataset,
    val_dataset=validation_dataset,
    learning_rate=remote_sensing_config.LEARNING_RATE,
    epochs=1,
    layers="heads",
)

model_path = "remote_sensing_mask_rcnn_trained.h5"
model.keras_model.save_weights(model_path)
