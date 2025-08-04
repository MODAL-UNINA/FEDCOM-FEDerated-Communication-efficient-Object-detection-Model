import xml.etree.ElementTree as ET
from collections import defaultdict
import json
import os
import cv2
import glob


def convert_voc_to_yolo(input_folder, output_folder, class_mapping):
    """
    Converts all VOC XML annotation files in a folder to YOLO .txt format.
    
    Parameters:
        input_folder (str): Path to folder containing .xml files.
        output_folder (str): Path to save YOLO-format .txt files.
        class_mapping (dict): Dictionary mapping class names to YOLO class IDs.
    """
    
    def convert_to_yolo_bbox(xmin, ymin, xmax, ymax, img_width, img_height):
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        return x_center, y_center, width, height

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.endswith(".xml"):
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".xml", ".txt"))

        tree = ET.parse(input_path)
        root = tree.getroot()

        img_width = int(root.find("size/width").text)
        img_height = int(root.find("size/height").text)

        yolo_labels = []

        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in class_mapping:
                continue  # Skip unknown classes

            class_id = class_mapping[class_name]
            bbox = obj.find("bndbox")
            xmin, ymin, xmax, ymax = map(int, [
                bbox.find("xmin").text,
                bbox.find("ymin").text,
                bbox.find("xmax").text,
                bbox.find("ymax").text
            ])
            x_center, y_center, width, height = convert_to_yolo_bbox(xmin, ymin, xmax, ymax, img_width, img_height)
            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        with open(output_path, "w") as f:
            f.writelines(yolo_labels)

    print("✅ VOC-to-YOLO conversion completed!")



def convert_masks_to_yolo(masks_dir, output_dir, class_id=0):
    """
    Converts binary mask PNG files into YOLO-format annotation .txt files.
    """
    os.makedirs(output_dir, exist_ok=True)

    def process_mask(mask_path):
        image_name = os.path.splitext(os.path.basename(mask_path))[0]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None or cv2.countNonZero(mask) == 0:
            print(f"⚠️ Empty or invalid mask: {mask_path}")
            return
        
        img_h, img_w = mask.shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        yolo_lines = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Normalize coordinates for YOLO format
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        if yolo_lines:
            txt_path = os.path.join(output_dir, f"{image_name}.txt")
            with open(txt_path, "w") as f:
                f.write("\n".join(yolo_lines))
            print(f"✅ Saved: {txt_path}")
        else:
            print(f"⚠️ No objects found in: {mask_path}")

    # === Execution ===
    mask_files = glob.glob(os.path.join(masks_dir, "*.png"))
    print(f"Found {len(mask_files)} mask(s)")

    for mask_file in mask_files:
        process_mask(mask_file)



def convert_polygons_to_yolo(annotations_dir, output_dir, class_map):
    """
    Converts polygon-based JSON annotations into YOLO-format .txt files.
    """

    os.makedirs(output_dir, exist_ok=True)

    def polygon_to_bbox(points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return min(xs), min(ys), max(xs), max(ys)

    def convert_to_yolo_bbox(x_min, y_min, x_max, y_max, img_w, img_h):
        x_center = (x_min + x_max) / 2 / img_w
        y_center = (y_min + y_max) / 2 / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h
        return x_center, y_center, w, h

    # === MAIN PROCESSING ===
    for file in os.listdir(annotations_dir):
        if not file.endswith(".json"):
            continue

        json_path = os.path.join(annotations_dir, file)
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract image dimensions
        img_w = data["size"]["width"]
        img_h = data["size"]["height"]

        # Get filename base (e.g. IMG_0983.jpg.json → IMG_0983)
        filename_with_ext = os.path.splitext(file)[0]
        filename = os.path.splitext(filename_with_ext)[0]

        yolo_lines = []

        for obj in data.get("objects", []):
            if obj.get("geometryType") != "polygon":
                continue

            class_title = obj.get("classTitle")
            class_id = class_map.get(class_title)
            if class_id is None:
                print(f"⚠️ Unknown class '{class_title}' in file: {file}")
                continue

            exterior = obj["points"]["exterior"]
            x_min, y_min, x_max, y_max = polygon_to_bbox(exterior)
            x_center, y_center, w, h = convert_to_yolo_bbox(x_min, y_min, x_max, y_max, img_w, img_h)

            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            yolo_lines.append(yolo_line)

        if yolo_lines:
            output_path = os.path.join(output_dir, f"{filename}.txt")
            with open(output_path, "w") as f:
                f.write("\n".join(yolo_lines))
            print(f"✅ Converted {file} → {filename}.txt")
        else:
            print(f"⚠️ No valid objects in {file}, skipping.")



def convert_json_to_yolo(annotations_dir, output_dir, class_map):
    """
    Converts rectangle-based JSON annotations into YOLO-format .txt files.
    """

    os.makedirs(output_dir, exist_ok=True)

    def convert_to_yolo_bbox(x_min, y_min, x_max, y_max, img_w, img_h):
        x_center = (x_min + x_max) / 2 / img_w
        y_center = (y_min + y_max) / 2 / img_h
        width = (x_max - x_min) / img_w
        height = (y_max - y_min) / img_h
        return x_center, y_center, width, height

    def strip_image_extensions(filename):
        name = filename
        for ext in [".jpg.png.json", ".jpg.json"]:
            if name.endswith(ext):
                name = name.replace(ext, "")
                break
        return name

    for file in os.listdir(annotations_dir):
        if not file.endswith(".json"):
            continue

        json_path = os.path.join(annotations_dir, file)
        with open(json_path, "r") as f:
            data = json.load(f)

        img_w = data["size"]["width"]
        img_h = data["size"]["height"]

        base_name = strip_image_extensions(file)
        txt_path = os.path.join(output_dir, f"{base_name}.txt")

        yolo_lines = []

        for obj in data.get("objects", []):
            if obj.get("geometryType") != "rectangle":
                continue

            class_title = obj.get("classTitle")
            class_id = class_map.get(class_title)
            if class_id is None:
                print(f"⚠️ Unknown class '{class_title}' in {file}")
                continue

            exterior = obj["points"]["exterior"]
            (x_min, y_min), (x_max, y_max) = exterior

            x_center, y_center, w, h = convert_to_yolo_bbox(x_min, y_min, x_max, y_max, img_w, img_h)
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        if yolo_lines:
            with open(txt_path, "w") as f:
                f.write("\n".join(yolo_lines))
            print(f"✅ Converted {file} → {base_name}.txt")
        else:
            print(f"⚠️ No rectangle objects in {file}, skipping.")



def convert_coco_to_yolo(json_path, output_dir):
    """
    Convert a COCO-format annotation JSON to YOLO annotation text files.
    Each image gets its own .txt file with YOLO-formatted lines.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Parse categories into a mapping: category_id -> yolo_class_id
    categories = data.get("categories", [])
    category_map = {}
    class_names = []

    for cat in categories:
        # Handle case where 'name' is a list (like in your JSON)
        if isinstance(cat["name"], list):
            for idx, name in enumerate(cat["name"]):
                category_map[idx + 1] = idx  # category_id starts from 1
                class_names.append(name)
        else:
            category_map[cat["id"]] = len(class_names)
            class_names.append(cat["name"])

    # Map image_id -> {file_name, width, height}
    image_info = {img["id"]: img for img in data["images"]}

    # Group annotations per image_id
    annotations_per_image = defaultdict(list)
    for ann in data["annotations"]:
        annotations_per_image[ann["image_id"]].append(ann)

    # Process each image's annotations
    for image_id, anns in annotations_per_image.items():
        img = image_info[image_id]
        file_name = img["file_name"]
        width, height = img["width"], img["height"]

        base_name = os.path.splitext(file_name)[0]
        txt_path = os.path.join(output_dir, f"{base_name}.txt")

        lines = []
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in category_map:
                continue
            class_id = category_map[cat_id]
            x, y, w, h = ann["bbox"]
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        with open(txt_path, "w") as f:
            f.write("\n".join(lines))

        print(f"Saved: {txt_path}")

