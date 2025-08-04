import albumentations as A
import cv2
import os


# Augmenting data for object detection using Albumentations
def augment_data(image_folder, annotation_folder, output_image_folder, output_annotation_folder, num_augmentations_per_image=3):
    """
    Augments images and their corresponding annotations for object detection.

    Args:
        image_folder: Path to the folder containing images.
        annotation_folder: Path to the folder containing annotation files (.txt - YOLO format).
        output_image_folder: Path to save augmented images.
        output_annotation_folder: Path to save augmented annotations.
        num_augmentations_per_image: Number of augmented versions to create for each image.
    """

    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_annotation_folder):
        os.makedirs(output_annotation_folder)

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.2), 
        A.CLAHE(p=0.2),
        A.Blur(blur_limit=3, p=0.2) 
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Warning: Could not read image {filename}. Skipping.")
                continue

            annotation_filename = os.path.splitext(filename)[0] + ".txt"
            annotation_path = os.path.join(annotation_folder, annotation_filename)

            if not os.path.exists(annotation_path):
                print(f"Warning: Annotation file {annotation_filename} not found for {filename}. Skipping.")
                continue

            try:
                with open(annotation_path, 'r') as f:
                    bboxes = []
                    class_labels = []  # Store class labels for each bounding box
                    for line in f:
                        parts = line.strip().split()
                        class_id = parts[0] # int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        bboxes.append([x_center, y_center, width, height, class_id]) # Include class label
                        class_labels.append(class_id) # Append class label

                for i in range(num_augmentations_per_image):
                    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    augmented_image = augmented['image']
                    augmented_bboxes = augmented['bboxes']
                    augmented_class_labels = augmented['class_labels']

                    new_filename = f"{os.path.splitext(filename)[0]}_aug{i}.jpg" # Consistent jpg output
                    output_image_path = os.path.join(output_image_folder, new_filename)
                    cv2.imwrite(output_image_path, augmented_image)

                    new_annotation_filename = f"{os.path.splitext(filename)[0]}_aug{i}.txt"
                    output_annotation_path = os.path.join(output_annotation_folder, new_annotation_filename)

                    with open(output_annotation_path, 'w') as outfile:
                        for bbox, class_label in zip(augmented_bboxes, augmented_class_labels):
                            x_center, y_center, width, height, _ = bbox # Extract without class label
                            outfile.write(f"{class_label} {x_center} {y_center} {width} {height}\n")

            except FileNotFoundError:
                print(f"Warning: Annotation file {annotation_filename} not found. Skipping.")
            except ValueError:
                print(f"Warning: Invalid annotation format in {annotation_filename}. Skipping.")
            except Exception as e:
                print(f"An error occurred processing {filename}: {e}")

