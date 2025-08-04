import shutil
import os
from pathlib import Path


def main():
    from preprocessing.config_conversion import (
        convert_coco_to_yolo,
        convert_masks_to_yolo,
        convert_polygons_to_yolo,
        convert_json_to_yolo,
        convert_voc_to_yolo,
    )
    from preprocessing.data_augmentation import augment_data
    from preprocessing.data_conversion import split_dataset

    root_dir = Path().cwd().parent
    data_dir = root_dir / "data"
    original_datasets_dir = data_dir / "original_datasets"
    preprocessed_datasets_dir = data_dir / "preprocessed_datasets"

    # --- Dataset(s) for data augmentation ---
    all_datasets = [
        "TomatoOD",
        "StrawDI_Db1",
        "LaboroTomato",
        "Purple_grapes",
        "Strawberry_dataset_for_object_detection",
        "Cherry_Tomato_Dataset",
        "Grapes_dataset",
        "Green_grapes",
        "Strawberry_DS",
        "TomatoPlantfactoryDataset",
        "wgisd",
        "Strawberry_Robotflow",
    ]

    # Define preprocessing actions per dataset
    datasets_to_preprocess = {
        "TomatoOD": lambda: convert_coco_to_yolo(
            str(original_datasets_dir / "TomatoOD/tomatOD_train.json"),
            str(original_datasets_dir / "TomatoOD/yolo_output_train_txt"),
        ),
        "StrawDI_Db1": lambda: convert_masks_to_yolo(
            str(original_datasets_dir / "StrawDI_Db1/masks"),
            str(preprocessed_datasets_dir / "StrawDI_Db1/yolo_labels"),
            class_id=0,
        ),
        "LaboroTomato": lambda: convert_polygons_to_yolo(
            str(original_datasets_dir / "LaboroTomato/annotations"),
            str(preprocessed_datasets_dir / "LaboroTomato/yolo_labels"),
            {
                "b_green": 0,
                "l_green": 1,
                "l_fully_ripened": 2,
                "b_half_ripened": 3,
                "l_half_ripened": 4,
                "b_fully_ripened": 5,
            },
        ),
        "Strawberry_dataset_for_object_detection": lambda: convert_json_to_yolo(
            str(original_datasets_dir / "Strawberry/annotations"),
            str(preprocessed_datasets_dir / "Strawberry/yolo_labels"),
            {"ripe": 0, "peduncle": 1, "unripe": 2},
        ),
        "Purple_grapes": lambda: convert_voc_to_yolo(
            str(original_datasets_dir / "Purple_grapes/annotations"),
            str(preprocessed_datasets_dir / "Purple_grapes/yolo_labels"),
            {"purple_grape": 0},
        ),
    }

    # --- Execute label conversion and data splitting ---
    for dataset_name, preprocess_func in datasets_to_preprocess.items():
        print(f"Preprocessing: {dataset_name}")
        preprocess_func()

        if dataset_name == "TomatoOD":
            dataset_path = preprocessed_datasets_dir / "TomatoOD"
            split_dataset_path = preprocessed_datasets_dir / "TomatoOD_split"

            print(f"Splitting dataset: {dataset_name}")
            split_dataset(
                train_images_dir=str(dataset_path / "train"),
                train_labels_dir=str(dataset_path / "labels"),
                val_images_dir=str(dataset_path / "test"),
                val_labels_dir=str(dataset_path / "labels"),
                dest_dir=str(split_dataset_path),
            )
            # remove the original dataset_name folder and rename the split dataset folder
            # with the original name

            shutil.rmtree(dataset_path)
            os.rename(str(split_dataset_path), str(dataset_path))

    # --- Execute augmentation on selected datasets ---
    for dataset_name in all_datasets:
        dataset_path = preprocessed_datasets_dir / dataset_name

        print(f"Augmenting dataset: {dataset_name}")
        augment_data(
            image_folder=str(dataset_path / "train" / "images"),
            annotation_folder=str(dataset_path / "train" / "labels"),
            output_image_folder=str(dataset_path / "train_aug" / "images"),
            output_annotation_folder=str(dataset_path / "train_aug" / "labels"),
            num_augmentations_per_image=2,
        )


if __name__ == "__main__":
    main()
