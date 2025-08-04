import os
import shutil
import random

def split_dataset(train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, dest_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # Create output folders
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dest_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, split, 'labels'), exist_ok=True)

    # Helper to collect image-label pairs from a source
    def collect_pairs(images_dir, labels_dir):
        pairs = []
        for f in os.listdir(images_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                label_name = os.path.splitext(f)[0] + ".txt"
                label_path = os.path.join(labels_dir, label_name)
                if os.path.exists(label_path):
                    pairs.append((os.path.join(images_dir, f), label_path))
        return pairs

    # Combine all image-label pairs
    all_pairs = collect_pairs(train_images_dir, train_labels_dir) + collect_pairs(val_images_dir, val_labels_dir)
    random.shuffle(all_pairs)

    # Split
    total = len(all_pairs)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    train_pairs = all_pairs[:train_count]
    val_pairs = all_pairs[train_count:train_count + val_count]
    test_pairs = all_pairs[train_count + val_count:]

    # Copy function
    def copy_pairs(pairs, split):
        for img_path, label_path in pairs:
            img_name = os.path.basename(img_path)
            label_name = os.path.basename(label_path)
            shutil.copy(img_path, os.path.join(dest_dir, split, 'images', img_name))
            shutil.copy(label_path, os.path.join(dest_dir, split, 'labels', label_name))

    # Execute copy
    copy_pairs(train_pairs, 'train')
    copy_pairs(val_pairs, 'val')
    copy_pairs(test_pairs, 'test')

    print(f"✅ Dataset successfully split and copied: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test.")
