import os
import json
import requests
from tqdm import tqdm
import zipfile
import shutil

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            f.write(data)

def extract_apple_annotations(coco_annotations_path, output_path):
    with open(coco_annotations_path, 'r') as f:
        coco_data = json.load(f)

    apple_class_id = 53
    new_categories = [cat for cat in coco_data['categories'] if cat['id'] == apple_class_id]
    new_annotations = [ann for ann in coco_data['annotations'] if ann['category_id'] == apple_class_id]
    apple_image_ids = set(ann['image_id'] for ann in new_annotations)
    new_images = [img for img in coco_data['images'] if img['id'] in apple_image_ids]
    apple_coco = {
        'info': coco_data['info'],
        'licenses': coco_data['licenses'],
        'images': new_images,
        'annotations': new_annotations,
        'categories': new_categories
    }
    with open(output_path, 'w') as f:
        json.dump(apple_coco, f)
    
    return apple_image_ids

def download_coco_apple_dataset():
    """Download and prepare COCO dataset with only apple images and annotations"""
    os.makedirs('coco_apple', exist_ok=True)
    os.makedirs('coco_apple/images', exist_ok=True)
    os.makedirs('coco_apple/annotations', exist_ok=True)
    os.makedirs('coco_apple/images/train2017', exist_ok=True)
    os.makedirs('coco_apple/images/val2017', exist_ok=True)
    print("Downloading COCO annotations...")
    train_ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    annotations_zip = "coco_apple/annotations_trainval2017.zip"
    download_file(train_ann_url, annotations_zip)
    print("Extracting annotations...")
    with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
        zip_ref.extractall("coco_apple/")
    print("Processing annotations to keep only apples...")
    train_apple_ids = extract_apple_annotations(
        "coco_apple/annotations/instances_train2017.json", 
        "coco_apple/annotations/instances_train2017_apple.json"
    )
    
    val_apple_ids = extract_apple_annotations(
        "coco_apple/annotations/instances_val2017.json", 
        "coco_apple/annotations/instances_val2017_apple.json"
    )
    with open("coco_apple/train_apple_images.txt", "w") as f:
        for img_id in train_apple_ids:
            f.write(f"http://images.cocodataset.org/train2017/{img_id:012d}.jpg\n")
    
    with open("coco_apple/val_apple_images.txt", "w") as f:
        for img_id in val_apple_ids:
            f.write(f"http://images.cocodataset.org/val2017/{img_id:012d}.jpg\n")

    print("Downloading training images with apples...")
    with open("coco_apple/train_apple_images.txt", "r") as f:
        for i, line in enumerate(tqdm(f.readlines())):
            url = line.strip()
            filename = url.split('/')[-1]
            download_file(url, f"coco_apple/images/train2017/{filename}")
    
    print("Downloading validation images with apples...")
    with open("coco_apple/val_apple_images.txt", "r") as f:
        for i, line in enumerate(tqdm(f.readlines())):
            url = line.strip()
            filename = url.split('/')[-1]
            download_file(url, f"coco_apple/images/val2017/{filename}")

    print("Converting annotations to YOLO format...")
    convert_coco_to_yolo("coco_apple/annotations/instances_train2017_apple.json", 
                         "coco_apple/images/train2017", 
                         "coco_apple/labels/train2017")
    
    convert_coco_to_yolo("coco_apple/annotations/instances_val2017_apple.json", 
                         "coco_apple/images/val2017", 
                         "coco_apple/labels/val2017")
    
    print("Creating YAML config file...")
    create_dataset_yaml()
    
    print("COCO apple dataset preparation complete!")
    return "coco_apple"

def convert_coco_to_yolo(ann_file, img_dir, out_dir):
    """Convert COCO annotations to YOLO format"""
    os.makedirs(out_dir, exist_ok=True)
    
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)

    images = {img['id']: img for img in coco_data['images']}

    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    for img_id, anns in annotations_by_image.items():
        img = images[img_id]
        img_width = img['width']
        img_height = img['height']

        filename = img['file_name']
        basename = os.path.splitext(filename)[0]
        yolo_filename = os.path.join(out_dir, f"{basename}.txt")
        
        with open(yolo_filename, 'w') as f:
            for ann in anns:
                class_id = 0
                x_min, y_min, width, height = ann['bbox']
                x_center = (x_min + width / 2) / img_width
                y_center = (y_min + height / 2) / img_height
                norm_width = width / img_width
                norm_height = height / img_height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")

def create_dataset_yaml():

    yaml_content = """
# COCO Apple dataset
path: coco_apple  # dataset root dir
train: images/train2017  # train images (relative to 'path')
val: images/val2017  # val images (relative to 'path')

# Classes
names:
  0: apple
"""
    
    with open("coco_apple.yaml", "w") as f:
        f.write(yaml_content.strip())

if __name__ == "__main__":
    download_coco_apple_dataset()