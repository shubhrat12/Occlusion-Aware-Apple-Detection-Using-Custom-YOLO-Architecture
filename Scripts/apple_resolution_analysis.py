import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from PIL import Image
import yaml
import json
from sklearn.metrics import precision_recall_curve, average_precision_score
from ultralytics import YOLO 

BASE_DIR = r"Resolution"
DATASET_DIR = os.path.join(BASE_DIR, "MinneApple.v6-augmented.yolov5pytorch")
TEST_IMAGES_PATH = os.path.join(DATASET_DIR, "test", "images")
OUTPUT_PATH = os.path.join(BASE_DIR, "apple_resolution_results")
os.makedirs(OUTPUT_PATH, exist_ok=True)

DATA_YAML_PATH = os.path.join(DATASET_DIR, "data.yaml")

RESOLUTIONS = [
    (320, 320),
    (416, 416),
    (512, 512),
    (640, 640),
    (832, 832),
    (1024, 1024),
    (1280, 1280)
]

def load_class_names():
    try:
        with open(DATA_YAML_PATH, 'r') as f:
            data = yaml.safe_load(f)
            return data.get('names', ['apple']) 
    except Exception as e:
        print(f"Warning: Could not load class names from {DATA_YAML_PATH}. Using default 'apple'. Error: {e}")
        return ['apple']

def load_model():
    model_path = r"yolov8x.pt"  # Update this path

    from ultralytics import YOLO
    model = YOLO(model_path)

    model.conf = 0.25
    model.iou = 0.45
    
    print("Loaded MinneApple-specific YOLOv8 model")
    return model

def get_image_files(dir_path):
    return [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

def estimate_apple_distance(detection, img_width, img_height):
    """Estimate distance of apple based on bounding box size relative to image"""
    box = detection['bbox']
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    box_area_ratio = (box_width * box_height) / (img_width * img_height)
    if box_area_ratio < 0.01:
        return "far"
    elif box_area_ratio < 0.05:
        return "medium"
    else:
        return "close"

def load_ground_truth(img_file, label_dir):
    label_file = os.path.splitext(img_file)[0] + '.txt'
    label_path = os.path.join(label_dir, label_file)
    
    ground_truth = []
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    ground_truth.append({
                        "class_id": class_id,
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": width,
                        "height": height
                    })
    
    return ground_truth

def calculate_iou(box1, box2):
    """
    Calculate IoU between box1 and box2
    box format: [x1, y1, x2, y2]
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou

def calculate_metrics(detections, ground_truths, iou_threshold=0.5):
    """Calculate precision, recall, and F1 score"""
    true_positives = 0
    detected = len(detections)
    actual = len(ground_truths)
    
    gt_boxes = []
    for gt in ground_truths:
        x_center, y_center = gt["x_center"], gt["y_center"]
        width, height = gt["width"], gt["height"]
        
        x1 = x_center - width/2
        y1 = y_center - height/2
        x2 = x_center + width/2
        y2 = y_center + height/2
        
        gt_boxes.append([x1, y1, x2, y2])
    
    matched_gt = set()
    
    for det in detections:
        det_box = det["bbox"]
        
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt_box in enumerate(gt_boxes):
            if i in matched_gt:
                continue
                
            iou = calculate_iou(det_box, gt_box)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        if best_iou >= iou_threshold:
            true_positives += 1
            matched_gt.add(best_gt_idx)
    
    precision = true_positives / detected if detected > 0 else 0
    recall = true_positives / actual if actual > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": detected - true_positives,
        "false_negatives": actual - true_positives
    }

def detect_apples_at_resolutions(model, img_path, gt_path=None):
    """Run detection on an image at multiple resolutions"""
    original_img = cv2.imread(img_path)
    if original_img is None:
        print(f"Failed to load image: {img_path}")
        return None
        
    original_height, original_width = original_img.shape[:2]
    img_name = os.path.basename(img_path)
    
    ground_truth = None
    if gt_path:
        ground_truth = load_ground_truth(img_name, gt_path)
    
    image_results = {
        "image_path": img_path,
        "original_size": (original_width, original_height),
        "detections_by_resolution": {},
        "min_successful_resolution": None,
        "metrics_by_resolution": {}
    }
    
    for width, height in sorted(RESOLUTIONS):
        resized_img = cv2.resize(original_img, (width, height))
        
        results = model(resized_img)
        
        apple_detections = []
        
        if len(results) > 0:
            result = results[0]  # Get first result
            boxes = result.boxes
            
            for box in boxes:
                # Get coordinates (YOLOv8 format)
                # For debugging, print the class
                cls = int(box.cls[0])
                # We expect cls to be 0 for apple in the MinneApple dataset
                
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                
                # Convert coordinates to original image scale
                x1 = x1 * original_width / width
                y1 = y1 * original_height / height
                x2 = x2 * original_width / width
                y2 = y2 * original_height / height
                
                apple_detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "class": cls
                })
        
        # Store detection results for this resolution
        image_results["detections_by_resolution"][(width, height)] = {
            "apple_count": len(apple_detections),
            "detections": apple_detections
        }
        
        # Update minimum successful resolution if apples were detected
        if len(apple_detections) > 0 and image_results["min_successful_resolution"] is None:
            image_results["min_successful_resolution"] = (width, height)
        
        # Calculate metrics if ground truth is available
        if ground_truth:
            metrics = calculate_metrics(apple_detections, ground_truth)
            image_results["metrics_by_resolution"][(width, height)] = metrics
    
    # Add distance estimates for detected apples (at highest resolution for reliability)
    if RESOLUTIONS[-1] in image_results["detections_by_resolution"]:
        highest_res = RESOLUTIONS[-1]
        for i, det in enumerate(image_results["detections_by_resolution"][highest_res]["detections"]):
            det["distance"] = estimate_apple_distance(det, original_width, original_height)
    
    return image_results

# Function to analyze and visualize results
def analyze_results(all_results):
    """Generate metrics and visualizations from detection results"""
    print("Analyzing results...")
    
    # Extract data for analysis
    detection_data = []
    for img_result in all_results:
        if img_result is None:
            continue
            
        img_path = img_result["image_path"]
        img_name = os.path.basename(img_path)
        
        for res, res_data in img_result["detections_by_resolution"].items():
            width, height = res
            resolution = width * height  # Total pixels
            detection_data.append({
                "image": img_name,
                "width": width,
                "height": height,
                "resolution": resolution,
                "resolution_text": f"{width}x{height}",
                "apples_detected": res_data["apple_count"],
                "detected": res_data["apple_count"] > 0
            })
            
            # Add metrics if available
            if res in img_result.get("metrics_by_resolution", {}):
                metrics = img_result["metrics_by_resolution"][res]
                for metric_name, metric_value in metrics.items():
                    detection_data[-1][metric_name] = metric_value
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(detection_data)
    
    # 1. Calculate detection rate at each resolution
    detection_rates = df.groupby("resolution_text")["detected"].mean().reset_index()
    detection_rates["resolution"] = detection_rates["resolution_text"].apply(
        lambda x: int(x.split("x")[0]) * int(x.split("x")[1])
    )
    detection_rates = detection_rates.sort_values(by="resolution")
    
    # 2. Calculate minimum successful resolution distribution
    min_res_data = []
    for img_result in all_results:
        if img_result is None or img_result["min_successful_resolution"] is None:
            continue
            
        width, height = img_result["min_successful_resolution"]
        min_res_data.append({
            "image": os.path.basename(img_result["image_path"]),
            "min_width": width,
            "min_height": height,
            "min_resolution": width * height,
            "min_resolution_text": f"{width}x{height}"
        })
    
    min_res_df = pd.DataFrame(min_res_data)
    
    # 3. Analyze by distance (using highest resolution detections)
    distance_data = []
    for img_result in all_results:
        if img_result is None:
            continue
            
        highest_res = RESOLUTIONS[-1]
        if highest_res in img_result["detections_by_resolution"]:
            res_data = img_result["detections_by_resolution"][highest_res]
            
            for det in res_data["detections"]:
                if "distance" in det:
                    distance_data.append({
                        "image": os.path.basename(img_result["image_path"]),
                        "distance": det["distance"],
                        "min_resolution": None
                    })
            
            # Add minimum resolution info
            if img_result["min_successful_resolution"]:
                for i in range(len(distance_data) - len(res_data["detections"]), len(distance_data)):
                    width, height = img_result["min_successful_resolution"]
                    distance_data[i]["min_resolution"] = width * height
    
    distance_df = pd.DataFrame(distance_data)
    
    # 4. Calculate metrics by resolution if available
    metric_columns = ["precision", "recall", "f1"]
    has_metrics = all(col in df.columns for col in metric_columns)
    
    if has_metrics:
        metrics_by_res = df.groupby("resolution_text")[metric_columns].mean().reset_index()
        metrics_by_res["resolution"] = metrics_by_res["resolution_text"].apply(
            lambda x: int(x.split("x")[0]) * int(x.split("x")[1])
        )
        metrics_by_res = metrics_by_res.sort_values(by="resolution")
    
    # Create visualizations
    # 1. Detection rate vs resolution
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=detection_rates, x="resolution_text", y="detected")
    plt.title("Apple Detection Rate by Image Resolution")
    plt.xlabel("Resolution")
    plt.ylabel("Detection Rate")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "detection_rate_by_resolution.png"))
    
    # 2. Distribution of minimum successful resolutions
    if not min_res_df.empty:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=min_res_df, x="min_resolution_text")
        plt.title("Distribution of Minimum Successful Resolution")
        plt.xlabel("Resolution")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, "min_resolution_distribution.png"))
    
    # 3. Minimum resolution by apple distance
    if not distance_df.empty and distance_df["min_resolution"].notna().any():
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=distance_df, x="distance", y="min_resolution")
        plt.title("Minimum Successful Resolution by Apple Distance")
        plt.xlabel("Estimated Apple Distance")
        plt.ylabel("Resolution (pixels)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, "min_resolution_by_distance.png"))
    
    # 4. Precision, recall, F1 by resolution if metrics are available
    if has_metrics:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        sns.lineplot(data=metrics_by_res, x="resolution_text", y="precision")
        plt.title("Precision by Resolution")
        plt.xlabel("")
        plt.ylabel("Precision")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.subplot(3, 1, 2)
        sns.lineplot(data=metrics_by_res, x="resolution_text", y="recall")
        plt.title("Recall by Resolution")
        plt.xlabel("")
        plt.ylabel("Recall")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.subplot(3, 1, 3)
        sns.lineplot(data=metrics_by_res, x="resolution_text", y="f1")
        plt.title("F1 Score by Resolution")
        plt.xlabel("Resolution")
        plt.ylabel("F1 Score")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, "metrics_by_resolution.png"))
    
    # Create summary results
    summary = {
        "detection_rates": detection_rates.to_dict(orient="records"),
        "minimum_resolution_recommendations": {}
    }
    
    # Add metrics to summary if available
    if has_metrics:
        summary["metrics_by_resolution"] = metrics_by_res.to_dict(orient="records")
    
    # Calculate recommended minimum resolutions
    if not distance_df.empty and distance_df["min_resolution"].notna().any():
        for distance in distance_df["distance"].unique():
            subset = distance_df[distance_df["distance"] == distance]
            # Use 90th percentile as recommendation (covers 90% of cases)
            rec_resolution = np.percentile(subset["min_resolution"].dropna(), 90)
            
            # Find closest standard resolution
            closest_res = min([(w*h, (w,h)) for w,h in RESOLUTIONS], 
                             key=lambda x: abs(x[0] - rec_resolution))
            
            summary["minimum_resolution_recommendations"][distance] = {
                "recommended_pixels": int(rec_resolution),
                "closest_standard": f"{closest_res[1][0]}x{closest_res[1][1]}"
            }
    
    # Save summary as JSON
    with open(os.path.join(OUTPUT_PATH, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    
    # Create overall recommendation text
    recommendations = []
    for distance, rec in summary.get("minimum_resolution_recommendations", {}).items():
        recommendations.append(f"- For apples at {distance} distance: {rec['closest_standard']} resolution")
    
    if recommendations:
        with open(os.path.join(OUTPUT_PATH, "recommendations.txt"), "w") as f:
            f.write("# Apple Detection Resolution Recommendations\n\n")
            f.write("Based on our analysis, we recommend the following minimum resolutions:\n\n")
            f.write("\n".join(recommendations))
    
    return summary

# Visualize some sample detections at different resolutions
def visualize_sample_detections(img_path, results, num_samples=3):
    """Create visualization of detection results at different resolutions"""
    print("Creating detection visualizations...")
    
    original_img = cv2.imread(img_path)
    img_name = os.path.basename(img_path)
    
    # Select resolutions to visualize
    if len(RESOLUTIONS) <= num_samples:
        sample_resolutions = RESOLUTIONS
    else:
        # Select low, medium, and high resolutions
        indices = [0, len(RESOLUTIONS) // 2, -1]
        sample_resolutions = [RESOLUTIONS[i] for i in indices]
    
    # Create figure
    fig, axs = plt.subplots(1, len(sample_resolutions), figsize=(15, 5))
    if len(sample_resolutions) == 1:
        axs = [axs]
    
    for i, res in enumerate(sample_resolutions):
        width, height = res
        
        # Resize image
        resized_img = cv2.resize(original_img, (width, height))
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        
        # Get detections for this resolution
        detections = results["detections_by_resolution"].get((width, height), {}).get("detections", [])
        
        # Display image
        axs[i].imshow(resized_img)
        axs[i].set_title(f"Resolution: {width}x{height}\nDetections: {len(detections)}")
        axs[i].axis('off')
        
        # Draw bounding boxes scaled to this resolution
        for det in detections:
            box = det["bbox"]
            x1, y1, x2, y2 = box
            
            # Scale to current resolution
            x1_scaled = x1 * width / original_img.shape[1]
            y1_scaled = y1 * height / original_img.shape[0]
            x2_scaled = x2 * width / original_img.shape[1]
            y2_scaled = y2 * height / original_img.shape[0]
            
            # Draw rectangle
            rect = plt.Rectangle((x1_scaled, y1_scaled), 
                               x2_scaled - x1_scaled, 
                               y2_scaled - y1_scaled,
                               linewidth=2, edgecolor='r', facecolor='none')
            axs[i].add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, f"detection_vis_{os.path.splitext(img_name)[0]}.png"))
    plt.close()

# Main execution function
def main():
    print("Loading YOLOv5x model...")
    model = load_model()
    
    print("Getting image files...")
    test_images = get_image_files(TEST_IMAGES_PATH)
    print(f"Found {len(test_images)} test images")
    
    # Ground truth labels path
    test_labels_path = os.path.join(DATASET_DIR, "test", "labels")
    
    # Process each image
    all_results = []
    for img_file in tqdm(test_images, desc="Processing images"):
        img_path = os.path.join(TEST_IMAGES_PATH, img_file)
        results = detect_apples_at_resolutions(model, img_path, test_labels_path)
        
        if results:
            all_results.append(results)
            
            # Save individual image results
            with open(os.path.join(OUTPUT_PATH, f"{os.path.splitext(img_file)[0]}_results.json"), "w") as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = {}
                for key, value in results.items():
                    if isinstance(value, dict):
                        json_results[key] = {}
                        for k, v in value.items():
                            if isinstance(k, tuple):
                                json_results[key][f"{k[0]}x{k[1]}"] = v
                            else:
                                json_results[key][k] = v
                    elif isinstance(value, tuple):
                        json_results[key] = list(value)
                    else:
                        json_results[key] = value
                        
                json.dump(json_results, f, indent=4)
            
            # Create visualization for first few images
            if len(all_results) <= 5:
                visualize_sample_detections(img_path, results)
    
    # Analyze and visualize results
    if all_results:
        summary = analyze_results(all_results)
        
        print("\nAnalysis complete! Results saved to:", OUTPUT_PATH)
        
        # Print key findings
        print("\nKey findings:")
        for distance, rec in summary.get("minimum_resolution_recommendations", {}).items():
            print(f"- For apples at {distance} distance: Recommended resolution is {rec['closest_standard']}")
    else:
        print("No valid results to analyze. Please check the image paths and model.")

if __name__ == "__main__":
    main()