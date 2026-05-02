from ultralytics import YOLO
import cv2
import numpy as np
import os
import urllib.request
import torch
from torchvision.ops import nms

model = YOLO("best.pt")
image_path = "a1.jpg"  # Update this to your image path

def enhance_for_fruit_detection(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask3 = cv2.inRange(hsv, lower_yellow, upper_yellow)

    apple_mask = cv2.bitwise_or(mask1, mask2)
    apple_mask = cv2.bitwise_or(apple_mask, mask3)

    kernel = np.ones((5,5), np.uint8)
    apple_mask = cv2.dilate(apple_mask, kernel, iterations=1)

    enhanced = img.copy()
    apple_regions = cv2.bitwise_and(img, img, mask=apple_mask)

    enhanced_orig = enhance_image_basic(img)

    alpha = 0.7
    enhanced = cv2.addWeighted(enhanced_orig, alpha, apple_regions, 1-alpha, 0)
    
    return enhanced

def enhance_image_basic(img):

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    a = clahe.apply(a)

    enhanced_lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    return enhanced

def enhance_image_resolution(img_path, target_size=4096):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load image from {img_path}")
        return None, None

    height, width = img.shape[:2]
    
    if max(height, width) < target_size:

        if width > height:
            new_width = target_size
            new_height = int(height * (target_size / width))
        else:
            new_height = target_size
            new_width = int(width * (target_size / height))
            

        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        enhanced = enhance_for_fruit_detection(resized)
        
        enhanced_path = "enhanced_" + os.path.basename(img_path)
        cv2.imwrite(enhanced_path, enhanced)
        print(f"Enhanced image with improved methods to {new_width}x{new_height}")
        return enhanced_path, enhanced
    else:
        print("Image already large enough, using original with fruit enhancement")
        enhanced = enhance_for_fruit_detection(img)
        enhanced_path = "enhanced_" + os.path.basename(img_path)
        cv2.imwrite(enhanced_path, enhanced)
        return enhanced_path, enhanced

def multi_scale_detection(img_path, img, model, scales=[0.5, 0.75, 1.0, 1.25, 1.5]):
    all_boxes = []
    all_confs = []
    all_cls = []

    print(f"Running multi-scale detection with scales: {scales}")
    height, width = img.shape[:2]
    
    for scale in scales:
        new_h, new_w = int(height * scale), int(width * scale)
        
        resized = cv2.resize(img, (new_w, new_h))
        
        temp_path = f"temp_scale_{scale}_{os.path.basename(img_path)}"
        cv2.imwrite(temp_path, resized)
        
        scale_results = model.predict(
            source=temp_path,
            conf=0.15,  
            iou=0.45,
            classes=[47],
            augment=True,
            max_det=300
        )
        
        for r in scale_results:
            if r.boxes.xyxy is not None and len(r.boxes.xyxy) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * (width / new_w)
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * (height / new_h)
                
                all_boxes.append(boxes)
                all_confs.append(r.boxes.conf.cpu().numpy())
                if r.boxes.cls is not None:
                    all_cls.append(r.boxes.cls.cpu().numpy())
        
        os.remove(temp_path)
    
    combined_boxes = []
    combined_confs = []
    
    if all_boxes:
        combined_boxes = np.vstack(all_boxes)
        combined_confs = np.concatenate(all_confs)
        
        print(f"Combined {len(combined_boxes)} detections from all scales")
    
    return combined_boxes, combined_confs

def soft_nms(boxes, scores, iou_threshold=0.5, score_threshold=0.2, sigma=0.5):
    """Improved Soft-NMS implementation for occluded objects"""
    if len(boxes) == 0:
        return []
    
    boxes_tensor = torch.from_numpy(boxes).float()
    scores_tensor = torch.from_numpy(scores).float()
    
    order = torch.argsort(scores_tensor, descending=True)
    keep = []
    
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        
        if order.numel() == 1:
            break
            
        curr_box = boxes_tensor[i]
        other_boxes = boxes_tensor[order[1:]]
        
        xx1 = torch.max(curr_box[0], other_boxes[:, 0])
        yy1 = torch.max(curr_box[1], other_boxes[:, 1])
        xx2 = torch.min(curr_box[2], other_boxes[:, 2])
        yy2 = torch.min(curr_box[3], other_boxes[:, 3])
        
        w = torch.max(torch.zeros(1).to(xx1.device), xx2 - xx1)
        h = torch.max(torch.zeros(1).to(yy1.device), yy2 - yy1)
        intersection = w * h
        
        box_area = lambda box: (box[2] - box[0]) * (box[3] - box[1])
        area1 = box_area(curr_box)
        area2 = torch.tensor([box_area(box) for box in other_boxes])
        
        union = area1 + area2 - intersection
        iou = intersection / union
        
        gaussian_penalty = torch.exp(-(iou * iou) / sigma)
        scores_tensor[order[1:]] *= gaussian_penalty
        
        very_similar_mask = iou > 0.7  
        if very_similar_mask.any():
            scores_tensor[order[1:][very_similar_mask]] = 0
            
        remaining = (scores_tensor[order[1:]] > score_threshold).nonzero().flatten()
        order = order[1:][remaining]
    
    return keep

def cluster_detections(boxes, scores, distance_threshold=50):
    """Group nearby detections that are likely the same apple"""
    if len(boxes) <= 1:
        return boxes, scores
    
    centers = []
    for box in boxes:
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        centers.append((cx, cy))
    
    groups = []
    group_scores = []
    used = set()
    
    for i in range(len(boxes)):
        if i in used:
            continue
            
        group = [i]
        used.add(i)
        
        for j in range(len(boxes)):
            if j in used or i == j:
                continue
                
            dist = np.sqrt((centers[i][0] - centers[j][0])**2 + 
                          (centers[i][1] - centers[j][1])**2)
            
            if dist < distance_threshold:
                group.append(j)
                used.add(j)
        
        groups.append(group)
    
    final_boxes = []
    final_scores = []
    
    for group in groups:
        best_idx = max(group, key=lambda idx: scores[idx])
        final_boxes.append(boxes[best_idx])
        final_scores.append(scores[best_idx])
    
    return np.array(final_boxes), np.array(final_scores)

def sliding_window_detection(image, model, window_size=800, overlap=0.5):
    height, width = image.shape[:2]
    stride = int(window_size * (1 - overlap))
    
    all_boxes = []
    all_confs = []
    
    print(f"Running sliding window detection with window size {window_size} and overlap {overlap}")
    
    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            window = image[y:y+window_size, x:x+window_size]
            
            temp_path = f"temp_window_{x}_{y}.jpg"
            cv2.imwrite(temp_path, window)
            
            results = model.predict(
                source=temp_path, 
                conf=0.15,  
                classes=[47],
                augment=True
            )
            
            for r in results:
                if r.boxes.xyxy is not None and len(r.boxes.xyxy) > 0:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    boxes[:, [0, 2]] += x
                    boxes[:, [1, 3]] += y
                    
                    all_boxes.append(boxes)
                    all_confs.append(r.boxes.conf.cpu().numpy())
            
            os.remove(temp_path)
    
    combined_boxes = []
    combined_confs = []
    
    if all_boxes:
        combined_boxes = np.vstack(all_boxes)
        combined_confs = np.concatenate(all_confs)
        
        print(f"Combined {len(combined_boxes)} detections from sliding windows")
    
    return combined_boxes, combined_confs

print("Starting enhanced apple detection pipeline for occlusion handling...")

processed_path, processed_img = enhance_image_resolution(image_path)

if processed_img is not None:
    print("Running enhanced occlusion detection pipeline...")
    
    height, width = processed_img.shape[:2]
    
    all_detections = []
    all_confidences = []
    
    print("Running standard detection...")
    results = model.predict(
        source=processed_path,
        conf=0.15, 
        iou=0.40,
        imgsz=min(4096, max(height, width)),  
        augment=True,
        max_det=300,
        classes=[47]  
    )
    
    for r in results:
        if r.boxes.xyxy is not None and len(r.boxes.xyxy) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            all_detections.append(boxes)
            all_confidences.append(confs)
    
    multi_boxes, multi_confs = multi_scale_detection(processed_path, processed_img, model)
    if len(multi_boxes) > 0:
        all_detections.append(multi_boxes)
        all_confidences.append(multi_confs)
    
    if max(height, width) > 1500:
        slide_boxes, slide_confs = sliding_window_detection(processed_img, model, overlap=0.5)
        if len(slide_boxes) > 0:
            all_detections.append(slide_boxes)
            all_confidences.append(slide_confs)
    
    final_boxes = []
    final_confs = []
    
    if all_detections:
        final_boxes = np.vstack(all_detections)
        final_confs = np.concatenate(all_confidences)
        
        print(f"Combined total of {len(final_boxes)} detections from all methods")
        
        keep_indices = soft_nms(final_boxes, final_confs)
        
        final_boxes = final_boxes[keep_indices]
        final_confs = final_confs[keep_indices]
        
        print(f"After Soft-NMS: {len(final_boxes)} detections remaining")
        
        final_boxes, final_confs = cluster_detections(final_boxes, final_confs, distance_threshold=50)
        print(f"After clustering: {len(final_boxes)} unique apples detected")
    else:
        print("No detections found in any method")
    
    original_for_drawing = cv2.imread(processed_path)
    
    if len(final_boxes) > 0:
        for i, box in enumerate(final_boxes):
            x1, y1, x2, y2 = map(int, box)
            conf = final_confs[i]
            
            cv2.rectangle(original_for_drawing, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            label = f"Apple: {conf:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(original_for_drawing, 
                          (x1, y1 - text_size[1] - 10), 
                          (x1 + text_size[0], y1), 
                          (0, 0, 0), -1)
            cv2.putText(original_for_drawing, label, (x1, y1 - 7), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    result_image_path = "improved_occlusion_detection_" + os.path.basename(image_path)
    cv2.imwrite(result_image_path, original_for_drawing)
    print(f"Saved improved occlusion detection result to {result_image_path}")
    print(f"Total apples detected: {len(final_boxes)}")
    
    try:
        original_img = cv2.imread(image_path)
        
        if original_img is not None and original_for_drawing is not None:
            target_height = min(800, original_for_drawing.shape[0])
            
            original_aspect = original_img.shape[1] / original_img.shape[0]
            original_new_width = int(target_height * original_aspect)
            resized_original = cv2.resize(original_img, (original_new_width, target_height))
            
            detection_aspect = original_for_drawing.shape[1] / original_for_drawing.shape[0]
            detection_new_width = int(target_height * detection_aspect)
            resized_detection = cv2.resize(original_for_drawing, (detection_new_width, target_height))
            
            label_height = 40
            
            original_with_label = np.zeros((target_height + label_height, original_new_width, 3), dtype=np.uint8)
            original_with_label[label_height:, :, :] = resized_original
            cv2.putText(original_with_label, "Original Image", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            detection_with_label = np.zeros((target_height + label_height, detection_new_width, 3), dtype=np.uint8)
            detection_with_label[label_height:, :, :] = resized_detection
            cv2.putText(detection_with_label, f"Enhanced Detection ({len(final_boxes)} apples)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            comparison = np.hstack((original_with_label, detection_with_label))
            
            cv2.imwrite("improved_occlusion_comparison.jpg", comparison)
            print("Saved full comparison to improved_occlusion_comparison.jpg")
            
            cv2.imshow("Original vs Improved Occlusion Detection", comparison)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.imshow("Apple Detection Result", original_for_drawing)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error creating comparison visualization: {str(e)}")
        cv2.imshow("Apple Detection Result", original_for_drawing)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print("Failed to process image")