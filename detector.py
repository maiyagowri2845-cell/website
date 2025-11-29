import cv2
import numpy as np
import time
import os
from ultralytics import YOLO 

# Import modules from your project files
from config import (
    CLF_CLASS_NAMES, 
    ENSEMBLE_VOTING_THRESHOLD, OD_IOU_THRESHOLD, 
    OD_MODEL_PATH_B, OD_MODEL_PATH_C, 
    DISEASE_REMEDIES,         
    OD_TO_CLF_MAP_B, OD_TO_CLF_MAP_C, 
    DEBUG_OD_CONF_THRESHOLD 
)
# Update ensemble_hard_voting for the 2-model OD-Only signature
from ensemble import ensemble_hard_voting

# Global variables for the dynamically loaded OD classes and models
OD_CLASS_NAMES_B_DYNAMIC = []
OD_CLASS_NAMES_C_DYNAMIC = []

# Initialize models
od_model_B = None   # Model B (Object Detection - YOLO 1)
od_model_C = None   # Model C (Object Detection - YOLO 2)

# --- 1. Model Loading ---
def load_dynamic_classes(model, model_name):
    """Dynamically loads class names from YOLO model metadata."""
    if model and hasattr(model, 'names') and isinstance(model.names, dict):
        dynamic_names = list(model.names.values())
        print(f"---------------------------------------------------------")
        print(f"{model_name} Class Names dynamically loaded from model metadata ({len(dynamic_names)} classes).")
        print(f"{model_name} Model's Class List: {dynamic_names}")
        print(f"---------------------------------------------------------")
        return dynamic_names
    else:
        print(f"Warning: Failed to load {model_name} class names from model metadata.")
        return []

# --- Load Models (B, C) ---
try:
    # 1a. Load Object Detection Model (Model B) - YOLO 1
    od_model_B = YOLO(OD_MODEL_PATH_B) 
    print(f"Object Detection Model (B) loaded successfully from {OD_MODEL_PATH_B}")
    
    # 1b. Load Object Detection Model (Model C) - YOLO 2
    od_model_C = YOLO(OD_MODEL_PATH_C) 
    print(f"Object Detection Model (C) loaded successfully from {OD_MODEL_PATH_C}")
    
    # --- CRITICAL: Dynamically load OD class names from the model files ---
    OD_CLASS_NAMES_B_DYNAMIC = load_dynamic_classes(od_model_B, "Model B")
    OD_CLASS_NAMES_C_DYNAMIC = load_dynamic_classes(od_model_C, "Model C")

except Exception as e:
    # Fatal error: exit if essential models fail
    print(f"FATAL Error loading essential models (B or C): {e}")
    print("Please ensure your model paths in model_config.py are correct and the files exist.")
    exit() 


# --- 2. OpenCV Setup ---
cap = cv2.VideoCapture(0) # 0 is typically the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set a window name
WINDOW_NAME = 'Real-Time Tomato Disease Detector (YOLO Ensemble B & C)'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

# --- Helper for Multi-line Text (Remedies) ---
def display_multiline_text(img, text, start_y, color=(255, 255, 255), line_height=20):
    """Draws multi-line text (like remedies) on the image."""
    for i, line in enumerate(text.split('\n')):
        # Use simple text rendering to avoid complex text layout issues
        cv2.putText(img, line.strip(), (10, start_y + i * line_height), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img

# --- Helper for OD Prediction ---
def run_od_prediction(model, frame, conf_threshold, model_names):
    """Runs a YOLO model prediction and extracts results."""
    od_labels = []
    od_confidences = []
    od_boxes = []

    # OD_CLASS_NAMES_X_DYNAMIC is the source of truth for the model's output index
    max_od_idx = len(model_names) - 1 

    # Use the provided confidence threshold (conf_threshold) to filter detections
    # Note: We use DEBUG_OD_CONF_THRESHOLD here to capture all possible boxes for aggregation later
    # The actual ENSEMBLE_VOTING_THRESHOLD filter is applied below in the main loop.
    yolo_results = model(
        frame, conf=conf_threshold, iou=OD_IOU_THRESHOLD, imgsz=416, verbose=False, device='cpu' 
    )
    
    if yolo_results and len(yolo_results[0].boxes) > 0:
        
        boxes_obj = yolo_results[0].boxes
        boxes = boxes_obj.xyxy.cpu().numpy().astype(int)
        confidences = boxes_obj.conf.cpu().numpy()
        class_indices = boxes_obj.cls.cpu().numpy().astype(int)

        for box, conf, cls_idx in zip(boxes, confidences, class_indices):
            
            if cls_idx > max_od_idx or cls_idx < 0:
                print(f"Warning: OD Model returned out-of-range index {cls_idx}. Skipping.")
                continue
            
            label = model_names[cls_idx]
            
            od_boxes.append(tuple(box))
            od_confidences.append(conf)
            od_labels.append(label)
            
    return od_labels, od_confidences, od_boxes


# --- 3. Main Detection Loop ---
start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from webcam.")
        break
    
    # These lists now hold the raw output from both models (B and C)
    od_labels_B, od_confidences_B, od_boxes_B = run_od_prediction(
        od_model_B, frame, DEBUG_OD_CONF_THRESHOLD, OD_CLASS_NAMES_B_DYNAMIC 
    )
    od_labels_C, od_confidences_C, od_boxes_C = run_od_prediction(
        od_model_C, frame, DEBUG_OD_CONF_THRESHOLD, OD_CLASS_NAMES_C_DYNAMIC
    )
    
    # --- 4. Ensemble Voting (OD-Only) ---
    
    od_predictions_list = [
        (od_labels_B, od_confidences_B, OD_TO_CLF_MAP_B),
        (od_labels_C, od_confidences_C, OD_TO_CLF_MAP_C),
    ]

    # Get the final winning class and its confidence
    ensembled_label, ensembled_conf = ensemble_hard_voting(
        od_predictions_list, 
    )
    
    # --- 5. Visualization & Display Setup ---
    
    # Calculate FPS
    frame_count += 1
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    # --- A. Determine Status, Color, and Remedy ---
    
    # Default lookup key is for uncertain status 
    remedy_lookup_key = "UNCERTAIN_MONITOR" 
    header_color = (0, 165, 255) # Default Orange/Uncertain
    
    # Check if a high confidence ensemble prediction was reached
    is_confident_detection = "UNCERTAIN" not in ensembled_label and ensembled_conf >= ENSEMBLE_VOTING_THRESHOLD
    
    if is_confident_detection:
        
        # Case 1: High confidence detection (Healthy or Disease)
        remedy_lookup_key = ensembled_label
        
        # Assign color based on result
        if "Healthy" in remedy_lookup_key:
            header_color = (0, 255, 0) # Green
        else:
            header_color = (0, 255, 255) # Yellow/Disease Detected
    
    elif len(od_labels_B) > 0 or len(od_labels_C) > 0:
        # Case 2: Low confidence or Mixed Signal (Some detection, but no clear winner)
        ensembled_label = "UNCERTAIN/MIXED SIGNAL" 
        # remedy_lookup_key remains "UNCERTAIN_MONITOR"
        # header_color remains Orange
        
    else:
        # Case 3: No leaf detected (Default to Healthy status and Green color)
        ensembled_label = "BACKGROUND/NO LEAF DETECTED"
        remedy_lookup_key = "Tomato_Healthy"
        header_color = (0, 255, 0) # Green, since we can't confirm disease
    
    # Get Remedy Info based on the determined key (use .get() for safety)
    remedy_info = DISEASE_REMEDIES.get(remedy_lookup_key, DISEASE_REMEDIES["UNCERTAIN_MONITOR"])
    status_text = remedy_info["status"]
    remedy_text = remedy_info["remedies"]

    # --- B. Calculate and Draw Final Aggregated Bounding Box ---
    contributing_boxes = []

    if is_confident_detection:
        
        # 1. Collect contributing boxes from Model B
        for label, conf, box in zip(od_labels_B, od_confidences_B, od_boxes_B):
            if conf >= ENSEMBLE_VOTING_THRESHOLD:
                mapped_label = OD_TO_CLF_MAP_B.get(label, label)
                if mapped_label == ensembled_label:
                    contributing_boxes.append(box)

        # 2. Collect contributing boxes from Model C
        for label, conf, box in zip(od_labels_C, od_confidences_C, od_boxes_C):
            if conf >= ENSEMBLE_VOTING_THRESHOLD:
                mapped_label = OD_TO_CLF_MAP_C.get(label, label)
                if mapped_label == ensembled_label:
                    contributing_boxes.append(box)

        # 3. Aggregate boxes (Union) and Draw the Final Box
        if contributing_boxes:
            # Calculate union box: min x1, min y1, max x2, max y2
            x1_min = min(b[0] for b in contributing_boxes)
            y1_min = min(b[1] for b in contributing_boxes)
            x2_max = max(b[2] for b in contributing_boxes)
            y2_max = max(b[3] for b in contributing_boxes)
            
            # Draw the thick union box in the final result color
            box_color = header_color 
            cv2.rectangle(frame, (x1_min, y1_min), (x2_max, y2_max), box_color, 4)
            
            # Draw the final ensembled label and confidence above the box
            final_text = f"{remedy_info['label']}: {ensembled_conf:.2f}"
            cv2.putText(frame, final_text, (x1_min, y1_min - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2, cv2.LINE_AA)


    # --- C. Draw Ensemble Result and FPS (Main Header) ---
        
    header_text = f"FPS: {fps:.2f} | Status: {status_text} | Confidence: {ensembled_conf:.2f}"
    # Draw the main header text near the top of the frame
    cv2.putText(frame, header_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, header_color, 2)
    
    # --- D. Draw Remedies (Bottom Display) ---
    frame_h, frame_w, _ = frame.shape
    
    # Calculate required height for the text block
    remedy_lines = remedy_text.split('\n')
    text_block_height = len(remedy_lines) * 20 + 20 # 20px per line + 20px padding
    
    # Draw a black background rectangle for remedies for better readability
    cv2.rectangle(frame, (0, frame_h - text_block_height), (frame_w, frame_h), (0, 0, 0), -1)
    
    # Display multi-line remedy text
    frame = display_multiline_text(frame, remedy_text, frame_h - text_block_height + 20, color=(255, 255, 255))
    
    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 6. Cleanup ---
cap.release()
cv2.destroyAllWindows()