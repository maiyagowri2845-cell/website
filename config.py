CLF_CLASS_NAMES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_Healthy"
]

# --- Configuration for Model B (Object Detection/YOLO 1) ---
OD_MODEL_PATH_B = "best.pt"
# Model B Class Names are loaded dynamically. OD_TO_CLF_MAP_B uses the unified CLF names.
OD_TO_CLF_MAP_B = {
    "Bacterial Spot": "Tomato_Bacterial_spot", 
    "Early_Blight": "Tomato_Early_blight",
    "Healthy": "Tomato_Healthy",
    "Late_blight": "Tomato_Late_blight",
    "Leaf Mold": "Tomato_Leaf_Mold",
    "Target_Spot": "Tomato_Target_Spot",
    "black spot": "Tomato_Septoria_leaf_spot" # Assuming 'black spot' maps to Septoria
    # If the OD model has other classes, they won't vote unless added here
}

# --- Configuration for Model C (Object Detection/YOLO 2) ---
OD_MODEL_PATH_C = "best2.pt" 
# Model C Class Names are loaded dynamically. OD_TO_CLF_MAP_C filters non-tomato classes.
OD_TO_CLF_MAP_C = {
    "Tomato Early blight leaf": "Tomato_Early_blight",
    "Tomato Septoria leaf spot": "Tomato_Septoria_leaf_spot",
    "Tomato leaf": "Tomato_Healthy", # Assume general 'Tomato leaf' is healthy
    "Tomato leaf bacterial spot": "Tomato_Bacterial_spot",
    "Tomato leaf late blight": "Tomato_Late_blight",
    "Tomato leaf mosaic virus": "Tomato_Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato_Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato_Spider_mites_Two_spotted_spider_mite",
    # All other Model C classes (Apple, Corn, etc.) are intentionally omitted and will not vote.
}

# --- General Ensemble & Detection Parameters ---
# Minimum confidence score for a prediction to be included in the ensemble vote.
ENSEMBLE_VOTING_THRESHOLD = 0.75 
# IOU threshold for Object Detection models
OD_IOU_THRESHOLD = 0.45 
# Confidence threshold for drawing OD bounding boxes (use a low value for debugging)
DEBUG_OD_CONF_THRESHOLD = 0.25 
# YOLO input size for fast real-time inference
YOLO_FAST_SIZE = 416 


# --- DISEASE REMEDIES (Used by detector.py for display) ---
# NOTE: The keys now use the unified CLF_CLASS_NAMES.
DISEASE_REMEDIES = {
    "Tomato_Healthy": {"label": "Healthy", "status": "No disease detected.", "remedies": "Keep up the good work! Continue with standard care, good ventilation, and proper watering."},
    
    "UNCERTAIN_MONITOR": {"label": "Uncertain Detection", "status": "Low Confidence/Mixed Signal Detected. Monitor closely.", "remedies": 
        "If you see symptoms of disease, re-scan the leaf.\n"
        "Check for visual symptoms: spots, mold, or yellowing.\n"
        "Monitor humidity and ensure good air circulation."
    },
    
    "Tomato_Bacterial_spot": {"label": "Bacterial Spot", "status": "Disease Detected: Tomato Bacterial Spot", "remedies": 
        "Remedies for Bacterial Spot:\n\n"
        "1. Discard or destroy any affected plants (Do not compost them).\n"
        "2. Rotate your tomato plants yearly to prevent re-infection.\n"
        "3. Apply copper fungicides strictly following the manufacturer's instructions."
    },
    "Tomato_Early_blight": {"label": "Early Blight", "status": "Disease Detected: Tomato Early Blight", "remedies": 
        "Remedies for Early Blight:\n\n"
        "1. Remove infected leaves immediately.\n"
        "2. Apply fungicides containing chlorothalonil or maneb.\n"
        "3. Improve air circulation and stake plants to keep foliage off the ground."
    },
    "Tomato_Late_blight": {"label": "Late Blight", "status": "Disease Detected: Tomato Late Blight", "remedies": 
        "Remedies for Late Blight:\n\n"
        "1. Late blight spreads rapidly; destroy infected plants immediately.\n"
        "2. Apply preventative fungicides (e.g., copper-based) during wet, cool weather.\n"
        "3. Ensure good drainage and avoid overhead watering."
    },
    "Tomato_Leaf_Mold": {"label": "Leaf Mold", "status": "Disease Detected: Tomato Leaf Mold", "remedies": 
        "Remedies for Leaf Mold:\n\n"
        "1. Improve air circulation by pruning dense foliage.\n"
        "2. Reduce humidity by proper ventilation (especially in greenhouses).\n"
        "3. Apply approved fungicides containing copper or other suitable active ingredients."
    },
    "Tomato_Septoria_leaf_spot": {"label": "Septoria Leaf Spot", "status": "Disease Detected: Tomato Septoria Leaf Spot", "remedies": 
        "Remedies for Septoria Leaf Spot:\n\n"
        "1. Remove and destroy infected lower leaves to reduce spore count.\n"
        "2. Apply fungicides containing chlorothalonil or copper.\n"
        "3. Ensure good air circulation and avoid overhead watering."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {"label": "Spider Mites", "status": "Pest Detected: Spider Mites", "remedies": 
        "Remedies for Spider Mites:\n\n"
        "1. Use insecticidal soaps or horticultural oils.\n"
        "2. Increase humidity around the plants (mites prefer dry conditions).\n"
        "3. Introduce beneficial predators (e.g., predatory mites)."
    },
    "Tomato_Target_Spot": {"label": "Target Spot", "status": "Disease Detected: Tomato Target Spot", "remedies": 
        "Remedies for Target Spot:\n\n"
        "1. Remove infected debris and practice crop rotation.\n"
        "2. Apply fungicides (e.g., chlorothalonil) during favorable conditions.\n"
        "3. Water at the base of the plant to keep foliage dry."
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {"label": "Curl Virus", "status": "Disease Detected: Tomato Yellow Leaf Curl Virus", "remedies": 
        "Remedies for Curl Virus:\n\n"
        "1. The primary control is managing the whitefly vector that spreads the virus.\n"
        "2. Monitor the field, remove, and destroy infected plants (once infected, recovery is rare).\n"
        "3. Use insecticide sprays targeted at whiteflies."
    },
    "Tomato_Tomato_mosaic_virus": {"label": "Mosaic Virus", "status": "Disease Detected: Tomato Mosaic Virus", "remedies": 
        "Remedies for Mosaic Virus:\n\n"
        "1. Monitor the field and handpick diseased plants immediately.\n"
        "2. Wash hands and tools after handling infected plants, as it spreads easily by contact.\n"
        "3. There is no chemical cure; focusing on prevention and hygiene is key."
    }
}