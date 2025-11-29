import numpy as np
# Import the specific configuration variables, including all mapping dictionaries
from config import (
    CLF_CLASS_NAMES, ENSEMBLE_VOTING_THRESHOLD, 
    OD_TO_CLF_MAP_B, OD_TO_CLF_MAP_C # OD Mappings
) 

def ensemble_hard_voting(od_predictions_list, threshold=ENSEMBLE_VOTING_THRESHOLD):
    """
    Combines predictions from two Object Detection Models (Model B & C) 
    using a Hard Voting scheme and class mapping.

    Args:
        od_predictions_list (list[tuple]): A list containing (labels, confidences, mapping_dict) 
                                           for each OD model: [(B_labels, B_confs, Map_B), (C_labels, C_confs, Map_C)].
        threshold (float): Minimum confidence for a prediction to be included in the vote.

    Returns:
        tuple[str, float]: The final ensembled prediction label and confidence.
    """
    
    # 1. Initialize Vote Tally based on the unified CLF class names
    vote_tally = {name: 0 for name in CLF_CLASS_NAMES}
    
    # Track all contributing confidence scores for the winning label
    all_contributing_scores = [] 

    # --- 2. Votes from Object Detection Models (Model B and Model C) ---
    
    # Iterate through all provided OD models (Model B and Model C)
    for od_labels, od_confidences, od_to_clf_map in od_predictions_list:
        
        # We process every detection from every model
        for od_label, confidence in zip(od_labels, od_confidences):
            
            if confidence >= threshold:
                
                # --- APPLY CLASS MAPPING ---
                # This maps the specific OD model label (e.g., 'Early_Blight') to the unified label ('Tomato_Early_blight')
                final_voting_label = od_to_clf_map.get(od_label, od_label)
                
                # Check if the resulting label exists in the CLF/Unified set before voting.
                if final_voting_label in vote_tally:
                    # Grant the vote (each detection counts as one vote)
                    vote_tally[final_voting_label] += 1
                    # Store confidence for final score calculation
                    all_contributing_scores.append((final_voting_label, confidence))
            
    # --- 3. Determine Final Ensembled Prediction ---
    
    max_votes = 0
    final_label = "UNCERTAIN/NO DETECTIONS"
    
    # Find the winning label (the one with the most high-confidence detections)
    for label, votes in vote_tally.items():
        if votes > max_votes:
            max_votes = votes
            final_label = label
        elif votes == max_votes and max_votes > 0:
            # If there is a tie, mark it as uncertain
            final_label = "UNCERTAIN/MIXED SIGNAL" 
            break

    # --- 4. Calculate Final Confidence ---
    
    if max_votes > 0 and "UNCERTAIN" not in final_label:
        winning_confidences = []
        
        # Collect all confidences that voted for the winning label
        for label, conf in all_contributing_scores:
            if label == final_label:
                winning_confidences.append(conf)
                
        # The final confidence is the average of the contributing scores for the winning label
        final_confidence = np.mean(winning_confidences) if winning_confidences else 0.0
        
    else:
        # No model reached the threshold or there was a tie
        final_confidence = 0.0
        if max_votes > 0:
             # If there were detections but they tied
             final_label = "UNCERTAIN/MIXED SIGNAL"
        else:
             # If there were no detections above the threshold
             final_label = "UNCERTAIN/NO DETECTIONS"


    return final_label, final_confidence