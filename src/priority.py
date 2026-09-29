import yaml

def get_priority_level(predicted_category: str, mapping: dict = None) -> str:
    """
    Maps predicted message category to an academic emergency priority level.
    """
    if mapping is None:
        mapping = {
            "Rescue Request": "CRITICAL",
            "Casualties / Injuries": "CRITICAL",
            "Infrastructure Damage": "HIGH",
            "Not Relevant": "LOW"
        }
    return mapping.get(predicted_category, "LOW")

def evaluate_review_status(confidence_score: float, threshold: float = 0.60) -> str:
    """
    Flags predictions below confidence threshold for human operator review.
    """
    if confidence_score < threshold:
        return "Human Review Recommended"
    return "Model Prediction"

def format_prediction_result(category: str, probabilities: dict, threshold: float = 0.60) -> dict:
    """
    Packages category prediction, priority level, confidence score,
    top alternative predictions, and review status.
    """
    priority = get_priority_level(category)
    top_confidence = probabilities.get(category, max(probabilities.values()) if probabilities else 0.0)
    review_status = evaluate_review_status(top_confidence, threshold)
    
    # Sort top alternative classes
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "predicted_category": category,
        "priority_level": priority,
        "confidence_score": top_confidence,
        "confidence_percentage": f"{top_confidence * 100:.1f}%",
        "review_status": review_status,
        "top_predictions": [
            {"category": cat, "probability": prob, "percentage": f"{prob * 100:.1f}%"}
            for cat, prob in sorted_probs
        ]
    }
