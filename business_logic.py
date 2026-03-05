def risk_bucket(prob):
    if prob < 0.3:
        return "Low", "Stable"
    elif prob <= 0.6:
        return "Medium", "Monitor closely"
    else:
        return "High", "Immediate action required"