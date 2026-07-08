from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_direction_metrics(y_true_dir, y_pred_dir) -> dict:
    """
    이진 분류(상승=1, 하락/횡보=0)에 대한 분류 지표를 계산합니다.
    """
    acc = accuracy_score(y_true_dir, y_pred_dir)
    prec = precision_score(y_true_dir, y_pred_dir, zero_division=0)
    rec = recall_score(y_true_dir, y_pred_dir, zero_division=0)
    f1 = f1_score(y_true_dir, y_pred_dir, zero_division=0)
    
    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1_Score": round(f1, 4)
    }