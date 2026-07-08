import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_price_metrics(y_true, y_pred) -> dict:
    """
    연속형 변수(가격) 예측에 대한 오차 지표를 계산합니다.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE 계산 (0으로 나누는 에러 방지)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100
    
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4)
    }