import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def flag_outliers(df: pd.DataFrame, threshold_pct: float = 0.20) -> pd.DataFrame:
    """
    일일 수익률을 계산하여 특정 임계치(예: 20%)를 초과하는 급등락 구간을 이상치로 플래깅합니다.
    """
    if df.empty or 'close' not in df.columns:
        return df
        
    df = df.copy()
    
    # 일일 수익률(Daily Return) 계산
    df['daily_return'] = df['close'].pct_change()
    
    # 이상치 플래그: 수익률의 절댓값이 임계치(20%)를 초과하면 1, 아니면 0
    df['is_outlier'] = (df['daily_return'].abs() > threshold_pct).astype(int)
    
    outlier_count = df['is_outlier'].sum()
    logger.info(f"이상치 탐지 완료: 총 {len(df)}일 중 {outlier_count}건 플래깅됨 (기준: {threshold_pct*100}%)")
    
    return df