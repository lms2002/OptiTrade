import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    엄격한 결측치(NaN) 처리 룰을 적용합니다. (Forward Fill 적용 후 나머지 0 처리)
    """
    if df.empty:
        return df
        
    df = df.copy()
    
    # 1단계: 시계열 데이터 특성을 고려하여 이전 값으로 채우기 (Forward Fill)
    df = df.ffill()
    
    # 2단계: 맨 앞단 데이터라 ffill로도 안 채워진 값들은 0으로 보간
    df = df.fillna(0)
    
    missing_count = df.isna().sum().sum()
    logger.info(f"결측치 처리 완료: 현재 남은 결측치 총 {missing_count}개")
    
    return df