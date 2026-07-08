import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_fundamental_features(df: pd.DataFrame, fundamentals_metric: dict) -> pd.DataFrame:
    """
    주가 데이터에 PSR 및 어닝 서프라이즈 관련 피처를 추가합니다.
    """
    if df.empty:
        return df
        
    df = df.copy()
    
    # Finnhub metric에서 TTM 기준 PSR 값 추출
    psr_value = fundamentals_metric.get('psTTM', None)
    
    # 시계열 백테스트를 위해 우선 현재 시점의 정적 값을 일괄 매핑 
    # (추후 유료 API나 분기별 과거 데이터 확보 시 병합 로직으로 고도화 예정)
    df['psr'] = psr_value if psr_value is not None else 0.0
    
    # 어닝 서프라이즈 퍼센티지 (초기 Placeholder)
    df['earning_surprise_pct'] = 0.0 

    logger.info("기본적 재무 피처(PSR 등) 생성 완료")
    return df