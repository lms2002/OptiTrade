import pandas as pd
import pandas_ta as ta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV 데이터프레임에 기술적 지표를 추가합니다.
    (SMA, RSI, Bollinger Bands, 지지/저항선)
    """
    if df.empty:
        return df
        
    df = df.copy()
    
    # 시간순 정렬 보장
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # 1. 이동평균선 (SMA: 20일, 50일)
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    
    # 2. RSI (14일)
    df.ta.rsi(length=14, append=True)
    
    # 3. 볼린저 밴드 (20일 기준)
    df.ta.bbands(length=20, append=True)

    # 4. 매물대 지지/저항선 Proxy (최근 20일 최저/최고점)
    df['support_20d'] = df['low'].rolling(window=20).min()
    df['resistance_20d'] = df['high'].rolling(window=20).max()

    df.reset_index(inplace=True)
    # 컬럼명 소문자 통일 (pandas-ta 생성 컬럼 포함)
    df.columns = [col.lower() for col in df.columns]
    
    logger.info(f"기술적 지표 생성 완료 (총 {len(df)}건)")
    return df