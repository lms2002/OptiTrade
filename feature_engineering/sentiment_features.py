import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HISTORICAL_FILE = "data_pipeline/news/historical/historical_sentiment.parquet"
REALTIME_FILE = "data_pipeline/news/realtime/realtime_sentiment.parquet"

def build_sentiment_features() -> pd.DataFrame:
    """
    과거 및 실시간 감성 스코어를 통합하고 선반영 모멘텀(Delta) 피처를 생성합니다.
    """
    dfs = []
    
    if os.path.exists(HISTORICAL_FILE):
        dfs.append(pd.read_parquet(HISTORICAL_FILE))
    if os.path.exists(REALTIME_FILE):
        dfs.append(pd.read_parquet(REALTIME_FILE))

    if not dfs:
        logger.warning("감성 스코어 Parquet 파일이 존재하지 않습니다.")
        return pd.DataFrame()

    # 데이터 병합 및 중복 제거
    df = pd.concat(dfs, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date']).drop_duplicates(subset=['date', 'ticker'])

    # 5일 이동평균 감성 스코어
    df['sentiment_ma_5'] = df.groupby('ticker')['sentiment_score'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    # 5일 감성 누적 변화량 (Delta - 선반영 모멘텀)
    df['sentiment_delta_5d'] = df.groupby('ticker')['sentiment_score'].transform(
        lambda x: x.diff(periods=5)
    )

    logger.info(f"감성 통합 피처 생성 완료 (총 {len(df)}일치 데이터)")
    return df