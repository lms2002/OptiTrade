import os
import sys
import pandas as pd
import logging
from datetime import datetime

# 프로젝트 최상위 경로를 path에 추가하여 다른 모듈들을 import할 수 있게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.ingestion.price_collector import fetch_ohlcv
from data_pipeline.ingestion.fundamentals_collector import fetch_basic_financials
from feature_engineering.technical_indicators import add_technical_indicators
from feature_engineering.sentiment_features import build_sentiment_features
from feature_engineering.fundamental_features import add_fundamental_features
from feature_engineering.outlier_detector import flag_outliers
from feature_engineering.missing_value_handler import handle_missing_values

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "feature_engineering/output"
OUTPUT_FILE = f"{OUTPUT_DIR}/final_features.parquet"

def build_final_feature_set(ticker: str = "AAPL"):
    """
    주가, 기술적 지표, 재무, 감성 스코어를 모두 병합하여 최종 피처셋을 생성합니다.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"[{ticker}] 최종 Feature Set 구축 시작...")

    # 1. 주가 데이터 로드 (2010년부터 현재까지 넉넉하게)
    end_date = datetime.now().strftime('%Y-%m-%d')
    price_df = fetch_ohlcv(ticker, start_date="2010-01-01", end_date=end_date)
    if price_df.empty:
        logger.error("주가 데이터를 불러오지 못했습니다.")
        return

    # 2. 기술적 지표 추가
    df = add_technical_indicators(price_df)

    # 3. 재무 데이터 추가
    fundamentals = fetch_basic_financials(ticker)
    df = add_fundamental_features(df, fundamentals)

    # 4. 감성 스코어 병합
    sentiment_df = build_sentiment_features()
    if not sentiment_df.empty:
        # ticker가 일치하는 감성 데이터만 필터링
        sentiment_df = sentiment_df[sentiment_df['ticker'] == ticker]
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        df['date'] = pd.to_datetime(df['date'])
        
        # Left Join으로 주가 데이터 기준 날짜에 감성 스코어 결합
        df = pd.merge(df, sentiment_df, on=['date', 'ticker'], how='left')
    else:
        logger.warning("감성 스코어 데이터를 찾을 수 없어 병합을 건너뜁니다.")

    # 5. 이상치 플래깅
    df = flag_outliers(df)

    # 6. 결측치 처리 (항상 맨 마지막에 수행)
    df = handle_missing_values(df)

    # 최종 저장
    df.to_parquet(OUTPUT_FILE, index=False)
    logger.info(f"🎉 최종 Feature Set 생성 완료! 저장 경로: {OUTPUT_FILE}")
    print(df.tail())

if __name__ == "__main__":
    build_final_feature_set("AAPL")