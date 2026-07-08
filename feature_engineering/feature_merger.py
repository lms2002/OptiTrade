import os
import sys
import pandas as pd
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.ingestion.price_collector import fetch_ohlcv
from data_pipeline.ingestion.fundamentals_collector import fetch_basic_financials
from data_pipeline.ingestion.macro_collector import fetch_fred_series
from data_pipeline.ingestion.alt_data_collector import fetch_insider_sentiment
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"[{ticker}] 강력한 Feature Set 재구축 시작 (Macro & Alt Data 포함)...")

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = "2010-01-01"

    # 1. 주가 데이터 로드
    df = fetch_ohlcv(ticker, start_date=start_date, end_date=end_date)
    if df.empty:
        return

    # 2. 기술적 지표 추가
    df = add_technical_indicators(df)

    # 3. 재무 데이터 추가
    fundamentals = fetch_basic_financials(ticker)
    df = add_fundamental_features(df, fundamentals)

    df['date'] = pd.to_datetime(df['date'])

    # 4. 감성 스코어 병합
    sentiment_df = build_sentiment_features()
    if not sentiment_df.empty:
        sentiment_df = sentiment_df[sentiment_df['ticker'] == ticker]
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        df = pd.merge(df, sentiment_df, on=['date', 'ticker'], how='left')

    # 5. 거시 경제 지표 병합 (미국 10년물 국채 금리)
    macro_df = fetch_fred_series("DGS10", start_date)
    if not macro_df.empty:
        macro_df['date'] = pd.to_datetime(macro_df['date'])
        macro_df = macro_df.rename(columns={'value': 'treasury_10y'})[['date', 'treasury_10y']]
        df = pd.merge(df, macro_df, on='date', how='left')

    # 6. 대체 데이터 병합 (내부자 거래 동향)
    insider_df = fetch_insider_sentiment(ticker, start_date, end_date)
    if not insider_df.empty:
        insider_df['date'] = pd.to_datetime(insider_df['date'])
        df = pd.merge(df, insider_df[['date', 'insider_mspr', 'insider_net_change']], on='date', how='left')
        # 월간 데이터를 일별로 확장 (이전 값으로 채우기)
        df['insider_mspr'] = df['insider_mspr'].ffill().fillna(0)
        df['insider_net_change'] = df['insider_net_change'].ffill().fillna(0)

    # 7. 이상치 플래깅 & 결측치 처리
    df = flag_outliers(df)
    df = handle_missing_values(df)

    # 최종 저장
    df.to_parquet(OUTPUT_FILE, index=False)
    logger.info(f"🎉 Macro & Alt Data 통합 Feature Set 생성 완료! 저장 경로: {OUTPUT_FILE}")

if __name__ == "__main__":
    build_final_feature_set("AAPL")