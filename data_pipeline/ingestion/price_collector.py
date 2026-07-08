import yfinance as yf
import pandas as pd
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_ohlcv(ticker: str, start_date: str, end_date: str, retries: int = 3) -> pd.DataFrame:
    """
    yfinance를 활용하여 OHLCV 데이터를 수집합니다.
    Rate Limit 대응을 위해 Exponential Backoff를 적용합니다.
    """
    for i in range(retries):
        try:
            # yfinance 다운로드 (progress bar 비활성화)
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                logger.warning(f"[{ticker}] 데이터를 찾을 수 없습니다.")
                return pd.DataFrame()
            
            df.reset_index(inplace=True)
            
            # MultiIndex 컬럼일 경우 단일 레벨로 평탄화 (yfinance 최신 버전 대응)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
                
            # DB 적재를 위한 스키마 정규화
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            df['ticker'] = ticker
            
            logger.info(f"[{ticker}] OHLCV 데이터 {len(df)}건 수집 완료")
            return df
            
        except Exception as e:
            wait_time = 2 ** i
            logger.error(f"[{ticker}] 데이터 수집 실패: {e}. {wait_time}초 후 재시도...")
            time.sleep(wait_time)
            
    return pd.DataFrame()

# 테스트용 실행 블록
if __name__ == "__main__":
    # 기준 종목인 AAPL 외에 SMCI, HOOD 등 관심 종목으로도 API 연동 테스트가 가능해
    test_df = fetch_ohlcv("AAPL", "2023-01-01", "2023-12-31")
    print(test_df.head())