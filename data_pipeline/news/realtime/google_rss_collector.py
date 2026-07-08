import os
import requests
import pandas as pd
import time
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_finnhub_news(ticker: str, start_date: str, end_date: str, retries: int = 3) -> pd.DataFrame:
    """
    Finnhub API를 사용하여 특정 기간의 종목 뉴스를 수집합니다. (2024~현재 갭 메우기 용도)
    과거 뉴스(FNSPID)와 동일한 스키마로 반환합니다.
    """
    if not FINNHUB_API_KEY:
        logger.error("FINNHUB_API_KEY가 설정되지 않았습니다.")
        return pd.DataFrame()

    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={start_date}&to={end_date}&token={FINNHUB_API_KEY}"

    for i in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data:
                logger.warning(f"[{ticker}] {start_date} ~ {end_date} 기간의 뉴스가 없습니다.")
                return pd.DataFrame()

            df = pd.DataFrame(data)
            
            # Finnhub datetime은 UNIX timestamp(초)이므로 변환
            df['date'] = pd.to_datetime(df['datetime'], unit='s').dt.strftime('%Y-%m-%d')
            df['ticker'] = ticker
            
            # 스키마 정규화 (summary -> body_summary)
            df = df.rename(columns={'summary': 'body_summary'})

            # 필요한 컬럼만 추출
            cols_to_keep = ['date', 'ticker', 'headline', 'body_summary']
            df = df[cols_to_keep]

            logger.info(f"[{ticker}] {start_date} ~ {end_date} 뉴스 {len(df)}건 수집 완료")
            return df

        except requests.exceptions.RequestException as e:
            wait_time = 2 ** i
            logger.error(f"[{ticker}] Finnhub API 호출 실패: {e}. {wait_time}초 후 재시도...")
            time.sleep(wait_time)

    return pd.DataFrame()

if __name__ == "__main__":
    # 2024년 1월 1일부터 현재까지의 빈 갭을 수집 테스트
    end_str = datetime.now().strftime("%Y-%m-%d")
    start_str = "2024-01-01"
    
    news_df = fetch_finnhub_news("AAPL", start_str, end_str)
    
    if not news_df.empty:
        os.makedirs("data_pipeline/news/realtime/raw_data", exist_ok=True)
        save_path = "data_pipeline/news/realtime/raw_data/finnhub_aapl_news.csv"
        news_df.to_csv(save_path, index=False)
        logger.info(f"데이터 저장 완료: {save_path}")
        print(news_df.head())