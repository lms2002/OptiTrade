import os
import requests
import pandas as pd
import time
import logging
from dotenv import load_dotenv

load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
logger = logging.getLogger(__name__)

def fetch_insider_sentiment(ticker: str, start_date: str, end_date: str, retries: int = 3) -> pd.DataFrame:
    """
    Finnhub API를 사용하여 내부자 거래(Insider Sentiment) 데이터를 수집합니다.
    뉴스 감성만으로 놓치는 내부 수급 신호를 보완하는 대체 데이터입니다.
    """
    if not FINNHUB_API_KEY:
        logger.error("FINNHUB_API_KEY가 설정되지 않았습니다.")
        return pd.DataFrame()

    url = f"https://finnhub.io/api/v1/stock/insider-sentiment?symbol={ticker}&from={start_date}&to={end_date}&token={FINNHUB_API_KEY}"

    for i in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data or 'data' not in data or not data['data']:
                logger.warning(f"[{ticker}] 내부자 거래 데이터를 찾을 수 없습니다.")
                return pd.DataFrame()

            df = pd.DataFrame(data['data'])
            
            # 년/월 데이터를 일별 데이터프레임과 병합하기 위해 매월 1일 자로 날짜 생성
            df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
            
            # mspr: Monthly Share Purchase Ratio (매수 비율), change: Net Buying (순매수량)
            df = df[['date', 'symbol', 'mspr', 'change']].rename(
                columns={'symbol': 'ticker', 'mspr': 'insider_mspr', 'change': 'insider_net_change'}
            )
            
            logger.info(f"[{ticker}] 내부자 거래 대체 데이터 {len(df)}개월 치 수집 완료")
            return df

        except requests.exceptions.RequestException as e:
            wait_time = 2 ** i
            logger.error(f"[{ticker}] 내부자 데이터 API 호출 실패: {e}. {wait_time}초 후 재시도...")
            time.sleep(wait_time)

    return pd.DataFrame()