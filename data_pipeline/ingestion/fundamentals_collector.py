import os
import requests
import pandas as pd
import time
import logging
from dotenv import load_dotenv

load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
logger = logging.getLogger(__name__)

def fetch_basic_financials(ticker: str, retries: int = 3) -> dict:
    """
    Finnhub API를 사용하여 종목의 기본 재무 데이터(Metric)를 수집합니다.
    """
    if not FINNHUB_API_KEY:
        logger.error("FINNHUB_API_KEY가 설정되지 않았습니다.")
        return {}

    url = f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all&token={FINNHUB_API_KEY}"
    
    for i in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            metric_data = data.get('metric', {})
            logger.info(f"[{ticker}] 재무 펀더멘털 데이터 수집 완료 (Key 개수: {len(metric_data)})")
            return metric_data
            
        except requests.exceptions.RequestException as e:
            wait_time = 2 ** i
            logger.error(f"[{ticker}] Finnhub API 호출 실패: {e}. {wait_time}초 후 재시도...")
            time.sleep(wait_time)
            
    return {}

# 테스트용 실행 블록
if __name__ == "__main__":
    financials = fetch_basic_financials("AAPL")
    # 반환된 딕셔너리에서 주요 키워드(예: PSR 계산을 위한 psTTM 등) 확인 가능
    if financials:
        print(f"52주 최고가: {financials.get('52WeekHigh')}")
        print(f"TTM P/S: {financials.get('psTTM')}")