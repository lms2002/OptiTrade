import os
import requests
import pandas as pd
import time
import logging
from dotenv import load_dotenv

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
logger = logging.getLogger(__name__)

def fetch_fred_series(series_id: str, start_date: str, retries: int = 3) -> pd.DataFrame:
    """
    FRED API를 호출하여 거시 경제 시계열 데이터를 수집합니다.
    예: DGS10 (10년물 국채 금리), VIXCLS (VIX 지수)
    """
    if not FRED_API_KEY:
        logger.error("FRED_API_KEY가 설정되지 않았습니다.")
        return pd.DataFrame()

    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json&observation_start={start_date}"
    
    for i in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data['observations'])
            # 필요한 date, value 컬럼만 추출 후 타입 캐스팅
            df = df[['date', 'value']].copy()
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df['series_id'] = series_id
            
            logger.info(f"[{series_id}] 거시 지표 {len(df)}건 수집 완료")
            return df
            
        except requests.exceptions.RequestException as e:
            wait_time = 2 ** i
            logger.error(f"[{series_id}] FRED API 호출 실패: {e}. {wait_time}초 후 재시도...")
            time.sleep(wait_time)
            
    return pd.DataFrame()

# 테스트용 실행 블록
if __name__ == "__main__":
    # 미국 10년물 국채 금리 (DGS10) 테스트
    macro_df = fetch_fred_series("DGS10", "2023-01-01")
    print(macro_df.head())