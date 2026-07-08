import os
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 임시 저장용 디렉토리 (git에 올라가지 않도록 주의)
DATA_DIR = "data_pipeline/news/historical/raw_data"
os.makedirs(DATA_DIR, exist_ok=True)

def download_fnspid_csv():
    """
    GitHub에서 FNSPID 데이터셋(nasdaq_exteral_data.csv)을 스트리밍 방식으로 다운로드합니다.
    이미 존재하면 다운로드를 건너뜁니다.
    """
    # Zdong104/FNSPID_Financial_News_Dataset 의 Raw URL (예시 샘플 경로, 실제 URL로 조정 필요)
    url = "https://raw.githubusercontent.com/Zdong104/FNSPID_Financial_News_Dataset/main/Data/nasdaq_exteral_data.csv"
    file_path = os.path.join(DATA_DIR, "nasdaq_exteral_data.csv")
    
    if os.path.exists(file_path):
        logger.info(f"데이터셋이 이미 존재합니다: {file_path}")
        return file_path
        
    logger.info("FNSPID 데이터 다운로드 시작 (시간이 다소 소요될 수 있습니다)...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"다운로드 완료: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"다운로드 실패: {e}")
        # 용량이 너무 커서 다운로드 실패시, 브라우저에서 직접 다운받아 raw_data 폴더에 넣도록 안내
        logger.info("직접 다운로드 링크: https://huggingface.co/datasets/Zihan1004/FNSPID/tree/main")
        return None

if __name__ == "__main__":
    download_fnspid_csv()