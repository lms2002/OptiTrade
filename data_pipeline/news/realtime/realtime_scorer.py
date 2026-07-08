import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_FILE = "data_pipeline/news/realtime/raw_data/finnhub_aapl_news.csv"
OUTPUT_FILE = "data_pipeline/news/realtime/realtime_sentiment.parquet"

def run_realtime_scoring(batch_size: int = 32):
    """
    수집된 실시간(최근) 뉴스 데이터를 FinBERT로 스코어링하여 Parquet로 저장합니다.
    """
    if not os.path.exists(INPUT_FILE):
        logger.error("수집된 뉴스 파일이 없습니다. 실시간 뉴스 수집기를 먼저 실행하세요.")
        return

    logger.info("FinBERT 모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    df = pd.read_csv(INPUT_FILE)
    df['sentiment_score'] = 0.0
    
    # 결측치 처리 (본문이 없으면 헤드라인 사용)
    df['text_to_score'] = df['body_summary'].fillna(df['headline']).astype(str)

    logger.info(f"총 {len(df)}건의 실시간 뉴스 스코어링 시작 (Device: {device})...")
    
    scores = []
    
    for i in range(0, len(df), batch_size):
        batch_texts = df['text_to_score'].iloc[i:i+batch_size].tolist()
        
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            batch_scores = (probs[:, 0] - probs[:, 1]).cpu().numpy()
            scores.extend(batch_scores)
            
        if i % (batch_size * 5) == 0 and i > 0:
            logger.info(f"{i} / {len(df)} 건 처리 완료...")

    df['sentiment_score'] = scores
    
    final_df = df[['date', 'ticker', 'sentiment_score']].copy()
    
    # 일별 평균 스코어 집계
    daily_sentiment = final_df.groupby(['date', 'ticker'], as_index=False)['sentiment_score'].mean()
    
    daily_sentiment.to_parquet(OUTPUT_FILE, index=False)
    logger.info(f"실시간 뉴스 스코어링 완료! Parquet 저장됨 -> {OUTPUT_FILE}")
    logger.info(daily_sentiment.head())

if __name__ == "__main__":
    run_realtime_scoring()