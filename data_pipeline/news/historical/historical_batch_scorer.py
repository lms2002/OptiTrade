import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PARSED_FILE = "data_pipeline/news/historical/raw_data/parsed_aapl_news.csv"
OUTPUT_FILE = "data_pipeline/news/historical/historical_sentiment.parquet"

def run_batch_scoring(batch_size: int = 32):
    """
    FinBERT를 활용하여 파싱된 뉴스 본문의 감성 스코어를 배치 단위로 계산합니다.
    """
    if not os.path.exists(PARSED_FILE):
        logger.error("파싱된 뉴스 파일이 없습니다. fnspid_parser.py를 먼저 실행하세요.")
        return

    logger.info("FinBERT 모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    df = pd.read_csv(PARSED_FILE)
    df['sentiment_score'] = 0.0
    
    # 결측치 처리 (본문이 없으면 헤드라인 사용)
    df['text_to_score'] = df['body_summary'].fillna(df['headline']).astype(str)

    logger.info(f"총 {len(df)}건의 뉴스 스코어링 시작 (Device: {device})...")
    
    scores = []
    
    for i in range(0, len(df), batch_size):
        batch_texts = df['text_to_score'].iloc[i:i+batch_size].tolist()
        
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # FinBERT 클래스: 0(Positive), 1(Negative), 2(Neutral)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Score = Positive 확률 - Negative 확률 (-1.0 ~ 1.0)
            batch_scores = (probs[:, 0] - probs[:, 1]).cpu().numpy()
            scores.extend(batch_scores)
            
        if i % (batch_size * 10) == 0 and i > 0:
            logger.info(f"{i} / {len(df)} 건 처리 완료...")

    df['sentiment_score'] = scores
    
    # 최종 스키마에 맞춰 정리 및 Parquet 저장
    final_df = df[['date', 'ticker', 'sentiment_score']].copy()
    
    # 일별, 종목별 스코어 평균으로 집계 (일봉 데이터와 조인하기 위함)
    daily_sentiment = final_df.groupby(['date', 'ticker'], as_index=False)['sentiment_score'].mean()
    
    daily_sentiment.to_parquet(OUTPUT_FILE, index=False)
    logger.info(f"배치 스코어링 완료! Parquet 저장됨 -> {OUTPUT_FILE}")
    logger.info(daily_sentiment.head())

if __name__ == "__main__":
    run_batch_scoring()