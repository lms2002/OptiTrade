import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_FILE = "data_pipeline/news/historical/raw_data/nasdaq_exteral_data.csv"
PARSED_FILE = "data_pipeline/news/historical/raw_data/parsed_aapl_news.csv"

def parse_and_filter(ticker: str = "AAPL"):
    """
    대용량 CSV를 청크 단위로 읽어 특정 종목만 필터링하고 스키마를 정규화합니다.
    """
    if not os.path.exists(RAW_FILE):
        logger.error("원본 CSV 파일이 없습니다. fnspid_downloader.py를 먼저 실행하세요.")
        return

    logger.info(f"[{ticker}] 종목 필터링 및 스키마 정규화 시작...")
    
    chunk_size = 50000
    filtered_chunks = []
    
    # FNSPID 데이터 스키마에 맞게 컬럼 매핑 (원본 스키마명에 따라 수정 필요)
    # 가정: 원본 컬럼이 Date, Article_title, Stock_symbol, Article 이라고 가정
    try:
        for chunk in pd.read_csv(RAW_FILE, chunksize=chunk_size, low_memory=False):
            # 대소문자 무시를 위해 컬럼명 소문자로 통일
            chunk.columns = chunk.columns.str.lower()
            
            # 종목 필터링 (보통 stock_symbol 이나 symbol 컬럼에 위치)
            symbol_col = 'stock_symbol' if 'stock_symbol' in chunk.columns else 'symbol'
            
            if symbol_col in chunk.columns:
                target_df = chunk[chunk[symbol_col] == ticker].copy()
                
                if not target_df.empty:
                    # 스키마 정규화 (date, ticker, headline, body_summary)
                    target_df = target_df.rename(columns={
                        symbol_col: 'ticker',
                        'article_title': 'headline',
                        'title': 'headline',
                        'article': 'body_summary',
                        'text': 'body_summary'
                    })
                    
                    # 필요한 컬럼만 추출
                    cols_to_keep = [c for c in ['date', 'ticker', 'headline', 'body_summary'] if c in target_df.columns]
                    filtered_chunks.append(target_df[cols_to_keep])
                    
        if filtered_chunks:
            final_df = pd.concat(filtered_chunks, ignore_index=True)
            # 날짜 형식 통일
            final_df['date'] = pd.to_datetime(final_df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
            final_df = final_df.dropna(subset=['date', 'headline'])
            
            final_df.to_csv(PARSED_FILE, index=False)
            logger.info(f"필터링 완료! 총 {len(final_df)}건의 {ticker} 뉴스 파싱됨. -> {PARSED_FILE}")
        else:
            logger.warning(f"{ticker} 종목의 뉴스를 찾을 수 없습니다.")
            
    except Exception as e:
        logger.error(f"파싱 중 에러 발생: {e}")

if __name__ == "__main__":
    parse_and_filter("AAPL")