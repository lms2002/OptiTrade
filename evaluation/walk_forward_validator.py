import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WalkForwardValidator:
    def __init__(self, train_window_size: int = 252, test_window_size: int = 21):
        """
        시간순 전진 검증(Walk-Forward Validation) 엔진입니다.
        미래 참조(Look-ahead bias)를 방지하기 위해 훈련 데이터와 테스트 데이터를 슬라이딩합니다.
        - train_window_size: 훈련에 사용할 일수 (기본 252일 = 약 1년치 거래일)
        - test_window_size: 예측 및 평가에 사용할 일수 (기본 21일 = 약 1개월치 거래일)
        """
        self.train_window_size = train_window_size
        self.test_window_size = test_window_size

    def validate(self, df: pd.DataFrame, model, feature_cols: list, target_col: str) -> pd.DataFrame:
        """
        데이터셋을 순회하며 모델을 훈련하고 예측 결과를 수집하여 반환합니다.
        """
        if df.empty or len(df) < self.train_window_size + self.test_window_size:
            logger.error("데이터가 부족하여 Walk-Forward 검증을 수행할 수 없습니다.")
            return pd.DataFrame()

        # 시간순 정렬 보장 (가장 중요한 부분)
        df = df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

        results = []
        total_steps = (len(df) - self.train_window_size) // self.test_window_size

        logger.info(f"Walk-Forward 검증 시작 (총 {total_steps} 스텝 슬라이딩 예정)...")

        for step in range(total_steps):
            train_start = step * self.test_window_size
            train_end = train_start + self.train_window_size
            test_end = train_end + self.test_window_size

            # 데이터 범위를 벗어나면 루프 종료
            if test_end > len(df):
                break

            # Train / Test 엄격한 분할
            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[train_end:test_end]

            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_test = test_df[feature_cols]
            # y_test는 모델에 넣지 않고 결과 비교용으로만 유지

            # 모델 훈련 및 예측 (sklearn API의 fit, predict 호환 가정)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # 해당 스텝의 예측 결과 저장
            step_result = test_df[['date', 'ticker', 'close', target_col]].copy()
            step_result['predicted'] = preds
            results.append(step_result)

        if not results:
            return pd.DataFrame()

        final_results = pd.concat(results, ignore_index=True)
        logger.info(f"Walk-Forward 검증 완료! (총 {len(final_results)}일 치의 미래 예측 누적)")
        return final_results