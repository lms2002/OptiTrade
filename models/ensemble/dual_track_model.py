import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DualTrackModel:
    def __init__(self, ev_threshold=0.0005):
        """
        분류(방향성 확률) + 회귀(예상 등락폭) 이중 트랙 모델
        ev_threshold: 매수를 결정할 최소 기대수익률 (기본값 0.05% 안전마진)
        """
        self.ev_threshold = ev_threshold
        
        # 1. 분류 트랙 (상승 확률용 - 기존 Optuna 튜닝값 적용)
        self.classifier = lgb.LGBMClassifier(
            n_estimators=112, max_depth=5, learning_rate=0.079, 
            subsample=0.588, random_state=42, verbose=-1
        )
        
        # 2. 회귀 트랙 (예상 등락폭용 - 보수적 세팅)
        self.regressor = xgb.XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
        )

    def fit(self, X_train: pd.DataFrame, y_train_dir: pd.Series, y_train_ret: pd.Series):
        """두 모델을 각각의 타겟(방향성, 수익률)으로 동시 학습시킵니다."""
        self.classifier.fit(X_train, y_train_dir)
        self.regressor.fit(X_train, y_train_ret)
        logger.info("Dual Track (분류+회귀) 학습 완료")

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        # 1. P(Up): 상승할 확률 (0 ~ 1)
        prob_up = self.classifier.predict_proba(X_test)[:, 1]
        
        # 2. E[Return]: 예상 수익률
        pred_return = self.regressor.predict(X_test)
        
        # 3. 기대값(Expected Value) 연산
        expected_value = prob_up * pred_return
        
        # 4. 기대값이 안전마진(Threshold) 이상일 때만 매수(1), 아니면 보류(0)
        decisions = (expected_value > self.ev_threshold).astype(int)
        
        return decisions