import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import logging

from models.regime_classifier.regime_model import RegimeClassifier
from models.advanced.lstm_model import LSTMModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StackingEnsemble:
    def __init__(self, lgb_best_params=None):
        """
        다양한 모델의 예측값을 메타 모델이 취합하여 최종 결과를 내는 Stacking 알고리즘입니다.
        """
        # Optuna로 찾은 최적 파라미터 적용 (이전에 터미널에 나온 결과값 기반)
        if lgb_best_params is None:
            lgb_best_params = {
                'n_estimators': 112,
                'max_depth': 5,
                'learning_rate': 0.079,
                'subsample': 0.588,
                'colsample_bytree': 0.878
            }
            
        self.lgb_base = lgb.LGBMClassifier(random_state=42, verbose=-1, **lgb_best_params)
        self.regime_clf = RegimeClassifier()
        self.lstm_model = LSTMModel(sequence_length=10, epochs=15)
        
        # 메타 모델 (최종 결정권자)
        self.meta_model = LogisticRegression()

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        # 1. 개별 베이스 모델 학습
        self.regime_clf.fit(X_train)
        self.lgb_base.fit(X_train, y_train)
        self.lstm_model.fit(X_train, y_train)
        
        # 2. 메타 특성(Meta-features) 추출
        regime_probs = self.regime_clf.predict_proba(X_train) # 하락/횡보/상승 확률
        lgb_preds = self.lgb_base.predict_proba(X_train)[:, 1] # 상승 확률
        lstm_preds = self.lstm_model.predict_proba(X_train) # 딥러닝 상승 확률
        
        meta_X = pd.DataFrame({
            'regime_down': regime_probs[:, 0],
            'regime_flat': regime_probs[:, 1],
            'regime_up': regime_probs[:, 2],
            'lgb_pred': lgb_preds,
            'lstm_pred': lstm_preds
        })
        
        # 3. 메타 모델 학습
        self.meta_model.fit(meta_X, y_train)
        logger.info("Stacking Ensemble 메타 모델 학습 완료")

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        # 테스트 데이터에 대해서도 동일하게 메타 피처 생성 후 최종 예측
        regime_probs = self.regime_clf.predict_proba(X_test)
        lgb_preds = self.lgb_base.predict_proba(X_test)[:, 1]
        lstm_preds = self.lstm_model.predict_proba(X_test)
        
        meta_X = pd.DataFrame({
            'regime_down': regime_probs[:, 0],
            'regime_flat': regime_probs[:, 1],
            'regime_up': regime_probs[:, 2],
            'lgb_pred': lgb_preds,
            'lstm_pred': lstm_preds
        })
        
        return self.meta_model.predict(meta_X)