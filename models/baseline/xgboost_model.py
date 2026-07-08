import xgboost as xgb
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XGBoostBaseline:
    def __init__(self, task_type='classification'):
        """
        XGBoost 베이스라인 모델
        기본적으로 다음날 주가의 상승(1) / 하락(0)을 예측하는 분류 모델로 설정합니다.
        """
        self.task_type = task_type
        if self.task_type == 'classification':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                random_state=42,
                eval_metric='logloss'
            )
        else:
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                random_state=42
            )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        return self.model.predict(X_test)