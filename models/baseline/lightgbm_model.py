import lightgbm as lgb
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightGBMBaseline:
    def __init__(self, task_type='classification'):
        self.task_type = task_type
        if self.task_type == 'classification':
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                random_state=42,
                verbose=-1
            )
        else:
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                random_state=42,
                verbose=-1
            )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        return self.model.predict(X_test)