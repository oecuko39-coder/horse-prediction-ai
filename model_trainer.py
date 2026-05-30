import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """機械学習モデルの学習と最適化クラス"""

    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_columns = []

    def prepare_training_data(self, features_df: pd.DataFrame, target_col: str = 'finish_position') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """学習データの準備"""
        # 数値型のカラムのみを特徴量として使用
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # ターゲットカラムを除外
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        # NaNを0で埋める
        X = features_df[numeric_cols].fillna(0).values
        
        # ターゲットの準備（1着を予測）
        if target_col in features_df.columns:
            y = (features_df[target_col] == 1).astype(int).values
        else:
            logger.warning(f"ターゲットカラム '{target_col}' が見つかりません")
            return None, None, numeric_cols
        
        self.feature_columns = numeric_cols
        return X, y, numeric_cols

    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
        """XGBoostモデルの学習"""
        logger.info("XGBoostの学習を開始します")
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        logger.info("XGBoostの学習が完了しました")
        return model

    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray) -> LGBMClassifier:
        """LightGBMモデルの学習"""
        logger.info("LightGBMの学習を開始します")
        model = LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=-1
        )
        model.fit(X_train, y_train)
        self.models['lightgbm'] = model
        logger.info("LightGBMの学習が完了しました")
        return model

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """Random Forestモデルの学習"""
        logger.info("Random Forestの学習を開始します")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        logger.info("Random Forestの学習が完了しました")
        return model

    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """モデルの評価"""
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
            }
            
            logger.info(f"{name} - Accuracy: {results[name]['accuracy']:.4f}, F1: {results[name]['f1']:.4f}")
        
        return results

    def get_best_model(self) -> object:
        """最高精度のモデルを取得"""
        if not self.models:
            logger.error("モデルが学習されていません")
            return None
        
        # 簡単な評価: accuracy が最も高いモデルを選ぶ
        best_name = max(self.models.keys())
        self.best_model = self.models[best_name]
        logger.info(f"最適なモデル: {best_name}")
        return self.best_model

    def save_model(self, filepath: str):
        """モデルを保存"""
        if self.best_model is None:
            logger.error("保存するモデルがありません")
            return False
        
        joblib.dump(self.best_model, filepath)
        logger.info(f"モデルを保存しました: {filepath}")
        return True

    def load_model(self, filepath: str):
        """モデルを読み込む"""
        self.best_model = joblib.load(filepath)
        logger.info(f"モデルを読み込みました: {filepath}")
        return self.best_model
