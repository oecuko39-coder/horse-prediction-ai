import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """CSVデータの読み込み、統合、クリーニングを処理するクラス"""

    def __init__(self):
        self.departure_data = None
        self.past_race_data = None
        self.historical_results = None
        self.reference_races = None
        self.merged_data = None

    def load_departure_csv(self, file_path: str) -> pd.DataFrame:
        """出馬表CSVを読み込む"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"出馬表を読み込みました: {len(df)} 行")
            self.departure_data = df
            return df
        except Exception as e:
            logger.error(f"出馬表の読み込みエラー: {e}")
            return None

    def load_past_race_csv(self, file_path: str) -> pd.DataFrame:
        """出走馬過去レースCSVを読み込む"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"出走馬過去レースを読み込みました: {len(df)} 行")
            self.past_race_data = df
            return df
        except Exception as e:
            logger.error(f"出走馬過去レースの読み込みエラー: {e}")
            return None

    def load_historical_results_csv(self, file_path: str) -> pd.DataFrame:
        """過去n年結果CSVを読み込む（オプション）"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"過去結果を読み込みました: {len(df)} 行")
            self.historical_results = df
            return df
        except Exception as e:
            logger.warning(f"過去結果の読み込み失敗（オプション）: {e}")
            return None

    def load_reference_races_csv(self, file_path: str) -> pd.DataFrame:
        """参考レースCSVを読み込む（オプション）"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"参考レースを読み込みました: {len(df)} 行")
            self.reference_races = df
            return df
        except Exception as e:
            logger.warning(f"参考レースの読み込み失敗（オプション）: {e}")
            return None

    def clean_numeric_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """数値カラムをクリーニング"""
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def clean_departure_data(self) -> pd.DataFrame:
        """出馬表データをクリーニング"""
        if self.departure_data is None:
            logger.error("出馬表データが読み込まれていません")
            return None

        df = self.departure_data.copy()
        numeric_cols = ['waku', 'umaban', 'carried_weight', 'odds', 'popularity']
        df = self.clean_numeric_columns(df, numeric_cols)

        if 'horse_id' in df.columns:
            df['horse_id'] = df['horse_id'].astype(str)

        if 'race_id' in df.columns:
            df['race_id'] = df['race_id'].astype(str)

        df = df.dropna(subset=['horse_id', 'race_id'])
        logger.info(f"出馬表クリーニング完了: {len(df)} 行")
        return df

    def clean_past_race_data(self) -> pd.DataFrame:
        """出走馬過去レースデータをクリーニング"""
        if self.past_race_data is None:
            logger.error("過去レースデータが読み込まれていません")
            return None

        df = self.past_race_data.copy()
        numeric_cols = ['着順', '枠', '馬番', '斤量', '距離', '人気']
        df = self.clean_numeric_columns(df, numeric_cols)

        if '日付' in df.columns:
            df['日付'] = pd.to_datetime(df['日付'], errors='coerce')

        df = df.dropna(subset=['馬名'])
        logger.info(f"過去レースクリーニング完了: {len(df)} 行")
        return df

    def merge_all_data(self) -> pd.DataFrame:
        """全データを統合"""
        if self.departure_data is None:
            logger.error("出馬表データが必須です")
            return None

        merged = self.departure_data.copy()
        self.merged_data = merged
        logger.info(f"データ統合完了: {len(merged)} 行")
        return merged

    def get_processed_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """処理済みデータを返す"""
        departure = self.clean_departure_data()
        past_races = self.clean_past_race_data()
        self.merge_all_data()
        return departure, past_races, self.merged_data
