import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineering:
    """特徴量生成クラス"""

    def __init__(self):
        self.past_race_data = None
        self.historical_results = None
        self.reference_races = None
        self.features = None

    def set_data(self, past_races: pd.DataFrame = None, historical: pd.DataFrame = None, reference: pd.DataFrame = None):
        """データセットを設定"""
        self.past_race_data = past_races
        self.historical_results = historical
        self.reference_races = reference

    def extract_horse_stats(self, horse_name: str) -> Dict:
        """馬の過去成績統計を抽出"""
        stats = {
            'win_count': 0,
            'win_rate': 0.0,
            'place_count': 0,
            'place_rate': 0.0,
            'show_count': 0,
            'show_rate': 0.0,
            'avg_finish': 10.0,
            'races_count': 0,
            'recent_form': 0.0,
        }

        if self.past_race_data is None:
            return stats

        horse_races = self.past_race_data[self.past_race_data['馬名'] == horse_name]
        if horse_races.empty:
            return stats

        races_count = len(horse_races)
        stats['races_count'] = races_count

        if races_count == 0:
            return stats

        try:
            finish_positions = pd.to_numeric(horse_races['着順'], errors='coerce')
            finish_positions = finish_positions.dropna()

            if len(finish_positions) > 0:
                stats['win_count'] = len(finish_positions[finish_positions == 1])
                stats['place_count'] = len(finish_positions[finish_positions == 2])
                stats['show_count'] = len(finish_positions[finish_positions == 3])
                stats['avg_finish'] = finish_positions.mean()

                stats['win_rate'] = stats['win_count'] / races_count
                stats['place_rate'] = (stats['win_count'] + stats['place_count']) / races_count
                stats['show_rate'] = (stats['win_count'] + stats['place_count'] + stats['show_count']) / races_count

                recent_5 = finish_positions.tail(5)
                stats['recent_form'] = (1 / (recent_5 + 1)).mean()
        except Exception as e:
            logger.warning(f"馬{horse_name}の統計抽出エラー: {e}")

        return stats

    def extract_jockey_stats(self, jockey_name: str) -> Dict:
        """騎手の成績統計を抽出"""
        stats = {
            'jockey_win_rate': 0.0,
            'jockey_place_rate': 0.0,
            'jockey_races': 0,
        }

        if self.past_race_data is None:
            return stats

        jockey_races = self.past_race_data[self.past_race_data['騎手'] == jockey_name]
        if jockey_races.empty:
            return stats

        races_count = len(jockey_races)
        stats['jockey_races'] = races_count

        try:
            finish_positions = pd.to_numeric(jockey_races['着順'], errors='coerce')
            finish_positions = finish_positions.dropna()

            if len(finish_positions) > 0:
                win_count = len(finish_positions[finish_positions == 1])
                place_count = len(finish_positions[finish_positions == 2])

                stats['jockey_win_rate'] = win_count / len(finish_positions)
                stats['jockey_place_rate'] = (win_count + place_count) / len(finish_positions)
        except Exception as e:
            logger.warning(f"騎手{jockey_name}の統計抽出エラー: {e}")

        return stats

    def extract_trainer_stats(self, trainer_name: str) -> Dict:
        """厩舎の成績統計を抽出"""
        stats = {
            'trainer_win_rate': 0.0,
            'trainer_place_rate': 0.0,
            'trainer_races': 0,
        }

        if self.past_race_data is None:
            return stats

        trainer_races = self.past_race_data[self.past_race_data['厩舎'] == trainer_name]
        if trainer_races.empty:
            return stats

        races_count = len(trainer_races)
        stats['trainer_races'] = races_count

        try:
            finish_positions = pd.to_numeric(trainer_races['着順'], errors='coerce')
            finish_positions = finish_positions.dropna()

            if len(finish_positions) > 0:
                win_count = len(finish_positions[finish_positions == 1])
                place_count = len(finish_positions[finish_positions == 2])

                stats['trainer_win_rate'] = win_count / len(finish_positions)
                stats['trainer_place_rate'] = (win_count + place_count) / len(finish_positions)
        except Exception as e:
            logger.warning(f"厩舎{trainer_name}の統計抽出エラー: {e}")

        return stats

    def generate_features(self, departure_data: pd.DataFrame) -> pd.DataFrame:
        """全特徴量を生成"""
        features = departure_data.copy()

        for idx, row in features.iterrows():
            horse_name = row.get('horse_name', '')
            jockey_name = row.get('jockey_name', '')
            trainer_name = row.get('trainer_name', '')

            horse_stats = self.extract_horse_stats(horse_name)
            features.loc[idx, 'horse_win_rate'] = horse_stats['win_rate']
            features.loc[idx, 'horse_place_rate'] = horse_stats['place_rate']
            features.loc[idx, 'horse_avg_finish'] = horse_stats['avg_finish']
            features.loc[idx, 'horse_recent_form'] = horse_stats['recent_form']
            features.loc[idx, 'horse_races_count'] = horse_stats['races_count']

            jockey_stats = self.extract_jockey_stats(jockey_name)
            features.loc[idx, 'jockey_win_rate'] = jockey_stats['jockey_win_rate']
            features.loc[idx, 'jockey_place_rate'] = jockey_stats['jockey_place_rate']

            trainer_stats = self.extract_trainer_stats(trainer_name)
            features.loc[idx, 'trainer_win_rate'] = trainer_stats['trainer_win_rate']
            features.loc[idx, 'trainer_place_rate'] = trainer_stats['trainer_place_rate']

        self.features = features
        return features
