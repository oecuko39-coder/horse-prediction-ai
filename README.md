# Horse Racing Prediction AI

競馬予想AIシステム - 機械学習を用いた複合予想（1-3着予測）

## 機能

- **複数レース対応**: 日本ダービーを含む全レースに対応
- **CSV自動処理**: 出馬表・過去レース・参考レースCSVを自動統合
- **柔軟なデータ対応**: 参考レース(④)・過去成績(②)がなくても動作
- **コース傾向学習**: 競馬場・コース・天気・馬場状態の自動分析
- **インタラクティブUI**: Streamlitによるレース条件入力
- **複合予想**: 各馬の1着～3着確率を予測

## 必要なデータ

### 必須
- ① 出馬表CSV（race_id, horse_id, odds, popularity等）
- ③ 出走馬過去レースCSV（馬の過去成績）

### オプション（なくても動作）
- ② 過去n年分結果CSV（同レースの過去成績）
- ④ 参考レースCSV（同条件レースの結果）

## インストール

```bash
git clone https://github.com/oecuko39-coder/horse-prediction-ai.git
cd horse-prediction-ai
pip install -r requirements.txt
```

## 使用方法

```bash
streamlit run app.py
```

1. CSV ファイルをアップロード（出馬表、過去レース、参考レース等）
2. レース条件を入力（競馬場、天気、馬場、距離等）
3. 予想実行
4. 結果表示（各馬の1-3着確率、信頼度スコア等）

## ファイル構成

```
horse-prediction-ai/
├── requirements.txt           # 依存ライブラリ
├── README.md
├── app.py                    # Streamlit メインアプリ
├── data_processor.py         # CSV処理・データ統合
├── feature_engineering.py    # 特徴量生成
├── model_trainer.py          # モデル学習・最適化
├── predictor.py              # 予測エンジン
├── utils.py                  # ユーティリティ関数
└── models/                   # 学習済みモデル保存先
```

## 予測モデル

- **XGBoost**（メインモデル）
- **LightGBM**（高速予測）
- **Random Forest**（解釈性重視）

複数モデルのアンサンブル予測

## 特徴量

### 馬の特徴
- 過去成績統計（勝率、連対率、複勝率、平均着順）
- タイムレーティング
- 馬体重・体重差のパターン
- 負担斤量

### 騎手・厩舎特徴
- 騎手の成績統計
- 厩舎の成績統計
- ペアリング成功率

### コース傾向
- 競馬場別勝率
- コース別勝率
- 天気別勝率
- 馬場状態別勝率
- 距離適性

## ライセンス

MIT
