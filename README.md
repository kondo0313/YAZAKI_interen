<<<<<<< HEAD
# JA向けミニトマト出荷量予測アプリの開発

## 概要  
- 現状、2週間後のミニトマトの出荷量の予測の精度が芳しくなく、イレギュラーな出荷爆発によって価格の暴落が起きることがある
- 本アプリケーションはAIを用いた精度の高い予測を提供することにより、価格の暴落を抑え、生産者の利益を安定させる効果が期待される
- また、予測をアプリ化することにより、予測業務の属人化を防ぐことも副次効果として期待される

## 予測の使い道や構想
- JAでは出荷量の予測が難しいために、出荷先を複数用意して急激な増加の際に投げ売りとならないような戦略をとっている
- 理想は平均単価の高い東京青果へ多く売ること
- しかし、東京青果へ多く売るためには、前もってある程度正確な量を市場に伝える必要がある
- 上記の「ある程度正確な量」をなるべく高い精度で、専門性を必要とせずに提供する方法として予測モデルを用いたアプリケーション開発を行う

## 特徴
- 特徴1:プロジェクトの主要な機能
- 特徴2:ユーザーがどのようにプロジェクトから利益を得るか
- 特徴3:技術的な突出点やイノベーションを強調

## 技術スタック
- **言語とフレームワーク**: Python, React.js, Django
- **データベース**: csv, pandas
- **機械学習ライブラリ**: Tensorflow, scikit-learn
- **その他ツール**: git 

## ディレクトリ構造
- **/src**: アプリケーションとモデルのソースコード。
- **/data**: トレーニング、検証、テストデータセットを格納。
- **/notebooks**: 実験用のJupyterノートブック。
- **/models**: トレーニング済みモデルとモデルアーキテクチャ。
- **/scripts**: データ処理やモデルトレーニングの自動化スクリプト。
- **/docs**: プロジェクト関連文書とAPIドキュメント。
- **/tests**: 単体テストや統合テストのコード。
- **/config**: 実行環境ごとの設定ファイル。

## 環境構築
### セットアップ手順
1. リポジトリをクローン:  
``` git clone https://github.com/Bib2bib2/JA.git ```
2. Dockerの立ち上げ：  
``` docker compose up -d --build ```

## 使用方法
アプリの実際の使い方を詳細に記載する。必要に応じてスクリーンショットやコマンドの例を挿入

## 内部リンク
- [アプリURL](https://ja-web.onrender.com/pred_ai)

## git周りのルール

mainブランチへのプルリクエストの際は以下のような対応をお願いします。

- プルリクエストのレビュワーは自分以外の2名とする（ex. yuuki330のプルリクエストのレビュワー → githubfreezing、kondo0313）
- レビュワーに指定された人は以下の観点を特に注意してフィードバックする
1. そのコードは何のために追加されたのか？（レビュワーが理解していること）
2. webアプリの動作に不具合を起こさないか？（特に上野さんにチェックしていただきたいです）
3. その他細かな疑問点はあるか？
4. プルリクエストのレビューは、出されてから１出勤以内で行う
=======
# YAZAKI_interen
>>>>>>> f77d95b1d0484a1890166673dc335ae5727dabc0