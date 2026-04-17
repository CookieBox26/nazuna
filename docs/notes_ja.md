# ノート


### Autoformer モデルについて

#### モデル構造の詳細
公式実装 [51c7d41](https://github.com/thuml/Autoformer/tree/51c7d416ae120b805fd5beef2f4ccf7de496a6ff) に倣っている。

- バッチから x_enc (B×L_in×C), x_mark_enc (B×L_in×4), x_mark_dec (B×(L_label+L_out)×4) を抽出する。ただし、x_mark_enc, x_mark_dec はタイムスタンプの hour, day of week, day of month, day of year を -0.5～0.5 の値に正規化したものである。
- x_enc のトレンド成分 (B×L_in×C) の末尾側 L_label ステップと x_enc の時間平均値を L_out ステップ繰り返したもの (B×L_out×C) を concat してデコーダ入力のトレンド成分 trend_init (B×(L_label+L_out)×C) とする。
- x_enc の季節成分 (B×L_in×C) の末尾側 L_label ステップをゼロパディングしてデコーダ入力の季節成分 seasonal_init (B×(L_label+L_out)×C) とする。

#### 公式実装との相違点

- 以下の相違点があるが、既定のハイパーパラメータでは公式実装の実験設定と同じである。
    - モデル全体への入力を訓練期間の四分位点で正規化し、出力を逆正規化することもできる。
    - 系列分解層は移動平均を重ね掛けもできる。
    - 自己相関層は相関の強いトップ k のラグを「ヘッド共通」ではなく「ヘッドごと」にもとれる。
    - エンコーダ層・デコーダ層の活性化関数が `GELU` 固定にしている (公式実装では選択でき、デフォルト引数が `relu` だが、実験スクリプトでは `GELU` が指定されている)。
- 以下の相違点は、既定のハイパーパラメータで公式実装の実験設定と異なる (同じにはできる)。
    - ソフトマックス後アテンションをドロップアウトしている。
    - 各箇所のドロップアウトの割合を個別設定可能にしてあり、既定値で変えている。
- 以下の相違点は、公式実装と異なる。
    - top-k の計算式が、公式実装は int(factor * math.log(length)) であるところを、max(1, topk_factor * math.log(l+1)) にしている。
