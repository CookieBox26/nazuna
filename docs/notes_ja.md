# ノート


## `Workflow` クラスについて

- このプロジェクトでは `Workflow` インスタンスを生成して実行することを基本とする。`Workflow` インスタンスを実行すると指定の `TaskRunner` インスタンスのリストが作成され順に実行される。
    - `TaskRunner` インスタンスも `TimeSeriesDataManager` インスタンスを与えれば直接実行できるが、`TimeSeriesDataManager` インスタンス生成機能やレポート機能をもたない。また、 `Workflow` では前方の `TaskRunner` の成果物 (最良エポック数や訓練済パラメータ) を引継げる。
- `nazuna.workflow.run` に設定を記入した TOML パス / TOML 文字列 / 辞書オブジェクトを渡すと `Workflow` インスタンスを生成して実行できる。便利のため、以下の記法を用意している。
    - (設定を TOML パスで渡したとき限定) `out_dir` キーに `"__CONFIG_STEM__"` を指定すると、その TOML パスの拡張子を取ったパスを設定する (`Workflow` クラスに渡す前に解決される)。
    - **definition 記法** &ndash; `tasks` 設定で繰り返し使う設定値を定義しておいて名前で参照できる。さらに、定義自体を記述する際にも、ベースとする定義名とそれに対する差分で定義することができる (`Workflow` インスタンスを実行し、各 `TaskRunner` インスタンスが生成される時解決される)。
    - **template 記法** &ndash; `tasks` キーを記述する代わりに `template` キーで典型的な `TaskRunner` のリストを設定する (`Workflow` クラスに渡す前に解決される)。


## 各モデルクラスについて

### `Autoformer` クラスについて

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
