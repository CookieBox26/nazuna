# ノート


## `Workflow` クラスのノート

- このプロジェクトでは `Workflow` インスタンスを生成して実行することを基本とする。`Workflow` インスタンスを実行すると指定の `TaskRunner` インスタンスのリストが作成され順に実行される。
    - `TaskRunner` インスタンスも `TimeSeriesDataManager` インスタンスを与えれば直接実行できるが、`TimeSeriesDataManager` インスタンス生成機能やレポート機能をもたない。また、 `Workflow` では前方の `TaskRunner` の成果物 (最良エポック数や訓練済パラメータ) を引継げる。
- `nazuna.workflow.run` に設定を記入した TOML パス / TOML 文字列 / 辞書オブジェクトを渡すと `Workflow` インスタンスを生成して実行できる。設定記入の便利のため、以下の記法を用意している。
    - (設定を TOML パスで渡したとき限定) `out_dir` キーに `"__CONFIG_STEM__"` を指定すると、その TOML パスの拡張子を取ったパスを設定する (`Workflow` クラスに渡す前に解決される)。
    - **Definition 記法** &ndash; `tasks` 設定で繰り返し使う設定値を定義しておいて名前で参照できる。さらに、定義自体を記述する際にも、ベースとする定義名とそれに対する差分で定義することができる (`Workflow` インスタンスを実行し、各 `TaskRunner` インスタンスが生成される時解決される)。
    - **Template 記法** &ndash; `tasks` キーを記述する代わりに `template` キーで典型的な `TaskRunner` のリストを設定する (`Workflow` クラスに渡す前に解決される)。

---

## 予測モデルクラスのノート(全般)

- `BaseModel`
- `BasicBaseModel` を継承するクラスは、モデル全体への入力を訓練期間の四分位点で正規化し、出力を逆正規化することができる。

### 備考

- Token ベクトルを標準化する層を `TokenNorm` と名付ける (目的語 + 動詞)。
- Series ベクトルからその平均を差し引く層を `SeriesDemean` と名付ける (目的語 + 動詞)。

---

## 予測モデルクラスのノート (個別)

### モデルクラス横断のノート

- Autoformer, PatchTST では原理上、入力チャネル数 `c_in` と 出力チャネル数 `c_out` を変更できる。そのため、内部実装では変数名が分けられている。ただし、このプロジェクトでは入力チャネル数と出力チャネル数が異なるケースの取り扱いを定めていないため、別々の値に設定することはできない。

### `Autoformer` クラス

公式実装 [51c7d41](https://github.com/thuml/Autoformer/tree/51c7d416ae120b805fd5beef2f4ccf7de496a6ff) に倣っている。

#### モデル構造の詳細

[こちらのページ](https://cookiebox26.github.io/cookipedia/articles/haixu_wu_et_al_2021.html)を参照。

- バッチから **x_enc** (B, L_in, C), **x_mark_enc** (B, L_in, 4), **x_mark_dec** (B, L_label+L_out, n_feat) を抽出する。ただし、x_mark_enc, x_mark_dec はタイムスタンプの hour, day of week, day of month, day of year などの特徴を -0.5～0.5 の値に正規化したものである (抽出する特徴は時間間隔による)。
- x_enc を各時点ごとに `Conv1d(c_in, d_model, kernel_size=3, keep_len=True, padding_mode='circular')` で埋め込み、それに x_mark_enc を `Linear(4, d_model)` で埋め込んだものを加算し、ドロップアウトして**エンコーダ入力** (B, C, d_model) とする。
- x_enc を SeriesDecomp 層で分解する。その内トレンド成分 (B, L_in, C) の末尾側 L_label ステップに、x_enc の時間平均値を L_out ステップ繰り返したもの (B, L_out, C) を concat して**デコーダ入力のトレンド成分** (B, L_label+L_out, C) とする。季節成分 (B, L_in, C) の末尾側 L_label ステップに、L_out ステップ分のゼロテンソルを concat して**デコーダ入力の季節成分** (B, L_label+L_out, C) とする。
- **エンコーダ** &ndash; 以下のエンコーダ層を e_layers 層積み重ねる。

#### Nazuna 版の実装上の相違点

- 原版は **x_dec** を受け取るが、形状しか参照されないため Nazuna では受け取らない。
- 原版では `DataEmbedding_wo_pos`層がタイムスタンプ埋込み

#### Nazuna 版の機能上の相違点

- 以下の相違点があるが、既定のハイパーパラメータでは公式実装の実験設定と同じである。
    - 系列分解層は移動平均を重ね掛けもできる。
    - 自己相関層は相関の強いトップ k のラグを「ヘッド共通」ではなく「ヘッドごと」にもとれる。
    - エンコーダ層・デコーダ層の活性化関数が `GELU` 固定にしている (公式実装では選択でき、デフォルト引数が `relu` だが、実験スクリプトでは `GELU` が指定されている)。
- 以下の相違点は、既定のハイパーパラメータで公式実装の実験設定と異なる (同じにはできる)。
    - ソフトマックス後アテンションをドロップアウトしている。
    - 各箇所のドロップアウトの割合を個別設定可能にしてあり、既定値で変えている。
- 以下の相違点は、公式実装と異なる。
    - top-k の計算式が、公式実装は int(factor * math.log(length)) であるところを、max(1, topk_factor * math.log(l+1)) にしている。

- AutoCorrelation 層

### `PatchTST` クラス

公式実装 [204c21e](https://github.com/yuqinie98/PatchTST/tree/204c21efe0b39603ad6e2ca640ef5896646ab1a9) に倣っている。

#### モデル構造の詳細

[こちらのページ](https://cookiebox26.github.io/cookipedia/articles/yuqi_nie_et_al_2023.html)を参照。

#### Nazuna 版の実装上の相違点

#### Nazuna 版の機能上の相違点

- 


### `iTransformer` クラス


