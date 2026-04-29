# Nazuna の設計に関するノート


## `Workflow` クラスと `TaskRunner` クラス

### 設計方針

Nazuna では `Workflow` インスタンスを生成して実行することを基本とする。`nazuna.workflow.run` に設定を記入した TOML パス / TOML 文字列 / 辞書オブジェクトを渡すと設定にしたがって `Workflow` インスタンスを生成して実行することができる。実行すると `tasks` キーの設定にしたがって `TaskRunner` インスタンスのリストが生成され各タスクが順に実行される。

なお、`TaskRunner` インスタンスも `TimeSeriesDataManager` インスタンスを与えれば直接実行できるよう設計する。が、以下の機能は `Workflow` 側が担うため、実用的には `Workflow` を実行する。

- 各タスクが共通に使用する `TimeSeriesDataManager` インスタンスを生成する。
- 先行タスクの成果物 (最良エポック数や訓練済パラメータ) を引き継ぐ。
- レポートを生成する。

!!! Tip

    `Workflow` の設定記入の便利のため、以下の記法を用意している。

    - (設定を TOML パスで渡したとき限定) `out_dir` キーに `"__CONFIG_STEM__"` を指定すると、その TOML パスの拡張子を取ったパスを設定する (`Workflow` クラスに渡す前に解決される)。
    - `definitions` キー &ndash; `tasks` で繰り返し使う設定値を定義して名前で参照できる。さらに、定義自体の記入にも、ベースとする定義名とそれに対する差分で定義することができる (`Workflow` インスタンスを実行し、各 `TaskRunner` インスタンスが生成される時に展開される)。
    - `template` キー &ndash; 典型的なシナリオの `tasks` を自動生成する (`Workflow` インスタンス生成前に `WorkflowTemplateResolver` によって展開される)。


## `BaseModel` クラス

### 設計方針

`BaseModel` クラスのメソッドのうち、以下は派生クラスでのオーバーライドを想定しない (これらのメソッドではなく、これらのメソッドが内部で呼び出すメソッドをオーバーライドする)。

!!! info "`BaseModel` クラスのメソッド (派生クラスでオーバーライドしないもの)"

    - コンストラクタ &ndash; モデルを構築し、実行デバイスに移動する。
    - `create(device, state_path, **setup_args)` (クラスメソッド) &ndash; `state_path` が渡された場合にはインスタンス初期化後に訓練済パラメータをロードする。
        - `scaler` をもつモデルの場合、スケール係数テンソルのロード直前に、予め訓練済パラメータ内のスケール係数と同じ形のゼロテンソルをセットする (便利のためスケーラにはチャネル数の遅延セットを許容しており、最初のバッチが流れてくるまでチャネル数をもたないための措置である)。
    - `get_loss_and_backward(batch, criterion)` &ndash; 勾配を取るべき損失 (`TimeSeriesError` インスタンス) を取得して、バックワードする (＝損失の勾配を各パラメータの `grad` にセットする)。このとき、`TimeSeriesError.grad_target` の勾配を計算する (このデータメンバが未セットの場合はデフォルトで `TimeSeriesError.batch_mean` の勾配を計算する)。

`BaseModel` クラスのメソッドのうち、以下は派生クラスで実装される必要がある (ただし、後述の `BasicBaseModel` を継承すれば **A2, A4, A5, A6** が実装済みである)。

!!! warning "`BaseModel` クラスの抽象メソッド (派生クラスでオーバーライドする必要がある)"

    - **A1.** `_setup(**setup_args)` &ndash; モデル構築に必要なハイパーパラメータを受け取り、モデルを構築する。このメソッドはコンストラクタからよばれる。
    - **A2.** `_extract_input(batch) -> Any` &ndash; バッチからそのモデルの `forward` が要求する入力を抽出する。もしモデルがスケールや差分系列化などのプレ処理を必要とするならば、ここで予め適用する想定である。このメソッドは各モデルの `forward` の引数を原実装に近いものにするためのインターフェースの位置づけである。
    - **A3.** `forward(input_) -> tuple[torch.Tensor, dict[str, Any]]` &ndash; モデル処理本体である。
    - **A4.** `get_loss(batch, criterion) -> TimeSeriesError` &ndash; 勾配を取るべき損失を返す。典型的には以下のステップからなる (が、特殊な目的関数でモデルを訓練する場合はこの限りでない)。このメソッドの役割は `get_loss_and_backward` に損失を供給することである。
        - `input_ = self._extract_input(batch)` によるモデル入力の抽出。
        - `output, info = self.forward(input_)` によるモデル出力の取得。
        - `target = self.extract_true(batch)` (後述) などによる、モデル出力が目指すターゲットの抽出。
        - `output` のリスケール (元の入力空間で損失を計算する場合) 又は `target` のスケール (スケールされた空間で損失を計算する場合)。
        - `loss = criterion(output, target)` による損失の算出。
    - **A5.** `predict(batch) -> tuple[torch.Tensor, dict[str, Any]]` &ndash; バッチから予測値を返す。典型的には以下のステップからなる。このメソッドの役割は criterion が受け取る `pred` を出力することである。
        - `input_ = self._extract_input(batch)` によるモデル入力の抽出。
        - `output, info = self.forward(input_)` によるモデル出力の取得。
        - `output` のリスケール。
    - **A6.** `extract_true(batch) -> Any` &ndash; バッチから真値を抽出する。このメソッドの役割は criterion が受け取る `true` を出力することである。


## `BasicBaseModel` クラス

### 設計方針

`BasicBaseModel(BaseModel)` クラスは、ハイパーパラメータ `seq_len`, `pred_len` をもち、現時点までの `seq_len` ステップの観測値から未来の `pred_len` ステップを予測する基本的なモデルの規定クラスとして用意したものである。

このクラスでは **A2, A4, A5, A6** が以下のように実装済みなので、最低限 **A1, A3** を実装すれば済む。ただし、タイムスタンプを要するクラスの場合は **A2** をオーバーライドしてさらに `bath.tsta[:, -self.seq_len:]` を抽出するなどする必要がある。

!!! info "`BasicBaseModel` クラスでオーバーライド済の抽象メソッド"

    - **A2.** `_extract_input(batch) -> torch.Tensor` &ndash; `batch.data[:, -self.seq_len:]` を抽出し、
        - `scaler` があればスケールする。
        - `prep_type` が指定されていれば系列変換する (差分系列化等)。
    - **A4.** `get_loss(batch, criterion) -> TimeSeriesError` &ndash; 以下のステップからなる。
        - `input_ = self._extract_input(batch)` によるモデル入力の抽出。
        - `output, info = self.forward(input_)` によるモデル出力の取得および `prep_type` 逆変換。
        - `target = self.extract_true(batch)` による、モデル出力が目指すターゲットの抽出。
        - `output` のリスケール (デフォルト) 又は `target` のスケール (`rescale_loss` を `False` にした場合)。
        - `loss = criterion(output, target)` による損失の算出。
    - **A5.** `predict(batch) -> tuple[torch.Tensor, dict[str, Any]]` &ndash; 以下のステップからなる。
        - `input_ = self._extract_input(batch)` によるモデル入力の抽出。
        - `output, info = self.forward(input_)` によるモデル出力の取得および `prep_type` 逆変換。
        - `output` のリスケール。
    - **A6.** `extract_true(batch) -> torch.Tensor` &ndash; `batch.data_future[:, :self.pred_len]` を抽出。


## 各モデルクラスのノート

### モデルクラス横断のノート

- Token ベクトルを標準化する層を `TokenNorm` と名付ける (目的語 + 動詞)。
- Series ベクトルからその平均を差し引く層を `SeriesDemean` と名付ける (目的語 + 動詞)。
- Autoformer, PatchTST では原理上、入力チャネル数 `c_in` と 出力チャネル数 `c_out` を変更できる。そのため、内部実装では変数名が分けられている。ただし、このプロジェクトでは入力チャネル数と出力チャネル数が異なるケースの取り扱いを定めていないため、別々の値に設定することはできない。

### `Autoformer` クラス

公式実装 [51c7d41](https://github.com/thuml/Autoformer/tree/51c7d416ae120b805fd5beef2f4ccf7de496a6ff) に倣っている。

#### モデル構造の詳細

公式実装の詳細は[こちら](https://cookiebox26.github.io/cookipedia/articles/haixu_wu_et_al_2021.html)を参照。

- バッチから **x_enc** (B, L_in, C), **x_mark_enc** (B, L_in, 4), **x_mark_dec** (B, L_label+L_out, n_feat) を抽出する。ただし、x_mark_enc, x_mark_dec はタイムスタンプの hour, day of week, day of month, day of year などの特徴を -0.5～0.5 の値に正規化したものである (抽出する特徴は時間間隔による)。
- x_enc を各時点ごとに `Conv1d(c_in, d_model, kernel_size=3, keep_len=True, padding_mode='circular')` で埋め込み、それに x_mark_enc を `Linear(4, d_model)` で埋め込んだものを加算し、ドロップアウトして**エンコーダ入力** (B, C, d_model) とする。
- x_enc を SeriesDecomp 層で分解する。その内トレンド成分 (B, L_in, C) の末尾側 L_label ステップに、x_enc の時間平均値を L_out ステップ繰り返したもの (B, L_out, C) を concat して**デコーダ入力のトレンド成分** (B, L_label+L_out, C) とする。季節成分 (B, L_in, C) の末尾側 L_label ステップに、L_out ステップ分のゼロテンソルを concat して**デコーダ入力の季節成分** (B, L_label+L_out, C) とする。
- **エンコーダ** &ndash; 以下のエンコーダ層を e_layers 層積み重ねる。

#### Nazuna 版の相違点

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

公式実装の詳細は[こちら](https://cookiebox26.github.io/cookipedia/articles/yuqi_nie_et_al_2023.html)を参照。

#### Nazuna 版の相違点

### `iTransformer` クラス

公式実装 [c2426e6](https://github.com/thuml/iTransformer/tree/c2426e68ca13f74aaec08045c5c724d8ad328124) に倣っている。

#### モデル構造の詳細

公式実装の詳細は[こちら](https://cookiebox26.github.io/cookipedia/articles/yong_liu_et_al_2024.html)を参照。

- 位置エンコーディングや時間特徴エンコーディングはない (転置によって各ステップが各チャネルになったため、これらのようなエンコーディングは意味をなさなくなる)。

#### Nazuna 版の相違点

## その他

- 各タスクの `result.toml` の `env` キーにそのタスクを実行したホスト名、PyTorch のバージョン、デバイス名のリストを記録するが、GPU 環境で敢えて CPU 実行したとしてもデバイス名には GPU 名のリストが記録される。
- モデルのうちクラスをセットアップ引数として要求するものでも、クラスパス文字列でなくクラスを受け取るようにし、`TaskRunner` インスタンス生成時に解決する。クラスパス文字列としないのは、設定ファイルの都合をモデルに押し付けることになるためと、モデル生成時までクラスパスが不正であることを検知できなくなるためである。
