# Reference (Models)

## Models

::: nazuna.models.simple_average.SimpleAverage

=== "en"

    SimpleAverage is a baseline model that predicts future values by computing a weighted average of past periodic segments. It assumes that the input time series has a periodic structure (e.g., daily patterns with a 24-hour period).

    **Input Processing:**

    1. **Extract input**: The model extracts the last `seq_len` timesteps from `batch.data`. The shape is `[batch_size, seq_len, n_channel]`.

    2. **Reshape into periods**: The input is reshaped into periodic segments. For example, if `seq_len=96` and `period_len=24`, the input is reshaped to `[batch_size, 4, 24, n_channel]`, representing 4 periods of 24 timesteps each.

    3. **Compute weighted average**: Each period is assigned a weight based on a decay rate. More recent periods receive higher weights (when `decay_rate < 1.0`). The weights are normalized to sum to 1. The weighted average is computed across periods using `torch.einsum`.

    4. **Output**: The result is the weighted average of the periodic segments, with shape `[batch_size, period_len, n_channel]`. This represents the predicted pattern for one period, which is used as the forecast.

=== "ja"

    SimpleAverage は、過去の周期的なセグメントの加重平均を計算して将来の値を予測するベースラインモデルです。入力時系列が周期的な構造 (例：24時間周期の日次パターン) を持つことを前提としています。

    **入力データの処理:**

    1. **入力の抽出**: `batch.data` から最後の `seq_len` タイムステップを抽出します。形状は `[batch_size, seq_len, n_channel]` です。

    2. **周期ごとに再形成**: 入力を周期的なセグメントに再形成します。例えば、`seq_len=96`、`period_len=24` の場合、入力は `[batch_size, 4, 24, n_channel]` に再形成され、24タイムステップの4周期分を表します。

    3. **加重平均の計算**: 各周期には減衰率に基づいて重みが割り当てられます。より新しい周期ほど高い重みを受けます（`decay_rate < 1.0` の場合）。重みは合計が1になるように正規化されます。`torch.einsum` を使用して周期間の加重平均を計算します。

    4. **出力**: 結果は周期的セグメントの加重平均で、形状は `[batch_size, period_len, n_channel]` です。これは1周期分の予測パターンを表し、予測として使用されます。


::: nazuna.models.circular.Circular

=== "en"

    Circular is a model that uses periodic sinusoidal features for time-series forecasting. Instead of using past data values, it uses the future timesteps to generate sin/cos features for multiple periods, then applies a linear transformation to produce predictions.

    **Input Processing:**

    1. **Extract input**: Unlike other models, Circular extracts `batch.tste_future` (future timesteps) instead of `batch.data`. The timesteps are integer values representing time indices (e.g., 0, 1, 2, ...).

    2. **Generate sin/cos features**: For each timestep, the model generates sin and cos values for multiple periods (default: 2 to 24):
       - For each period `n` and timestep `t`, compute the phase: `phase = t mod n`
       - Generate `sin(2π * phase / n)` and `cos(2π * phase / n)`
       - Note: For period 2, sin is always 0, so only cos is used.
       - The features are pre-computed and cached for efficiency.

    3. **Stack features**: All sin/cos features are stacked to form a feature vector. The total number of features is `2 * len(periods) - 1` (since period 2's sin is excluded).

    4. **Linear transformation**: The stacked features `[batch_size, pred_len, len_features]` are passed through a linear layer that maps `len_features` → `n_channel`.

    **Key Difference**: This model captures time-based periodic patterns (e.g., hour of day, day of week) rather than learning from past data values. It's useful for modeling seasonality that depends purely on the time index.

=== "ja"

    Circular は、時系列予測のために周期的な正弦波特徴量を使用するモデルです。過去のデータ値を使用する代わりに、将来のタイムステップを使用して複数の周期のsin/cos特徴量を生成し、線形変換を適用して予測を生成します。

    **入力データの処理:**

    1. **入力の抽出**: 他のモデルとは異なり、Circularは `batch.data` ではなく `batch.tste_future`（将来のタイムステップ）を抽出します。タイムステップは時間インデックスを表す整数値（例：0, 1, 2, ...）です。

    2. **sin/cos特徴量の生成**: 各タイムステップに対して、モデルは複数の周期（デフォルト：2から24）のsinとcos値を生成します：
       - 各周期 `n` とタイムステップ `t` に対して、位相を計算：`phase = t mod n`
       - `sin(2π * phase / n)` と `cos(2π * phase / n)` を生成
       - 注意：周期2の場合、sinは常に0なので、cosのみが使用されます。
       - 特徴量は効率化のために事前計算されキャッシュされます。

    3. **特徴量のスタック**: すべてのsin/cos特徴量がスタックされて特徴量ベクトルを形成します。特徴量の総数は `2 * len(periods) - 1` です（周期2のsinが除外されるため）。

    4. **線形変換**: スタックされた特徴量 `[batch_size, pred_len, len_features]` は、`len_features` → `n_channel` をマッピングする線形層に渡されます。

    **主な違い**: このモデルは過去のデータ値から学習するのではなく、時間ベースの周期的パターン（例：1日のうちの時間、曜日）を捉えます。純粋に時間インデックスに依存する季節性をモデル化するのに有用です。


::: nazuna.models.dlinear.DLinear

=== "en"

    DLinear is a decomposition-based linear model for time-series forecasting. It decomposes the input time series into trend and seasonal (residual) components, then applies separate linear transformations to each component.

    **Input Processing:**

    1. **Extract input and scale**: The model extracts the last `seq_len` timesteps from `batch.data`. Then it applies IQR-based scaling using quantiles (q1, q2, q3) obtained from the batch. The scaling formula is: `scaled = (x - q2) / (q3 - q1)`, where q2 is the median and (q3 - q1) is the interquartile range. The quantile source is determined by `quantile_mode` ('full', 'cum', or 'rolling').

    2. **Series decomposition**: The scaled input is decomposed into trend and seasonal components using a moving average filter:
       - **Trend component**: Computed by applying an average pooling with `kernel_size`. The input is padded by repeating the first and last values to maintain the sequence length.
       - **Seasonal component**: Computed as the residual: `seasonal = input - trend`.

    3. **Linear transformation**: Each component is permuted to `[batch_size, n_channel, seq_len]` and passed through a separate linear layer:
       - `Linear_Seasonal`: Maps `seq_len` → `pred_len` for the seasonal component.
       - `Linear_Trend`: Maps `seq_len` → `pred_len` for the trend component.

    4. **Combine and output**: The outputs of the two linear layers are summed and permuted back to `[batch_size, pred_len, n_channel]`.

    5. **Rescale for prediction**: When using `predict()`, the output is rescaled back to the original scale using the inverse transformation: `output = scaled * (q3 - q1) + q2`.

=== "ja"

    DLinear は、時系列予測のための分解ベースの線形モデルです。入力時系列をトレンド成分と季節性（残差）成分に分解し、各成分に別々の線形変換を適用します。

    **入力データの処理:**

    1. **入力の抽出とスケーリング**: `batch.data` から最後の `seq_len` タイムステップを抽出します。次に、バッチから取得した分位点（q1, q2, q3）を使用してIQRベースのスケーリングを適用します。スケーリングの式は：`scaled = (x - q2) / (q3 - q1)` です。ここで q2 は中央値、(q3 - q1) は四分位範囲です。分位点のソースは `quantile_mode`（'full', 'cum', 'rolling'）で決定されます。

    2. **時系列分解**: スケーリングされた入力は移動平均フィルタを使用してトレンド成分と季節性成分に分解されます：
       - **トレンド成分**: `kernel_size` での平均プーリングを適用して計算されます。系列長を維持するために、最初と最後の値を繰り返してパディングします。
       - **季節性成分**: 残差として計算されます：`seasonal = input - trend`

    3. **線形変換**: 各成分は `[batch_size, n_channel, seq_len]` に並べ替えられ、別々の線形層に渡されます：
       - `Linear_Seasonal`: 季節性成分に対して `seq_len` → `pred_len` をマッピング
       - `Linear_Trend`: トレンド成分に対して `seq_len` → `pred_len` をマッピング

    4. **結合と出力**: 2つの線形層の出力は加算され、`[batch_size, pred_len, n_channel]` に並べ替えられます。

    5. **予測時のリスケール**: `predict()` を使用する場合、出力は逆変換を使用して元のスケールに戻されます：`output = scaled * (q3 - q1) + q2`


::: nazuna.models.patchtst.PatchTST

=== "en"

    PatchTST is a Transformer-based model that uses the PatchTST architecture. It divides the input time series into patches and processes them using a Transformer encoder. Each channel is processed independently (channel-independent design).

    **Input Processing:**

    1. **Extract input and scale**: The model extracts the last `seq_len` timesteps from `batch.data` and applies IQR-based scaling (same as DLinear).

    2. **Patchify**: The scaled input `[batch_size, seq_len, n_channel]` is divided into overlapping patches:
       - Transpose to `[batch_size, n_channel, seq_len]`
       - Apply `unfold` with `patch_len=8` and `stride=4` to get `[batch_size, n_channel, n_patches, patch_len]`
       - Number of patches: `n_patches = 1 + (seq_len - patch_len) // stride`

    3. **Patch normalization**: Each patch is normalized using LayerNorm (if `use_layernorm_patch=True`).

    4. **Reshape for Transformer**: Reshape to `[batch_size * n_channel, n_patches, patch_len]` to process each channel independently.

    5. **Patch projection**: Each patch is projected to the model dimension using a linear layer: `patch_len` → `d_model` (32 by default).

    6. **Positional encoding**: Sinusoidal positional encoding is added to encode the position of each patch within the sequence.

    7. **Transformer encoder**: The patches are processed by a Transformer encoder with:
       - `n_layers=2` encoder layers
       - `n_heads=2` attention heads
       - `d_ff=128` feedforward dimension
       - Pre-LayerNorm architecture (`norm_first=True`)
       - GELU activation

    8. **Pooling**: Either the last patch representation or the mean of all patches is taken (controlled by `pool` parameter, default is 'last').

    9. **Prediction head**: A linear layer maps from `d_model` to `pred_len`.

    10. **Reshape output**: The output is reshaped from `[batch_size * n_channel, pred_len]` to `[batch_size, pred_len, n_channel]`.

    11. **Rescale for prediction**: When using `predict()`, the output is rescaled back to the original scale.

=== "ja"

    PatchTST は、PatchTSTアーキテクチャを使用したTransformerベースのモデルです。入力時系列をパッチに分割し、Transformerエンコーダで処理します。各チャンネルは独立して処理されます（チャンネル独立設計）。

    **入力データの処理:**

    1. **入力の抽出とスケーリング**: `batch.data` から最後の `seq_len` タイムステップを抽出し、IQRベースのスケーリングを適用します（DLinearと同じ）。

    2. **パッチ化**: スケーリングされた入力 `[batch_size, seq_len, n_channel]` はオーバーラップするパッチに分割されます：
       - `[batch_size, n_channel, seq_len]` に転置
       - `patch_len=8`、`stride=4` で `unfold` を適用し、`[batch_size, n_channel, n_patches, patch_len]` を取得
       - パッチ数：`n_patches = 1 + (seq_len - patch_len) // stride`

    3. **パッチ正規化**: 各パッチはLayerNormで正規化されます（`use_layernorm_patch=True` の場合）。

    4. **Transformer用に再形成**: 各チャンネルを独立して処理するために `[batch_size * n_channel, n_patches, patch_len]` に再形成します。

    5. **パッチ投影**: 各パッチは線形層を使用してモデル次元に投影されます：`patch_len` → `d_model`（デフォルトは32）。

    6. **位置エンコーディング**: シーケンス内の各パッチの位置をエンコードするために、正弦波位置エンコーディングが追加されます。

    7. **Transformerエンコーダ**: パッチは以下の構成のTransformerエンコーダで処理されます：
       - `n_layers=2` エンコーダ層
       - `n_heads=2` アテンションヘッド
       - `d_ff=128` フィードフォワード次元
       - Pre-LayerNormアーキテクチャ（`norm_first=True`）
       - GELU活性化

    8. **プーリング**: 最後のパッチ表現または全パッチの平均が取られます（`pool` パラメータで制御、デフォルトは 'last'）。

    9. **予測ヘッド**: 線形層が `d_model` から `pred_len` にマッピングします。

    10. **出力の再形成**: 出力は `[batch_size * n_channel, pred_len]` から `[batch_size, pred_len, n_channel]` に再形成されます。

    11. **予測時のリスケール**: `predict()` を使用する場合、出力は元のスケールに戻されます。
