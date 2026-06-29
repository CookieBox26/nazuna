# 更新情報

- nazuna.models.UniTSTLike にクラスメソッド calc_dists を追加した。
- nazuna.optimizers.Adam を追加した。オプティマイザとしての動作は torch.optim.Adam と同じだが、訓練中の勾配や更新幅の履歴の記録機能をもつ。実際に記録する場合は `record_norms=True` を指定する。
