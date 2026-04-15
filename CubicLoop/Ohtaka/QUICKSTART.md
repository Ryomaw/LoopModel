# 大規模並列計算 クイックスタートガイド

## 概要
L=16, n=1～100, x=0.1～5.0 で、ts/os4通りの大規模Monte Carlo計算を実行します。

- **計算規模**: 2548個のシミュレーション
- **予想時間**: ~7時間（32並列）
- **必要環境**: Ohtaka スーパーコンピュータ

---

## 1分で始める

### パターン A: すぐに実行
```bash
cd ~/MonteCarlo/Ohtaka
sbatch run_cubiclon_l16_parallel.sh
```

### パターン B: インタラクティブ投入
```bash
cd ~/MonteCarlo/Ohtaka
./submit_job.sh
# メニューから選択 (選択肢 1～5)
```

### パターン C: テスト実行（推奨）
```bash
cd ~/MonteCarlo/Ohtaka
./submit_job.sh
# メニューから「2) Quick run」を選択 (~2時間で完了)
```

---

## 進捗確認方法

### ジョブ状態確認
```bash
squeue -u $(whoami)
```

### ログファイルを監視
```bash
tail -f CubicLOn_L16_parallel_*.log
```

### ノード状態確認
```bash
sinfo
```

---

## 計算完了後

### 結果の確認
```bash
ls -lh /work/users/$(whoami)/CubicLOn_L16/results/
cat /work/users/$(whoami)/CubicLOn_L16/results/results.csv | head -10
```

### 結果の解析
```bash
cd ~/MonteCarlo/Ohtaka

# サマリー表示
python analyze_results.py /work/users/$(whoami)/CubicLOn_L16/results/results.csv --summary

# n ごとに分割
python analyze_results.py /work/users/$(whoami)/CubicLOn_L16/results/results.csv --export-by-n ./results_by_n

# ts/os ごとに分割
python analyze_results.py /work/users/$(whoami)/CubicLOn_L16/results/results.csv --export-by-ts-os ./results_by_config
```

---

## ファイル構成

```
Ohtaka/
├── run_cubiclon_l16_parallel.sh    ← メイン実行スクリプト
├── submit_job.sh                    ← インタラクティブ投入ツール
├── analyze_results.py               ← 結果解析ツール
├── README_Schedule.md               ← 詳細ドキュメント
└── Partition_Ohtaka.txt             ← ノード情報（参考）

MonteCarlo/
├── CubicLOn_parallel_MC.py          ← 並列実行版コード
├── CubicLOn_rev_MC.py               ← 元のコード（参考）
└── Ohtaka/                          ← (このディレクトリ)
```

---

## トラブル時の対応

### ジョブがペンディング状態
```bash
# パーティション状態確認
sinfo -p F72cpu
sinfo -p B72cpu

# 別パーティションで再投入
sbatch -p B72cpu ~/MonteCarlo/Ohtaka/run_cubiclon_l16_parallel.sh
```

### ジョブがキャンセルされた
```bash
# ログ確認
cat CubicLOn_L16_parallel_JOBID.err

# 時間不足の場合
sbatch --time=96:00:00 run_cubiclon_l16_parallel.sh

# メモリ不足の場合（実装により）
sbatch --mem=256G run_cubiclon_l16_parallel.sh
```

### 部分的に再計算したい
```bash
python ~/MonteCarlo/CubicLOn_parallel_MC.py \
    --lattice_size 16 \
    --n_values 1 2 3 \
    --x_min 0.1 --x_max 5.0 --x_step 0.1 \
    --output_dir /work/users/$(whoami)/results_subset \
    --num_workers 32
```

---

## パラメータのカスタマイズ

スクリプトの拡張例：

```bash
# n値を減らして高速化
./submit_job.sh
# →選択肢 3 (Custom parameters)
# n_values: 1 2 5 10 50 100
# x_step: 0.2
```

---

## 詳細情報

詳細な計算スケジュール、パフォーマンス最適化、トラブルシューティング等は
`README_Schedule.md` を参照してください。

---

**最終確認**:
- [ ] `run_cubiclon_l16_parallel.sh` が実行可能か確認
- [ ] Ohtaka のログインノードからアクセス可能か確認
- [ ] `/work/users/$(whoami)/` ディレクトリにアクセス権があるか確認

準備完了！さあ、計算を開始しましょう！
