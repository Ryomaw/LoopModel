# Ohtaka 大規模並列計算スケジュール

## 計算概要

### シミュレーション条件
- **格子サイズ**: L = 16
- **n 値**: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100 (13個)
- **x 値**: 0.1 ～ 5.0 (刻み幅 0.1) = 49個
- **ts/os 構成**: 4通り
  - ts=10000, os=40000   (短時間: ~1分/実行)
  - ts=40000, os=40000   (中程度: ~2分/実行)
  - ts=10000, os=160000  (長時間: ~3分/実行)
  - ts=40000, os=160000  (超長時間: ~5分/実行)

### 計算規模
- **総実行数**: 13 × 49 × 4 = 2548 シミュレーション
- **予想計算時間**: 
  - 平均 2-3 分/実行 × 2548 = 5,096-7,644 分≈ 3.5-5.3 日
  - マルチプロセス（32並列）で: ~5-7 時間で全完了
- **必要メモリ**: 240 GB（ノード全体）
- **並列数**: 32 なし（1ノード = 128コア）

---

## 使用ファイル

```
Ohtaka/
├── run_cubiclon_l16_parallel.sh      # 実行スクリプト（このファイル）
└── Partition_Ohtaka.txt               # ノード情報（参考）

MonteCarlo/
├── CubicLOn_parallel_MC.py            # 並列実行版Python
├── CubicLOn_rev_MC.py                 # 元のコード（参考）
└── Ohtaka/ → このディレクトリ
```

---

## ジョブ投入方法

### ステップ 1: ジョブスクリプトの準備

```bash
cd /path/to/MonteCarlo/Ohtaka
chmod +x run_cubiclon_l16_parallel.sh
```

### ステップ 2: ジョブ投入

```bash
sbatch run_cubiclon_l16_parallel.sh
```

出力例：
```
Submitted batch job 12345678
```

### ステップ 3: 進捗確認

リアルタイム監視：
```bash
tail -f CubicLOn_L16_parallel_12345678.log
```

ジョブ状態確認：
```bash
squeue -j 12345678
sinfo -n c01u01n1  # ノード状態確認
```

キャンセル（必要に応じて）：
```bash
scancel 12345678
```

---

## 計算スケジュールの詳細

### フェーズ 1: セットアップ (~5 分)
- 作業ディレクトリ作成
- Python 仮想環境構築（初回のみ）
- 依存パッケージインストール

### フェーズ 2: メイン計算 (~5-7 時間)

**並列戦略**: 
- 32 プロセスで同時実行
- タスク数: 2548
- 各ジョブ: Independent (共有状態なし)
- Seed 管理: 自動で衝突回避

**タイムライン（概算）**:
- n=1: 49×4 = 196 タスク → ~12 分
- n=2: 196 タスク → ~12 分
- ...
- n=100: 196 タスク → ~12 分
- 合計: 2548 タスク / 32 並列 ≈ 80 バッチ × ~5 分 = **~400 分 ≈ 6.7 時間**

### フェーズ 3: 後処理 (~5 分)
- CSV 統合
- 結果サマリー出力
- ファイル削除・整理

---

## リソース効率

### 使用リソース
| リソース | 設定値 | 備考 |
|---------|-------|------|
| ノード数 | 1 | F72cpu パーティション |
| CPU数 | 32 | cpus-per-task |
| メモリ | 240 GB | ノード全体割り当て |
| 時間制限 | 72:00:00 | 十分な余裕 |
| ストレージ | 計算後 ~500MB-1GB | CSV + ログ |

### メモリ使用量（実際）
- 基本オーバーヘッド: ~20 GB
- 並列プロセス x32: 各 ~5-10 MB
- 結果バッファ: ~50 GB（最大）
- **合計**: ~70-100 GB / 240 GB → **30-40% 利用**

### CPU利用率
- 期待値: **~90-95%** （32プロセス ÷ 128コア = 25%, ただしマルチスレッド効果で向上）

---

## 出力結果

### ディレクトリ構成（完了後）
```
/work/users/$(whoami)/CubicLOn_L16/
└── results/
    ├── results.csv           # メイン結果ファイル (2548行)
    └── ...その他バッチ処理ファイル
```

### CSV フォーマット

各行: 1シミュレーション結果
- カラム数: 41
- サンプル行:
```
lattice_size,n,x,thermalization,observation,average_length,...
16,1,0.1,10000,40000,3.245,0.123,...
16,1,0.2,10000,40000,4.521,0.156,...
...
```

### 分析例
```python
import pandas as pd

df = pd.read_csv('/work/users/username/CubicLOn_L16/results/results.csv')

# n=1, x=1.0 のデータ抽出
subset = df[(df['n'] == 1) & (df['x'] == 1.0)]

# 平均長の ts/os 依存性
ts_os_dep = df[df['x'] == 1.0].groupby(['thermalization', 'observation'])['average_length'].mean()
print(ts_os_dep)
```

---

## トラブルシューティング

### ジョブが Pending 状態で進まない
```bash
# ノード状態確認
sinfo -N -l

# パーティション確認
sinfo -p F72cpu
sinfo -p B72cpu

# 別パーティションで再投入
sbatch -p B72cpu run_cubiclon_l16_parallel.sh
```

### ジョブがす停止した場合
```bash
# ログ確認
cat CubicLOn_L16_parallel_JOBID.err

# よくある原因:
# 1. メモリ不足: --mem を増やす（現在 240G は十分）
# 2. タイムアウト: --time を延長
# 3. プロセス数: --cpus-per-task を減らす
```

### 部分的に再実行（推奨）
```bash
# 一部の n のみ再計算
python CubicLOn_parallel_MC.py \
    --lattice_size 16 \
    --n_values 1 2 3 \
    --x_min 0.1 --x_max 5.0 --x_step 0.1 \
    --output_dir /work/users/username/CubicLOn_L16/results_subset \
    --num_workers 32
```

---

## パフォーマンス最適化のヒント

### 1. マルチプロセス数の調整
```bash
# cpus-per-task を変更（1-64を推奨）
#SBATCH --cpus-per-task=48  # 最大リソース
#SBATCH --cpus-per-task=16  # 最小リソース
```

### 2. ノード数拡大（将来的）
```bash
# 複数ノード対応版（MPI）は追加実装が必要
#SBATCH --nodes=4
#SBATCH --tasks-per-node=8
```

### 3. 計算時間削減
- `ts` を減らす（収束性を損なわない範囲で）
- `block_size` を 100 → 50 に削減
- `x_step` を 0.1 → 0.2 に粗くする

---

## 推奨スケジュール

### シナリオ A: 最短実行（24 時間以内）
```bash
# ts を半減、n を絞る
python CubicLOn_parallel_MC.py \
    --lattice_size 16 \
    --n_values 1 2 5 10 50 100 \   # 6個のみ
    --x_min 0.1 --x_max 5.0 --x_step 0.2 \  # 25個のポイント
    --output_dir results_quick \
    --num_workers 32
# 予想: ~2 時間
```

### シナリオ B: 標準実行（48 時間以内）
```bash
# 提供スクリプト（2548 全タスク）
sbatch run_cubiclon_l16_parallel.sh
# 予想: ~7 時間
```

### シナリオ C: 高精度実行（72 時間以上）
```bash
# ts/os を倍に
ts_os_configs = [
    (20000, 80000),
    (80000, 80000),
    (20000, 320000),
    (80000, 320000),
]
# 予想: ~14 時間
```

---

## 運用ガイド

### ビジネスアワー外での投入（推奨）
```bash
# 夜間投入: 7PM-9AM の利用
sbatch --begin=22:00 run_cubiclon_l16_parallel.sh
```

### 定期実行（cron）
```bash
# ~1 月ごとに新パラメータで実行
0 22 * * 0 cd /path/to/MonteCarlo && sbatch Ohtaka/run_cubiclon_l16_parallel.sh
```

### 結果の定期的なバックアップ
```bash
# 完了後、ローカルにダウンロード
rsync -avz \
    user@ohtaka:/work/users/username/CubicLOn_L16/results/ \
    ~/backup_cubiclon_L16_$(date +%Y%m%d)/
```

---

## その他の注意事項

1. **セキュリティ**: スクリプト内のメールアドレスを更新してください
2. **バージョン管理**: Python 3.11 以上推奨
3. **ライセンス**: NumPy, PIL は MIT ライセンス（商用利用可）
4. **結果再現性**: 同じ seed で同じ結果が得られます

---

**最終更新**: 2026-04-15
**推奨投入時期**: 即座（スケジュール許可後）
