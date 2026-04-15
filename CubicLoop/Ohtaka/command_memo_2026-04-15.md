# Ohtaka 実行準備コマンド備忘録（2026-04-15）

## 1. ローカルでのファイル準備

```bash
cd /Users/ryoma/Library/Mobile\ Documents/com~apple~CloudDocs/川島研/MonteCarlo
cp CubicLOn_parallel_MC.py Ohtaka/
ls -lh "/Users/ryoma/Library/Mobile Documents/com~apple~CloudDocs/川島研/MonteCarlo/Ohtaka/"
```

## 2. リモート転送（初回）

```bash
scp -i /Users/ryoma/.ssh/id_rsa_ohtaka_2026 /Users/ryoma/Library/Mobile\ Documents/com~apple~CloudDocs/川島研/MonteCarlo/Ohtaka/* i002007@ohtaka.issp.u-tokyo.ac.jp:Cubic/
```

## 3. sbatch 実行とエラー対応で確認した情報

```bash
# リモート側で実行
sbatch run_cubiclon_l16_parallel.sh

# QOS/制限の確認
sacctmgr show qos
scontrol show job 2875797
```

## 4. ローカル側でジョブスクリプトを修正した内容

### 4-1. 実行時間を 72h -> 24h に変更

```bash
# run_cubiclon_l16_parallel.sh 内
#SBATCH --time=24:00:00
```

### 4-2. パーティションを修正

```bash
# 誤: part_f1cpu
# 正: F1cpu
#SBATCH --partition=F1cpu
```

### 4-3. メモリ要求を修正

```bash
# 変更前
#SBATCH --mem=240G

# 変更後
#SBATCH --mem=230G
```

## 5. 修正版ジョブスクリプトの再転送

```bash
scp -i /Users/ryoma/.ssh/id_rsa_ohtaka_2026 "/Users/ryoma/Library/Mobile Documents/com~apple~CloudDocs/川島研/MonteCarlo/Ohtaka/run_cubiclon_l16_parallel.sh" i002007@ohtaka.issp.u-tokyo.ac.jp:Cubic/

scp -i /Users/ryoma/.ssh/id_rsa_ohtaka_2026 "/Users/ryoma/Library/Mobile Documents/com~apple~CloudDocs/川島研/MonteCarlo/Ohtaka/run_cubiclon_l16_parallel.sh" i002007@ohtaka.issp.u-tokyo.ac.jp:Cubic/ 2>&1 && echo "✅ 送信完了"
```

## 6. 最終的に使う実行コマンド（リモート側）

```bash
cd ~/Cubic
sbatch run_cubiclon_l16_parallel.sh
```

## 7. 実行監視コマンド（リモート側）

```bash
squeue -j 2875797
squeue --start -j 2875797
tail -f /home/i0020/i002007/Cubic/CubicLOn_L16_parallel_2875797.log
```

## 8. 今回の確定ジョブ情報

- JobId: 2875797
- JobName: CubicLOn_L16_parallel
- Partition: F1cpu
- NumNodes: 1
- NumCPUs: 32
- ReqMem: 230G
- TimeLimit: 1-00:00:00
- State: PENDING (Reason=Priority)
