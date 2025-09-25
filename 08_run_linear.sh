#!/bin/bash
#PBS -q E40
#PBS -l select=1:ncpus=40:mem=128gb
#PBS -o naam_linear.log
#PBS -j oe

# === 環境変数設定 ===
export OMP_NUM_THREADS=1
export KMP_AFFINITY=disabled

# === 作業ディレクトリに移動 ===
cd $PBS_O_WORKDIR

# === conda activate ===
source ~/miniconda3/etc/profile.d/conda.sh  # 環境に合わせて変更
conda activate gam

# === パラメータ ===
SUCCESS_LIST="/your/directory/path/success_list.txt"
FAILED_LIST="/your/directory/path/failed_list_quadratic.txt"

# === 失敗リストを初期化 ===
echo "" > $FAILED_LIST

# === エラーチェック付き並列実行 ===
cat $SUCCESS_LIST | \
while read iy ix
do
  echo "$iy $ix"
done | xargs -n 2 -P 40 bash -c '
  iy=$0
  ix=$1
  if ! Rscript /your/directory/path/07_linear.R "$iy" "$ix"; then
    echo "FAILED: iy=$iy ix=$ix" >> "$FAILED_LIST"
    echo "FAILED: iy=$iy ix=$ix" >&2
  fi
'

