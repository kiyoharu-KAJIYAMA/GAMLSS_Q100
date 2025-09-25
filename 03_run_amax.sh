#!/bin/bash
#please run without qsub

export OMP_NUM_THREADS=1  # 各プロセス内の並列を防止

for year in $(seq 1981 1996)
do
  python ./02_amax_per_year.py $year &
  if [[ $(jobs -r -p | wc -l) -ge 16 ]]; then  # 最大16並列
    wait -n
  fi
done

wait  # 全部終了まで待つ
