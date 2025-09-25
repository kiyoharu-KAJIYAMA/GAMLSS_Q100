import os
import numpy as np
import csv
from collections import defaultdict

def main():
    # ディレクトリ設定
    threshold = 10000
    input_dir = "/your/directory/path/ACC_ssp245_amax/tmp_yearwise"
    output_root = f"/your/directory/path/ACC_ssp245_amax/basin_{threshold}"
    os.makedirs(output_root, exist_ok=True)

    # reach_list 読み込み
    csv_path = f"/your/directory/path/reach/reach_list_{threshold}.csv"
    reach_points = []
    with open(csv_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            reach_points.append((float(row["subbasin_id"]), int(row["iy"]), int(row["ix"])))

    # 集計用ディクショナリ
    results = defaultdict(list)

    # 各年の中間ファイルを読み込み
    for year in range(1981, 2015):
        filestr = os.path.join(input_dir, f"amax_year{year}.npy")
        data = np.load(filestr)
        for sub_id, iy, ix, year_val, maxval in data:
            key = (sub_id, int(iy), int(ix))
            results[key].append([int(year_val), maxval])
        print(f"Loaded {filestr}")

    for year in range(2015, 2021):
        filestr = os.path.join(input_dir, f"amax_year{year}.npy")
        data = np.load(filestr)
        for sub_id, iy, ix, year_val, maxval in data:
            key = (sub_id, int(iy), int(ix))
            results[key].append([int(year_val), maxval])
        print(f"Loaded {filestr}")

    for year in range(2021, 2101):
        filestr = os.path.join(input_dir, f"amax_year{year}.npy")
        data = np.load(filestr)
        for sub_id, iy, ix, year_val, maxval in data:
            key = (sub_id, int(iy), int(ix))
            results[key].append([int(year_val), maxval])
        print(f"Loaded {filestr}")

    # 各 reach_point ごとに保存
    for (sub_id, iy, ix), year_flow in results.items():
        array = np.array(sorted(year_flow, key=lambda x: x[0]), dtype="float32")
        fname = f"ix{ix:04d}_iy{iy:04d}"
        fpath = os.path.join(output_root, fname)
        array.tofile(f"{fpath}.bin")
        print(f"Saved: {fpath}.bin")

if __name__ == "__main__":
    main()

