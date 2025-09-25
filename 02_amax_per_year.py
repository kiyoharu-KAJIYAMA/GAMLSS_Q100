import os
import csv
import numpy as np

def load_outflow_one_year(year):
    ny, nx = 1800, 3600
    if year < 2015:
        path = f"/your/directory/path/C6-g06M_ACC_hist/outflw{year}.bin"
    elif 2015 <= year < 2021:
        path = f"/your/directory/path/C6-g06M_ACC_ssp245_y2015-2022/outflw{year}.bin"
    else:
        path = f"/your/directory/path/C6-g06M_ACC_ssp245/outflw{year}.bin"
    data = np.fromfile(path, "float32").reshape(-1, ny, nx)
    return data

def main(year):
    threshold = 10000
    ny, nx = 1800, 3600
    output_root = "/your/directory/path/ACC_ssp245_amax/tmp_yearwise"
    os.makedirs(output_root, exist_ok=True)

    csv_path = f"/your/directory/path/reach/reach_list_{threshold}.csv"
    reach_points = []
    with open(csv_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            reach_points.append((float(row["subbasin_id"]), int(row["iy"]), int(row["ix"])))

    dat = load_outflow_one_year(year)
    results = []

    for sub_id, iy, ix in reach_points:
        maxval = np.max(dat[:, iy, ix])
        results.append([sub_id, iy, ix, year, maxval])

    np.save(os.path.join(output_root, f"amax_year{year}.npy"), np.array(results, dtype="float32"))
    print(f"Year {year} processed and saved.")

if __name__ == "__main__":
    import sys
    year = int(sys.argv[1])
    main(year)

