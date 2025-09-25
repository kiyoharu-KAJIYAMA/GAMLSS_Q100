import csv
import rasterio
import numpy as np

class PREPROCESS():
    def __init__(self):
        self.size_thres = 10000 # km2

        self.datadir = "/your/directory/path/"
        self.camadir = f'{self.datadir}/MERIT/topography/glb_06min' # Please download it from https://hydro.iis.u-tokyo.ac.jp/~yamadai/cama-flood/
        self.rivara_path = f'{self.camadir}/uparea.bin'
        self.gamdir = f'{self.datadir}/gamlss/dat'

        self.ny, self.nx = 1800, 3600
        self.rivnum_path = f'{self.gamdir}/subbsn/subbsn_{self.size_thres}.bin'
        self.savefile = f"{self.gamdir}/reach/reach_list_{self.size_thres}.csv"


    def reach_coord(self):
        rivara = np.fromfile(self.rivara_path, 'float32').reshape(self.ny, self.nx)
        subbasin_id_map = np.fromfile(self.rivnum_path, 'int32').reshape(self.ny, self.nx) # int32 is important
        
        reach_list = []
        uid = np.unique(subbasin_id_map)
        uid = [i for i in uid if i > 0]
        total_grid_count = np.sum(subbasin_id_map > 0)
        print(f"basins: {len(uid)}, grids: {total_grid_count}")

        for sub_id in uid:
            sub_mask = (subbasin_id_map == sub_id)  
            rivara_sub = np.where(sub_mask, rivara, -1)     
            max_idx_flat = np.argmax(rivara_sub)
            iy, ix = np.unravel_index(max_idx_flat, rivara_sub.shape)
        
            reach_list.append({
                "subbasin_id": sub_id,
                "iy": iy,
                "ix": ix,
            })
        return reach_list                                                                                                                            

    def save_file(self):
        reach_list = self.reach_coord()
        with open(self.savefile, "w", newline="") as csvfile:
            fieldnames = ["subbasin_id", "iy", "ix"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(reach_list)


if __name__ == '__main__':
    preprocess = PREPROCESS()
    preprocess.save_file()
