import numpy as np
import pypcd
from pypcd import pypcd
from logging import getLogger
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

logger = getLogger(__name__)


class PcdToBinConvertor:
    def __init__(self, pcd_dir: Path, bin_dir: Path, height_offset=0.0):
        self.pcd_dir = pcd_dir
        self.pcd_dir.mkdir(parents=True, exist_ok=True)
        self.bin_dir = bin_dir
        self.height_offset = height_offset

    def process_pcd_file(self, pcd_path: Path):
        if not pcd_path.name.endswith(".pcd"):
            raise ValueError(f"Frame {pcd_path} is not a .pcd file!")

        try:
            bin_name = pcd_path.name.replace(".pcd", ".bin")
            bin_path = self.bin_dir / bin_name
            pcd_data = pypcd.PointCloud.from_path(pcd_path)

            points = np.zeros([pcd_data.width, 4], dtype=np.float32)
            points[:, 0] = pcd_data.pc_data["x"].copy() 
            points[:, 1] = pcd_data.pc_data["y"].copy() 
            points[:, 2] = pcd_data.pc_data["z"].copy()
            points[:, 3] = pcd_data.pc_data["intensity"].copy().astype(np.float32)
            points[:, 2] += self.height_offset
            points.tofile(bin_path)
        except Exception as e:
            raise ValueError(f"Error processing {pcd_path}: {e}")

    def convert(self):
        if not self.pcd_dir.is_dir():
            raise ValueError(f"Invalid directory path {self.pcd_dir}")
        if not self.bin_dir.is_dir():
            self.bin_dir.mkdir(parents=True, exist_ok=True)

        pcd_files = list(self.pcd_dir.glob("*.pcd"))
        for pcd_file in tqdm(pcd_files):
            self.process_pcd_file(pcd_file)
        logger.error(f"Converted {len(pcd_files)} .pcd files to .bin format")

    def convert_multi_process(self, num_processes=4):
        if not self.pcd_dir.is_dir():
            raise ValueError(f"Invalid directory path {self.pcd_dir}")
        if not self.bin_dir.is_dir():
            self.bin_dir.mkdir(parents=True, exist_ok=True)

        pcd_files = list(self.pcd_dir.glob("*.pcd"))
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            list(
                tqdm(
                    executor.map(self.process_pcd_file, pcd_files), total=len(pcd_files)
                )
            )
        logger.error(f"Converted {len(pcd_files)} .pcd files to .bin format")


def convert_pcd_to_bin(
    pcd_dir: str,
    bin_dir: str,
    height_offset: float = 0.0,
    is_multi_process=False,
    num_processes=4,
):
    convertor = PcdToBinConvertor(
        Path(pcd_dir), Path(bin_dir), height_offset=height_offset
    )
    if is_multi_process:
        convertor.convert_multi_process(num_processes=num_processes)
    else:
        convertor.convert()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--pcd_dir",
        type=str,
        required=True,
        help="Path to the directory containing the .pcd files",
    )
    parser.add_argument(
        "--bin_dir",
        type=str,
        required=True,
        help="Path to save the converted .bin files",
    )
    parser.add_argument(
        "--height_offset",
        type=float,
        default=0.0,
        help="Height offset to add to the z-axis",
    )
    parser.add_argument(
        "--is_multi_process",
        action="store_true",
        help="Use multiple processes for conversion",
    )
    args = parser.parse_args()
    convert_pcd_to_bin(args.pcd_dir, args.bin_dir, height_offset=args.height_offset)
