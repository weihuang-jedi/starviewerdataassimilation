import os
import glob
import torch
import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import RegularGridInterpolator


class GFSAIDADataset(Dataset):
    """
    PyTorch Dataset for AI-Data Assimilation.
    Pairs GFS Forecast NetCDF (x_b) with Observation NetCDF files.
    
    Returns:
      x_b_tensor: (C, H, Lat, Lon) -> Background forecast grid
      inno_tensor: (C, H, Lat, Lon) -> Grid-mapped observation innovations [y - H(x_b)]
      dx_target_tensor: (C, H, Lat, Lon) -> Target increment [x_a_target - x_b]
    """
    def __init__(
        self,
        gfs_dir: str,
        obs_dir: str,
        target_dir: str = None,
        var_names: list = ["t"],
        preload_to_ram: bool = False,
    ):
        super().__init__()
        self.gfs_dir = gfs_dir
        self.obs_dir = obs_dir
        self.target_dir = target_dir
        self.var_names = var_names
        self.preload_to_ram = preload_to_ram

        # Match files by timestamp identifier in filename (e.g., gfs_20220411_06z.nc)
        self.gfs_files = sorted(glob.glob(os.path.join(gfs_dir, "*.nc")))
        
        if len(self.gfs_files) == 0:
            raise FileNotFoundError(f"No GFS NetCDF files found in '{gfs_dir}'")

        # Extract fixed spatial coordinates from the first file
        with xr.open_dataset(self.gfs_files[0]) as ds:
            self.heights = ds["height"].values.astype(np.float32)
            self.lats = ds["latitude"].values.astype(np.float32)
            self.lons = ds["longitude"].values.astype(np.float32)
            
            # Ensure ascending latitude for interpolation indexing
            if np.any(np.diff(self.lats) < 0):
                self.lats = self.lats[::-1]
                self.flip_lat = True
            else:
                self.flip_lat = False

        self.grid_shape = (len(self.heights), len(self.lats), len(self.lons))
        
        # Optional RAM Caching for small-to-medium datasets
        self.cache = {} if preload_to_ram else None

    def _map_obs_to_grid(self, obs_path: str, x_b_grid: np.ndarray) -> np.ndarray:
        """
        Reads point observations (y_obs) and computes grid innovation [y - H(x_b)],
        mapping sparse point residuals onto the 3D target grid tensor.
        """
        inno_grid = np.zeros_like(x_b_grid, dtype=np.float32)

        if not os.path.exists(obs_path):
            # Return zero grid if observation file doesn't exist for timestamp
            return inno_grid

        with xr.open_dataset(obs_path) as ds_obs:
            obs_h = ds_obs["height"].values
            obs_lat = ds_obs["latitude"].values
            obs_lon = ds_obs["longitude"].values
            y_obs = ds_obs["observation_value"].values

        if len(y_obs) == 0:
            return inno_grid

        # 1. Forward Operator H(x_b): Interpolate x_b to observation points
        interp = RegularGridInterpolator(
            (self.heights, self.lats, self.lons),
            x_b_grid[0],  # Assuming channel 0
            bounds_error=False,
            fill_value=None,
        )
        H_xb = interp(np.column_stack((obs_h, obs_lat, obs_lon)))

        # 2. Innovation vector d = y_obs - H(x_b)
        innovations = y_obs - H_xb

        # 3. Simple 3D nearest-neighbor mapping of innovation onto grid
        # (For production, replace with Gaussian Kernel Splatting or H^T)
        k_idx = np.searchsorted(self.heights, obs_h) - 1
        j_idx = np.searchsorted(self.lats, obs_lat) - 1
        i_idx = np.searchsorted(self.lons, obs_lon) - 1

        k_idx = np.clip(k_idx, 0, self.grid_shape[0] - 1)
        j_idx = np.clip(j_idx, 0, self.grid_shape[1] - 1)
        i_idx = np.clip(i_idx, 0, self.grid_shape[2] - 1)

        # Accumulate innovations onto 3D grid
        np.add.at(inno_grid[0], (k_idx, j_idx, i_idx), innovations)

        return inno_grid

    def __len__(self):
        return len(self.gfs_files)

    def __getitem__(self, idx: int):
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]

        gfs_path = self.gfs_files[idx]
        filename = os.path.basename(gfs_path)

        # Construct matching paths for observations and targets
        obs_path = os.path.join(self.obs_dir, filename.replace("gfs_", "obs_"))
        
        # 1. Read Background State (x_b)
        with xr.open_dataset(gfs_path) as ds_gfs:
            x_b_list = []
            for var in self.var_names:
                arr = ds_gfs[var].values.astype(np.float32)
                if self.flip_lat:
                    arr = np.flip(arr, axis=1)
                x_b_list.append(arr)
            x_b = np.stack(x_b_list, axis=0)  # Shape: (C, Height, Lat, Lon)

        # 2. Map Observation Innovations to Grid [y - H(x_b)]
        inno_grid = self._map_obs_to_grid(obs_path, x_b)

        # 3. Read or Compute Target Increment (dx)
        if self.target_dir:
            target_path = os.path.join(self.target_dir, filename.replace("gfs_", "analysis_"))
            with xr.open_dataset(target_path) as ds_target:
                x_a_list = []
                for var in self.var_names:
                    arr = ds_target[var].values.astype(np.float32)
                    if self.flip_lat:
                        arr = np.flip(arr, axis=1)
                    x_a_list.append(arr)
                x_a_target = np.stack(x_a_list, axis=0)
            dx_target = x_a_target - x_b
        else:
            # Fallback: Zero target increment if no target provided
            dx_target = np.zeros_like(x_b)

        # Convert to PyTorch Tensors
        x_b_tensor = torch.from_numpy(x_b)
        inno_tensor = torch.from_numpy(inno_grid)
        dx_target_tensor = torch.from_numpy(dx_target)

        sample = (x_b_tensor, inno_tensor, dx_target_tensor)

        if self.cache is not None:
            self.cache[idx] = sample

        return sample


# ==============================================================================
# DATALOADER CREATION FUNCTION
# ==============================================================================
def create_aida_dataloader(
    gfs_dir: str,
    obs_dir: str,
    target_dir: str = None,
    batch_size: int = 2,
    num_workers: int = 4,
    shuffle: bool = True,
):
    dataset = GFSAIDADataset(
        gfs_dir=gfs_dir,
        obs_dir=obs_dir,
        target_dir=target_dir,
        var_names=["t"],
        preload_to_ram=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Speeds up CPU-to-GPU memory transfer
        persistent_workers=(num_workers > 0),
    )

    return dataloader
