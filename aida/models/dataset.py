import numpy as np
import torch
from torch.utils.data import Dataset

LOG_STATE_VARS = [
    'ln_t_icosahedral',
    'u_icosahedral',
    'v_icosahedral',
    'w_icosahedral',
    'q_icosahedral',
    'ln_rho_icosahedral',
    'ln_p_icosahedral'
]


class LogStateZarrDataset(Dataset):
    """
    Dataset wrapper for multi-variable Zarr stores with separate 3D arrays:
    Keys: ['ln_t_icosahedral', 'u_icosahedral', 'v_icosahedral',
           'w_icosahedral', 'q_icosahedral', 'ln_rho_icosahedral', 'ln_p_icosahedral']
    Array shape per variable: [Time=1460, Levels=32, Nodes=2562]
    Output shape per sample:   [Vars=7, Levels=32, Nodes=2562]
    """
    def __init__(self, zarr_path: str):
        super().__init__()
        self.zarr_path = zarr_path
        self.var_keys = LOG_STATE_VARS

        try:
            import zarr
        except ImportError:
            raise ImportError("zarr library is required. Run 'pip install zarr'.")

        self.root = zarr.open(zarr_path, mode='r')

        available_keys = list(self.root.array_keys())
        for k in self.var_keys:
            if k not in available_keys:
                raise KeyError(
                    f"Expected key '{k}' not found in Zarr store at '{zarr_path}'. "
                    f"Found: {available_keys}"
                )

        first_arr = self.root[self.var_keys[0]]
        self.num_time_steps = first_arr.shape[0] - 1  # t -> t+1 pairs
        self.num_vars = len(self.var_keys)
        self.num_levels = first_arr.shape[1]  # 32
        self.num_nodes = first_arr.shape[2]   # 2562

        print(f"[DATASET] Loaded Multi-Array Zarr dataset from '{zarr_path}'")
        print(f"          Variables ({self.num_vars}): {self.var_keys}")
        print(
            f"          Dimensions: Time={self.num_time_steps + 1}, "
            f"Levels={self.num_levels}, Nodes={self.num_nodes}"
        )

    def __len__(self):
        return self.num_time_steps

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_list = [np.array(self.root[key][idx], dtype=np.float32) for key in self.var_keys]
        y_list = [np.array(self.root[key][idx + 1], dtype=np.float32) for key in self.var_keys]

        x = np.stack(x_list, axis=0)
        y = np.stack(y_list, axis=0)

        # Basic NaN safeguard on data read
        x = np.nan_to_num(x, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)

        return torch.from_numpy(x), torch.from_numpy(y)


class SyntheticAIDAStateDataset(Dataset):
    """Fallback dataset simulating log-state atmospheric variables on mesh."""
    def __init__(self, num_samples: int = 80, num_nodes: int = 2562, num_levels: int = 8):
        super().__init__()
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.num_levels = num_levels
        self.num_vars = len(LOG_STATE_VARS)

        np.random.seed(42)
        self.data_x = np.random.randn(num_samples, 7, num_levels, num_nodes).astype(np.float32)
        self.data_y = self.data_x * 0.98 + 0.02 * np.random.randn(num_samples, 7, num_levels, num_nodes).astype(np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.data_x[idx]), torch.from_numpy(self.data_y[idx])
