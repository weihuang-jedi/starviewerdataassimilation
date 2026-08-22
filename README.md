# starviewerdataassimilation
echo "# starviewerdataassimilation" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M develop
git remote add origin https://github.com/weihuang-jedi/starviewerdataassimilation.git
git push -u origin develop

0. setup env

   EAGLEhome=/scratch5/purged/Wei.Huang/src/EAGLE
   source ${EAGLEhome}/conda/etc/profile.d/conda.sh
   eval "$(mamba shell hook --shell bash)"
   mamba activate anemoi

1. build graph mesh

   for lvl in 0 1 2 3 4 5 6
   do
      python build_icosahedral_mesh.py -k ${lvl} -o graph-grid
   done

   python build_icosahedral_hierarchy.py -k 6 -o graph-grid

2. check connection

   python check_pt_graph.py -f graph-grid/icosahedral_edge_index_m6.pt
   python verify_boundary_connectivity.py

3. append static geography data to mesh

   for lvl in 0 1 2 3 4 5 6
   do
      python append_static_to_icosahedral.py \
         --mesh graph-grid/global_icosahedral_mesh_m${lvl}.nc \
         --etopo /scratch4/NAGAPE/epic/Wei.Huang/src/starviewergraphcast/data/etopo/ETOPO_2022_v1_60s_N90W180_geoid.nc
   done

4. interpolate to icosahedral grid

   4.1 interpolate to terrain following height grid:
       python interpolate_to_terrain_heights.py \
          --etopo /scratch4/NAGAPE/epic/Wei.Huang/src/starviewergraphcast/data/etopo/ETOPO_2022_v1_60s_N90W180_geoid.nc \
          --input tmp_20240101_gfs.t00z.pgrb2b.1p00.f000 \
          --output terrain-regular-grid/gfs.20240101.t00z.1p00.f000.nc

   python interpolate2icosahedral.py \
      --input gfs.20210101.t00z.1p00.f000.nc \
      --mesh ../graph/graph-grid/global_icosahedral_mesh_m4.nc \
      --output icosahedral-grid/global_icosahedral_m4.20210101.t00z.1p00.f000.nc

5. pack to Zarr

   python pack_icosahedral_zarr.py -o starviewergraphcast-zarr/global_icosahedral_m4.zarr -c 32 -l 3

6. build graph

7. train

   python python training_on_icosahedral_grid.py -c config.yaml

   check GPU performance: nvidia-smi -l 1

   check zarr file:
      python -c "import xarray as xr; ds = xr.open_zarr('/scratch4/NAGAPE/epic/Wei.Huang/src/starviewergraphcast/data/starviewergraphcast-zarr/global_icosahedral_m4.zarr'); print([v for v in ds.data_vars if 'icosahedral' in v or len(ds[v].dims) >= 2])"
['elevation', 'face_nodes', 'icosahedral_mesh', 'land_sea_mask', 'p_icosahedral', 'q_icosahedral', 't_icosahedral', 'u_icosahedral', 'v_icosahedral', 'w_icosahedral', 'x_cartesian', 'y_cartesian', 'z_cartesian']


8. forecast/inference

   8.0 Prepare a 2-step initial condition NetCDF file for GraphCast from 3D icosahedral inputs.

       python make_init.py \
          -p gim/global_icosahedral_m4.20260131.t18z.1p00.f000.nc \
          -c gim/global_icosahedral_m4.20260201.t00z.1p00.f000.nc \
          -o src/init_icosahedral_m4.20260201.t00z.1p00.f000.nc


   8.1 Make forecast

       python starviewergraphcast_forecast.py \
          -c config_inference.yaml \
          -m lightning_logs/version_6/checkpoints/epoch=1-step=2916.ckpt \
          -s 12 \
          -o first_test_forecast.nc

   8.2 check checkpoints

       python -c "import torch; ckpt = torch.load('lightning_logs/version_7/checkpoints/epoch=1-step=2916.ckpt', map_location='cpu'); print(ckpt['hyper_parameters'])"

   8.3 evaluation

       (anemoi) [Wei.Huang@ufe02 src]$ python ../utils/evaluate_acc.py -t "../data/gim/global_icosahedral_m4.20260201.t12z.1p00.f000.nc ../data/gim/global_icosahedral_m4.20260201.t18z.1p00.f000.nc ../data/gim/global_icosahedral_m4.20260202.t00z.1p00.f000.nc ../data/gim/global_icosahedral_m4.20260202.t06z.1p00.f000.nc ../data/gim/global_icosahedral_m4.20260202.t12z.1p00.f000.nc ../data/gim/global_icosahedral_m4.20260202.t18z.1p00.f000.nc ../data/gim/global_icosahedral_m4.20260203.t00z.1p00.f000.nc ../data/gim/global_icosahedral_m4.20260203.t06z.1p00.f000.nc ../data/gim/global_icosahedral_m4.20260203.t12z.1p00.f000.nc" -p forecast_icosahedral_m4.20260201.t00z.1p00.nc
Loading forecast rollout and ground truth records...
/scratch4/NAGAPE/epic/Wei.Huang/src/starviewergraphcast/src/../utils/evaluate_acc.py:31: FutureWarning: In a future version of xarray the default value for coords will change from coords='different' to coords='minimal'. This is likely to lead to different results when multiple datasets have matching variables with overlapping values. To opt in to new defaults and get rid of these warnings now use `set_options(use_new_combine_kwarg_defaults=True) or set coords explicitly.
  self.ds_true = xr.open_mfdataset(
Aligning matching timestamps across files...
 -> Detected default 1970 timestamps in forecast. Reconstructing real datetimes...
 -> Found 9 aligned overlapping forecast steps for evaluation.
 -> Mapping True Variable Core: 'v_icosahedral'
 -> Mapping Predicted Variable Core: 'v_icosahedral'

Calculating Diurnal-Corrected Anomaly Correlation Coefficient (ACC) step drift...
===================================================================================================================
Lead Time       | Forecast Frame (UTC)      | Ground Truth Frame (UTC)  | ACC
===================================================================================================================
+12 Hours        | 2026-02-01 12:00:00       | 2026-02-01 12:00:00       | 0.1711
+18 Hours        | 2026-02-01 18:00:00       | 2026-02-01 18:00:00       | 0.0466
+24 Hours        | 2026-02-02 00:00:00       | 2026-02-02 00:00:00       | 0.0227
+30 Hours        | 2026-02-02 06:00:00       | 2026-02-02 06:00:00       | 0.0150
+36 Hours        | 2026-02-02 12:00:00       | 2026-02-02 12:00:00       | -0.0603
+42 Hours        | 2026-02-02 18:00:00       | 2026-02-02 18:00:00       | -0.0172
+48 Hours        | 2026-02-03 00:00:00       | 2026-02-03 00:00:00       | -0.0080
+54 Hours        | 2026-02-03 06:00:00       | 2026-02-03 06:00:00       | -0.0067
+60 Hours        | 2026-02-03 12:00:00       | 2026-02-03 12:00:00       | -0.0000
===================================================================================================================


9. reconstruct lat-lon grid

   python ../utils/icosahedral2regular.py \
      -i gim/global_icosahedral_m4.20260201.t00z.1p00.f000.nc \
      -o reconstructed_global.t00z.1p00.f000.nc
      -r 1.0


   python ../utils/icosahedral2regular.py \
      -i forecast_icosahedral_m4.20260201.t00z.1p00.nc \
      -o reconstructed_forecast.20260201.t00z.1p00.nc \
      -r 1.0
   

10. compare with original lat-lon grid data

    python ../../utils/plot_comparison.py \
       --original ../../utils/regular_init_20000701T00.nc \
       --reconstructed reconstructed_forecast_20000701T00.nc \
       --image comparison_july_2000_icosahedral_forecast.png \
       --time_index 0

11. plot icosahedral gripython ../utils/plot_regular_grid.py -f python ../utils/plot_regular_grid.py -f d data

    python plot_icosahedral_grid_data.py -f gim/global_icosahedral_m4.20260201.t00z.1p00.f000.nc -v p_icosahedral --height 2

Opening dataset: gim/global_icosahedral_m4.20260201.t00z.1p00.f000.nc
Extracting 'p_icosahedral' data at height index 2...

    python ../utils/plot_regular_grid.py -f reconstructed_global.t00z.1p00.f000.nc -v p --height 2

    python ../utils/plot_regular_grid.py -f reconstructed_forecast.20260201.t00z.1p00.nc  -v p --height 2
