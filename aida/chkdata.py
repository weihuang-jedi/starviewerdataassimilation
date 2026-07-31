import zarr
import numpy as np

z = zarr.open("../data/icosahedral_2023.zarr", mode="r+")

# Take the first time slice [0] to get 1D node arrays (shape: 2562)
x = z["x_cartesian"][0]
y = z["y_cartesian"][0]
z_cart = z["z_cartesian"][0]

r = np.sqrt(x**2 + y**2 + z_cart**2)
lats = np.degrees(np.arcsin(np.clip(z_cart / r, -1.0, 1.0)))
lons = np.degrees(np.arctan2(y, x))
lons = np.where(lons < 0, lons + 360, lons)

# Write 1D arrays matching shape (2562,)
z["latitude"][:] = lats
z["longitude"][:] = lons

print('lats shape:', lats.shape)
print('lons shape:', lons.shape)
print('lats =', lats)
print('lons =', lons)

print("Updated latitude and longitude in icosahedral_2023.zarr successfully!")

