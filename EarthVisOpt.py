import PIL
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

file_name = "bluemarble.jpg"
filepath = Path(__file__).parent.joinpath(file_name) #goes from file directory, up to the parent folder, then back to the "data" folder", then to the filename

# Load image (without resizing)
bm = PIL.Image.open(filepath)
bm_array = np.array(bm) / 256.0  # Normalize pixel values

# Earth's radius (in km)
earth_radius = 6371  

# Generate latitude and longitude grids
lons = np.linspace(-180, 180, bm_array.shape[1]) * np.pi / 180  
lats = np.linspace(-90, 90, bm_array.shape[0]) * np.pi / 180  

# Create meshgrid for sphere mapping
lon_grid, lat_grid = np.meshgrid(lons, lats)

# Convert to 3D Cartesian coordinates (scaled to Earth's radius)
x = earth_radius * np.cos(lat_grid) * np.cos(lon_grid)
y = earth_radius * np.cos(lat_grid) * np.sin(lon_grid)
z = earth_radius * np.sin(lat_grid)

# Create 3D plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Render the Earth
ax.plot_surface(x, y, z, facecolors=bm_array, rstride=5, cstride=5, shade=False, antialiased=False)

# ⭐ Add Stars (Reduced & Closer)
num_stars = 80  # Fewer stars
star_dist = 10 * earth_radius  # Bring stars closer

star_x = star_dist * (np.random.rand(num_stars) - 0.5)
star_y = star_dist * (np.random.rand(num_stars) - 0.5)
star_z = star_dist * (np.random.rand(num_stars) - 0.5)

ax.scatter(star_x, star_y, star_z, s=0.5, c='white')  

#figure set 
fig.set_facecolor('black')
ax.set_facecolor('black')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False





plt.show()
