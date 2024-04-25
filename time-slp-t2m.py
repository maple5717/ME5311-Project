import numpy as np
import pandas as pd
import xarray as xr
import os
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import linregress
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize


# Load sea level pressure data
slp_file_path = os.path.abspath('slp.nc')
slp_ds = xr.open_dataset(slp_file_path)

# Slice longitude and latitude to reduce resolution to 2°×2°
low_res_slp = slp_ds.sel(longitude=slice(None, None, 2), latitude=slice(None, None, 2))

# Get time snapshots
time_snapshots = low_res_slp['time'].values
dates = pd.to_datetime(time_snapshots)

# Load sea surface temperature data
t2m_file_path = os.path.abspath('t2m.nc')
t2m_ds = xr.open_dataset(t2m_file_path)

# Slice longitude and latitude to reduce resolution to 2°×2°
low_res_t2m = t2m_ds.sel(longitude=slice(None, None, 2), latitude=slice(None, None, 2))

# Get time snapshots
time_snapshots2 = low_res_t2m['time'].values
dates2 = pd.to_datetime(time_snapshots2)


def calculate_correlation(start_year, n):

    # Loop to calculate the mean for each month
    monthly_pressure_means = []
    monthly_temperature_means = []
    for year in range(start_year, start_year + n):
        for month in range(1, 13):  # 12 months
            # Select data for the current year and month
            ds_subset = low_res_slp.sel(time=dates[(dates.year == year) & (dates.month == month)])
            ds2_subset = low_res_t2m.sel(time=dates2[(dates2.year == year) & (dates2.month == month)])

            # Calculate the mean sea level pressure for the current year and month, keeping the time dimension
            monthly_pressure_mean = ds_subset.mean(dim='time', keep_attrs=True)['msl']
            monthly_temperature_mean = ds2_subset.mean(dim='time', keep_attrs=True)['t2m']

            # Add year and month as new dimensions
            monthly_pressure_mean.coords['year_month'] = pd.to_datetime(f'{year}-{month:02}', format='%Y-%m')
            monthly_temperature_mean.coords['year_month'] = pd.to_datetime(f'{year}-{month:02}', format='%Y-%m')

            # Append to the result lists
            monthly_pressure_means.append(monthly_pressure_mean)
            monthly_temperature_means.append(monthly_temperature_mean)

    # Merge the result lists into a single dataset
    monthly_pressure_means_ds = xr.concat(monthly_pressure_means, dim='year_month')
    monthly_temperature_means_ds = xr.concat(monthly_temperature_means, dim='year_month')


    # Calculate correlation coefficients

    # Get the shape of sea level pressure and temperature
    n_latitudes, n_longitudes, n_months = monthly_pressure_means_ds.shape

    # Create an array to store correlation coefficients
    correlation_coefficients = np.zeros((n_latitudes, n_longitudes))

    # Iterate over each pixel
    for i in range(n_latitudes):
        for j in range(n_longitudes):
            # Get sea level pressure and temperature data for that pixel
            pressure = monthly_pressure_means_ds[i, j]
            temperature = monthly_temperature_means_ds[i, j]

            # Calculate Pearson correlation coefficient
            # The closer the absolute value is to 1, the stronger the linear correlation
            correlation_coefficients[i, j], _ = pearsonr(pressure.values.flatten(), temperature.values.flatten())

    return correlation_coefficients


start_year = 1979
end_year = 2022
n = 2 #As n increases, the correlation analysis becomes increasingly affected by noise, especially when n reaches 43 (noise-correlation_plot_1979-2021.png). The best processing performance is achieved when n equals 2.

# List to store correlation matrices
correlation_matrices = []

# Create map projection
projection = ccrs.PlateCarree()

# Loop to calculate correlation coefficients for each time period
for year in range(start_year, end_year, n):
    correlation_matrix = calculate_correlation(year, n)
    correlation_matrices.append((year, year + n - 1, correlation_matrix))  # Store year range and correlation matrix

# Directory to save images
output_dir = './correlation_plots 7/'

# Ensure the output directory exists, create it if it doesn't
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through each time period, plot and save correlation coefficient images
for i, (start, end, correlation_matrix) in enumerate(correlation_matrices):
    # Create map projection
    projection = ccrs.PlateCarree()

    # Plot the map
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=projection)
    ax.coastlines()
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.gridlines()

    # Define normalizer to map correlation coefficients to colormap range
    norm = Normalize(vmin=-1, vmax=0)

    # Plot correlation coefficient data
    im = ax.imshow(correlation_matrix, extent=[70, 150, -10, 40], cmap='viridis', norm=norm, origin='lower')

    # Add color bar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)
    cbar.set_label('Pearson Correlation Coefficient')

    # Show the map
    plt.title(f'Pearson Correlation Coefficients Map ({start}-{end})')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Construct filename and save the image
    filename = os.path.join(output_dir, f'correlation_plot_{start}-{end}.png')
    plt.savefig(filename, dpi=300)  # Save the image with 300 dpi resolution
    plt.close()  # Close the current plot to proceed to the next one
