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



# Data Preprocessing

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





# Correlation Calculation Function

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
            correlation_coefficients[i, j], _ = pearsonr(pressure.values.flatten(), temperature.values.flatten())

    return correlation_coefficients





# Main Program

start_year = 1979
end_year = 2022
n = 2

# List to store correlation matrices
correlation_matrices = []

# Loop to calculate correlation coefficients for each time period
for year in range(start_year, end_year, n):
    correlation_matrix = calculate_correlation(year, n)
    correlation_matrices.append((year, year + n - 1, correlation_matrix))  # Store year range and correlation matrix

# List to store latitude and longitude coordinates of all selected pixels
selected_points = []

# Choose any correlation matrix to get the shape and latitude-longitude information
example_correlation_matrix = correlation_matrices[0][2]
n_latitudes, n_longitudes = example_correlation_matrix.shape

# Strong linear correlation point filtering
# Based on 'time-slp-t2m' , it was observed that the Pearson correlation coefficients for all pixels are consistently negative, indicating a persistent negative correlation. However, due to the wide range of variations in most coefficients over time, clustering analysis is not suitable for identifying spatial correlations. Therefore, it is proposed to restrict the range and select pixels with strong correlation coefficients (which may could serve as prior models).

# Loop through each pixel
for i in range(n_latitudes):
    for j in range(n_longitudes):
        # Initialize a flag to indicate whether the current pixel satisfies the condition
        satisfies_condition = True

        # Loop through each correlation matrix
        for start, end, correlation_matrix in correlation_matrices:
            # Get the correlation coefficient for the current pixel
            correlation_coefficient = correlation_matrix[i, j]

            # If the correlation coefficient is not within the specified range, set the flag to False and break out of the inner loop
            # when -1 <= correlation_coefficient <= -0.7, define it as a strong negative correlation
            if not (-1 <= correlation_coefficient <= -0.7):
                satisfies_condition = False
                break

        # If the current pixel satisfies the condition in all correlation matrices, record its latitude and longitude coordinates
        if satisfies_condition:
            selected_points.append((i, j))

print(selected_points)

# Strong Linear Correlation Point Visualization Analysis

# Create map projection
projection = ccrs.PlateCarree()

# Plot the map
plt.figure(figsize=(10, 8))
ax = plt.axes(projection=projection)
ax.coastlines()
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.gridlines()

# Define a normalizer to map correlation coefficients to the colormap range
norm = Normalize(vmin=-1, vmax=0)

# Plot the correlation coefficient data
im = ax.imshow(correlation_matrices[0][2], extent=[70, 150, -10, 40], cmap='viridis', norm=norm, origin='lower')

# Choose one correlation matrix as an example
example_start, example_end, example_correlation_matrix = correlation_matrices[0]

# Get the shape of the example correlation matrix
n_latitudes, n_longitudes = example_correlation_matrix.shape

# Get the values of latitude and longitude corresponding to the example correlation matrix
example_latitudes = low_res_slp.coords['latitude'].values
example_longitudes = low_res_slp.coords['longitude'].values

# Convert the pixel coordinates of selected points to latitude and longitude coordinates
selected_latitudes = []
selected_longitudes = []
for point in selected_points:
    # Get the pixel coordinates of the selected point
    pixel_latitude, pixel_longitude = point

    # Calculate latitude and longitude coordinates
    latitude_index = int((pixel_latitude / n_latitudes) * len(example_latitudes))
    longitude_index = int((pixel_longitude / n_longitudes) * len(example_longitudes))
    latitude = example_latitudes[latitude_index]
    longitude = example_longitudes[longitude_index]

    # Append to latitude and longitude lists
    selected_latitudes.append(latitude)
    selected_longitudes.append(longitude)

# Plot the selected points
ax.scatter(selected_longitudes, selected_latitudes, color='red', transform=ccrs.PlateCarree(), label='Selected Points')

# Add colorbar
cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)
cbar.set_label('Pearson Correlation Coefficient')

# Show the map
plt.title(f'Pearson Correlation Coefficients Map ({start_year}-{end_year})')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()  # Show legend
plt.show()



