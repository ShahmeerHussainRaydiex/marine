import os
import copy
import shutil
import numpy as np

import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import geometry_mask
from rasterio.crs import CRS
from rasterio.mask import mask
from rasterio.transform import from_origin
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt


from osgeo import gdal, ogr
import geopandas as gpd
import fiona
from shapely.geometry import Point
from pyproj import Proj, Transformer


def gdal_vec_to_ras(in_s, out_r, template_r, field):
    '''
    Converts a polygon to a raster and matches a template raster

    Parameters:
        in_s {string}: path to the shapefile to be converted
        out_r {string}: desired output path for the converted raster
        template_r {string}: path to the template raster
        field {string}: name of the field that should be used when converting the raster   _____ Address if there is no field 4444 
    '''
    shapefile = ogr.Open(in_s)
    layer = shapefile.GetLayer()

    template = gdal.Open(template_r)
    output_geo_transform = template.GetGeoTransform()
    x_min, x_max, y_min, y_max = layer.GetExtent()

    driver = gdal.GetDriverByName("GTiff")
    output = driver.Create(out_r, template.RasterXSize, template.RasterYSize, 1, gdal.GDT_Float32)

    output.SetProjection(template.GetProjection())
    output.SetGeoTransform(output_geo_transform)

    gdal.RasterizeLayer(output, [1], layer, options=[f"ATTRIBUTE={field}"])

    output = None
    template = None
    shapefile = None


def resample_rasters(input_folder, output_folder, target_resolution):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.tif', '.tiff', '.img')):
            input_path = os.path.join(input_folder, filename)
            
            with rasterio.open(input_path) as src:
                # Calculate new dimensions
                new_width = int((src.bounds.right - src.bounds.left) / target_resolution)
                new_height = int((src.bounds.top - src.bounds.bottom) / target_resolution)
                
                # Choose resampling method based on filename
                if 'seabed' in filename.lower() or 'country' in filename.lower() or 'closest_gridconnect' in filename.lower():
                    resampling_method = Resampling.mode
                else:
                    resampling_method = Resampling.bilinear
                
                # Create new transform
                new_transform = from_origin(src.bounds.left, src.bounds.top, 
                                            target_resolution, target_resolution)
                
                # Perform resampling
                data = src.read(
                    out_shape=(src.count, new_height, new_width),
                    resampling=resampling_method
                )
                
                # Prepare the output path
                output_filename = f"{filename}"
                output_path = os.path.join(output_folder, output_filename)
                
                kwargs = src.meta.copy()
                kwargs.update({
                    'transform': new_transform,
                    'width': new_width,
                    'height': new_height,
                    'driver': 'GTiff'
                })
                
                # Write the resampled raster to the output folder
                with rasterio.open(output_path, 'w', **kwargs) as dst:
                    dst.write(data)
    

def reproject_raster(input, output, epsg):

    '''
    Reprojects a raster to a predetermined epsg

    Parameters:
        input {string}: filepath of the raster to be reprojected
        output {string}: filepath for the output raster
        epsg {string}: destination epsg. In 'EPSG:xxxx' format
    '''

    dst_crs = CRS.from_string(epsg)

    with rasterio.open(input) as src:
        src_transform = src.transform

        # calculate the transform matrix for the output
        dst_transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,  # unpacks outer boundaries (left, bottom, right, top)
        )

        # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update(
            {
                "crs": dst_crs,
                "transform": dst_transform,
                "width": width,
                "height": height,
                "nodata": 0,  # replace 0 with np.nan
            }
        )

        with rasterio.open(output, "w", **dst_kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
    print(f'Raster has been reprojected from {src.crs} to {dst_crs}')


def mask_tifs(in_f, out_f, shape_path, flag):
    
    '''
    Masks all rasters from a folder to a shapefile -- effectively a batch clip

    Parameters:
        in_f {string}: folder with rasters
        out_f {string}: output folder name to add masked rasters to
        shape_path {string}: path to the shapefile to be used for the masking
        flag {int}: To determine whether or not the raster will be what is within, or outside of the shapefile
            1 = Extracts the values within the bounds of the shapefile and removes what is outside
            2 = Masks out the data that is within the bounds of the shapefile
    '''
    
    with fiona.open(shape_path, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    for filename in os.listdir(in_f):
        if filename.endswith(".tif"):
           
            tif = os.path.join(in_f, filename)
            tif_filename = os.path.basename(tif)
            out_tif = os.path.join(out_f, tif_filename)

            with rasterio.open(tif) as src:
                if flag == 1: # Data from within the mask
                    out_image, out_transform = mask(src, shapes, crop=True, invert=False)
                    out_meta = src.meta
                elif flag == 2: # Data from outside the mask
                    out_image, out_transform = mask(src, shapes, crop=False, invert=True)
                    out_meta = src.meta
            out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})

            with rasterio.open(out_tif, "w", **out_meta) as dest:
                dest.write(out_image)


def reclassify_raster(in_f, out_f, table):
    '''
    Reclassifies a raster with simple integers as the cell values.

    Parameters:
        in_f {string}: Path to the raster to be reclassified
        out_f {strinf}: Path to the reclassified final raster
        table {dict}: keys are the original cell, values are the reclassified  
    '''


    with rasterio.open(in_f) as src:
        raster_array = src.read(1)
        _copy = copy.deepcopy(raster_array)

        for old_value, new_value in table.items():
            raster_array[_copy == old_value] = new_value

        profile = src.profile

    with rasterio.open(out_f, 'w', **profile) as dst:
        dst.write(raster_array, 1)


def raster_zeros(template, output):
    with rasterio.open(template) as src:
        # Get metadata from the template raster
        meta = src.meta.copy()

        # Create a new raster with every cell assigned to 0
        data = np.zeros((meta['height'], meta['width']), dtype=np.uint8)

        with rasterio.open(output, 'w', **meta) as dst:
            dst.write(data, 1)

    #print('Raster of zeros created.')


def convert_coordinates(lat_lon_tuple, raster_path):

    lat, lon = lat_lon_tuple

    with rasterio.open(raster_path) as src:
        template_raster_crs = src.crs.to_proj4()

    projector = Proj(template_raster_crs)
    transformer = Transformer.from_proj(Proj('EPSG:4326'), projector)
    x, y = transformer.transform(lon, lat)

    #print(f"The input coordinates of: {lat}, {lon} have been succesfully transformed to: {x}, {y}")

    return x, y


def burn_point(raster_path, point, flag):
    
    with rasterio.open(raster_path, 'r+') as raster:
        
        if flag == 0: # Point from tuple
            
            point_utm = convert_coordinates(point, raster_path)

            shape_point = Point(point_utm[0], point_utm[1])

            gdf = gpd.GeoDataFrame(geometry=[shape_point])

            point_geometry = gdf.geometry.values[0]

            mask = geometry_mask([point_geometry], out_shape=raster.shape, transform=raster.transform, invert=True)

            raster.write(np.ones_like(raster.read(1)) * mask, 1)

        else:

            mask = geometry_mask([point], out_shape=raster.shape, transform=raster.transform, invert=True)
            raster.write(np.ones_like(raster.read(1)) * mask, 1)
    
    #print('The point has been successfully burned into the raster')


def calc_distance_raster(input, output):
    '''
    Takes in a raster of zeros with one point that is 1 and burns the euclidean distance to that '1' point for every cell

    Parameters:
        input {string}: path to the input raster of zeros and a 1
        output {string}: path for the created raster 
    '''

    original_raster = gdal.Open(input)
    original_array = original_raster.GetRasterBand(1).ReadAsArray()

    # Find coordinates of the cell with value 1
    one_point_coords = np.argwhere(original_array == 1)[0]

    # Get geotransform information
    geotransform = original_raster.GetGeoTransform()
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]

    # Calculate the Euclidean distance in meters
    rows, cols = original_array.shape
    euclidean_distance = np.zeros_like(original_array, dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            x_distance = (i - one_point_coords[0]) * pixel_width
            y_distance = (j - one_point_coords[1]) * pixel_height
            euclidean_distance[i, j] = np.sqrt(x_distance**2 + y_distance**2)

    # Set the distance of the point to 0
    euclidean_distance[one_point_coords[0], one_point_coords[1]] = 1

    # Create new tiff
    driver = gdal.GetDriverByName('GTiff')
    output_raster = driver.Create(output, original_raster.RasterXSize, original_raster.RasterYSize, 1, gdal.GDT_Float32)
    output_raster.SetGeoTransform(geotransform)
    output_raster.SetProjection(original_raster.GetProjection())

    # Write the Euclidean distance array to the output raster
    output_raster.GetRasterBand(1).WriteArray(euclidean_distance)

    # Close rasters
    output_raster.FlushCache()
    output_raster = None
    original_raster = None


def create_euclidean_distance(template, output_folder, points, working_folder):

    if not os.path.exists(working_folder):
        os.mkdir(working_folder)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    try:
        # create blank raster from template into the working folder
        working_raster = os.path.join(working_folder, 'working_raster.tif')
        raster_zeros(template, working_raster)

        
        if isinstance(points, list): ## WORKS BUT THE POINTS MUST BE WITHIN THE RASTER
            
            for i in points:
                new_file = os.path.join(output_folder, f'{i[0]}_distance.tif')

                coord = i[1], i[2]

                burn_point(working_raster, coord, 0)

                calc_distance_raster(working_raster, new_file)


        elif points.endswith('.shp') or points.endswith('.geojson'):
            
            gdf = gpd.read_file(points)

            for i in range(len(gdf)):
                
                new_file = str(output_folder + str(gdf.loc[i, 'name']) + '_distance.tif')

                burn_point(working_raster, gdf.geometry.values[i], 1)

                calc_distance_raster(working_raster, new_file)
        else:
            print('Unsupported point format')


    except Exception as e:
        print(f'Error: {e}')
        

    finally:
        shutil.rmtree(working_folder, ignore_errors=True)


def mosaic_rasters(input_folder, output_file):
    # Get list of raster files in the input folder
    raster_files = [f for f in os.listdir(input_folder) if f.endswith(('.tif', '.tiff', '.img'))]
    
    if not raster_files:
        print("No raster files found in the input folder.")
        return
    
    # Create a virtual raster (VRT) dataset
    vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest', separate=False)
    vrt_dataset = gdal.BuildVRT('', [os.path.join(input_folder, f) for f in raster_files], options=vrt_options)
    
    # Create the output mosaic file
    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.CreateCopy(output_file, vrt_dataset, 0)
    
    # Close the datasets
    vrt_dataset = None
    output_dataset = None
    
    print(f"Mosaic created successfully: {output_file}")


def create_distance_from_poly_raster(shapefile_path, template_raster_path, output_raster_path):
    # Open the template raster and get its properties
    template_ds = gdal.Open(template_raster_path)
    if template_ds is None:
        raise ValueError("Could not open template raster")
    
    geotransform = template_ds.GetGeoTransform()
    projection = template_ds.GetProjection()
    num_cols = template_ds.RasterXSize
    num_rows = template_ds.RasterYSize
    
    # Create a new raster in memory
    mem_driver = gdal.GetDriverByName('MEM')
    target_ds = mem_driver.Create('', num_cols, num_rows, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geotransform)
    target_ds.SetProjection(projection)
    
    # Rasterize the shapefile
    shapefile = ogr.Open(shapefile_path)
    if shapefile is None:
        raise ValueError("Could not open shapefile")
    
    layer = shapefile.GetLayer()
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[1])
    
    # Get the rasterized data as a numpy array
    rasterized_array = target_ds.ReadAsArray()
    
    # Calculate the Euclidean distance transform
    distance_array = distance_transform_edt(1 - rasterized_array)
    
    # Create the output raster
    driver = gdal.GetDriverByName('GTiff')
    output_ds = driver.Create(output_raster_path, num_cols, num_rows, 1, gdal.GDT_Float32)
    output_ds.SetGeoTransform(geotransform)
    output_ds.SetProjection(projection)
    
    # Write the distance array to the output raster
    output_band = output_ds.GetRasterBand(1)
    output_band.WriteArray(distance_array)
    
    # Close datasets
    template_ds = None
    target_ds = None
    output_ds = None
    shapefile = None
    
    print(f"Distance raster created successfully: {output_raster_path}")


def interpolate_raster_nans_idw(raster_path, output_path):
    # Read the raster
    with rasterio.open(raster_path) as src:
        raster_data = src.read(1)  # Assuming single band raster
        profile = src.profile

    # Create a mask of valid (non-NaN) cells
    mask = ~np.isnan(raster_data)

    # Get the coordinates of valid cells
    rows, cols = np.where(mask)
    valid_coords = np.column_stack((rows, cols))

    # Get the values of valid cells
    valid_values = raster_data[mask]

    # Create a grid of all coordinates
    all_rows, all_cols = np.meshgrid(np.arange(raster_data.shape[0]), np.arange(raster_data.shape[1]), indexing='ij')
    all_coords = np.column_stack((all_rows.ravel(), all_cols.ravel()))

    # Perform IDW interpolation
    interpolated_data = griddata(valid_coords, valid_values, all_coords, method='cubic', fill_value=np.nan)
    interpolated_data = interpolated_data.reshape(raster_data.shape)

    # Fill any remaining NaNs with nearest neighbor interpolation
    if np.isnan(interpolated_data).any():
        nn_interpolated = griddata(valid_coords, valid_values, all_coords, method='nearest')
        nn_interpolated = nn_interpolated.reshape(raster_data.shape)
        interpolated_data = np.where(np.isnan(interpolated_data), nn_interpolated, interpolated_data)

    # Write the interpolated raster
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(interpolated_data.astype(rasterio.float32), 1)
