import geojson
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Class Reads Geojson file containing buoy coordinates and returns locations for a specified position and tilesize

class GetGeoData():
    def __init__(self, file="/home/marten/Uni/Semester_4/src/BuoyAssociation/utility/data/noaa_navigational_aids.geojson", tile_size = 0.5):
        self.file = file    # path to geojson file
        self.tile_size =0.5     # size of tile for which buoys are returned (in degrees)
        try:
            f = open(file)
            self.data = geojson.load(f)
        except:
            raise ValueError(f"Cannot open Geojson File: {self.file}")

    def getBuoyLocations(self, pos_lat, pos_lng):
        # Function returns buoy info for all buoys within self.tile_size from given pos
        # Arguments: pos_lat & pos_lng are geographical coordinates
        buoys = []
        for buoy in self.data["features"]:
            buoy_lat = buoy["geometry"]["coordinates"][1]
            buoy_lng = buoy["geometry"]["coordinates"][0]
            if abs(buoy_lng - pos_lng) < self.tile_size and abs(buoy_lat - pos_lat) < self.tile_size:
                buoys.append(buoy)
        return buoys

    def plotBuoyLocations(self, buoyList):
        # Function plots all Buoys specified in buoyList
        # Expects Listitems to be in geojson format
        df = pd.DataFrame([    {**buoy['properties'], 'geometry': Point(buoy['geometry']['coordinates'])}
                                for buoy in buoyList])

        gdf = gpd.GeoDataFrame(df, geometry='geometry')

        # Plot the GeoDataFrame
        gdf.plot(marker='o', color='green', markersize=5)
        plt.title("Buoy Locations")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()