from OpenGL.arrays import returnPointer
import geojson
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import math

from utility.Transformations import haversineDist

# Class Reads Geojson file containing buoy coordinates and returns locations for a specified position and tilesize

class GetGeoData():
    def __init__(self, file="utility/data/noaa_navigational_aids.geojson", tile_size = 0.1):
        self.file = file    # path to geojson file
        self.tile_size = tile_size     # size of tile for which buoys are returned (in degrees)
        try:
            f = open(file)
            self.data = geojson.load(f)
        except:
            raise ValueError(f"Cannot open Geojson File: {self.file}")
        self.tile_center = None

    def getBuoyLocations(self, pos_lat, pos_lng):
        # Function returns buoy info for all buoys within self.tile_size from given pos
        # Arguments: pos_lat & pos_lng are geographical coordinates
        self.tile_center = {"lat": pos_lat, "lng": pos_lng}
        buoys = []
        for buoy in self.data["features"]:
            buoy_lat = buoy["geometry"]["coordinates"][1]
            buoy_lng = buoy["geometry"]["coordinates"][0]
            if abs(buoy_lng - pos_lng) < self.tile_size and abs(buoy_lat - pos_lat) < self.tile_size:
                buoys.append(buoy)
        return buoys

    def getBuoyID(self, buoy_lat, buoy_lng):
        # Function returns buoy id for given coordinates lat & lng, if bouy exits inside database
        # Arguments: buoy_lat & buoy_lng are geographical coordinates
        self.tile_center = {"lat": buoy_lat, "lng": buoy_lng}
        res = None
        min_dist = math.inf
        for buoy in self.data["features"]:
            buoy2_lat = buoy["geometry"]["coordinates"][1]
            buoy2_lng = buoy["geometry"]["coordinates"][0]
            dist = haversineDist(buoy_lat, buoy_lng, buoy2_lat, buoy2_lng)
            if dist < min_dist:
                min_dist = dist
                res = buoy

        if round(min_dist) > 5:
            print(f"Warning: Min dist of matched buoys is {min_dist}")
        return res["properties"]["id"]

    def getBuoyByID(self, ID):
        for buoy in self.data["features"]:
            if buoy["properties"]["id"] == ID:
                return buoy
        return None

    def getBuoyLocationsThreading(self, pos_lat, pos_lng, results_list, event):
        # Function to get BuoyLocations in a thread -> saves locations in a results_list and sets event flag
        # Arguments: pos_lat & pos_lng are geographical coordinates
        self.tile_center = {"lat": pos_lat, "lng": pos_lng}
        for buoy in self.data["features"]:
            buoy_lat = buoy["geometry"]["coordinates"][1]
            buoy_lng = buoy["geometry"]["coordinates"][0]
            if abs(buoy_lng - pos_lng) < self.tile_size and abs(buoy_lat - pos_lat) < self.tile_size:
                results_list.append(buoy)
        event.set() 

    def checkForRefresh(self, lat, lng):
        # function checks whether lat & lng are too close to tile edge
        # returns true if this is the case, else false
        if abs(lat-self.tile_center["lat"]) > self.tile_size*0.5 or abs(lng-self.tile_center["lng"]) > self.tile_size*0.5:
            return True
        else:
            return False

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
        plt.savefig("buoyLocations.pdf")
        #plt.show()
