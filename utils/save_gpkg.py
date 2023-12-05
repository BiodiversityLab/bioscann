import geopandas as geopd
import json

def append_lists_to_dataframe(names,data, geodataframe):
    for i, p in enumerate(data):
        for name in names:
            #geodataframe.loc[i,[name]] = "".join([str(x)+" " for x in data[i][name]])
            geodataframe.loc[i,[name]] = json.dumps(data[i][name])
    return geodataframe



def save(geodf_separata_polygoner, filename, layer='BoundingBoxes'):
        #Write to geopackage file
    polys = []
    for poly in geodf_separata_polygoner['geometry']:
        polys.append(poly)
        
    f = geopd.GeoDataFrame(geodf_separata_polygoner,geometry=polys,crs='EPSG:3006')

    f.to_file(driver='GPKG', filename=filename, layer=layer, encoding='utf-8', mode='w')
