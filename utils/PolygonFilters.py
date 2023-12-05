import pandas as pd
import geopandas as geopd
import pdb

def indata_version_1(geodf_polygoner, name):
    lista = []
    for index, polygon1 in enumerate(geodf_polygoner.iterrows()):
        polygon1 = polygon1[1]
        # THIS WILL GET ALL THE RELEVANT NATURA2000 POLYGONS
        if 'KARTERINGS' in polygon1 and polygon1['KARTERINGS'] is not None:
            if '3 - Besökt i fält' == polygon1['KARTERINGS'] or '4 - Inventerad i fält' == polygon1['KARTERINGS']:
                if polygon1['NATURTYPSS']=='1 - Fullgod Natura-naturtyp' or polygon1['NATURTYPSS']=='2 - Icke fullgod Natura-naturtyp':
                    polygon1['TARGET_FID']=index
                    if polygon1['geometry'].geom_type == 'MultiPolygon':
                        # extract polygons out of multipolygon
                        for p1 in polygon1['geometry'].geoms:
                            lista.append({'geometry':p1, 'TARGET_FID':polygon1['TARGET_FID'], "NATURTYPSS": polygon1['NATURTYPSS'], "KARTERINGS":polygon1['KARTERINGS']})
                        # lista.append({'geometry':p1, 'TARGET_FID':polygon1['TARGET_FID'], "NATURTYPSS":1})
                    elif polygon1['geometry'].geom_type == 'Polygon':
                        lista.append({'geometry':polygon1, 'TARGET_FID':polygon1['TARGET_FID'], "NATURTYPSS": polygon1['NATURTYPSS'], "KARTERINGS":polygon1['KARTERINGS']})
        # THIS WILL GET ALL THE NYCKELBIOTOP POLYGONS
        elif 'Beteckn' in polygon1 and polygon1['Beteckn'] is not None:
            if polygon1['geometry'].geom_type == 'MultiPolygon':
                # extract polygons out of multipolygon
                for p1 in polygon1['geometry'].geoms:
                    lista.append({'geometry':p1, 'TARGET_FID':polygon1['TARGET_FID'], "NATURTYPSS": "NyckelBiotop", "KARTERINGS":polygon1['KARTERINGS']})
                # lista.append({'geometry':p1, 'TARGET_FID':polygon1['TARGET_FID'], "NATURTYPSS":1})
            elif polygon1['geometry'].geom_type == 'Polygon':
                lista.append({'geometry':polygon1, 'TARGET_FID':polygon1['TARGET_FID'], "NATURTYPSS": "NyckelBiotop", "KARTERINGS":polygon1['KARTERINGS']})
        # THIS WILL GET ALL THE LOW NATURVARDEN POLYGONS
        else:
            #low values polygon
            if polygon1['geometry'].geom_type == 'MultiPolygon':
                    # extract polygons out of multipolygon
                    for p1 in polygon1['geometry'].geoms:
                        lista.append({'geometry':p1, 'TARGET_FID':name+str(polygon1['OBJECTID']), "NATURTYPSS": '', "KARTERINGS":'Tidigare Hygge'})
            elif polygon1['geometry'].geom_type == 'Polygon':
                lista.append({'geometry':polygon1, 'TARGET_FID':name+str(polygon1['OBJECTID']), "NATURTYPSS": '', "KARTERINGS":'Tidigare Hygge'})

    df_separata_polygoner = pd.DataFrame(lista)
    all_polygons = geopd.GeoDataFrame(df_separata_polygoner) 
    return all_polygons


def indata_version_2(geodf_polygoner, name):
    lista = []
    for index, polygon1 in enumerate(geodf_polygoner.iterrows()):
        polygon1 = polygon1[1]
        filename = '_'
        if polygon1['filename'] is not None:
            filename = polygon1['filename']
        # THIS WILL GET ALL THE RELEVANT NATURA2000 POLYGONS
        #TODO: metod som filtrerar bort den disjunkta mängden av det den ska vara istället för att kolla efter olika alternativ av vad den kan vara
        #Förslag: Använd bara: isinstance(polygon1['KARTERINGS'],str) and len(polygon1['KARTERINGS'])>1
        if 'KARTERINGS' in polygon1 and polygon1['KARTERINGS'] is not None and isinstance(polygon1['KARTERINGS'],str) and len(polygon1['KARTERINGS'])>1:
            if '3 - Besökt i fält' == polygon1['KARTERINGS'] or '4 - Inventerad i fält' == polygon1['KARTERINGS'] or '2 - Granskad vid skrivbordet' == polygon1['KARTERINGS']:
                if polygon1['NATURTYPSS']=='1 - Fullgod Natura-naturtyp' or polygon1['NATURTYPSS']=='2 - Icke fullgod Natura-naturtyp':
                    polygon1['TARGET_FID']=index
                    if polygon1['geometry'].geom_type == 'MultiPolygon':
                        # extract polygons out of multipolygon
                        for p1 in polygon1['geometry'].geoms:
                            lista.append({'geometry':p1, 'TARGET_FID':polygon1['TARGET_FID'], "NATURTYPSS": polygon1['NATURTYPSS'], "KARTERINGS":polygon1['KARTERINGS'], 'filename':filename})
                        # lista.append({'geometry':p1, 'TARGET_FID':polygon1['TARGET_FID'], "NATURTYPSS":1})
                    elif polygon1['geometry'].geom_type == 'Polygon':
                        lista.append({'geometry':polygon1, 'TARGET_FID':polygon1['TARGET_FID'], "NATURTYPSS": polygon1['NATURTYPSS'], "KARTERINGS":polygon1['KARTERINGS'], 'filename':filename})
        # THIS WILL GET ALL THE NYCKELBIOTOP POLYGONS
        elif 'Beteckn' in polygon1 and polygon1['Beteckn'] is not None and isinstance(polygon1['Beteckn'],str) and len(polygon1['Beteckn'])>1:
            if polygon1['geometry'].geom_type == 'MultiPolygon':
                # extract polygons out of multipolygon
                for p1 in polygon1['geometry'].geoms:
                    lista.append({'geometry':p1, 'TARGET_FID':polygon1['TARGET_FID'], "NATURTYPSS": "NyckelBiotop", "KARTERINGS":"NyckelBiotop", 'filename':filename})
                # lista.append({'geometry':p1, 'TARGET_FID':polygon1['TARGET_FID'], "NATURTYPSS":1})
            elif polygon1['geometry'].geom_type == 'Polygon':
                lista.append({'geometry':polygon1, 'TARGET_FID':polygon1['TARGET_FID'], "NATURTYPSS": "NyckelBiotop", "KARTERINGS":"NyckelBiotop", 'filename':filename})
        elif 'Tabell' in polygon1 and polygon1['Tabell'] is not None and polygon1['Tabell']=='NYCKELSVEASKOGYTA':
            #print("yes")
            if polygon1['geometry'].geom_type == 'MultiPolygon':
                # extract polygons out of multipolygon
                for p1 in polygon1['geometry'].geoms:
                    lista.append({'geometry':p1, 'TARGET_FID':polygon1['TARGET_FID'], "NATURTYPSS": "NyckelBiotop", "KARTERINGS":"NyckelBiotop", 'filename':filename})
                # lista.append({'geometry':p1, 'TARGET_FID':polygon1['TARGET_FID'], "NATURTYPSS":1})
            elif polygon1['geometry'].geom_type == 'Polygon':
                lista.append({'geometry':polygon1, 'TARGET_FID':polygon1['TARGET_FID'], "NATURTYPSS": "NyckelBiotop", "KARTERINGS":"NyckelBiotop", 'filename':filename})
        # THIS WILL GET ALL THE LOW NATURVARDEN POLYGONS
        elif filename.startswith('NB'):
            if polygon1['geometry'].geom_type == 'MultiPolygon':
                # extract polygons out of multipolygon
                for p1 in polygon1['geometry'].geoms:
                    lista.append({'geometry':p1, 'TARGET_FID':polygon1['TARGET_FID'], "NATURTYPSS": "NyckelBiotop", "KARTERINGS":"NyckelBiotop", 'filename':filename})
                # lista.append({'geometry':p1, 'TARGET_FID':polygon1['TARGET_FID'], "NATURTYPSS":1})
            elif polygon1['geometry'].geom_type == 'Polygon':
                lista.append({'geometry':polygon1, 'TARGET_FID':polygon1['TARGET_FID'], "NATURTYPSS": "NyckelBiotop", "KARTERINGS":"NyckelBiotop", 'filename':filename})
        else:
            #low values polygon
            if polygon1['geometry'].geom_type == 'MultiPolygon':
                    # extract polygons out of multipolygon
                    for p1 in polygon1['geometry'].geoms:
                        lista.append({'geometry':p1, 'TARGET_FID':name+str(polygon1['OBJECTID']), "NATURTYPSS": '', "KARTERINGS":'Tidigare Hygge', 'filename':filename})
            elif polygon1['geometry'].geom_type == 'Polygon':
                lista.append({'geometry':polygon1, 'TARGET_FID':name+str(polygon1['OBJECTID']), "NATURTYPSS": '', "KARTERINGS":'Tidigare Hygge', 'filename':filename})

    df_separata_polygoner = pd.DataFrame(lista)
    all_polygons = geopd.GeoDataFrame(df_separata_polygoner) 
    return all_polygons

def annotation_version_1(poly):
    if poly['NATURTYPSS']=='1 - Fullgod Natura-naturtyp' or poly['NATURTYPSS']=='2 - Icke fullgod Natura-naturtyp' or poly['NATURTYPSS']=='NyckelBiotop':
        return True
    else: return False


def forest_detection_1970_indata_filtering(geodf_polygoner, name):
    lista = []
    for index, polygon1 in enumerate(geodf_polygoner.iterrows()):
        polygon1 = polygon1[1]
        if polygon1['geometry'] is not None:
            if 'class' in polygon1 and polygon1['class'] is 1:
                if polygon1['geometry'].geom_type == 'MultiPolygon':
                    for p1 in polygon1['geometry'].geoms:
                        if 'OBJECTID' in polygon1:
                            lista.append({'geometry':p1, 'TARGET_FID':polygon1['OBJECTID'], "CLASS": "FOREST"})
                        else:
                            lista.append({'geometry':p1, 'TARGET_FID':index, "CLASS": "FOREST"})
                elif polygon1['geometry'].geom_type == 'Polygon':
                    if 'OBJECTID' in polygon1:
                        lista.append({'geometry':polygon1,'TARGET_FID':polygon1['OBJECTID'], "CLASS": "FOREST"})
                    else:
                        lista.append({'geometry':polygon1,'TARGET_FID':index, "CLASS": "FOREST"})

            else:  
                if polygon1['geometry'].geom_type == 'MultiPolygon':
                    for p1 in polygon1['geometry'].geoms:
                        if 'OBJECTID' in polygon1:
                            lista.append({'geometry':p1, 'TARGET_FID':polygon1['OBJECTID'], "CLASS": "NO_FOREST"})
                        else:
                            #pdb.set_trace()
                            lista.append({'geometry':p1, 'TARGET_FID':index, "CLASS": "NO_FOREST"})
                elif polygon1['geometry'].geom_type == 'Polygon':
                    if 'OBJECTID' in polygon1:
                        lista.append({'geometry':polygon1,'TARGET_FID':polygon1['OBJECTID'], "CLASS": "NO_FOREST"})
                    else:
                        lista.append({'geometry':polygon1,'TARGET_FID':index, "CLASS": "NO_FOREST"})

    df_separata_polygoner = pd.DataFrame(lista)
    all_polygons = geopd.GeoDataFrame(df_separata_polygoner) 
    return all_polygons

def annotation_forest_detection_1970(poly):
    if poly['CLASS']=='FOREST':
        return True
    else: return False


configurations = {
    "version_1": {"indata_filtering": indata_version_1, "annotation_filtering": annotation_version_1},
    "version_2": {"indata_filtering": indata_version_2, "annotation_filtering": annotation_version_1},
    "forest_detection_1970": {"indata_filtering": forest_detection_1970_indata_filtering, "annotation_filtering": annotation_forest_detection_1970}
}