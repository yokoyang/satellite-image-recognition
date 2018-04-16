# Import flask dependencies
import base64
import json
import math
from functools import partial

import cv2
import numpy as np
import pyproj
import requests
import shapely.ops as ops
import tifffile as tiff
from PIL import Image
from flask import Blueprint, request, render_template
from shapely.geometry import Polygon
from skimage import measure, morphology

from app import set_classes
from app import u_net

zoom = 18
lat = 31.284091
lng = 121.503364

dic_color = {}
dic_color['water'] = [48, 93, 254]
dic_color['tree'] = [12, 169, 64]
dic_color['playground'] = [139, 69, 19]
dic_color['road'] = [47, 79, 79]
dic_color['building_yard'] = [255, 255, 255]
dic_color['bare_land'] = [239, 156, 119]
dic_color['general_building'] = [249, 255, 25]
dic_color['countryside'] = [227, 23, 33]
dic_color['factory'] = [48, 254, 254]
dic_color['shadow'] = [255, 1, 255]

mod_diag = Blueprint('diag', __name__, url_prefix='/diag')
img_path = 'app/mod_diag/upload/img'


@mod_diag.route('/')
def home():
    return render_template("diag/citylens.html")


@mod_diag.route('/upload', methods=['GET', 'POST'])
def diagnose():
    # if request.method == 'POST':
    #     saveImage(request.values.get("imgData"), img_path +'.jpg')
    #     img, image = processImage(img_path)

    data = ''
    zoom = request.values.get("zoom")
    lat = request.values.get("lat")
    lng = request.values.get("lng")
    qtype = request.values.get("type")

    url = 'https://maps.googleapis.com/maps/api/staticmap?center=' + lat + ',' + lng + '&zoom=' + zoom + '&size=640x640&maptype=satellite&key=AIzaSyCdhjAB0Twe3dBRdVXXG1MJV6t0k4vkaXk'
    # print("start save")
    save_request_img(url, img_path + '.png')
    # print("finish save")

    image = processImage(img_path)
    out = u_net.predict_target_classes(set_classes, image)
    tiff.imsave("app/mod_diag/upload/out.tif", out)
    image = Image.open("app/mod_diag/upload/out.tif")
    contours = findContours(image)
    out = getPloyandArea(contours, float(zoom), float(lat), float(lng))
    out = getMerit(out)

    # with open('test.json', 'w') as outfile:
    #     json.dump(out, outfile)

    return json.dumps(out)


def changeCBack(block_list, w, h):
    nlist = []
    for x in block_list:
        one = [x[0] - w, x[1] - h]
        nlist.append(one)
    return nlist


def save_request_img(url, img_path):
    html = requests.get(url)
    with open(img_path, 'wb') as file:  # 以byte形式将图片数据写入
        file.write(html.content)
        file.flush()
    file.close()  # 关闭文件


def findContours(image):
    result = {}
    xdata = np.array(image)
    for k in dic_color:

        tmp = xdata
        mask = (tmp[:, :, 0] == dic_color[k][0]) & (tmp[:, :, 1] == dic_color[k][1]) & (tmp[:, :, 2] == dic_color[k][2])

        dst = morphology.remove_small_objects(mask, min_size=150, connectivity=50)
        img = dst.astype(np.uint8)
        img *= 255

        old_im = Image.fromarray(img, mode='L')
        old_size = old_im.size

        new_size = (old_size[0] * 2, old_size[1] * 2)
        new_im = Image.new("L", new_size)
        new_im.paste(old_im, (int(old_size[0] / 2), int(old_size[0] / 2)))
        contours = measure.find_contours(new_im, 0, positive_orientation='high')

        al = []
        for x in contours:
            out = changeCBack(x, old_size[0] / 2, old_size[0] / 2)
            al.append(np.array(out))
        result[k] = al

    return result


def getPloyandArea(contours, zoom, lat, lng):
    res = {}
    box = [[[0, 0], [0, 640], [640, 640], [640, 0], [0, 0]]]
    aarea = getArea(box, zoom, lat, lng)

    for k in contours:
        del_li, matrix = getMatrix(contours[k])
        poly_outer = []
        poly_inner = []
        area_list = []

        hole_list = []
        for m in range(len(matrix)):
            foo_index = [n for n, x in enumerate(matrix[m]) if x == 1]
            hole_list += foo_index

        for x in range(len(contours[k])):
            if x not in del_li and x not in hole_list:
                index = np.where(np.array(matrix[x]) == 1)
                outer = getCoords([contours[k][x]], zoom, lat, lng)[0]
                inner = getCoords(np.array(contours[k])[index], zoom, lat, lng)
                myPloy = Polygon(outer, inner)

                geom_area = ops.transform(
                    partial(
                        pyproj.transform,
                        pyproj.Proj(init='EPSG:4326'),
                        pyproj.Proj(
                            proj='aea',
                            lat1=myPloy.bounds[1],
                            lat2=myPloy.bounds[3])),
                    myPloy)
                if int(geom_area.area) != 0:
                    poly_outer.append(outer)
                    poly_inner.append(inner)
                    area_list.append(int(geom_area.area))

        res[k] = {'gdata': getGeojson(poly_outer, poly_inner, area_list), 'number': len(area_list),
                  'area': sum(area_list), 'percentage': int(sum(area_list) * 1.0 / aarea * 100)}

    return res


def getMatrix(al):
    matrix = [[0 for i in range(len(al))] for j in range(len(al))]
    del_list = []

    for x in range(len(al)):
        for y in range(len(al)):
            if len(al[x]) <= 3:
                if x not in del_list:
                    del_list.append(x)
                continue
            if len(al[y]) <= 3:
                if y not in del_list:
                    del_list.append(y)
                continue

            polygon1 = Polygon(al[x])
            polygon2 = Polygon(al[y])

            if x != y and polygon1.contains(polygon2):
                matrix[x][y] = 1

    return del_list, matrix


def getCoords(data, zoom, lat, lng):
    result = []

    for x in range(len(data)):
        out = []
        for k in range(len(data[x])):
            res = getPointLatLng(data[x][k][1], data[x][k][0], zoom, lat, lng)
            out.append(res)

        result.append(out)
    return result


def getPointLatLng(x, y, zoom, lat, lng):
    h = 640
    w = 640

    parallelMultiplier = math.cos(lat * math.pi / 180)
    degreesPerPixelX = 360 / math.pow(2, zoom + 8)
    degreesPerPixelY = 360 / math.pow(2, zoom + 8) * parallelMultiplier
    pointLat = lat - degreesPerPixelY * (y - h / 2)
    pointLng = lng + degreesPerPixelX * (x - w / 2)

    return (pointLng, pointLat)


def getGeojson(poly_outer, poly_inner, area_list):
    feature = []

    for x in range(len(poly_outer)):
        out_coords = []
        for point in poly_outer[x]:
            out_coords.append([point[0], point[1]])
        out_coords.append(out_coords[0])

        inner_coords = []
        for p in poly_inner[x]:
            ner_coords = []
            for point in p:
                ner_coords.append([point[0], point[1]])
            ner_coords.append(ner_coords[0])
            inner_coords.append(ner_coords)

        f = [out_coords] + inner_coords
        cood = {"type": "Polygon", "coordinates": f}
        t = {"type": "Feature", "properties": {"area": area_list[x], "number": x}, "geometry": cood}
        feature.append(t)

    output = {"type": "FeatureCollection", "features": feature}

    return output


def getMerit(data):
    number = 0
    area = 0
    percentage = 0

    box = [[[0, 0], [0, 640], [640, 640], [640, 0], [0, 0]]]
    aarea = getArea(box, zoom, lat, lng)

    for x in data:
        number += data[x]['number']
        area += data[x]['area']
        percentage += data[x]['percentage']

    if percentage >= 100:
        percentage = 100

    data['all'] = {'number': number, 'area': area, 'percentage': percentage, 'aarea': round(aarea * 0.000001, 2)}
    return data


def getArea(data, zoom, lat, lng):
    inner = getCoords(data, zoom, lat, lng)
    myPloy = Polygon(inner[0])

    geom_area = ops.transform(
        partial(
            pyproj.transform,
            pyproj.Proj(init='EPSG:4326'),
            pyproj.Proj(
                proj='aea',
                lat1=myPloy.bounds[1],
                lat2=myPloy.bounds[3])),
        myPloy)

    return geom_area.area


def changeCBack(block_list):
    nlist = []
    for x in block_list:
        one = [x[0] - 384, x[1] - 384]
        nlist.append(one)
    return nlist


def findContours(image):
    result = {}
    xdata = np.array(image)
    for k in dic_color:

        tmp = xdata
        mask = (tmp[:, :, 0] == dic_color[k][0]) & (tmp[:, :, 1] == dic_color[k][1]) & (tmp[:, :, 2] == dic_color[k][2])

        dst = morphology.remove_small_objects(mask, min_size=150, connectivity=50)
        img = dst.astype(np.uint8)
        img *= 255

        old_im = Image.fromarray(img, mode='L')
        old_size = old_im.size

        new_size = (old_size[0] * 2, old_size[0] * 2)
        new_im = Image.new("L", new_size)
        new_im.paste(old_im, (384, 384))
        contours = measure.find_contours(new_im, 0, positive_orientation='high')

        al = []
        for x in contours:
            out = changeCBack(x)
            al.append(np.array(out))
        result[k] = al

    return result


def saveImage(data, img_path):
    imgData = data[23:]
    with open(img_path, 'wb') as f:
        imgData = base64.standard_b64decode(imgData)
        newFileByteArray = bytearray(imgData)
        f.write(newFileByteArray)


def processImage(img_path):
    img = Image.open(img_path + '.png')
    img.save(img_path + '.tif', "TIFF")
    a = cv2.imread(img_path + '.tif')
    cv2.imwrite(img_path + '.tif', a)
    image = tiff.imread(img_path + '.tif')
    return image
