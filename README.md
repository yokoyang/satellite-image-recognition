# satellite-image-recognition
satellite image recognition using U-Net model

## demo

![demo1][1]
![demo2][2]
![demo3][3]
![demo3][4]
![demo3][5]

## Data source

Get data from Google satellite map, and then labeled and splite. Then model is based on  54 * (160px*160px*3) RGB images data.

## train

Training process is using reflection,mirror method and U-net model
## RGB list
- ['water'] = [48, 93, 254]
- ['tree'] = [12, 169, 64]
- ['playground'] = [139, 69, 19]
- ['road'] = [47, 79, 79]
- ['building_yard'] = [255, 255, 255]
- ['bare_land'] = [239, 156, 119]
- ['general_building'] = [249, 255, 25]
- ['countryside'] = [227, 23, 33]
- ['factory'] = [48, 254, 254]
- ['shadow'] = [255, 1, 255]

## quote from

> Long J, Shelhamer E, Darrell T. Fully convolutional networks for semantic segmentation[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 3431-3440.

[1]: https://github.com/yokoyang/satellite-image-recognition/blob/master/img/1.gif

[2]: https://github.com/yokoyang/satellite-image-recognition/blob/master/img/2.gif

[3]: https://github.com/yokoyang/satellite-image-recognition/blob/master/img/3.gif

[4]: https://github.com/yokoyang/satellite-image-recognition/blob/master/img/4.gif

[5]: https://github.com/yokoyang/satellite-image-recognition/blob/master/img/5.gif