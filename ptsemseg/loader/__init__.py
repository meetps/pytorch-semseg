import json

from ptsemseg.loader.pascal_voc_loader import pascalVOCLoader
from ptsemseg.loader.camvid_loader import camvidLoader

def get_loader(name):
    return {
        'pascal': pascalVOCLoader,
        'camvid': camvidLoader,
    }[name]

def get_data_path(name):
    js = open('config.json').read()
    data = json.loads(js)
    return data[name]['data_path']