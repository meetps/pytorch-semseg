import json

from ptsemseg.loader.pascal_voc_loader import pascalVOCLoader
from ptsemseg.loader.camvid_loader import camvidLoader
from ptsemseg.loader.ade20k_loader import ADE20KLoader

def get_loader(name):
    return {
        'pascal': pascalVOCLoader,
        'camvid': camvidLoader,
        'ade20k': ADE20KLoader,
    }[name]

def get_data_path(name):
    js = open('config.json').read()
    data = json.loads(js)
    return data[name]['data_path']