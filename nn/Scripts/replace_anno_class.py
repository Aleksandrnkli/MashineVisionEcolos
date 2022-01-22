import os

try:
    from lxml import etree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="replace name tag and put plate number into plate tag")
    parser.add_argument("--anno_path", default="E:/DataSets/LicensePlate/Russian_KZ_LPR_detection_2_class/Annotations", help="annotations path")
    parser.add_argument("--anno_class", default="plate number", help="tag name class ")
    args = parser.parse_args()

    ANNO_CLASS = args.anno_class
    path = args.anno_class

    xmls_list = filter(lambda x: x.endswith('.xml'), os.listdir(path))
    for xml_file in xmls_list:
        tree = ET.parse(path + '/' + xml_file)
        print(xml_file)
        try:
            tree.find('./object').find('./name').text
        except AttributeError:
            continue
        objs = tree.findall('./object')
        for obj in objs:
            plate_number = obj.find('./name').text
            plate_elem = ET.fromstring(f'<plate>{plate_number}</plate>')
            obj.append(plate_elem)
            obj.find('./name').text = ANNO_CLASS
        tree.write(path + '/' + xml_file)

