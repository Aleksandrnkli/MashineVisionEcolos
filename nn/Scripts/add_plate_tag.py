import os
import argparse
try:
    from lxml import etree as ET
except ImportError:
    import xml.etree.ElementTree as ET



def add_plate_anno(args):
    path_without_tag = args.path_without_tag
    path_with_tag = args.path_with_tag

    xmls_list_without_plate = filter(lambda x: x.endswith('.xml'), os.listdir(path_without_tag))
    xmls_list_with_tag = filter(lambda x: x.endswith('.xml'), os.listdir(path_with_tag))

    for xml_file_1 in xmls_list_without_plate:
        for xml_file_2 in xmls_list_with_tag:
                if xml_file_1 == xml_file_2:
                    tree1 = ET.parse(path_without_tag + (f'/{xml_file_1}') )
                    tree2 = ET.parse(path_with_tag + f'/{xml_file_2}')
                    print(xml_file_1)
                    try:
                        tree2.find('./object').find('./plate').text
                    except AttributeError:
                        continue
                    objs1 = tree1.findall('./object')
                    objs2 = tree2.findall('./object')
                    for obj in enumerate(objs2):
                        plate_number = obj[1].find('./plate').text
                        plate_elem = ET.fromstring(f'<plate>{plate_number}</plate>')
                        if objs1[obj[0]].find('./plate'):
                            continue
                        else:
                            objs1[obj[0]].append(plate_elem)
                    tree1.write(xml_file_1)
                    break


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Add plate number into anno without number')
    parser.add_argument('--path_without_tag', default="E:/DataSets/LicensePlate/Russian_KZ_LPR_detection_2_class(copy)/Annotations", help='the annotations without tag path')
    parser.add_argument('--path_with_tag', default="E:/DataSets/LicensePlate/test/Russian_KZ_LPR_detection_2_class/Annotations", help='the annotations with tag path')
    args = parser.parse_args()
    replace_xml_anno_path(args)
