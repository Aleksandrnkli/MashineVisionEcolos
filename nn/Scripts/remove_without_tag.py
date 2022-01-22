import os
try:
    from lxml import etree as ET
except  ImportError:
    import xml.etree.ElementTree as ET
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="remove annotations without tag plate")
    path = parser.add_argument("--path", default="E:/DataSets/LicensePlate/Russian_KZ_LPR_detection_2_class/Annotations", help="folder with annotations path")

    xmls_list = filter(lambda x: x.endswith('.xml'), os.listdir(path))
    for xml_f in xmls_list:
        tree = ET.parse(path + "/" +xml_f)
        print(xml_f)
        try:
            objcts = tree.findall('.object')
            for obj in objcts:
               if obj.find('./plate') is not None:
                   continue
               else:
                    os.remove(path + "/" + xml_f)
                    break
            # name, ext = os.path.splitext(xml_f)
            # dif = len(name) - 8
            # if len(name) > 8:
            #     plate_elem = ET.fromstring(f'<plate>{name[:-dif]}</plate>')
            # else:
            #     plate_elem = ET.fromstring(f'<plate>{name}</plate>')
            # tree.find('.object').append(plate_elem)
        except AttributeError:
            continue

        tree.write(path + "/" + xml_f)


