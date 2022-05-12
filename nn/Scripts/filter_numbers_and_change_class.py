import os
import argparse
try:
    from lxml import etree as ET
except  ImportError:
    import xml.etree.ElementTree as ET



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filtering numbers and changing class in anno')
    parser.add_argument('--path',
                                       default="E:/DataSets/LicensePlate/Russian_KZ_LPR_detection_2_class/Annotations",
                                       help='the annotations path')
    parser.add_argument('--class_name',
                                       default="RU",
                                       help='class name RU or KZ')
    args = parser.parse_args()
    class_name = args.class_name
    path = args.path
    if class_name == "RU":
        ANNO_CLASS = 'RU plate number' #if changing to RUSSIAN lp
    elif class_name == "KZ":
        ANNO_CLASS = 'KZ plate number' #if changing to KZ lp

    xmls_list = filter(lambda x: x.endswith('.xml'), os.listdir(path))
    x = 0
    for xml_f in xmls_list:
        tree = ET.parse(path + '/' + xml_f)
        print(xml_f)
        objcts = tree.findall('.object')
        if objcts:
            try:
                for obj in objcts:
                    obj.find('./name').text = ANNO_CLASS
                    if obj.findall('./plate') and len(obj.find('./plate').text) > 7:
                                x+=1
                                plate = obj.find('./plate').text
                                print("\nNumber: "+ str(plate) +' '+ str(x))

                    else:
                        tree.write(path + '/' + xml_f)
                        os.remove(path + '/' + xml_f)
                        print(f'removed annotation with {args.class_name} plate class but inconsistent non-{args.class_name} number')
                        break
            except AttributeError:
                continue
        else:
            tree.write(path + '/' + xml_f)
            os.remove(path + '/' + xml_f)
            print('removed annotation without object tag')
            continue


            # name, ext = os.path.splitext(xml_f)
            # dif = len(name) - 8
            # if len(name) > 8:
            #     plate_elem = ET.fromstring(f'<plate>{name[:-dif]}</plate>')
            # else:
            #     plate_elem = ET.fromstring(f'<plate>{name}</plate>')
            # tree.find('.object').append(plate_elem)


    tree.write(path + '/' + xml_f)


