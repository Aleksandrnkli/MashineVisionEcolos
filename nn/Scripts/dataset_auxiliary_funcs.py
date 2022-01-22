import csv
import os

try:
    from lxml import etree as ET
except ImportError:
    import xml.etree.ElementTree as ET


def xml_to_csv_anno():	
    xmls_list = filter(lambda x: x.endswith('.xml'), os.listdir())

    for xml_file in xmls_list:
        root = ET.parse(xml_file).getroot()
        filename = root.find('filename').text

        annotations = []
        for annotation in root.findall('object'):
            params = []
            # [filename,x1,y1,x2,y2,class]
            params.append(filename)
            params.append(annotation.find('bndbox').find('xmin').text)
            params.append(annotation.find('bndbox').find('ymin').text)
            params.append(annotation.find('bndbox').find('xmax').text)
            params.append(annotation.find('bndbox').find('ymax').text)
            params.append(annotation.find('name').text)
            annotations.append(params)

        if len(annotations) == 0:
            # if image annotation has no target objects (persons, cars etc.)
            annotations.append([filename, '', '', '', '', ''])

        with open('dataset.csv', 'a', newline='', encoding='utf-8') as csv_file:
            annotation_writer = csv.writer(csv_file, delimiter=',')
            annotation_writer.writerows(annotations)



def replace_xml_anno_path(anno_path):
    xmls_list = filter(lambda x: x.endswith('.xml'), os.listdir())
    for xml_file in xmls_list:
        tree = ET.parse(xml_file)
        tree.find('./path').text = anno_path + tree.find('./filename').text
        tree.write(xml_file)


def delete_images_wthout_anno(anno_path, images_path, anno_format='.xml', photo_format='.jpg'):
    annotations = [item.replace(anno_format, '') for item in os.listdir(anno_path)]
    photos = [item.replace(photo_format, '') for item in os.listdir(images_path)]

    for photo in photos:
        if photo not in annotations:
            os.remove(images_path + '/' + photo + photo_format)


def change_xml_anno_class():
    xmls_list = filter(lambda x: x.endswith('.xml'), os.listdir())
    for xml_file in xmls_list:
        tree = ET.parse(xml_file)
        tree.find('./object').find('./name').text = 'defect'
        tree.write(xml_file)
