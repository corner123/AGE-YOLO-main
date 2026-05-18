import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

# GC10-DET class mapping: XML annotation name -> class ID
# Paper Section 4.2: 10 defect categories
XML_NAME_TO_ID = {
    '1_chongkong': 0,    # punching_hole (ph)
    '2_hanfeng': 1,      # weld_line (wl)
    '3_yueyawan': 2,     # crescent_gap (cg)
    '4_shuiban': 3,      # water_spot (ws)
    '5_youban': 4,       # oil_spot (os)
    '6_siban': 5,        # silk_spot (ss)
    '7_yiwu': 6,         # inclusion (in)
    '8_yahen': 7,        # rolled_pit (rp)
    '9_zhehen': 8,       # crease (cr)
    '10_yaozhe': 9,      # waist_folding (wf)
    '10_yaozhed': 9,     # waist_folding (wf) - typo variant
}

CLASSES = ['punching_hole', 'weld_line', 'crescent_gap', 'water_spot', 'oil_spot',
           'silk_spot', 'inclusion', 'rolled_pit', 'crease', 'waist_folding']


def convert_box(size, box):
    """Convert VOC coordinates (xmin, xmax, ymin, ymax) to YOLO format"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 * dw
    y = (box[2] + box[3]) / 2.0 * dh
    w = (box[1] - box[0]) * dw
    h = (box[3] - box[2]) * dh
    return (x, y, w, h)


def convert_xml_to_yolo(xml_dir, save_dir):
    """Read VOC XMLs and save labels in YOLO txt format"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]

    for xml_file in tqdm(xml_files, desc="Converting XML to YOLO"):
        tree = ET.parse(os.path.join(xml_dir, xml_file))
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        txt_name = xml_file.replace('.xml', '.txt')
        with open(os.path.join(save_dir, txt_name), 'w') as out_file:
            for obj in root.iter('object'):
                cls = obj.find('name').text.lower().strip()
                if cls not in XML_NAME_TO_ID:
                    continue
                cls_id = XML_NAME_TO_ID[cls]
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert_box((w, h), b)
                out_file.write(f"{cls_id} {' '.join([f'{a:.6f}' for a in bb])}\n")


if __name__ == "__main__":
    convert_xml_to_yolo(
        xml_dir='D:/wupengju/Code/workspace_acdemic/AGE-YOLO-main/dataset/GC10-DET/lable',
        save_dir='D:/wupengju/Code/workspace_acdemic/AGE-YOLO-main/dataset/GC10-DET/labels'
    )