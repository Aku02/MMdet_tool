import copy
import os

import mmcv
from tqdm import tqdm
from pycocotools.coco import COCO
import json

CATEGORIES = (
      'empty',
    'fox',
    'skunk',
    'rodent',
    'bird',
    'american crow',
    'american black bear',
    'chicken',
    'virginia opossum',
    'domestic cat',
    'grey fox',
    'rooster',
    'donkey',
    'raven',
    'petrel_chick',
    'goat',
    'pig',
    'shearwater',
    'iguana',
    'cat'
)
CAT2IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}


def init_coco():
    return {
        'info': {},
        'categories':
            [{
                'id': idx,
                'name': cat,
            } for cat, idx in CAT2IDX.items()]
    }


def to_multiclass(ann_file):
    img_infos = []
    ann_infos = []
    img_id = 0
    ann_id = 0
    coco = COCO(ann_file)
    with open('train.json', 'rt', encoding='UTF-8') as annotations:
        c = json.load(annotations)
    for _id in tqdm(coco.getImgIds()):
        # print(_id)
        # print(coco.loadImgs(f'{_id}'))
        # img_info = copy.deepcopy(coco.loadImgs(_id))
        # print(img_info)
        # img_info['id'] = img_id
        # filename = img_info['file_path']
        # subdir = filename#.split('_')[0]
        # img_info['file_name'] = filename#os.path.join(subdir, filename)
        # cat = subdir.lower()

        # img_infos.append(img_info)
        for ann_info in coco.loadAnns(coco.getAnnIds(_id)):
            ann_info = copy.deepcopy(ann_info)
            ann_info['image_id'] = img_id
            ann_info['id'] = ann_id
            # ann_info['category_id'] = CAT2IDX[cat]
            ann_infos.append(ann_info)
            ann_id += 1
        img_id += 1

    coco = init_coco()
    # coco['categories'] = train_coco.getCatIds()
    coco['images'] = c['images']#img_infos
    coco['annotations'] = ann_infos
    return coco


if __name__ == '__main__':
    mmcv.dump(
        to_multiclass('train.json'),
        'train_20.json'
    )
    # mmcv.dump(
    #     to_multiclass('../data/LIVECell_dataset_2021/livecell_coco_val.json'),
    #     '../data/LIVECell_dataset_2021/val_8class.json'
    # )
    # mmcv.dump(
    #     to_multiclass('../data/LIVECell_dataset_2021/livecell_coco_test.json'),
    #     '../data/LIVECell_dataset_2021/test_8class.json'
    # )