import json
import argparse
# import funcy
# from sklearn.model_selection import train_test_split
import mmcv
# from tqdm import tqdm
# from pycocotools.coco import COCO

# parser = argparse.ArgumentParser(description='Splits COCO annotations file into training and test sets.')
# parser.add_argument('--annotations', metavar='coco_annotations', type=str, default= 'train.json',
#                     help='Path to COCO annotations file.')
# parser.add_argument('--train', type=str, help='Where to store COCO training annotations')
# # parser.add_argument('test', type=str, help='Where to store COCO test annotations')
# # parser.add_argument('-s', dest='split', type=float, required=True,
# #                     help="A percentage of a split; a number in (0, 1)")
# parser.add_argument('--having-annotations', dest='having_annotations', action='store_true',
#                     help='Ignore all images without annotations. Keep only these with at least one annotation')

# args = parser.parse_args()

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
    
def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

# def filter_annotations(annotations, images):
#     image_ids = funcy.lmap(lambda i: str(i['id']), images)
#     return funcy.lfilter(lambda a: str(a['image_id']) in image_ids, annotations)
def multi(ann):
    c = {}
    with open(ann, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        # info = coco['info']
        annotations = coco['annotations']
    # categories = coco['categories']
    for i, ann in enumerate(coco["annotations"]):
        id = str(ann["image_id"])
        category_id = int(ann["category_id"])
        bbox = ann['bbox']
        # score = 1.0
    
    c['id'] = id
    c['category_id'] = category_id
    c['bbox'] = bbox
    c['score'] = 1.0
    # coco['images'] = img_infos
    # coco['annotations'] = ann_infos
    return c

def main():
    mmcv.dump(
        multi('data_wild/validation.json'),
        'val_20_class.json'
    )


        # images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

        # if having_annotations:
        #     images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        # x, y = train_test_split(images, train_size=args.split)

        # save_coco('train_coco.json', info, licenses, images, filter_annotations(annotations, images), categories)
        
        # save_coco(args.test, info, licenses, y, filter_annotations(annotations, y), categories)

        # print("Saved {} entries in {} and {} in {}".format(len(x), args.train, len(y), args.test))


if __name__ == "__main__":
    main()