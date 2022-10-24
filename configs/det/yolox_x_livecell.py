fp16 = dict(loss_scale=64.)
img_scale = (128,128)
num_last_epochs = 5

# model settings
model = dict(
    type='YOLOX',
    input_size=img_scale,
    random_size_range=(32, 64),
    random_size_interval=1,
    backbone=dict(
        type='YOLOPAFPNOfficial',
        depth=1.33,
        width=1.25,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/yolox_x_coco.pth',
            prefix='backbone'
        )
    ),
    neck=None,
    bbox_head=dict(
        type='YOLOXHeadOfficial',
        num_classes=498,
        width=1.25,
        in_channels=[256, 512, 1024],
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/yolox_x_coco.pth',
            prefix='head'
        )
    ),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65))
)

dataset_type = 'CellDataset'
classes = ('Bread, wholemeal', 'Jam', 'Water', 'Bread, sourdough', 'Banana', 'Soft cheese', 'Ham, raw', 'Hard cheese', 'Cottage cheese', 'Bread, half white', 'Coffee, with caffeine', 'Fruit salad', 'Pancakes', 'Tea', 'Salmon, smoked', 'Avocado', 'Spring onion / scallion', 'Ristretto, with caffeine', 'Ham', 'Egg', 'Bacon, frying', 'Chips, french fries', 'Juice, apple', 'Chicken', 'Tomato, raw ', 'Broccoli', 'Shrimp, boiled', 'Beetroot, steamed, without addition of salt', 'Carrot, raw', 'Chickpeas', 'French salad dressing', 'Pasta, Hörnli', 'Sauce, cream', 'Meat balls', 'Pasta', 'Tomato sauce', 'Cheese', 'Pear', 'Cashew nut', 'Almonds', 'Lentils', 'Mixed vegetables', 'Peanut butter', 'Apple', 'Blueberries', 'Cucumber', 'Cocoa powder', 'Greek Yaourt, yahourt, yogourt ou yoghourt', 'Maple syrup (Concentrate)', 'Buckwheat, grain peeled', 'Butter', 'Herbal tea', 'Mayonnaise', 'Soup, vegetable', 'Wine, red', 'Wine, white', 'Green bean, steamed, without addition of salt', 'Sausage', 'Pizza, Margherita, baked', 'Salami', 'Mushroom', '(bread, meat substitute, lettuce, sauce)', 'Tart', 'Tea, verveine', 'Rice', 'White coffee, with caffeine', 'Linseeds', 'Sunflower seeds', 'Ham, cooked', 'Bell pepper, red, raw ', 'Zucchini', 'Green asparagus', 'Tartar sauce', 'Lye pretzel (soft)', 'Cucumber, pickled ', 'Curry, vegetarian', 'Yaourt, yahourt, yogourt ou yoghourt, natural', 'Soup of lentils, Dahl (Dhal)', 'Soup, cream of vegetables', 'Balsamic vinegar', 'Salmon', 'Salt cake (vegetables, filled)', 'Bacon', 'Orange', 'Pasta, noodles', 'Cream', 'Cake, chocolate', 'Pasta, spaghetti', 'Black olives', 'Parmesan', 'Spaetzle', "Salad, lambs' ear", 'Salad, leaf / salad, green', 'Potatoes steamed', 'White cabbage', 'Halloumi', 'Beetroot, raw', 'Bread, grain', 'Applesauce, unsweetened, canned', 'Cheese for raclette', 'Mushrooms', 'Bread, white', 'Curds, natural, with at most 10% fidm', 'Bagel (without filling)', 'Quiche, with cheese, baked, with puff pastry', 'Soup, potato', 'Bouillon, vegetable', 'Beef, sirloin steak', 'Taboulé, prepared, with couscous', 'Eggplant', 'Bread', 'Turnover with meat (small meat pie, empanadas)', 'Mungbean sprouts', 'Mozzarella', 'Pasta, penne', 'Lasagne, vegetable, prepared', 'Mandarine', 'Kiwi', 'French beans', 'Tartar (meat)', 'Spring roll (fried)', 'Pork, chop', 'Caprese salad (Tomato Mozzarella)', 'Leaf spinach', 'Roll of half-white or white flour, with large void', 'Pasta, ravioli, stuffing', 'Omelette, plain', 'Tuna', 'Dark chocolate', 'Sauce (savoury)', 'Dried raisins', 'Ice tea', 'Kaki', 'Macaroon', 'Smoothie', 'Crêpe, plain', 'Chicken nuggets', 'Chili con carne, prepared', 'Veggie burger', 'Cream spinach', 'Cod', 'Chinese cabbage', 'Hamburger (Bread, meat, ketchup)', 'Soup, pumpkin', 'Sushi', 'Chestnuts', 'Coffee, decaffeinated', 'Sauce, soya', 'Balsamic salad dressing', 'Pasta, twist', 'Bolognaise sauce', 'Leek', 'Fajita (bread only)', 'Potato-gnocchi', 'Beef, cut into stripes (only meat)', 'Rice noodles/vermicelli', 'Tea, ginger', 'Tea, green', 'Bread, whole wheat', 'Onion', 'Garlic', 'Hummus', 'Pizza, with vegetables, baked', 'Beer', 'Glucose drink 50g', 'Chicken, wing', 'Ratatouille', 'Peanut', 'High protein pasta (made of lentils, peas, ...)', 'Cauliflower', 'Quiche, with spinach, baked, with cake dough', 'Green olives', 'Brazil nut', 'Eggplant caviar', 'Bread, pita', 'Pasta, wholemeal', 'Sauce, pesto', 'Oil', 'Couscous', 'Sauce, roast', 'Prosecco', 'Crackers', 'Bread, toast', 'Shrimp / prawn (small)', 'Panna cotta', 'Romanesco', 'Water with lemon juice', 'Espresso, with caffeine', 'Egg, scrambled, prepared', 'Juice, orange', 'Ice cubes', 'Braided white loaf', 'Emmental cheese', 'Croissant, wholegrain', 'Hazelnut-chocolate spread(Nutella, Ovomaltine, Caotina)', 'Tomme', 'Water, mineral', 'Hazelnut', 'Bacon, raw', 'Bread, nut', 'Black Forest Tart', 'Soup, Miso', 'Peach', 'Figs', 'Beef, filet', 'Mustard, Dijon', 'Rice, Basmati', 'Mashed potatoes, prepared, with full fat milk, with butter', 'Dumplings', 'Pumpkin', 'Swiss chard', 'Red cabbage', 'Spinach, raw', 'Naan (indien bread)', 'Chicken curry (cream/coconut milk. curry spices/paste))', 'Crunch Müesli', 'Biscuits', 'Bread, French (white flour)', 'Meatloaf', 'Fresh cheese', 'Honey', 'Vegetable mix, peas and carrots', 'Parsley', 'Brownie', 'Dairy ice cream', 'Tea, black', 'Carrot cake', 'Fish fingers (breaded)', 'Salad dressing', 'Dried meat', 'Chicken, breast', 'Mixed salad (chopped without sauce)', 'Feta', 'Praline', 'Tea, peppermint', 'Walnut', 'Potato salad, with mayonnaise yogurt dressing', 'Kebab in pita bread', 'Kolhrabi', 'Alfa sprouts', 'Brussel sprouts', 'Bacon, cooking', 'Gruyère', 'Bulgur', 'Grapes', 'Pork, escalope', 'Chocolate egg, small', 'Cappuccino', 'Zucchini, stewed, without addition of fat, without addition of salt', 'Crisp bread, Wasa', 'Bread, black', 'Perch fillets (lake)', 'Rosti', 'Mango', 'Sandwich (ham, cheese and butter)', 'Müesli', 'Spinach, steamed, without addition of salt', 'Fish', 'Risotto, without cheese, cooked', 'Milk Chocolate with hazelnuts', 'Cake (oblong)', 'Crisps', 'Pork', 'Pomegranate', 'Sweet corn, canned', 'Flakes, oat', 'Greek salad', 'Cantonese fried rice', 'Sesame seeds', 'Bouillon', 'Baked potato', 'Fennel', 'Meat', 'Bread, olive', 'Croutons', 'Philadelphia', 'Mushroom, (average), stewed, without addition of fat, without addition of salt', 'Bell pepper, red, stewed, without addition of fat, without addition of salt', 'White chocolate', 'Mixed nuts', 'Breadcrumbs (unspiced)', 'Fondue', 'Sauce, mushroom', 'Tea, spice', 'Strawberries', 'Tea, rooibos', 'Pie, plum, baked, with cake dough', 'Potatoes au gratin, dauphinois, prepared', 'Capers', 'Vegetables', 'Bread, wholemeal toast', 'Red radish', 'Fruit tart', 'Beans, kidney', 'Sauerkraut', 'Mustard', 'Country fries', 'Ketchup', 'Pasta, linguini, parpadelle, Tagliatelle', 'Chicken, cut into stripes (only meat)', 'Cookies', 'Sun-dried tomatoe', 'Bread, Ticino', 'Semi-hard cheese', 'Margarine', 'Porridge, prepared, with partially skimmed milk', 'Soya drink (soy milk)', 'Juice, multifruit', 'Popcorn salted', 'Chocolate, filled', 'Milk chocolate', 'Bread, fruit', 'Mix of dried fruits and nuts', 'Corn', 'Tête de Moine', 'Dates', 'Pistachio', 'Celery', 'White radish', 'Oat milk', 'Cream cheese', 'Bread, rye', 'Witloof chicory', 'Apple crumble', 'Goat cheese (soft)', 'Grapefruit, pomelo', 'Risotto, with mushrooms, cooked', 'Blue mould cheese', 'Biscuit with Butter', 'Guacamole', 'Pecan nut', 'Tofu', 'Cordon bleu, from pork schnitzel, fried', 'Paprika chips', 'Quinoa', 'Kefir drink', "M&M's", 'Salad, rocket', 'Bread, spelt', 'Pizza, with ham, with mushrooms, baked', 'Fruit coulis', 'Plums', 'Beef, minced (only meat)', 'Pizza, with ham, baked', 'Pineapple', 'Soup, tomato', 'Cheddar', 'Tea, fruit', 'Rice, Jasmin', 'Seeds', 'Focaccia', 'Milk', 'Coleslaw (chopped without sauce)', 'Pastry, flaky', 'Curd', 'Savoury puff pastry stick', 'Sweet potato', 'Chicken, leg', 'Croissant', 'Sour cream', 'Ham, turkey', 'Processed cheese', 'Fruit compotes', 'Cheesecake', 'Pasta, tortelloni, stuffing', 'Sauce, cocktail', 'Croissant with chocolate filling', 'Pumpkin seeds', 'Artichoke', 'Champagne', 'Grissini', 'Sweets / candies', 'Brie', 'Wienerli (Swiss sausage)', 'Syrup (diluted, ready to drink)', 'Apple pie', 'White bread with butter, eggs and milk', 'Savoury puff pastry', 'Anchovies', 'Tuna, in oil, drained', 'Lemon pie', 'Meat terrine, paté', 'Coriander', 'Falafel (balls)', 'Berries', 'Latte macchiato, with caffeine', 'Faux-mage Cashew, vegan chers', 'Beans, white', 'Sugar Melon', 'Mixed seeds', 'Hamburger', 'Hamburger bun', 'Oil & vinegar salad dressing', 'Soya Yaourt, yahourt, yogourt ou yoghourt', 'Chocolate milk, chocolate drink', 'Celeriac', 'Chocolate mousse', 'Cenovis, yeast spread', 'Thickened cream (> 35%)', 'Meringue', 'Lamb, chop', 'Shrimp / prawn (large)', 'Beef', 'Lemon', 'Croque monsieur', 'Chives', 'Chocolate cookies', 'Birchermüesli, prepared, no sugar added', 'Fish crunchies (battered)', 'Muffin', 'Savoy cabbage, steamed, without addition of salt', 'Pine nuts', 'Chorizo', 'Chia grains', 'Frying sausage', 'French pizza from Alsace, baked', 'Chocolate', 'Cooked sausage', 'Grits, polenta, maize flour', 'Gummi bears, fruit jellies, Jelly babies with fruit essence', 'Wine, rosé', 'Coca Cola', 'Raspberries', 'Roll with pieces of chocolate', 'Goat, (average), raw', 'Lemon Cake', 'Coconut milk', 'Rice, wild', 'Gluten-free bread', 'Pearl onions', 'Buckwheat pancake', 'Bread, 5-grain', 'Light beer', 'Sugar, glazing', 'Tzatziki', 'Butter, herb', 'Ham croissant', 'Corn crisps', 'Lentils green (du Puy, du Berry)', 'Cocktail', 'Rice, whole-grain', 'Veal sausage', 'Cervelat', 'Sorbet', 'Aperitif, with alcohol, apérol, Spritz', 'Dips', 'Corn Flakes', 'Peas', 'Tiramisu', 'Apricots', 'Cake, marble', 'Lamb', 'Lasagne, meat, prepared', 'Coca Cola Zero', 'Cake, salted', 'Dough (puff pastry, shortcrust, bread, pizza dough)', 'Rice waffels', 'Sekt', 'Brioche', 'Vegetable au gratin, baked', 'Mango dried', 'Processed meat, Charcuterie', 'Mousse', 'Sauce, sweet & sour', 'Basil', 'Butter, spread, puree almond', 'Pie, apricot, baked, with cake dough', 'Rusk, wholemeal', 'Beef, roast', 'Vanille cream, cooked, Custard, Crème dessert', 'Pasta in conch form', 'Nuts', 'Sauce, carbonara', 'Fig, dried', 'Pasta in butterfly form, farfalle', 'Minced meat', 'Carrot, steamed, without addition of salt', 'Ebly', 'Damson plum', 'Shoots', 'Bouquet garni', 'Coconut', 'Banana cake', 'Waffle', 'Apricot, dried', 'Sauce, curry', 'Watermelon, fresh', 'Sauce, sweet-salted (asian)', 'Pork, roast', 'Blackberry', 'Smoked cooked sausage of pork and beef meat sausag', 'bean seeds', 'Italian salad dressing', 'White asparagus', 'Pie, rhubarb, baked, with cake dough', 'Tomato, stewed, without addition of fat, without addition of salt', 'Cherries', 'Nectarine')
#('shsy5y', 'a172', 'bt474', 'bv2', 'huh7', 'mcf7', 'skov3', 'skbr3')
data_root = 'dataset/'
train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)
    ),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.5, 1.5),
        pad_val=114.0
    ),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18
    ),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad', pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))
    ),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val/annotations_val.json',
        img_prefix=data_root +
        'val/images',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))
            ),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ]
    )
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    # persistent_workers=True,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val/annotations_val.json',
        img_prefix=data_root +
        'val/images',
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val/annotations_val.json',
        img_prefix=data_root +
        'val/images',
        pipeline=test_pipeline,
    ),
)
optimizer = dict(
    type='SGD',
    lr=0.01 / 64,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0)
)
optimizer_config = dict(grad_clip=None)

evaluation = dict(
    interval=1, metric='bbox', classwise=True, proposal_nums=(100, 300, 2000)
)
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=1,
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05
)

runner = dict(type='EpochBasedRunner', max_epochs=15)
checkpoint_config = dict(interval=1)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48
    ),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=1,
        priority=48
    ),
    dict(type='ExpMomentumEMAHook', resume_from=resume_from, priority=49)
]