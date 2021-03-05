import json
import csv

from pycocotools.coco import COCO
# from mrcnn.cocoeval import COCOeval

ground_truth_path = r'/media/HDD_4TB_1/Datasets/AICrowd_newval/val/val_annotations_fixed.json'
predictions_file_path = r'/home/marcosmejia/UB/Github/mmdetection/TROUT_Ckp19_Val95_5_Balanced_1e-2/best_map/best_map_predictions.segm.json'

iouType = 'segm'  # set iouType to 'segm', 'bbox' or 'keypoints'

VALID_CATEGORIES = [2578, 1157, 2022, 1198, 2053, 1566, 2099, 1554, 1151, 2530, 2521, 2534, 1026, 1311, 2738, 1505,
                    1078, 1116, 1731, 1453, 1040, 1538, 2504, 1154, 1022, 1565, 2895, 2620, 1853, 1300, 1074, 1310,
                    1893, 1533, 2747, 1069, 1084, 1009, 2618, 2730, 1490, 1058, 1210, 2131, 1180, 1308, 1588, 2944,
                    1156, 2388, 1384, 1108, 1082, 1126, 1143, 1032, 1468, 2413, 1150, 2350, 2512, 1587, 633, 2836,
                    1560, 1010, 1482, 1831, 2934, 3080, 1670, 1203, 1013, 1098, 1024, 1942, 1007, 5641, 2952, 1956,
                    2634, 1214, 2939, 1060, 1889, 2303, 1520, 2729, 1469, 3358, 1065, 1033, 3532, 2742, 1496, 2562,
                    2841, 732, 1181, 387, 1061, 1536, 1483, 1728, 1506, 1513, 1212, 1352, 2973, 1068, 2807, 1789,
                    1478, 1545, 1228, 1569, 1323, 2501, 1422, 2750, 1788, 1327, 1187, 6404, 1980, 1348, 2237, 1014,
                    1085, 1138, 1094, 2734, 2923, 1221, 1294, 1967, 1056, 2728, 1038, 630, 1307, 2524, 2970, 1627,
                    1607, 1107, 3100, 2736, 1915, 1879, 1144, 1102, 2935, 1119, 2961, 2898, 1004, 3220, 50, 2711,
                    1113, 1237, 2498, 1166, 1551, 1050, 3115, 1092, 2073, 1557, 1229, 1321, 1916, 1152, 2930, 3630,
                    2103, 2454, 2376, 1220, 1614, 1794, 1170, 1727, 2741, 1986, 1383, 2954, 2714, 1070, 3332, 1213,
                    1919, 1793, 1455, 1561, 2269, 1523, 1948, 2580, 2920, 2446, 2873, 3042, 1568, 1547, 1487, 1280,
                    1019, 1467, 2905, 1724, 3249, 1730, 2172, 2134, 1186, 2470, 727, 2495, 1184, 1556, 1620, 3306,
                    1985, 2743, 2949, 2132, 1748, 1402, 2749, 1924, 2555, 3308, 3262, 2254, 1200, 1856, 1162, 1580,
                    2967, 2362, 1055, 1223, 1264, 2278, 1328, 2543, 1371, 1463, 1494, 1054, 1169, 1209, 2964, 2320,
                    1215, 1176, 1199, 1089, 1191, 1075, 1376, 3221, 1153, 1249, 1522, 1163, 2616, 578, 1123, 1124,
                    1190]


coco_pred = COCO(ground_truth_path)
img_ids = coco_pred.getImgIds()
print(len(img_ids))

submission_file = json.loads(open(predictions_file_path).read())

with open('gt_categories.csv', 'w') as f_obj:
    csv_writer = csv.writer(f_obj)
    for img_id in img_ids:
        annotationIds = coco_pred.getAnnIds(img_id)
        annotations = coco_pred.loadAnns(annotationIds)

        catIds = []
        for i in range(len(annotations)):
            entityId = annotations[i]['category_id']
            entity = coco_pred.loadCats([entityId])[0]['name']
            catIds.extend(coco_pred.getCatIds([entity]))

        predIds = []
        for entry in submission_file:
            if img_id == entry['image_id']:
                predIds.extend([entry['category_id']])  
        
        csv_writer.writerow([img_id, catIds, predIds])
        print(img_id)

"""
results = coco_pred.loadRes(submission_file)
cocoEval = COCOeval(coco_pred, results, iouType)
cocoEval.evaluate()
cocoEval.accumulate()
average_precision = cocoEval._summarize(ap=1, iouThr=0.5, areaRng="all", maxDets=100)
average_recall = cocoEval._summarize(ap=0, iouThr=0.5, areaRng="all", maxDets=100)
"""
