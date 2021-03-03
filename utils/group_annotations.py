import os
import json
from utils.groupings.bread import bread_groups

data_root_path = r'/media/HDD_3TB_2/bhalaji/challenges/aicrowd/food_recognition_challenge/data'


def create_data_subset(data_dir, cat_groups):
    with open(data_dir + "/" + data_dir.split('/')[-1] + "_annotations_fixed.json") as f:
        data = json.load(f)

    original_cats = list(cat_groups.keys())

    data_subset_cats = []
    for x in data["categories"]:
        if x["id"] in original_cats:
            mod_cat = next((item for item in data["categories"] if item["id"] == cat_groups[x['id']]), None)
            if mod_cat not in data_subset_cats:
                data_subset_cats.append(mod_cat)
        else:
            data_subset_cats.append(x)

    data_subset_anns = []
    for x in data["annotations"]:
        if x["category_id"] in original_cats:
            x['category_id'] = cat_groups[x['category_id']]
            data_subset_anns.append(x)
        else:
            data_subset_anns.append(x)

    reqd_dict = {}
    reqd_dict["info"] = data["info"]
    reqd_dict["images"] = data["images"]
    reqd_dict["categories"] = data_subset_cats
    reqd_dict["annotations"] = data_subset_anns

    with open(data_dir + "/annotations-bread-grouped.json", "w") as f:
        json.dump(reqd_dict, f)


create_data_subset(os.path.join(data_root_path, 'train'), bread_groups)
create_data_subset(os.path.join(data_root_path, 'val'), bread_groups)
