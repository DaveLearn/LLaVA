#%%
import json

data_path='./playground/data/llava_v1_5_mix665k.json'
list_data_dict = json.load(open(data_path, "r"))

#%%
from pathlib import Path
image_path = Path(data_path).parent
# %%
(image_path / list_data_dict[0]['image']).exists()
# %%
from tqdm import tqdm
#%%
images_changed = 0
for item in tqdm(list_data_dict):
    if not 'image' in item or not item['image'].startswith('ocr_vqa'):
        continue
    full_path = image_path / item['image']
    if not full_path.exists():
        #print(f'image missing: {item["image"]}')
        # find all files in image_path with the same name but different extension
        # and print them
        for f in full_path.parent.glob(full_path.stem + '.*'):
            alt = Path(item['image']).parent / f.name
            item['image'] = str(alt)
            images_changed += 1
            #print(f'alternative: {alt}')
print(f'images changed: {images_changed}')
# %%
fixed_data_path='./playground/data/llava_v1_5_mix665k_fixed.json'
json.dump(list_data_dict, open(fixed_data_path, "w"))
# %%
