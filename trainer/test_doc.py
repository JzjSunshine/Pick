## Script for preprocessing dataset
import os
import pandas
import json
import csv
import shutil

root_dir = "/home/hcxw/newDisk/jzj/Pick.Pytorch.Sroie/data_load/"
## Input dataset
data_path = root_dir + "/ICDAR-2019-SROIE/data/"
box_path = data_path + "box/"
img_path = data_path + "img/"
key_path = data_path + "key/"

## Output dataset
out_boxes_and_transcripts = root_dir + "content/boxes_and_transcripts/"
out_images = root_dir + "content/images/"
out_entities = root_dir + "content/entities/"

train_samples_list = []
for file in os.listdir(data_path + "box/"):

    ## Reading csv
    with open(box_path + file, "r") as fp:
        reader = csv.reader(fp, delimiter=",")
        ## arranging dataframe index ,coordinates x1_1,y1_1,x2_1,y2_1,x3_1,y3_1,x4_1,y4_1, transcript
        rows = [[1] + x[:8] + [','.join(x[8:]).strip(',')] for x in reader]
        df = pandas.DataFrame(rows)

    ## including ner label dataframe index ,coordinates x1_1,y1_1,x2_1,y2_1,x3_1,y3_1,x4_1,y4_1, transcript , ner tag
    df[10] = 'other'

    ##saving file into new dataset folder
    jpg = file.replace(".csv", ".jpg")
    entities = json.load(open(key_path + file.replace(".csv", ".json")))
    for key, value in sorted(entities.items()):
        idx = df[df[9].str.contains('|'.join(map(str.strip, value.split(','))))].index
        df.loc[idx, 10] = key

    shutil.copy(img_path + jpg, out_images)
    with open(out_entities + file.replace(".csv", ".txt"), "w") as j:
        print(json.dumps(entities), file=j)

    df.to_csv(out_boxes_and_transcripts + file.replace(".csv", ".tsv"), index=False, header=False, quotechar='',
              escapechar='\\', quoting=csv.QUOTE_NONE, )
    train_samples_list.append(['receipt', file.replace('.csv', '')])
train_samples_list = pandas.DataFrame(train_samples_list)
train_samples_list.to_csv("train_samples_list.csv")
