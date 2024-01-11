# coding=utf-8
import json
import os
import csv
import re
import shutil
from pathlib import Path
import datasets

from PIL import Image
import numpy as np

logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{yu2021pick,
               title={PICK: Processing key information extraction from documents using improved graph learning-convolutional networks},
               author={Yu, Wenwen and Lu, Ning and Qi, Xianbiao and Gong, Ping and Xiao, Rong},
               booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
               pages={4363--4370},
               year={2021},
               organization={IEEE}
}
"""
_DESCRIPTION = """\
The train ticket is fixed layout dataset, however, it contains background noise and imaging distortions.
It contains 1,530 synthetic images and 320 real images for training, and 80 real images for testing.
Every train ticket has eight key text fields including ticket number, starting station, train number, destination station, date, ticket rates, seat category, and name.
This dataset mainly consists of digits, English characters, and Chinese characters.
"""

_URL = """\
https://drive.google.com/file/d/1o8JktPD7bS74tfjz-8dVcZq_uFS6YEGh/view?usp=sharing
"""


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


class TrainTicketsConfig(datasets.BuilderConfig):
    """BuilderConfig for train_tickets"""

    def __init__(self, **kwargs):
        """BuilderConfig for train_tickets.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(TrainTicketsConfig, self).__init__(**kwargs)


class TrainTickets(datasets.GeneratorBasedBuilder):
    """train tickets"""

    BUILDER_CONFIGS = [
        TrainTicketsConfig(
            name="train_tickets-yu2020pick",
            version=datasets.Version("1.0.0"),
            description="Chinese train tickets",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(
                        datasets.Sequence(datasets.Value("int64"))
                    ),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "S-DATE",
                                "S-DESTINATION_STATION",
                                "S-NAME",
                                "S-SEAT_CATEGORY",
                                "S-STARTING_STATION",
                                "S-TICKET_NUM",
                                "S-TICKET_RATES",
                                "S-TRAIN_NUM",
                            ]
                        )
                    ),
                    "image_path": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/wenwenyu/PICK-pytorch",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_file = dl_manager.download_and_extract(
            "https://drive.google.com/uc?export=download&id=1o8JktPD7bS74tfjz-8dVcZq_uFS6YEGh"
        )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filelist": f"{downloaded_file}/train_tickets/synth1530_real320_baseline_trainset.csv"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filelist": f"{downloaded_file}/train_tickets/real80_baseline_testset.csv"
                },
            ),
        ]

    # based on https://github.com/wenwenyu/PICK-pytorch/blob/master/data_utils/documents.py#L229
    def _read_gt_file_with_box_entity_type(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            document_text = f.read()

        # match pattern in document: index,x1,y1,x2,y2,x3,y3,x4,y4,transcript,box_entity_type
        regex = (
            r"^\s*(-?\d+)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,"
            r"\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,(.*),(.*)\n?$"
        )

        matches = re.finditer(regex, document_text, re.MULTILINE)

        res = []
        for _, match in enumerate(matches, start=1):
            points = [int(match.group(i)) for i in range(2, 10)]
            x = points[0:8:2]
            y = points[1:8:2]
            x1 = min(x)
            y1 = min(y)
            x2 = max(x)
            y2 = max(y)
            transcription = str(match.group(10))
            entity_type = str(match.group(11))
            res.append((x1, y1, x2, y2, transcription, entity_type))
        return res

    def _generate_examples(self, filelist):
        logger.info("⏳ Generating examples from = %s", filelist)

        ann_dir = os.path.join(os.path.dirname(filelist), "boxes_trans")
        img_dir = os.path.join(os.path.dirname(filelist), "images1930")
        print(ann_dir)

        with open(filelist) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                guid = row[0]
                # document_type = row[1]
                filename = row[2]

                words = []
                bboxes = []
                ner_tags = []
                file_path = os.path.join(ann_dir, f"{filename}.tsv")
                data = self._read_gt_file_with_box_entity_type(file_path)
                image_path = os.path.join(img_dir, f"{filename}.jpg")
                print(image_path)
                print(size)
                _, size = load_image(image_path)
                for item in data:
                    box = item[0:4]
                    transcription, label = item[4:6]
                    words.append(transcription)
                    bboxes.append(normalize_bbox(box, size))
                    if label == "other":
                        ner_tags.append("O")
                    else:
                        ner_tags.append("S-" + label.upper())
                # print(
                #         guid, {
                #         "id": str(guid),
                #         "words": words,
                #         "bboxes": bboxes,
                #         "ner_tags": ner_tags,
                #         "image_path": image_path,
                #     }
                # )
                yield guid, {
                    "id": str(guid),
                    "words": words,
                    "bboxes": bboxes,
                    "ner_tags": ner_tags,
                    "image_path": image_path,
                }
    
    
    def generate_entities(self, filelist):
        print("⏳ Generating entities from = ", filelist)

        ann_dir = filelist + "boxes_trans" # os.path.join(os.path.dirname(filelist), "boxes_trans")
        img_dir = filelist + "images1930" # os.path.join(os.path.dirname(filelist), "images1930")
        print(ann_dir)

        csv_dir = filelist + "/synth1530_real320_baseline_trainset.csv" # os.path.join(os.path.dirname(filelist),"synth1530_real320_baseline_trainset.csv")
        # csv_dir = filelist + "/real80_baseline_testset.csv"
        
        with open(csv_dir) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            size_list = []
            for row in csv_reader:
                guid = row[0]
                # document_type = row[1]
                filename = row[2]

                words = []
                bboxes = []
                ner_tags = []
                file_path = os.path.join(ann_dir, f"{filename}.tsv")
                data = self._read_gt_file_with_box_entity_type(file_path)
                image_path = os.path.join(img_dir, f"{filename}.jpg")
                _, size = load_image(image_path)
                # print(size)
                print(image_path)
                print(size)
                # size_list.append(size)
                # return
                # print(data)
                # return
                entities_dict = {
                                "ticket_num":"",
                                "starting_station":"",
                                "train_num":"",
                                "destination_station":"",
                                "date":"",
                                "ticket_rates":"",
                                "seat_category":"",
                                "name":""
                }
                for item in data:
                    box = item[0:4]
                    transcription, label = item[4:6]
                    # words.append(transcription)
                    # bboxes.append(normalize_bbox(box, size))
                    # if label == "other":
                    #     ner_tags.append("O")
                    # else:
                    #     ner_tags.append("S-" + label.upper())
                    if label in entities_dict.keys():
                        entities_dict[label] = transcription
                # print(entities_dict)
                # print(filename)
                result_file =  filelist + "/entities/" + filename + ".txt" # output_path.joinpath(Path(test_dataset.files_list[image_index]).stem + '.txt')
                with open(result_file, "w") as f:
                    f.write(json.dumps(entities_dict, ensure_ascii=False))
                        
                # return
                # print(
                #         guid, {
                #         "id": str(guid),
                #         "words": words,
                #         "bboxes": bboxes,
                #         "ner_tags": ner_tags,
                #         "image_path": image_path,
                #     }
                # )
                # yield guid, {
                #     "id": str(guid),
                #     "words": words,
                #     "bboxes": bboxes,
                #     "ner_tags": ner_tags,
                #     "image_path": image_path,
                # }

            # print(size_list)

    def move_dataset(self, csv_file_path, src_folder, dst_folder):
        """_summary_

        Args:
            csv_file_path (_type_): _description_
            src_folder (_type_): _description_
            dst_folder (_type_): _description_
        """
        # 创建目标文件夹（如果不存在）
        src_images_folder = src_folder + '/images1930'
        src_entities_folder = src_folder + '/entities'
        src_boxes_and_transcripts_folder = src_folder + '/boxes_trans'
        
        dst_images_folder = dst_folder + '/images'
        os.makedirs(dst_images_folder, exist_ok=True)
        # shutil.rmtree(dst_images_folder, ignore_errors=True) if Path(dst_images_folder).exists() else Path(dst_images_folder).mkdir(parents=True, exist_ok=True)
        dst_entities_folder = dst_folder + '/entities'
        os.makedirs(dst_entities_folder, exist_ok=True)
        dst_boxes_and_transcripts_folder = dst_folder + '/boxes_and_transcripts'
        os.makedirs(dst_boxes_and_transcripts_folder, exist_ok=True)
        
        # 打开CSV文件并读取索引
        with open(csv_file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # 跳过标题行
            for row in csv_reader:
                index = row[2]  # 假设索引在CSV文件的第一列
                filename_img = f"{index}.jpg"  # 假设文件名为 index.jpg
                # 构建源文件和目标文件的路径
                image_source = os.path.join(src_images_folder, f"{index}.jpg")
                entities_source = os.path.join(src_entities_folder, f"{index}.txt")
                boxes_and_transcripts_source = os.path.join(src_boxes_and_transcripts_folder, f"{index}.tsv")

                # print(image_source)
                # print(entities_source)
                # print(boxes_and_transcripts_source)
                # return
               
                # 移动文件
                try:
                    shutil.move(image_source, dst_images_folder)
                    # if os.path.exists(entities_source):
                    # shutil.move(entities_source, dst_entities_folder)
                    # shutil.move(boxes_and_transcripts_source, dst_boxes_and_transcripts_folder)
                    print(f"Moved files for {index} to {image_source}")
                except FileNotFoundError:
                    print(f"File {index} not found in {image_source}")
                except Exception as e:
                    print(f"Error moving {index}: {e}")
                
                try:
                    # shutil.move(image_source, dst_images_folder)
                    shutil.move(entities_source, dst_entities_folder)
                    # shutil.move(boxes_and_transcripts_source, dst_boxes_and_transcripts_folder)
                    print(f"Moved files for {index} to {entities_source}")
                except FileNotFoundError:
                    print(f"File {index} not found in {entities_source}")
                except Exception as e:
                    print(f"Error moving {entities_source}: {e}")
                
                try:
                    # shutil.move(image_source, dst_images_folder)
                    # # if os.path.exists(entities_source):
                    # shutil.move(entities_source, dst_entities_folder)
                    shutil.move(boxes_and_transcripts_source, dst_boxes_and_transcripts_folder)
                    print(f"Moved files for {index} to {boxes_and_transcripts_source}")
                except FileNotFoundError:
                    print(f"File {index} not found in {boxes_and_transcripts_source}")
                except Exception as e:
                    print(f"Error moving {boxes_and_transcripts_source}: {e}")
        
if __name__ == '__main__':
    train_utl = TrainTickets()
    # 生成 entities
    # train_utl.generate_entities("/data/zjjing/DocumentAI/Pick/train_tickets_copy/")
    
    # 移动测试数据
    # train_utl.move_dataset("/data/zjjing/DocumentAI/Pick/train_tickets/real80_baseline_testset.csv",
    #                         "/data/zjjing/DocumentAI/Pick/train_tickets/",
    #                         "/data/zjjing/DocumentAI/Pick/train_tickets/test_data/")
    # 移动训练数据
    train_utl.move_dataset("/data/zjjing/DocumentAI/Pick/train_tickets/synth1530_real320_baseline_trainset.csv",
                            "/data/zjjing/DocumentAI/Pick/train_tickets/",
                            "/data/zjjing/DocumentAI/Pick/train_tickets/train_data/")
    # 移动实体
    
    