import json
import csv
import shutil
import os
from PIL import Image

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)

class JsonParser:
    def __init__(self, ann_file_dir, input_img_dir,output_csv_path, output_tsv_folder,output_img_folder,output_entity_folder,mode="train"):
        self.ann_file_dir = ann_file_dir
        self.input_img_dir = input_img_dir
        self.output_csv_path = output_csv_path
        self.output_tsv_folder = output_tsv_folder
        self.output_img_folder = output_img_folder
        self.output_entity_folder = output_entity_folder
        
        self.mode = mode
    def parse_and_save(self):
        
        if self.mode == "train":
            self.parse_and_save_train()
        else:
            self.parse_and_save_test()
    
    def parse_and_save_train(self):
        # 确保输出文件夹存在
        os.makedirs(self.output_tsv_folder, exist_ok=True)
        png_name_list = []
        # 遍历 JSON 文件
        for filename in os.listdir(self.ann_file_dir):
            if filename.endswith('.json'):
                json_file_path = os.path.join(self.ann_file_dir, filename)
                self.process_json_file(json_file_path, output_tsv_folder)

                img_file_path = os.path.join(self.input_img_dir,filename.split('.')[0]+".png")
                png_name_list.append(filename.split('.')[0]+".png")
                shutil.copy2(img_file_path, self.output_img_folder)
        # 写入 csv 文件
        # print(png_name_list)
        self.write_to_csv(png_name_list,self.output_csv_path)
        
        
    def process_json_file(self,json_file_path, output_tsv_folder):
        """写一个 json 文件

        Args:
            json_file_path (_type_): _description_
            output_tsv_folder (_type_): _description_
        """
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

            # 获取文件名（不包含扩展名）
            filename = os.path.splitext(os.path.basename(json_file_path))[0]

            # 生成输出 TSV 文件路径
            tsv_file_path = os.path.join(output_tsv_folder, f'{filename}.tsv')

            # 写入 TSV 文件
            with open(tsv_file_path, 'w', newline='', encoding='utf-8') as tsv_file:
                # tsv_writer = csv.writer(tsv_file, delimiter='\t')
                # tsv_writer.writerow(['index', 'box', 'text', 'label'])

                for form_entry in data['form']:
                    index = form_entry['id']
                    x1, y1, x2, y2 = form_entry['box']
                    text = form_entry['text']
                    label = form_entry['label']

                    # print(f"{index},{x1},{y1},{x2},{y1},{x2},{y2},{x1},{y2},{text},{label}\n")
                    # tsv_writer.writerow(f"{index},{x1},{y1},{x2},{y1},{x2},{y2},{x1},{y2},{text},{label}\n")
                    tsv_file.write(f"{index},{x1},{y1},{x2},{y1},{x2},{y2},{x1},{y2},{text},{label}\n")
    
    def parse_and_save_test(self):
        
        with open(self.input_file_path, "r", encoding="utf-8") as file:
            data = file.readlines()

            fileNameList = []
            # 处理每一行 JSON 数据
            for line in data:
                # 将每行 JSON 数据解析为字典
                json_data = json.loads(line)

                # 提取所需信息
                file_name = json_data["file_name"].split('/')[-1]
                fileNameList.append(file_name)
                receipt = "receipt"  # receipt 固定为 "receipt"
                polygons = json_data["annotations"]

                entity_dict = json_data["entity_dict"]
                
                print(entity_dict)
                entity_dict_keys = entity_dict.keys()
                # 处理每个 polygon
                for polygon_data in polygons:
                    polygon = polygon_data["polygon"]
                    text = polygon_data["text"]

                    # 提取有效的实体类型
                    entity_str = "other"
                    for key,value in entity_dict.items():
                        if text in value:
                            entity_str = key
                            break

                    # 将 polygon 转换为字符串格式
                    polygon_str = ",".join(map(str, polygon))


                    # 写入 TXT 文件
                    txt_file_path = f"{self.output_txt_folder}/{file_name.split('.')[0]}.tsv"
                    with open(txt_file_path, mode="a+", encoding="utf-8") as txt_file:
                        txt_file.write(f"{'1'},{polygon_str},{text},{entity_str}\n")

            self.write_to_csv(fileNameList,self.output_csv_path)
            
        print(f"Data has been written to {self.output_csv_path} and TXT files in {self.output_txt_folder}")

    def copy_img(self,src_folder,dst_folder):
        
        with open(self.input_file_path, "r", encoding="utf-8") as file:
            data = file.readlines()

            fileNameList = []
            # 处理每一行 JSON 数据
            for line in data:
                # 将每行 JSON 数据解析为字典
                json_data = json.loads(line)
                # 提取所需信息
                file_name = json_data["file_name"].split("/")[-1]
                fileNameList.append(file_name)
       
        # print(fileNameList)
        # 遍历文件名列表，拷贝文件
        for filename in fileNameList:
            source_path = os.path.join(src_folder, filename)
            destination_path = os.path.join(dst_folder, filename)

            # print(source_path)
            # print(dst_folder)
            # print()
            # 使用 shutil 拷贝文件
            shutil.copy2(source_path, destination_path)
                
        print(f"Data has been copy from {src_folder}  to {dst_folder}")
        
    def generate_entities(self,dst_folder):
        with open(self.input_file_path, "r", encoding="utf-8") as file:
            data = file.readlines()

            fileNameList = []
            # 处理每一行 JSON 数据
            for line in data:
                # 将每行 JSON 数据解析为字典
                json_data = json.loads(line)

                # 提取所需信息
                file_name = json_data["file_name"].split('/')[-1]
                fileNameList.append(file_name)
                entity_dict = json_data["entity_dict"]
                
                txt_file_path = f"{dst_folder}/{file_name.split('.')[0]}.txt"
                with open(txt_file_path, mode="w", encoding="utf-8") as txt_file:
                        txt_file.write(json.dumps(entity_dict, ensure_ascii=False))
            
        print(f"Data has been written to {self.output_csv_path} and TXT files in {self.output_txt_folder}")
        
    
    def write_to_csv(self,file_name_list, output_csv_path):
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            # fieldnames = ["id", "receipt", "filename"]
            csv_writer = csv.writer(csv_file)

            # 写入数据
            for idx, filename in enumerate(file_name_list):
                # print({
                #     "id": idx,
                #     "receipt": "form",
                #     "filename": filename
                # })
                csv_writer.writerow([idx,"form",filename])

def remove_cvs(file_path,dst_file):

    # 输入文件和输出文件路径
    input_csv_path = file_path
    output_csv_path = dst_file

    # 读取 CSV 文件
    with open(input_csv_path, "r", newline="", encoding="utf-8") as input_file:
        reader = csv.reader(input_file)
        data = list(reader)

    # 删除第一行
    data = data[1:]

    # 保存到新的 CSV 文件
    with open(output_csv_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerows(data) 
           
if __name__ == '__main__':
    
    # 读取训练数据 ann_file_dir, img_file_dir,output_csv_path, output_tsv_folder,output_img_folder,output_entity_folder
    # test
    # ann_file_dir = "/data/zjjing/DocumentAI/DataSet/funsd/testing_data/annotations/"
    # img_file_dir = "/data/zjjing/DocumentAI/DataSet/funsd/testing_data/images/"
    # output_csv_path = "/data/zjjing/DocumentAI/Pick/data/funsd/test_data/test_samples_list.csv" 
    # output_tsv_folder = "/data/zjjing/DocumentAI/Pick/data/funsd/test_data/boxes_and_transcripts/"
    # output_img_folder = "/data/zjjing/DocumentAI/Pick/data/funsd/test_data/images/"
    # output_entity_folder = "/data/zjjing/DocumentAI/Pick/data/funsd/test_data/entities"

    
    # train
    ann_file_dir = "/data/zjjing/DocumentAI/DataSet/funsd/training_data/annotations/"
    img_file_dir = "/data/zjjing/DocumentAI/DataSet/funsd/training_data/images/"
    output_csv_path = "/data/zjjing/DocumentAI/Pick/data/funsd/train_data/train_samples_list.csv" 
    output_tsv_folder = "/data/zjjing/DocumentAI/Pick/data/funsd/train_data/boxes_and_transcripts/"
    output_img_folder = "/data/zjjing/DocumentAI/Pick/data/funsd/train_data/images/"
    output_entity_folder = "/data/zjjing/DocumentAI/Pick/data/funsd/train_data/entities"

    parser = JsonParser(ann_file_dir, img_file_dir,output_csv_path, output_tsv_folder,output_img_folder,output_entity_folder)
    parser.parse_and_save()
    
    pass
    
    
    