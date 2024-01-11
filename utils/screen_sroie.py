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
    def __init__(self, input_file_path, output_csv_path, output_txt_folder,model="train"):
        self.input_file_path = input_file_path
        self.output_csv_path = output_csv_path
        self.output_txt_folder = output_txt_folder
        self.model = model 
    def parse_and_save(self):
        
        if self.model == "train":
            self.parse_and_save_train()
        else:
            self.parse_and_save_test()
    
    def parse_and_save_train(self):
        
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

                # 处理每个 polygon
                for polygon_data in polygons:
                    polygon = polygon_data["polygon"]
                    text = polygon_data["text"]
                    entity_types = polygon_data["entity"]

                    # 提取有效的实体类型
                    valid_entity_types = [entity.split("-")[1] for entity in entity_types if "-" in entity]

                    # 判断 entity 类型
                    # 判断 entity 类型
                    if "O" == entity_types[0]:
                        entity_str = "other"
                    else:
                        entity_str = entity_types[0].split('-')[-1]
                    # 提取有效的实体类型

                    # 将 polygon 转换为字符串格式
                    polygon_str = ",".join(map(str, polygon))


                    # 写入 TXT 文件
                    
                    txt_file_path = f"{self.output_txt_folder}/{file_name.split('.')[0]}.tsv"
                    with open(txt_file_path, mode="a+", encoding="utf-8") as txt_file:
                        txt_file.write(f"{'1'},{polygon_str},{text},{entity_str}\n")

            self.write_to_csv(fileNameList,self.output_csv_path)
            
        print(f"Data has been written to {self.output_csv_path} and TXT files in {self.output_txt_folder}")

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
        with open(output_csv_path, mode="w", encoding="utf-8", newline="") as csv_file:
            fieldnames = ["id", "receipt", "filename"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            # 写入数据
            for idx, filename in enumerate(file_name_list, start=1):
                writer.writerow({
                    "id": idx,
                    "receipt": "receipt",
                    "filename": filename
                })

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
    
    # 读取包含 JSON 内容的文件
    # # train 文件
    # file_path = "/data/zjjing/DocumentAI/Pick/data/sroie/e2e_format/train_update_screen.txt"  # 将文件路径替换为实际文件路径
    # output_txt_path = "/data/zjjing/DocumentAI/Pick/data/sroie/train_data/boxes_and_transcripts/"
    # output_csv_path = "/data/zjjing/DocumentAI/Pick/data/sroie/train_data/train_samples_list.csv"
    # model = "train"
    # src_folder = "/data/zjjing/DocumentAI/Pick/data/sroie/e2e_format/image_files/"
    # dst_folder = "/data/zjjing/DocumentAI/Pick/data/sroie/train_data/images"
    # dst_entity_folder = "/data/zjjing/DocumentAI/Pick/data/sroie/train_data/entities/"
    # parser = JsonParser(file_path,output_csv_path,output_txt_path,model)
    # # parser.parse_and_save() # 解析 txt 文件
    # # parser.copy_img(src_folder,dst_folder) # 拷贝文件
    # parser.generate_entities(dst_entity_folder)
    
    
    # test
    # file_path = "/data/zjjing/DocumentAI/Pick/data/sroie/e2e_format/test_screen.txt"  # 将文件路径替换为实际文件路径
    # output_txt_path = "/data/zjjing/DocumentAI/Pick/data/sroie/test_data/boxes_and_transcripts/"
    # output_csv_path = "/data/zjjing/DocumentAI/Pick/data/sroie/test_data/test_samples_list.csv"
    # model = "test"
    # parser = JsonParser(file_path,output_csv_path,output_txt_path,model)
    # # parser.parse_and_save()
    # src_folder = "/data/zjjing/DocumentAI/Pick/data/sroie/e2e_format/image_files/"
    # dst_folder = "/data/zjjing/DocumentAI/Pick/data/sroie/test_data/images"
    # dst_entity_folder = "/data/zjjing/DocumentAI/Pick/data/sroie/test_data/entities/"
    # # parser.copy_img(src_folder,dst_folder) # 拷贝文件
    # parser.generate_entities(dst_entity_folder)
    
    
    # 处理 CSV文件
    train_csv = "/data/zjjing/DocumentAI/Pick/data/sroie/test_data/test_samples_list.csv"
    train_csv_2 = "/data/zjjing/DocumentAI/Pick/data/sroie/test_data/test_samples_list_2.csv"
    remove_cvs(train_csv,train_csv_2)
    
    
    pass
    
    
    