#!/bin/bash

# 指定数据文件夹路径
data_folder="/data/zjjing/DocumentAI/Pick/data/sroie/"

# 遍历子文件夹
for subfolder in "$data_folder"/test_data "$data_folder"/train_data; do
    echo $subfolder
    # 检查是否是目录
    if [ -d "$subfolder" ]; then
        # 获取文件数量
        for subsubfolder in "$subfolder"/boxes_and_transcripts "$subfolder"/images "$subfolder"/entities; do
            # 检查是否是目录
            if [ -d "$subsubfolder" ]; then
                # 获取文件数量
                file_count=$(find "$subsubfolder" -type f | wc -l)
                
                # 输出结果
                echo "$subsubfolder: $file_count"
            fi
        done
    fi
done