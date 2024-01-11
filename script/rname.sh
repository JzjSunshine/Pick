folder_path="/data/zjjing/DocumentAI/Pick/data/sroie/test_data/boxes_and_transcripts"  # 替换为你的文件夹路径

for file in $folder_path/*.txt; do
    [ -f "$file" ] || continue  # 确保是文件
    new_file="${file%.txt}.tsv"
    mv "$file" "$new_file"
    echo "已将文件 $(basename "$file") 的后缀改为 .tsv"
done