# python ./test/test.py --checkpoint ./saved/models/PICK_Default/test_0821_222232/model_best.pth --boxes_transcripts ./data/test_data_example/boxes_and_transcripts/ \
#                --images_path ./data/test_data_example/images --output_folder ./test_output \
#                --gpu 0 --batch_size 2

python ./test/test.py --checkpoint ./saved/models/PICK_Default/test_0821_222232/model_best.pth --boxes_transcripts ./data/test_data_example/boxes_and_transcripts/ \
               --images_path ./data/test_data_example/images --output_folder ./test_output \
               --gpu 0 --batch_size 2