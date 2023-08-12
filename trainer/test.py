# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/13/2020 10:26 PM

import argparse
import torch
from tqdm import tqdm
from pathlib import Path

from torch.utils.data.dataloader import DataLoader
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans

from parse_config import ConfigParser
import model.pick as pick_arch_module
from data_utils.pick_dataset import PICKDataset
from data_utils.pick_dataset import BatchCollateFn
from utils.util import iob_index_to_str, text_index_to_str,iob_tags_to_union_iob_tags
from utils.metrics import MetricTracker, SpanBasedF1MetricTracker
from utils.class_utils import iob_labels_vocab_cls

def main(args):
    device = torch.device(f'cuda:{args.gpu}' if args.gpu != -1 else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)# 加载序列化内容

    config = checkpoint['config']
    state_dict = checkpoint['state_dict']
    monitor_best = checkpoint['monitor_best']
    print('Loading checkpoint: {} \nwith saved mEF {:.4f} ...'.format(args.checkpoint, monitor_best))

    # prepare model for testing
    pick_model = config.init_obj('model_arch', pick_arch_module)
    pick_model = pick_model.to(device)
    pick_model.load_state_dict(state_dict)
    pick_model.eval()

    # setup dataset and data_loader instances
    test_dataset = PICKDataset(boxes_and_transcripts_folder=args.bt,
                               images_folder=args.impt,
                               resized_image_size=(480, 960),
                               ignore_error=False,
                               training=False)
    test_data_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                  num_workers=2, collate_fn=BatchCollateFn(training=False))

    # setup output path
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # predict and save to file
    with torch.no_grad():
        for step_idx, input_data_item in tqdm(enumerate(test_data_loader)):
            print()
            print(step_idx)
            print(input_data_item.keys())
            for key, input_value in input_data_item.items():
                if input_value is not None:
                    input_data_item[key] = input_value.to(device)

            output = pick_model(**input_data_item)
            # print(output)
            # print("="*50)
            logits = output['logits']
            new_mask = output['new_mask']
            image_indexs = input_data_item['image_indexs']  # (B,)
            text_segments = input_data_item['text_segments']  # (B, num_boxes, T)
            mask = input_data_item['mask']
            # List[(List[int], torch.Tensor)]
            best_paths = pick_model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask, logits_batch_first=True)
            predicted_tags = []
            for path, score in best_paths:
                predicted_tags.append(path)

            # convert iob index to iob string
            decoded_tags_list = iob_index_to_str(predicted_tags)
            # union text as a sequence and convert index to string
            decoded_texts_list = text_index_to_str(text_segments, mask)

            for decoded_tags, decoded_texts, image_index in zip(decoded_tags_list, decoded_texts_list, image_indexs):
                # List[ Tuple[str, Tuple[int, int]] ]
                spans = bio_tags_to_spans(decoded_tags, [])
                spans = sorted(spans, key=lambda x: x[1][0])

                entities = []  # exists one to many case
                for entity_name, range_tuple in spans:
                    entity = dict(entity_name=entity_name,
                                  text=''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]))
                    entities.append(entity)

                result_file = output_path.joinpath(Path(test_dataset.files_list[image_index]).stem + '.txt')
                with result_file.open(mode='w') as f:
                    for item in entities:
                        f.write('{}\t{}\n'.format(item['entity_name'], item['text']))

def main_table(args):
    device = torch.device(f'cuda:{args.gpu}' if args.gpu != -1 else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)# 加载序列化内容

    config = checkpoint['config']
    state_dict = checkpoint['state_dict']
    monitor_best = checkpoint['monitor_best']
    print('Loading checkpoint: {} \nwith saved mEF {:.4f} ...'.format(args.checkpoint, monitor_best))

    # prepare model for testing
    pick_model = config.init_obj('model_arch', pick_arch_module)
    pick_model = pick_model.to(device)
    pick_model.load_state_dict(state_dict)
    pick_model.eval()

    # setup dataset and data_loader instances
    """
    "files_name":"/home/hcxw/newDisk/jzj/PICK-Pytorch/data/test_data_example/test_samples_list.csv",
            "boxes_and_transcripts_folder":"/home/hcxw/newDisk/jzj/Pick.Pytorch.Sroie/data/test_data_example/boxes_and_transcripts",
            "images_folder":"/home/hcxw/newDisk/jzj/Pick.Pytorch.Sroie/data/test_data_example/images",
            "entities_folder":"/home/hcxw/newDisk/jzj/Pick.Pytorch.Sroie/data/test_data_example/entities",
            "iob_tagging_type":"box_and_within_box_level",
            "resized_image_size": [480, 960],
            "ignore_error": false
    """
    # test_dataset = PICKDataset(
    #                            files_name="/home/hcxw/newDisk/jzj/PICK-pytorch/data/test_data_example/test_samples_list.csv",
    #                             boxes_and_transcripts_folder="/home/hcxw/newDisk/jzj/PICK-pytorch/data/test_data_example/boxes_and_transcripts",
    #                             images_folder="/home/hcxw/newDisk/jzj/PICK-pytorch/data/test_data_example/images",
    #                             entities_folder="/home/hcxw/newDisk/jzj/PICK-pytorch/data/test_data_example/entities",
    #                             iob_tagging_type="box_and_within_box_level",
    #                             resized_image_size= [480, 960],
    #                             ignore_error= False)
    
    # test_data_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
    #                               num_workers=2, collate_fn=BatchCollateFn())
    test_dataset = PICKDataset(boxes_and_transcripts_folder=args.bt,
                               images_folder=args.impt,
                               entities_folder="/home/hcxw/newDisk/jzj/PICK-pytorch/data/test_data_example/entities",
                               iob_tagging_type="box_and_within_box_level",
                               resized_image_size=(480, 960),
                               ignore_error=False,
                               training=False)
    test_data_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                  num_workers=2, collate_fn=BatchCollateFn(training=False))

    # setup output path
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # predict and save to file
    
    # 新增以下两行代码
    valid_f1_metrics = SpanBasedF1MetricTracker(iob_labels_vocab_cls)
    valid_f1_metrics.reset()
    
    with torch.no_grad():
        for step_idx, input_data_item in tqdm(enumerate(test_data_loader)):
            print(step_idx)
            # ['whole_image', 'relation_features', 'text_segments', 'text_length', 'boxes_coordinate', 'mask', 'image_indexs']
            # ['whole_image', 'relation_features', 'text_segments', 'text_length', 'boxes_coordinate', 'mask', 'image_indexs']
            # ['whole_image', 'relation_features', 'text_segments', 'text_length', 'boxes_coordinate', 'mask', 'iob_tags_label']
            # ['whole_image', 'relation_features', 'text_segments', 'text_length', 'boxes_coordinate', 'mask', 'iob_tags_label']
            print(input_data_item.keys())
            for key, input_value in input_data_item.items():
                if input_value is not None:
                    input_data_item[key] = input_value.to(device)

            output = pick_model(**input_data_item)
            # print(output)
            # print("="*50)
            logits = output['logits']
            new_mask = output['new_mask']
            image_indexs = input_data_item['image_indexs']  # (B,)
            text_segments = input_data_item['text_segments']  # (B, num_boxes, T)
            mask = input_data_item['mask']
            # List[(List[int], torch.Tensor)]
            best_paths = pick_model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask, logits_batch_first=True)
            predicted_tags = []
            for path, score in best_paths:
                predicted_tags.append(path)

            # convert iob index to iob string
            decoded_tags_list = iob_index_to_str(predicted_tags)
            # union text as a sequence and convert index to string
            decoded_texts_list = text_index_to_str(text_segments, mask)

            for decoded_tags, decoded_texts, image_index in zip(decoded_tags_list, decoded_texts_list, image_indexs):
                # List[ Tuple[str, Tuple[int, int]] ]
                spans = bio_tags_to_spans(decoded_tags, [])
                spans = sorted(spans, key=lambda x: x[1][0])

                entities = []  # exists one to many case
                for entity_name, range_tuple in spans:
                    entity = dict(entity_name=entity_name,
                                  text=''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]))
                    entities.append(entity)

                result_file = output_path.joinpath(Path(test_dataset.files_list[image_index]).stem + '.txt')
                with result_file.open(mode='w') as f:
                    for item in entities:
                        f.write('{}\t{}\n'.format(item['entity_name'], item['text']))
            predicted_tags_hard_prob = logits * 0
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    predicted_tags_hard_prob[i, j, tag_id] = 1

            golden_tags = input_data_item['iob_tags_label']
            mask = input_data_item['mask']
            union_iob_tags = iob_tags_to_union_iob_tags(golden_tags, mask)       
            valid_f1_metrics.update(predicted_tags_hard_prob.long(), union_iob_tags, new_mask)
            
        f1_result_dict = valid_f1_metrics.result()
        print(f1_result_dict)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch PICK Testing')
    args.add_argument('-ckpt', '--checkpoint', default="/home/hcxw/newDisk/jzj/Pick.Pytorch.Sroie/saved/models/PICK_Default/test_0525_002309/model_best.pth", type=str,
                      help='path to load checkpoint (default: None)')
    args.add_argument('--bt', '--boxes_transcripts', default="/home/hcxw/newDisk/jzj/PICK-pytorch/data/test_data_example/boxes_and_transcripts/", type=str,
                      help='ocr results folder including boxes and transcripts (default: None)')
    args.add_argument('--impt', '--images_path', default="/home/hcxw/newDisk/jzj/PICK-pytorch/data/test_data_example/images/", type=str,
                      help='images folder path (default: None)')
    args.add_argument('-output', '--output_folder', default='./saved/predict_results/', type=str,
                      help='output folder (default: predict_results)')
    args.add_argument('-g', '--gpu', default=0, type=int,
                      help='GPU id to use. (default: -1, cpu)')
    args.add_argument('--bs', '--batch_size', default=1, type=int,
                      help='batch size (default: 1)')
    args = args.parse_args()
    main_table(args)
