# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/8/2020 10:54 PM

from typing import *
from types import FunctionType

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .encoder import Encoder
from .graph import GLCN
from .decoder import Decoder
from utils.class_utils import keys_vocab_cls, iob_labels_vocab_cls
from utils.entities_list import Entities_list
# B：batch  T is the max length of transcripts; D is dimension of model
# N is the number of segments of the documents;
class PICKModel(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        embedding_kwargs = kwargs['embedding_kwargs']
        encoder_kwargs = kwargs['encoder_kwargs']
        graph_kwargs = kwargs['graph_kwargs']
        decoder_kwargs = kwargs['decoder_kwargs']
        self.kvc_kwargs = kwargs['kvc_kwargs']
        self.kvc_type = self.kvc_kwargs["kvc_type"]
        if self.kvc_type == "matrix":
            d_model = self.kvc_kwargs["embedding_dim"]
            
            self.entity_list = Entities_list
            # self.entity_cls_loss = torchvision.ops.sigmoid_focal_loss
            self.entity_cls_loss = sigmoid_focal_loss
            self.kvc_weights = self.kvc_kwargs["kvc_weights"]
            # (M,D)
            self.cls_embedding = nn.Embedding(1 + len(self.entity_list), d_model)
            self.kie_fuse_fc = nn.Linear((len(self.entity_list) + 1) * d_model, d_model)
            # cls_num = len(self.entity_list) + 1 # M
            # self.FC = nn.Linear(N, cls_num)
            self.entity_dict = {
                'O': 0,
                '<PAD>': 1,
            }
            
            # for ind, val in enumerate(load_dict(kie_dict_file), 1):
            for ind, val in enumerate(self.entity_list, 1):
                self.entity_dict['B-' + val] = 2 * ind
                self.entity_dict['I-' + val] = 2 * ind + 1
            self.rev_entity_dict = dict()
            for key, val in self.entity_dict.items():
                self.rev_entity_dict[val] = key
            
        self.make_model(embedding_kwargs, encoder_kwargs, graph_kwargs, decoder_kwargs)

    def make_model(self, embedding_kwargs, encoder_kwargs, graph_kwargs, decoder_kwargs):

        embedding_kwargs['num_embeddings'] = len(keys_vocab_cls)
        self.word_emb = nn.Embedding(**embedding_kwargs)

        encoder_kwargs['char_embedding_dim'] = embedding_kwargs['embedding_dim']
        self.encoder = Encoder(**encoder_kwargs)

        graph_kwargs['in_dim'] = encoder_kwargs['out_dim']
        graph_kwargs['out_dim'] = encoder_kwargs['out_dim']
        self.graph = GLCN(**graph_kwargs)

        decoder_kwargs['bilstm_kwargs']['input_size'] = encoder_kwargs['out_dim']
        if decoder_kwargs['bilstm_kwargs']['bidirectional']:
            decoder_kwargs['mlp_kwargs']['in_dim'] = decoder_kwargs['bilstm_kwargs']['hidden_size'] * 2
        else:
            decoder_kwargs['mlp_kwargs']['in_dim'] = decoder_kwargs['bilstm_kwargs']['hidden_size']
        decoder_kwargs['mlp_kwargs']['out_dim'] = len(iob_labels_vocab_cls)
        decoder_kwargs['crf_kwargs']['num_tags'] = len(iob_labels_vocab_cls)
        self.decoder = Decoder(**decoder_kwargs)

    def _aggregate_avg_pooling(self, input, text_mask):
        '''
        Apply mean pooling over time (text length), (B*N, T, D) -> (B*N, D)
        :param input: (B*N, T, D)
        :param text_mask: (B*N, T)
        :return: (B*N, D)
        '''
        # filter out padding value, (B*N, T, D)
        input = input * text_mask.detach().unsqueeze(2).float()
        # (B*N, D)
        sum_out = torch.sum(input, dim=1)
        # (B*N, )
        text_len = text_mask.float().sum(dim=1)
        # (B*N, D)
        text_len = text_len.unsqueeze(1).expand_as(sum_out)
        text_len = text_len + text_len.eq(0).float()  # avoid divide zero denominator
        # (B*N, D)
        mean_out = sum_out.div(text_len)
        return mean_out

    @staticmethod
    def compute_mask(mask: torch.Tensor):
        '''
        :param mask: (B, N, T)
        :return: True for masked key position according to pytorch official implementation of Transformer
        '''
        B, N, T = mask.shape
        mask = mask.reshape(B * N, T)
        mask_sum = mask.sum(dim=-1)  # (B*N,)

        # (B*N,)
        graph_node_mask = mask_sum != 0
        # (B * N, T)
        graph_node_mask = graph_node_mask.unsqueeze(-1).expand(B * N, T)  # True for valid node
        # If src key are all be masked (indicting text segments is null), atten_weight will be nan after softmax
        # in self-attention layer of Transformer.
        # So we do not mask all padded sample. Instead we mask it after Transformer encoding.
        src_key_padding_mask = torch.logical_not(mask.bool()) & graph_node_mask  # True for padding mask position
        return src_key_padding_mask, graph_node_mask

    def forward(self, **kwargs):
        # input
        whole_image = kwargs['whole_image']  # (B, 3, H, W)
        relation_features = kwargs['relation_features']  # initial relation embedding (B, N, N, 6) # np.zeros((boxes_num, boxes_num, 6))
        text_segments = kwargs['text_segments']  # text segments (B, N, T)
        text_length = kwargs['text_length']  # (B, N)
        iob_tags_label = kwargs['iob_tags_label'] if self.training else None  # (B, N, T)
        mask = kwargs['mask']  # (B, N, T)
        boxes_coordinate = kwargs['boxes_coordinate']  # (B, num_boxes, 8)
        num_boxes = kwargs['num_boxes'] # list[]
        
        ##### Forward Begin #####
        ### Encoder module ###
        # word embedding (B,N,T,Dim) [2, 70, 33, 512]
        text_emb = self.word_emb(text_segments) # 如何实现一个 (B,N,cls)的编码
        B,N,T,D = text_emb.shape
        
        # src_key_padding_mask is text padding mask, True is padding value (B*N, T)
        # graph_node_mask is mask for graph, True is valid node, (B*N, T)
        src_key_padding_mask, graph_node_mask = self.compute_mask(mask)

        # set of nodes, (B*N, T, D)
        x = self.encoder(images=whole_image, boxes_coordinate=boxes_coordinate, transcripts=text_emb,
                         src_key_padding_mask=src_key_padding_mask)

        ### Graph module ###
        # text_mask, True for valid, (including all not valid node), (B*N, T)
        text_mask = torch.logical_not(src_key_padding_mask).byte()
        # (B*N, T, D) -> (B*N, D)
        x_gcn = self._aggregate_avg_pooling(x, text_mask)
        # (B*N, 1)，True is valid node
        graph_node_mask = graph_node_mask.any(dim=-1, keepdim=True)
        # (B*N, D), filter out not valid node
        x_gcn = x_gcn * graph_node_mask.byte()
        # initial adjacent matrix (B, N, N)
        B, N, T = mask.shape
        init_adj = torch.ones((B, N, N), device=text_emb.device)
        boxes_num = mask[:, :, 0].sum(dim=1, keepdim=True)  # (B, 1)
        # (B, N, D)
        x_gcn = x_gcn.reshape(B, N, -1)
        # (B, N, D), (B, N, N), (B,)
        x_gcn, soft_adj, gl_loss = self.graph(x_gcn, relation_features, init_adj, boxes_num)
        adj = soft_adj * init_adj

        # 对 x_gcn 进行关系增强
        # 准备 entity feature
        if self.kvc_type == 'matrix':
            instance_feature = x_gcn.clone() #(B,N,D) 亦可用x
            cls_num = len(self.entity_list) + 1 # M
            # cls_num = N
            # (B,cls_num)->(B,cls_num,C)   理解为CFAM中的 L
            entity_feature = self.cls_embedding( # (B,M,D)
                nn.Parameter(torch.arange(0, cls_num, device=relation_features.device).unsqueeze(0).expand(B, cls_num),
                         requires_grad=False))
            entity_feature = entity_feature / entity_feature.norm(dim=1, keepdim=True)
            #(B, cls, D)-> (B, D, cls_num) # [1, 512, 22] (B,D,M)
            entity_feature = entity_feature.transpose(-1, -2)

            # try some encoder
            device = entity_feature.device
            FC = nn.Linear(N, cls_num).to(device)
            #  E 使用 FC 将维度转换为 (B,D,M) E`
            entity_feature = FC(instance_feature.reshape(B,D,-1))
            
            # 对比学习 S = (B,N,M) = (B, N, D)*(B, D, M) 
            entity_logits = torch.bmm(instance_feature, entity_feature) # [2, 93, 5]
            if self.training:
                # 在训练时，使用真值监督其参数的更新
                # gt_texts = labels['texts'],
                # gt_entities = labels.get('entities', None), gt_entities: (B, N, L)
                # gt_entities_emb = self.word_emb(iob_tags_label)
                kv_mask = self.prep_kvc_gt(iob_tags_label, num_boxes) # (B, N, T)
                #gt_entity:(B,cls,N)->(B,N,cls) 
                # CFAM：N num of instance 和 PICK中N is the number of segments of the documents
                gt_entity = kv_mask['attn_map_gt'] # (B,M,N)
                gt_entity = gt_entity.transpose(-1, -2) # (B,N,M) [2, 93, 5])
                # loss赋予一定权重
                loss_entity = self.kvc_weights * self.entity_cls_loss(entity_logits, gt_entity)
                #  # x_gcn:(B, N, D) [2, 93, 512]
                # x += entity_logits
            
            # 将 entity_feature(B,N,M) [2, 93, 5]  维度转换为 (B,N,D)
            # entity_logits = torch.mean(entity_logits, dim=2) # [2, 93]
            # B,N,M->B,M,N   可以认为是在后处理 数据格式后处理
            entity_logits = entity_logits.permute(0, 2, 1)
            # B, M, N -> B, M, 1, N
            entity_logits = entity_logits.unsqueeze(2) # [2, 5, 1, 93]
            x_logits = x.clone() # (B, N, T, D)
            # BN, T, C -> B, DT, N -> B, 1, DT, N
            x_logits = x_logits.reshape(B, -1, N).unsqueeze(1) # [2, 1, 23040, 93]
            # (B,1,CT,N) *(B,M,1,N)-> (B, M, CL, N) -> (B, NT, M, C) -> B, NT, M * C 
            entity_logits = (x_logits * entity_logits).permute(0, 2, 1, 3).reshape(B, N*T, -1) # [2, 4185, 2560]
            # B, NL, num_cls * C -> B, NL, C -> BN, L, D
            entity_logits = self.kie_fuse_fc(entity_logits).reshape(-1, T, D).reshape(B*N, T, D)
            # entity_logits = self.kie_fc(self.kie_norm(entity_logits))
            x += entity_logits
        
            
        
        ### Decoder module ###
        logits, new_mask, log_likelihood = self.decoder(x.reshape(B, N, T, -1), x_gcn, mask, text_length,
                                                        iob_tags_label)
        ##### Forward End #####

        output = {"logits": logits, "new_mask": new_mask, "adj": adj}

        if self.training:
            output['gl_loss'] = gl_loss
            crf_loss = -log_likelihood
            output['crf_loss'] = crf_loss
            
            if self.kvc_type == "matrix": 
                output['entity_loss'] = loss_entity
        return output

    def __str__(self):
        '''
        Model prints with number of trainable parameters
        '''
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def model_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params
    
    def prep_kvc_gt(self, entities, num_boxes):
        """_summary_

        Args:
            entities (_type_): _description_ (B,N,T)
            num_boxes (_type_): _description_
            gt_texts (_type_, optional): _description_. Defaults to None.
            img_metas (_type_, optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        kvc_mask = dict()
        assert 'O' not in self.entity_list
        if self.kvc_type == "matrix":
            if self.training:
                B, N, L = entities.shape
                cls_num = len(self.entity_list) + 1
                # B, cls_num
                entity_cls_gt = torch.zeros((B, cls_num), device=entities.device)
                # entity_cls_gt = torch.zeros((B, N), device=entities.device)
                # B, cls_num, N
                attn_map_gt = torch.zeros((B, cls_num, N), device=entities.device)
                non_entity_map_gt = torch.zeros((B, N), device=entities.device)
                for idx, cls in enumerate(self.entity_list):
                    # B, N, L
                    entity_mask = (entities == self.entity_dict[f'B-{cls}']) | (entities == self.entity_dict[f'I-{cls}'])
                    # B, N
                    entity_mask = entity_mask.sum(dim=-1) > 0

                    attn_map_gt[:, idx+1, :] = entity_mask
                    entity_cls_gt[:, idx+1] = entity_mask.sum(dim=-1) > 0
                    non_entity_map_gt = non_entity_map_gt + entity_mask

                # #### DEBUG ONLY ####
                # # convert text to transcripts
                # transcripts = []
                # for ind in range(gt_texts.shape[1]):
                #     line_ = ""
                #     for char_idx in gt_texts[0, ind]:
                #         if char_idx == 1:
                #             break
                #         line_ += self.rev_ocr_dict[char_idx.item()]
                #     transcripts.append(line_)
                # import ipdb
                # ipdb.set_trace()

                # for tag 'O'
                # B, N
                non_entity_map_gt = non_entity_map_gt == 0
                attn_map_gt[:, 0, :] = non_entity_map_gt
                entity_cls_gt[:, 0] = non_entity_map_gt.sum(dim=-1) > 0
                kvc_mask['attn_map_gt'] = attn_map_gt
                kvc_mask['entity_cls_gt'] = entity_cls_gt.long()
                # prepare instance mask
                N = max(num_boxes)
                ins_mask = torch.full((B, 1, N, N), fill_value=False, device=entities.device, dtype=torch.bool)
                for idx, num_box in enumerate(num_boxes):
                    ins_mask[idx, :num_box, :num_box] = True
                kvc_mask['ins_mask'] = ins_mask
            else:
                # prepare instance mask
                B = 1
                N = num_boxes[0]
                ins_mask = torch.full((B, 1, N, N), fill_value=False, device=entities.device, dtype=torch.bool)
                for idx, num_box in enumerate(num_boxes):
                    ins_mask[idx, :num_box, :num_box] = True
                kvc_mask['ins_mask'] = ins_mask
        else:
            raise NotImplementedError
        return kvc_mask

def _log_api_usage_once(obj: Any) -> None:

    """
    Logs API usage(module and name) within an organization.
    In a large ecosystem, it's often useful to track the PyTorch and
    TorchVision APIs usage. This API provides the similar functionality to the
    logging module in the Python stdlib. It can be used for debugging purpose
    to log which methods are used and by default it is inactive, unless the user
    manually subscribes a logger via the `SetAPIUsageLogger method <https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
    Please note it is triggered only once for the same API call within a process.
    It does not collect any data from open-source users since it is no-op by default.
    For more information, please refer to
    * PyTorch note: https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
    * Logging policy: https://github.com/pytorch/vision/issues/5052;

    Args:
        obj (class instance or method): an object to extract info from.
    """
    module = obj.__module__
    if not module.startswith("torchvision"):
        module = f"torchvision.internal.{module}"
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{module}.{name}")
    
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(sigmoid_focal_loss)
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss