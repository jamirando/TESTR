import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from adet.layers.deformable_transformer import DeformableTransformer

from adet.layers.pos_encoding import PositionalEncoding1D
from adet.utils.misc import NestedTensor, inverse_sigmoid_offset, nested_tensor_from_tensor_list, sigmoid_offset

import copy
import math
from methods.vidt.dct import ProcessorDCT
from methods.vidt.fpn_fusion import FPNFusionModule


def _get_clones(module, N):
    """ Clone a module N times """

    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class TESTR(nn.Module):
    """
    Same as :class:`detectron2.modeling.ProposalNetwork`.
    Use one stage detector and a second stage for instance-wise prediction.
    """
    def __init__(self, cfg, backbone):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = backbone
        
        # fmt: off
        self.d_model                 = cfg.MODEL.TRANSFORMER.HIDDEN_DIM
        self.nhead                   = cfg.MODEL.TRANSFORMER.NHEADS
        self.num_encoder_layers      = cfg.MODEL.TRANSFORMER.ENC_LAYERS
        self.num_decoder_layers      = cfg.MODEL.TRANSFORMER.DEC_LAYERS
        self.dim_feedforward         = cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD
        self.dropout                 = cfg.MODEL.TRANSFORMER.DROPOUT
        self.activation              = "relu"
        self.return_intermediate_dec = True
        self.num_feature_levels      = cfg.MODEL.TRANSFORMER.NUM_FEATURE_LEVELS
        self.dec_n_points            = cfg.MODEL.TRANSFORMER.ENC_N_POINTS
        self.enc_n_points            = cfg.MODEL.TRANSFORMER.DEC_N_POINTS
        self.num_proposals           = cfg.MODEL.TRANSFORMER.NUM_QUERIES
        self.pos_embed_scale         = cfg.MODEL.TRANSFORMER.POSITION_EMBEDDING_SCALE
        self.num_ctrl_points         = cfg.MODEL.TRANSFORMER.NUM_CTRL_POINTS
        self.num_classes             = 1
        self.max_text_len            = cfg.MODEL.TRANSFORMER.NUM_CHARS
        self.voc_size                = cfg.MODEL.TRANSFORMER.VOC_SIZE
        self.sigmoid_offset          = not cfg.MODEL.TRANSFORMER.USE_POLYGON

        self.text_pos_embed   = PositionalEncoding1D(self.d_model, normalize=True, scale=self.pos_embed_scale)
        # fmt: on
        
        self.transformer = DeformableTransformer(
            d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers, dim_feedforward=self.dim_feedforward,
            dropout=self.dropout, activation=self.activation, return_intermediate_dec=self.return_intermediate_dec,
            num_feature_levels=self.num_feature_levels, dec_n_points=self.dec_n_points, 
            enc_n_points=self.enc_n_points, num_proposals=self.num_proposals,
        )
        self.ctrl_point_class = nn.Linear(self.d_model, self.num_classes)
        self.ctrl_point_coord = MLP(self.d_model, self.d_model, 2, 3)
        self.bbox_coord = MLP(self.d_model, self.d_model, 4, 3)
        self.bbox_class = nn.Linear(self.d_model, self.num_classes)
        self.text_class = nn.Linear(self.d_model, self.voc_size + 1)

        # shared prior between instances (objects)
        self.ctrl_point_embed = nn.Embedding(self.num_ctrl_points, self.d_model)
        self.text_embed = nn.Embedding(self.max_text_len, self.d_model)

                
        if self.num_feature_levels > 1:
            strides = [8, 16, 32]
            num_channels = [512, 1024, 2048]
            num_backbone_outs = len(strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.d_model, kernel_size=1),
                    nn.GroupNorm(32, self.d_model),
                ))
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.d_model,
                              kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.d_model),
                ))
                in_channels = self.d_model
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            strides = [32]
            num_channels = [2048]
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(
                        num_channels[0], self.d_model, kernel_size=1),
                    nn.GroupNorm(32, self.d_model),
                )])
        self.aux_loss = cfg.MODEL.TRANSFORMER.AUX_LOSS

        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        self.ctrl_point_class.bias.data = torch.ones(self.num_classes) * bias_value
        self.bbox_class.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.ctrl_point_coord.layers[-1].weight.data, 0)
        nn.init.constant_(self.ctrl_point_coord.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = self.num_decoder_layers
        self.ctrl_point_class = nn.ModuleList(
            [self.ctrl_point_class for _ in range(num_pred)])
        self.ctrl_point_coord = nn.ModuleList(
            [self.ctrl_point_coord for _ in range(num_pred)])
        self.transformer.decoder.bbox_embed = None

        nn.init.constant_(self.bbox_coord.layers[-1].bias.data[2:], 0.0)
        self.transformer.bbox_class_embed = self.bbox_class
        self.transformer.bbox_embed = self.bbox_coord

        self.to(self.device)


    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_keypoints": The normalized keypoint coordinates for all queries, represented as
                               (x, y). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        # samples = nested_tensor_from_tensor_list(samples)
        # print(samples.tensors.shape)
        features, pos = self.backbone(samples)
        print(dim(features))

        if self.num_feature_levels == 1:
            features = [features[-1]]
            pos = [pos[-1]]

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = masks[0]
                mask = F.interpolate(
                    m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        # n_points, embed_dim --> n_objects, n_points, embed_dim
        ctrl_point_embed = self.ctrl_point_embed.weight[None, ...].repeat(self.num_proposals, 1, 1)
        text_pos_embed = self.text_pos_embed(self.text_embed.weight)[None, ...].repeat(self.num_proposals, 1, 1)
        text_embed = self.text_embed.weight[None, ...].repeat(self.num_proposals, 1, 1)

        hs, hs_text, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(
            srcs, masks, pos, ctrl_point_embed, text_embed, text_pos_embed, text_mask=None)

        outputs_classes = []
        outputs_coords = []
        outputs_texts = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid_offset(reference, offset=self.sigmoid_offset)
            outputs_class = self.ctrl_point_class[lvl](hs[lvl])
            tmp = self.ctrl_point_coord[lvl](hs[lvl])
            if reference.shape[-1] == 2:
                tmp += reference[:, :, None, :]
            else:
                assert reference.shape[-1] == 4
                tmp += reference[:, :, None, :2]
            outputs_texts.append(self.text_class(hs_text[lvl]))
            outputs_coord = sigmoid_offset(tmp, offset=self.sigmoid_offset)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_text = torch.stack(outputs_texts)

        out = {'pred_logits': outputs_class[-1],
               'pred_ctrl_points': outputs_coord[-1],
               'pred_texts': outputs_text[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_text)

        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        out['enc_outputs'] = {
            'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_text):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_ctrl_points': b, 'pred_texts': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_text[:-1])]

class TextDet(nn.Module):
    """
    Same as :class:`detectron2.modeling.ProposalNetwork`.
    Use one stage detector and a second stage for instance-wise prediction.
    """
    def __init__(self, cfg, backbone):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = backbone
        
        # fmt: off
        self.d_model                 = cfg.MODEL.TRANSFORMER.HIDDEN_DIM
        self.nhead                   = cfg.MODEL.TRANSFORMER.NHEADS
        self.num_encoder_layers      = cfg.MODEL.TRANSFORMER.ENC_LAYERS
        self.num_decoder_layers      = cfg.MODEL.TRANSFORMER.DEC_LAYERS
        self.dim_feedforward         = cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD
        self.dropout                 = cfg.MODEL.TRANSFORMER.DROPOUT
        self.activation              = "relu"
        self.return_intermediate_dec = True
        self.num_feature_levels      = cfg.MODEL.TRANSFORMER.NUM_FEATURE_LEVELS
        self.dec_n_points            = cfg.MODEL.TRANSFORMER.ENC_N_POINTS
        self.enc_n_points            = cfg.MODEL.TRANSFORMER.DEC_N_POINTS
        self.num_proposals           = cfg.MODEL.TRANSFORMER.NUM_QUERIES
        self.pos_embed_scale         = cfg.MODEL.TRANSFORMER.POSITION_EMBEDDING_SCALE
        self.num_ctrl_points         = cfg.MODEL.TRANSFORMER.NUM_CTRL_POINTS
        self.num_classes             = 1
        self.max_text_len            = cfg.MODEL.TRANSFORMER.NUM_CHARS
        self.voc_size                = cfg.MODEL.TRANSFORMER.VOC_SIZE
        self.sigmoid_offset          = not cfg.MODEL.TRANSFORMER.USE_POLYGON

        self.text_pos_embed   = PositionalEncoding1D(self.d_model, normalize=True, scale=self.pos_embed_scale)
        # fmt: on
        
        self.transformer = DeformableTransformer(
            d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers, dim_feedforward=self.dim_feedforward,
            dropout=self.dropout, activation=self.activation, return_intermediate_dec=self.return_intermediate_dec,
            num_feature_levels=self.num_feature_levels, dec_n_points=self.dec_n_points, 
            enc_n_points=self.enc_n_points, num_proposals=self.num_proposals,
        )
        self.ctrl_point_class = nn.Linear(self.d_model, self.num_classes)
        self.ctrl_point_coord = MLP(self.d_model, self.d_model, 2, 3)
        self.bbox_coord = MLP(self.d_model, self.d_model, 4, 3)
        self.bbox_class = nn.Linear(self.d_model, self.num_classes)
        self.text_class = nn.Linear(self.d_model, self.voc_size + 1)

        # shared prior between instances (objects)
        self.ctrl_point_embed = nn.Embedding(self.num_ctrl_points, self.d_model)
        self.text_embed = nn.Embedding(self.max_text_len, self.d_model)

                
        if self.num_feature_levels > 1:
            strides = [8, 16, 32]
            num_channels = [512, 1024, 2048]
            num_backbone_outs = len(strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.d_model, kernel_size=1),
                    nn.GroupNorm(32, self.d_model),
                ))
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.d_model,
                              kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.d_model),
                ))
                in_channels = self.d_model
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            strides = [32]
            num_channels = [2048]
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(
                        num_channels[0], self.d_model, kernel_size=1),
                    nn.GroupNorm(32, self.d_model),
                )])
        self.aux_loss = cfg.MODEL.TRANSFORMER.AUX_LOSS

        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        self.ctrl_point_class.bias.data = torch.ones(self.num_classes) * bias_value
        self.bbox_class.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.ctrl_point_coord.layers[-1].weight.data, 0)
        nn.init.constant_(self.ctrl_point_coord.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = self.num_decoder_layers
        self.ctrl_point_class = nn.ModuleList(
            [self.ctrl_point_class for _ in range(num_pred)])
        self.ctrl_point_coord = nn.ModuleList(
            [self.ctrl_point_coord for _ in range(num_pred)])
        self.transformer.decoder.bbox_embed = None

        nn.init.constant_(self.bbox_coord.layers[-1].bias.data[2:], 0.0)
        self.transformer.bbox_class_embed = self.bbox_class
        self.transformer.bbox_embed = self.bbox_coord

        self.to(self.device)


    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_keypoints": The normalized keypoint coordinates for all queries, represented as
                               (x, y). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        # samples = nested_tensor_from_tensor_list(samples)
        # print(samples.tensors.shape)
        features, pos = self.backbone(samples)
        # print(dim(features))

        if self.num_feature_levels == 1:
            features = [features[-1]]
            pos = [pos[-1]]

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = masks[0]
                mask = F.interpolate(
                    m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        # n_points, embed_dim --> n_objects, n_points, embed_dim
        ctrl_point_embed = self.ctrl_point_embed.weight[None, ...].repeat(self.num_proposals, 1, 1)
        text_pos_embed = self.text_pos_embed(self.text_embed.weight)[None, ...].repeat(self.num_proposals, 1, 1)
        text_embed = self.text_embed.weight[None, ...].repeat(self.num_proposals, 1, 1)

        hs, hs_text, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, ctrl_point_embed, text_embed, text_pos_embed, text_mask=None)

        # init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, ctrl_point_embed, text_mask=None)

        outputs_classes = []
        outputs_coords = []
        outputs_texts = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid_offset(reference, offset=self.sigmoid_offset)
            outputs_class = self.ctrl_point_class[lvl](hs[lvl])
            tmp = self.ctrl_point_coord[lvl](hs[lvl])
            if reference.shape[-1] == 2:
                tmp += reference[:, :, None, :]
            else:
                assert reference.shape[-1] == 4
                tmp += reference[:, :, None, :2]
            outputs_texts.append(self.text_class(hs_text[lvl]))
            outputs_coord = sigmoid_offset(tmp, offset=self.sigmoid_offset)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_text = torch.stack(outputs_texts)

        out = {'pred_logits': outputs_class[-1],
               'pred_ctrl_points': outputs_coord[-1],#}
               'pred_texts': outputs_text[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_text)

        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        out['enc_outputs'] = {
            'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_text):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_ctrl_points': b, 'pred_texts': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_text[:-1])]


class ViDT(nn.Module):
    
    def __init__(self, cfg, backbone, transformer):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = backbone
        
        # fmt: off
        self.d_model                 = cfg.MODEL.TRANSFORMER.REDUCED_DIM
        # self.nhead                   = cfg.MODEL.TRANSFORMER.NHEADS
        # self.num_decoder_layers      = cfg.MODEL.TRANSFORMER.DEC_LAYERS
        # self.dim_feedforward         = cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD
        # self.dropout                 = cfg.MODEL.TRANSFORMER.DROPOUT
        # self.activation              = "relu"
        # self.return_intermediate_dec = True
        # self.num_feature_levels      = cfg.MODEL.TRANSFORMER.NUM_FEATURE_LEVELS
        # self.dec_n_points            = cfg.MODEL.TRANSFORMER.DEC_N_POINTS
        # self.token_label            = cfg.MODEL.TRANSFORMER.TOKEN_LABEL
        # self.num_queries           = cfg.MODEL.TRANSFORMER.NUM_QUERIES
        self.num_classes             = 1
        self.sigmoid_offset          = not cfg.MODEL.TRANSFORMER.USE_POLYGON
        
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # DeformableTransformer(
        #     d_model=self.d_model, nhead=self.nhead, 
        #     num_decoder_layers=self.num_decoder_layers, dim_feedforward=self.dim_feedforward,
        #     dropout=self.dropout, activation=self.activation, return_intermediate_dec=self.return_intermediate_dec,
        #     num_feature_levels=self.num_feature_levels, dec_n_points=self.dec_n_points, #num_proposals=self.num_queries,
        # )

        self.class_embed = nn.Linear(self.d_model, self.num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.backbone = backbone

        # two essential techniques used [default use]
        self.aux_loss = cfg.MODEL.TRANSFORMER.AUX_LOSS
        self.with_box_refine = cfg.MODEL.TRANSFORMER.WITH_BOX_REFINE

        # For UQR module for ViDT+
        self.with_vector = cfg.MODEL.TRANSFORMER.WITH_VECTOR
        self.processor_dct = cfg.MODEL.TRANSFORMER.PROCESSOR_DCT
        if self.with_vector:
            print(f'Training with vector_hidden_dim {vector_hidden_dim}.', flush=True)
            self.vector_embed = MLP(hidden_dim, vector_hidden_dim, self.processor_dct.n_keep, 3)
        
        ############ Modified for ViDT+
        # For two additional losses for ViDT+
        self.iou_aware = cfg.MODEL.TRANSFORMER.IOU_AWARE
        self.token_label = cfg.MODEL.TRANSFORMER.TOKEN_LABEL

        # distillation
        self.distil = cfg.MODEL.TRANSFORMER.DISTIL


        # For EPFF module for ViDT+
        if not cfg.MODEL.TRANSFORMER.EPFF:
        # if epff is None:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    # This is 1x1 conv -> so linear layer
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)

            # initialize the projection layer for [PATCH] tokens
            for proj in self.input_proj:
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)
            self.fusion = None
        else:
            # the cross scale fusion module has its own reduction layers
            self.fusion = epff
        ############
                
        # channel dim reduction for [DET] tokens
        self.tgt_proj = nn.Sequential(
            # This is 1x1 conv -> so linear layer
            nn.Conv2d(self.backbone.num_channels[-2], hidden_dim, kernel_size=1),
            nn.GroupNorm(32, hidden_dim),
            )

        # channel dim reduction for [DET] learnable pos encodings
        self.query_pos_proj = nn.Sequential(
            # This is 1x1 conv -> so linear layer
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(32, hidden_dim)
            )

        # initialize detection head: box regression and classification
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # initialize projection layer for [DET] tokens and encodings
        nn.init.xavier_uniform_(self.tgt_proj[0].weight, gain=1)
        nn.init.constant_(self.tgt_proj[0].bias, 0)
        nn.init.xavier_uniform_(self.query_pos_proj[0].weight, gain=1)
        nn.init.constant_(self.query_pos_proj[0].bias, 0)

        ############ Added for UQR
        if self.with_vector:
            nn.init.constant_(self.vector_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.vector_embed.layers[-1].bias.data, 0)
        ############        

        # the prediction is mae for each decoding layers + the standalone detector (Swin with RAM)
        num_pred = transformer.decoder.num_layers + 1

        # set up all required nn.Module for additional techniques
        if self.with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = num_backbone_outs

        ############ Added for UQR
        if self.with_vector:
            nn.init.constant_(self.vector_embed.layers[-1].bias.data[2:], -2.0)
            self.vector_embed = nn.ModuleList([self.vector_embed for _ in range(num_pred)])
        ############

        if self.iou_aware:
            self.iou_embed = MLP(hidden_dim, hidden_dim, 1, 3)
            if with_box_refine:
                self.iou_embed = _get_clones(self.iou_embed, num_pred)
            else:
                self.iou_embed = nn.ModuleList([self.iou_embed for _ in range(num_pred)])

    def forward(self, samples: NestedTensor):
        """ The forward step of ViDT

        Parameters:
            The forward expects a NestedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        Returns:
            A dictionary having the key and value pairs below:
            - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
            - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
            - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
                            If iou_aware is True, "pred_ious" is also returns as one of the key in "aux_outputs"
            - "enc_tokens": If token_label is True, "enc_tokens" is returned to be used

            Note that aux_loss and box refinement is used in ViDT in default.
        """

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # print(samples)
        x = samples.tensors # RGB input
        # print(x.shape)
        mask = samples.mask # padding mask
        # print(mask.shape)
        # print(mask)

        # return multi-scale [PATCH] tokens along with final [DET] tokens and their pos encodings
        features, det_tgt, det_pos =  self.backbone(x, mask)
        # feature, det_tgt, det_pos = self.backbone(samples)

        # [DET] token and encoding projection to compact representation for the input to the Neck-free transformer
        det_tgt = self.tgt_proj(det_tgt.unsqueeze(-1)).squeeze(-1).permute(0,2,1)
        det_pos = self.query_pos_proj(det_pos.unsqueeze(-1)).squeeze(-1).permute(0,2,1)

        # [PATCH] token projection
        shapes = []
        for l, src in enumerate(features):
            shapes.append(src.shape[-2:])

        srcs = []
        if self.fusion is None:
            for l, src in enumerate(features):
                # print(l)
                # print(src)
                srcs.append(self.input_proj[l](src))
        else:
            # EPFF (multi-scale fusion) is used if fusion is activated
            srcs = self.fusion(features)

        masks = []
        for l, src in enumerate(srcs):
            # resize mask
            shapes.append(src.shape[-2:])
            _mask = F.interpolate(mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
            masks.append(_mask)
            assert mask is not None

        outputs_classes = []
        outputs_coords = []

        # return the output of the neck-free decoder
        hs, init_reference, inter_references, enc_token_class_unflat = self.transformer(srcs, masks, det_tgt, det_pos)

        # perform predictions via the detection head
        for lvl in range(hs.shape[0]):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = inverse_sigmoid_offset(reference, offset=self.sigmoid_offset)

            outputs_class = self.class_embed[lvl](hs[lvl])
            ## bbox output + reference
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            outputs_coord = sigmoid_offset(tmp, offset=self.sigmoid_offset)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        # stack all predictions made form each decoding layers
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        ############ Added for UQR
        outputs_vector = None
        if self.with_vector:
            outputs_vectors = []
            for lvl in range(hs.shape[0]):
                outputs_vector = self.vector_embed[lvl](hs[lvl])
                outputs_vectors.append(outputs_vector)
            outputs_vector = torch.stack(outputs_vectors)
        ############

        # final prediction is made by the last decoding layer
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        ############ Added for UQR
        if self.with_vector:
            out.update({'pred_vectors': outputs_vector[-1]})

        ############

        # aux loss is defined tby using the rest predictions
        if self.aux_loss and self.transformer.decoder.num_layers:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_vector)

        # iou awareness loss is defined for each decoding layer similar to auxiliary decoding loss
        if self.iou_aware:
            outputs_ious = []
            for lvl in range(hs.shape[0]):
                outputs_ious.append(self.iou_embed[lvl](hs[lvl]))
                outputs_iou = torch.stack(outputs_ious)
                out['pred_ious'] = outputs_iou[-1]

                if self.aux_loss:
                    for i, aux in enumerate(out['aux_outputs']):
                        aux['pred_ious'] = outputs_iou[i]

        # token label loss
        if self.token_label:
            out['enc_tokens'] = {'pred_logits': enc_token_class_unflat}

        if self.distil:
            # 'patch_token': multi-scale patch tokens from each stage
            # 'body_det_token' and 'neck_det_tgt': the input det_token for multiple detection heads
            out['distil_tokens'] = {'patch_token': srcs, 'body_det_token': det_tgt, 'neck_det_token': hs}

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_vector):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.

        if outputs_vector is None:
            return [{'pred_logits':a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        else:
            return [{'pred_logits':a, 'pred_boxes': b, 'pred_vectors':c} for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_vector[:-1])]        