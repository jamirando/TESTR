from typing import List
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess as d2_postprocesss
from detectron2.structures import ImageList, Instances

from adet.layers.pos_encoding import PositionalEncoding2D
# from adet.modeling.testr.losses import SetCriterion
# from adet.modeling.testr.matcher import build_matcher
from adet.modeling.testr.models import TESTR
from adet.utils.misc import NestedTensor, box_xyxy_to_cxcywh


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for _, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

class MaskedBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""
    # def __init__(self, cfg):
    def __init__(self,backbone):
        super().__init__()
        # self.backbone = build_backbone(cfg)
        self.backbone = backbone
        self.num_channels = backbone.num_channels
        # backbone_shape = self.backbone.output_shape()
        # self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]
        # self.num_channels = backbone_shape[list(backbone_shape.keys())[-1]].channels

    def forward(self, images):
        features = self.backbone(images.tensor)
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in features.values()],
            images.image_sizes,
            images.tensor.device,
        )
        assert len(features) == len(masks)
        for i, k in enumerate(features.keys()):
            features[k] = NestedTensor(features[k], masks[i])
        # return features, det_tgt, det_pos
        return features

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.feature_strides[idx])),
                    : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks


def detector_postprocess(results, output_height, output_width, mask_threshold=0.5):
    """
    In addition to the post processing of detectron2, we add scalign for 
    bezier control points.
    """
    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
    # results = d2_postprocesss(results, output_height, output_width, mask_threshold)

    # scale bezier points
    if results.has("beziers"):
        beziers = results.beziers
        # scale and clip in place
        h, w = results.image_size
        beziers[:, 0].clamp_(min=0, max=w)
        beziers[:, 1].clamp_(min=0, max=h)
        beziers[:, 6].clamp_(min=0, max=w)
        beziers[:, 7].clamp_(min=0, max=h)
        beziers[:, 8].clamp_(min=0, max=w)
        beziers[:, 9].clamp_(min=0, max=h)
        beziers[:, 14].clamp_(min=0, max=w)
        beziers[:, 15].clamp_(min=0, max=h)
        beziers[:, 0::2] *= scale_x
        beziers[:, 1::2] *= scale_y

    if results.has("polygons"):
        polygons = results.polygons
        polygons[:, 0::2] *= scale_x
        polygons[:, 1::2] *= scale_y

    return results


@META_ARCH_REGISTRY.register()
class TransformerDetector(nn.Module):
    """
    Same as :class:`detectron2.modeling.ProposalNetwork`.
    Use one stage detector and a second stage for instance-wise prediction.
    """
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        
        d2_backbone = MaskedBackbone(cfg)
        N_steps = cfg.MODEL.TRANSFORMER.HIDDEN_DIM // 2
        self.test_score_threshold = cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST
        self.use_polygon = cfg.MODEL.TRANSFORMER.USE_POLYGON
        backbone = Joiner(d2_backbone, PositionalEncoding2D(N_steps, normalize=True))
        backbone.num_channels = d2_backbone.num_channels
        self.testr = TESTR(cfg, backbone)
        # self.testr = TextDet(cfg, backbone)

        box_matcher, point_matcher = build_matcher(cfg)
        # box_matcher = build_matcher(cfg)
        
        loss_cfg = cfg.MODEL.TRANSFORMER.LOSS
        weight_dict = {'loss_ce': loss_cfg.POINT_CLASS_WEIGHT, 'loss_ctrl_points': loss_cfg.POINT_COORD_WEIGHT, 'loss_texts': loss_cfg.POINT_TEXT_WEIGHT}
        enc_weight_dict = {'loss_bbox': loss_cfg.BOX_COORD_WEIGHT, 'loss_giou': loss_cfg.BOX_GIOU_WEIGHT, 'loss_ce': loss_cfg.BOX_CLASS_WEIGHT}
        if loss_cfg.AUX_LOSS:
            aux_weight_dict = {}
            # decoder aux loss
            for i in range(cfg.MODEL.TRANSFORMER.DEC_LAYERS - 1):
                aux_weight_dict.update(
                    {k + f'_{i}': v for k, v in weight_dict.items()})
            # encoder aux loss
            aux_weight_dict.update(
                {k + f'_enc': v for k, v in enc_weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        enc_losses = ['labels', 'boxes']
        dec_losses = ['labels', 'ctrl_points', 'texts']

        self.criterion = SetCriterion(self.testr.num_classes, box_matcher, point_matcher,
                                      weight_dict, enc_losses, dec_losses, self.testr.num_ctrl_points, 
                                      focal_alpha=loss_cfg.FOCAL_ALPHA, focal_gamma=loss_cfg.FOCAL_GAMMA)

        # self.criterion = SetCriterion(self.testr.num_classes, box_matcher,
        #                               weight_dict, enc_losses, 
        #                               focal_alpha=loss_cfg.FOCAL_ALPHA, focal_gamma=loss_cfg.FOCAL_GAMMA)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        images = self.preprocess_image(batched_inputs)
        output = self.testr(images)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            ctrl_point_cls = output["pred_logits"]
            ctrl_point_coord = output["pred_ctrl_points"]
            text_pred = output["pred_texts"]
            results = self.inference(ctrl_point_cls, ctrl_point_coord, text_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            raw_ctrl_points = targets_per_image.polygons if self.use_polygon else targets_per_image.beziers
            gt_ctrl_points = raw_ctrl_points.reshape(-1, self.testr.num_ctrl_points, 2) / torch.as_tensor([w, h], dtype=torch.float, device=self.device)[None, None, :]
            gt_text = targets_per_image.text
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes, "ctrl_points": gt_ctrl_points, "texts": gt_text})
        return new_targets

    def inference(self, ctrl_point_cls, ctrl_point_coord, text_pred, image_sizes):
        assert len(ctrl_point_cls) == len(image_sizes)
        results = []

        text_pred = torch.softmax(text_pred, dim=-1)
        prob = ctrl_point_cls.mean(-2).sigmoid()
        scores, labels = prob.max(-1)

        for scores_per_image, labels_per_image, ctrl_point_per_image, text_per_image, image_size in zip(
            scores, labels, ctrl_point_coord, text_pred, image_sizes
        ):
            selector = scores_per_image >= self.test_score_threshold
            scores_per_image = scores_per_image[selector]
            labels_per_image = labels_per_image[selector]
            ctrl_point_per_image = ctrl_point_per_image[selector]
            text_per_image = text_per_image[selector]
            result = Instances(image_size)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            result.rec_scores = text_per_image
            ctrl_point_per_image[..., 0] *= image_size[1]
            ctrl_point_per_image[..., 1] *= image_size[0]
            if self.use_polygon:
                result.polygons = ctrl_point_per_image.flatten(1)
            else:
                result.beziers = ctrl_point_per_image.flatten(1)
            _, topi = text_per_image.topk(1)
            result.recs = topi.squeeze(-1)
            results.append(result)
        return results


from methods.swin_w_ram import swin_nano
from methods.vidt.matcher import build_matcher
from methods.vidt.criterion import SetCriterion
from methods.vidt.postprocessor import PostProcess
from methods.vidt.fpn_fusion import FPNFusionModule
import copy
import math
from methods.vidt.dct import ProcessorDCT
from adet.modeling.testr.models import ViDT
from adet.layers.deformable_transformer import build_deformable_transformer
from util import box_ops

@META_ARCH_REGISTRY.register()
class ViDTDetector(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.test_score_threshold = cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST        
        self.use_polygon = cfg.MODEL.TRANSFORMER.USE_POLYGON
        backbone, hidden_dim = swin_nano(pretrained=cfg.MODEL.TRANSFORMER.PRETRAINED)
        # self.testr = TESTR(cfg, backbone)
        
        backbone.finetune_det(method=cfg.MODEL.TRANSFORMER.METHOD,
                              det_token_num=cfg.MODEL.TRANSFORMER.DET_TOKEN_NUM,
                              pos_dim=cfg.MODEL.TRANSFORMER.REDUCED_DIM,
                              cross_indices=cfg.MODEL.TRANSFORMER.CROSS_INDICES,
            )

        # backbone = MaskedBackbone(backbone)

        epff = None
        if cfg.MODEL.TRANSFORMER.EPFF:
            epff = FPNFusionModule(backbone.num_channels, fuse_dim=cfg.MODEL.TRANSFORMER.REDUCED_DIM)

        deform_transformer = build_deformable_transformer(cfg)

        if cfg.MODEL.TRANSFORMER.WITH_VECTOR:
            processor_dct = ProcessorDCT()

        self.detector = ViDT(cfg, backbone, deform_transformer)

        matcher = build_matcher(cfg)
        
        loss_cfg = cfg.MODEL.TRANSFORMER.LOSS
        weight_dict = {'loss_bbox': loss_cfg.BOX_COORD_WEIGHT, 'loss_giou': loss_cfg.BOX_GIOU_WEIGHT, 'loss_ce': loss_cfg.BOX_CLASS_WEIGHT}
     
        ##
        if cfg.MODEL.TRANSFORMER.IOU_AWARE:
            weight_dict['loss_iouaware'] = cfg.MODEL.TRANSFORMER.LOSS.IOU_AWARE_COEFF

        if cfg.MODEL.TRANSFORMER.TOKEN_LABEL:
            weight_dict['loss_token_focal'] = cfg.MODEL.TRANSFORMER.LOSS.TOKEN_LOSS_COEF
            weight_dict['loss_token_dice'] = cfg.MODEL.TRANSFORMER.LOSS.TOKEN_LOSS_COEF


        # FOR UQR module
        if cfg.MODEL.TRANSFORMER.MASKS:
            weight_dict["loss_vector"] = 1

        if cfg.MODEL.TRANSFORMER.DISTIL_MODEL is not None:
            weight_dict['loss_distil'] = cfg.MODEL.TRANSFORMER.DISTIL_LOSS_COEF

        # aux decoding loss
        if cfg.MODEL.TRANSFORMER.AUX_LOSS:
            aux_weight_dict = {}
            for i in range(cfg.MODEL.TRANSFORMER.DEC_LAYERS - 1 + 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ['labels', 'boxes']

        if cfg.MODEL.TRANSFORMER.IOU_AWARE:
            losses += ['iouaware']

        # For UQR
        if cfg.MODEL.TRANSFORMER.MASKS:
            losses += ['masks']

        self.criterion = SetCriterion(self.detector.num_classes, matcher, weight_dict, losses,
                                 focal_alpha=cfg.MODEL.TRANSFORMER.LOSS.FOCAL_ALPHA,
                                 # For UQR
                                 with_vector=cfg.MODEL.TRANSFORMER.WITH_VECTOR,
                                 processor_dct=processor_dct if cfg.MODEL.TRANSFORMER.WITH_VECTOR else None,
                                 vector_loss_coef=cfg.MODEL.TRANSFORMER.LOSS.VECTOR_LOSS_COEF,
                                 no_vector_loss_norm=cfg.MODEL.TRANSFORMER.LOSS.NO_VECTOR_LOSS_NORM,
                                 vector_start_stage=cfg.MODEL.TRANSFORMER.VECTOR_START_STAGE)

        self.postprocessors = {'bbox': PostProcess(processor_dct=processor_dct if (cfg.MODEL.TRANSFORMER.WITH_VECTOR) else None)}
        
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)


    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        # images = [x["image"].to(self.device) for x in batched_inputs]
        images_list = ImageList.from_tensors(images)
        return images, images_list

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        
        images, images_list = self.preprocess_image(batched_inputs)
        # print(images)
        self.image_sizes = images_list.image_sizes
        # print(self.image_sizes)
        output = self.detector(images)
        # print('output:',output['pred_boxes'][0])

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            # print('target:',targets[0]['boxes'])
            loss_dict = self.criterion(output, targets)
            # print('loss dict:',loss_dict)
            weight_dict = self.criterion.weight_dict
            # print('weight dict:',weight_dict)
            for k in loss_dict.keys():
                if k in weight_dict:
                    #print(k)
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            ctrl_point_cls = output["pred_logits"]
            ctrl_point_coord = output["pred_boxes"]
            # text_pred = output["pred_texts"]
            # results = self.inference(ctrl_point_cls, ctrl_point_coord, images.image_sizes)
            # target_sizes = torch.stack([self.image_sizes[i] for i in range(len(self.image_sizes))],dim=0)
            results = self.inference(ctrl_point_cls, ctrl_point_coord, self.image_sizes)
            processed_results = []
            #processed_results = results
            #for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, self.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                # r = self.postprocessors['bbox'](results_per_image, image_size)
                processed_results.append({"instances": r})
            return processed_results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            # print('h: {}, w: {}'.format(h,w))
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            # raw_ctrl_points = targets_per_image.polygons if self.use_polygon else targets_per_image.beziers
            # gt_ctrl_points = raw_ctrl_points.reshape(-1, self.testr.num_ctrl_points, 2) / torch.as_tensor([w, h], dtype=torch.float, device=self.device)[None, None, :]
            # gt_text = targets_per_image.text
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        return new_targets

    def inference(self, ctrl_point_cls, ctrl_point_coord, image_sizes):
        assert len(ctrl_point_cls) == len(image_sizes)
        results = []

        text_pred = torch.softmax(text_pred, dim=-1)
        prob = ctrl_point_cls.mean(-2).sigmoid()
        scores, labels = prob.max(-1)

        #prob = ctrl_point_cls.sigmoid()
        #topk_values, topk_indexes = torch.topk(prob.view(ctrl_point_cls.shape[0], -1), 100, dim=1)
        #scores = topk_values
        #topk_boxes = topk_indexes // ctrl_point_cls.shape[2]
        #labels = topk_indexes % ctrl_point_cls.shape[2]
        #boxes = box_ops.box_cxcywh_to_xyxy(ctrl_point_coord)
        #boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        for scores_per_image, labels_per_image, ctrl_point_per_image, image_size in zip(
            scores, labels, ctrl_point_coord, image_sizes
        ):
        #for scores_per_image, labels_per_image, ctrl_point_per_image, image_size in zip(
        #    scores, labels, boxes, image_sizes
        #):
            selector = scores_per_image >= self.test_score_threshold
            scores_per_image = scores_per_image[selector]
            labels_per_image = labels_per_image[selector]
            ctrl_point_per_image = ctrl_point_per_image[selector]
            # text_per_image = text_per_image[selector]
            result = Instances(image_size)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            # result.rec_scores = text_per_image
            ctrl_point_per_image[..., 0] *= image_size[1]
            ctrl_point_per_image[..., 1] *= image_size[0]
            if self.use_polygon:
                result.polygons = ctrl_point_per_image.flatten(1)
            else:
                result.beziers = ctrl_point_per_image.flatten(1)
            # _, topi = text_per_image.topk(1)
            # result.recs = topi.squeeze(-1)
            
            results.append(result)
        print(results)
        return results
        #img_h, img_w = torch.as_tensor(image_sizes, dtype=torch.float, device=self.device).unbind(1)
        #scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(torch.float32)
        #boxes = boxes * scale_fct[:, None, :]

        #results = [{'scores': s, 'labels': l, 'boxes':b} for s, l, b in zip(scores, labels, boxes)]

        #return results
