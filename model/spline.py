import math
from typing import List, Dict
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY

from .util.layer import DFConv2d
from .util.iou_loss import IOULoss
from .output import BTextOutputs


INF = 100000000


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


@PROPOSAL_GENERATOR_REGISTRY.register()
class BText(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        # fmt: off
        self.in_features          = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides          = cfg.MODEL.FCOS.FPN_STRIDES
        self.focal_loss_alpha     = cfg.MODEL.FCOS.LOSS_ALPHA
        self.focal_loss_gamma     = cfg.MODEL.FCOS.LOSS_GAMMA
        self.center_sample        = cfg.MODEL.FCOS.CENTER_SAMPLE
        self.strides              = cfg.MODEL.FCOS.FPN_STRIDES
        self.radius               = cfg.MODEL.FCOS.POS_RADIUS
        self.pre_nms_thresh_train = cfg.MODEL.FCOS.INFERENCE_TH_TRAIN
        self.pre_nms_thresh_test  = cfg.MODEL.FCOS.INFERENCE_TH_TEST
        self.pre_nms_topk_train   = cfg.MODEL.FCOS.PRE_NMS_TOPK_TRAIN
        self.pre_nms_topk_test    = cfg.MODEL.FCOS.PRE_NMS_TOPK_TEST
        self.nms_thresh           = cfg.MODEL.FCOS.NMS_TH
        self.yield_proposal       = cfg.MODEL.FCOS.YIELD_PROPOSAL
        self.post_nms_topk_train  = cfg.MODEL.FCOS.POST_NMS_TOPK_TRAIN
        self.post_nms_topk_test   = cfg.MODEL.FCOS.POST_NMS_TOPK_TEST
        self.thresh_with_ctr      = cfg.MODEL.FCOS.THRESH_WITH_CTR
        # fmt: on
        self.iou_loss = IOULoss(cfg.MODEL.FCOS.LOC_LOSS_TYPE)
        # generate sizes of interest
        soi = []
        prev_size = -1
        for s in cfg.MODEL.FCOS.SIZES_OF_INTEREST:
            soi.append([prev_size, s])
            prev_size = s
        soi.append([prev_size, INF])
        self.sizes_of_interest = soi
        self.fcos_head = FCOSHead(cfg, [input_shape[f] for f in self.in_features])

    def forward_head(self, features, top_module=None):
        features = [features[f] for f in self.in_features]
        pred_class_logits, pred_deltas, pred_centerness, top_feats, bbox_towers = self.fcos_head(
            features, top_module, self.yield_proposal)
        return pred_class_logits, pred_deltas, pred_centerness, top_feats, bbox_towers

    def forward(self, images, features, gt_instances=None, top_module=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        features = [features[f] for f in self.in_features]
        locations = self.compute_locations(features)
        logits_pred, reg_pred, ctrness_pred, top_feats, bbox_towers = self.fcos_head(
            features, top_module, self.yield_proposal)

        if self.training:
            pre_nms_thresh = self.pre_nms_thresh_train
            pre_nms_topk = self.pre_nms_topk_train
            post_nms_topk = self.post_nms_topk_train
        else:
            pre_nms_thresh = self.pre_nms_thresh_test
            pre_nms_topk = self.pre_nms_topk_test
            post_nms_topk = self.post_nms_topk_test

        outputs = BTextOutputs(
            images,
            locations,
            logits_pred,
            reg_pred,
            ctrness_pred,
            top_feats,
            self.focal_loss_alpha,
            self.focal_loss_gamma,
            self.iou_loss,
            self.center_sample,
            self.sizes_of_interest,
            self.strides,
            self.radius,
            self.fcos_head.num_classes,
            pre_nms_thresh,
            pre_nms_topk,
            self.nms_thresh,
            post_nms_topk,
            self.thresh_with_ctr,
            gt_instances
        )

        results = {}
        if self.yield_proposal:
            results["features"] = {
                f: b for f, b in zip(self.in_features, bbox_towers)}

        if self.training:
            losses = outputs.losses()
            
            if top_module is not None:
                results["top_feats"] = top_feats
            if self.yield_proposal:
                with torch.no_grad():
                    results["proposals"] = outputs.predict_proposals(top_feats)
        else:
            losses = {}
            with torch.no_grad():
                proposals = outputs.predict_proposals(top_feats)
            if self.yield_proposal:
                results["proposals"] = proposals
            else:
                results = proposals
                
        return results, losses

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations