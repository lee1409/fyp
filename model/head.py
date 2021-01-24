def build_top_module(cfg):
    top_type = cfg.MODEL.TOP_MODULE.NAME
    if top_type == "conv":
        inp = cfg.MODEL.FPN.OUT_CHANNELS
        oup = cfg.MODEL.TOP_MODULE.DIM
        top_module = nn.Conv2d(
            inp, oup,
            kernel_size=3, stride=1, padding=1)
    else:
        top_module = None
    return top_module

class FCOSHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        # TODO: Implement the sigmoid version first.
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        head_configs = {"cls": (cfg.MODEL.FCOS.NUM_CLS_CONVS,
                                False),
                        "bbox": (cfg.MODEL.FCOS.NUM_BOX_CONVS,
                                 cfg.MODEL.FCOS.USE_DEFORMABLE),
                        "share": (cfg.MODEL.FCOS.NUM_SHARE_CONVS,
                                  cfg.MODEL.FCOS.USE_DEFORMABLE)}
        norm = None if cfg.MODEL.FCOS.NORM == "none" else cfg.MODEL.FCOS.NORM

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            if use_deformable:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d
            for i in range(num_convs):
                tower.append(conv_func(
                    in_channels, in_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=True
                ))
                if norm == "GN":
                    tower.append(nn.GroupNorm(32, in_channels))
                tower.append(nn.ReLU())
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.cls_logits = nn.Conv2d(
            in_channels, self.num_classes,
            kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3,
            stride=1, padding=1
        )
        self.ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3,
            stride=1, padding=1
        )
        self.top_module = build_top_module(cfg)

        if cfg.MODEL.FCOS.USE_SCALE:
            self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in self.fpn_strides])
        else:
            self.scales = None

        for modules in [
            self.cls_tower, self.bbox_tower,
            self.share_tower, self.cls_logits,
            self.bbox_pred, self.ctrness
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x, top_module=None, yield_bbox_towers=False):
        logits = []
        bbox_reg = []
        ctrness = []
        top_feats = []
        bbox_towers = []
        for l, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = self.cls_tower(feature)
            bbox_tower = self.bbox_tower(feature)
            if yield_bbox_towers:
                bbox_towers.append(bbox_tower)

            logits.append(self.cls_logits(cls_tower))
            ctrness.append(self.ctrness(bbox_tower))
            reg = self.bbox_pred(bbox_tower)
            if self.scales is not None:
                reg = self.scales[l](reg)
            # Note that we use relu, as in the improved FCOS, instead of exp.
            bbox_reg.append(F.relu(reg))
            if top_module is None:
                top_feats.append(self.top_module(bbox_tower))
        return logits, bbox_reg, ctrness, top_feats, bbox_towers