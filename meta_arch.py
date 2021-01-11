from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling import ProposalNetwork

@META_ARCH_REGISTRY.register()
class OneStageDetector(ProposalNetwork):
    """
    Same as :class:`detectron2.modeling.ProposalNetwork`.
    Uses "instances" as the return key instead of using "proposal".
    """
    def forward(self, batched_inputs):
        if self.training:
            return super().forward(batched_inputs)
        processed_results = super().forward(batched_inputs)
        processed_results = [{"instances": r["proposals"]} for r in processed_results]
        return processed_results


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