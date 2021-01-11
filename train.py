from config.default import _C
from util.train import Trainer

cfg = _C.clone()
cfg.merge_from_file("./base.yaml")
cfg.DATASETS.TRAIN = ("text_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.BASE_LR = 0.002  # pick a good LearningRate
cfg.SOLVER.MAX_ITER = 20000  #No. of iterations
cfg.SOLVER.IMS_PER_BATCH = 6
cfg.MODEL.FCOS.NUM_CLASSES = 1 # Only text class
cfg.INPUT.HFLIP_TRAIN = False
cfg.MODEL.FCOS.YIELD_PROPOSAL = False
cfg.MODEL.META_ARCHITECTURE = "OneStageDetector"
cfg.freeze()

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = Trainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()