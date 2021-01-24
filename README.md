## Requirement

1. Setting up detectron2

## Guide

### Training

1. Prepare a COCO dataset format json file (Ex: total_text.json)
2. The dataset should have annotations such as
   1. width
   2. height
   3. segmentation
   4. image_id
3. Download the dataset and create directory to place the training images (Ex: Images/Train)
4. Run `python train.py` to train the model

### Inference

1. For inference

`python inference.py --input <IMAGEDIR> --output <OUTPUTDIR> --weights-file <WEIGHTFILE> --config-file <CONFIG_FILE>`
