# Update
## 23.06.27 2x model train
### configs.py file : train_pipeline -> dict of type 'FixedCrop' -> crop_size change from (256, 256) to (128, 128)
### .local/lib/python3.8/site-packages/mmedit/models/restorers/real_basicvsr.py:
- def train_step : gt_clean interpolate parameter change(scale_factor, view factor 0.5)
### .local/lib/python3.8/site-packages/mmedit/models/backbones/sr_backbones/basicvsr_net.py:
- def __init__ : self.upsample2 layer pixelshuffle scale_factor change from 2 to 1
- def __init__ : self.img_upsample layer Upsample scale_factor from 4 to 2
