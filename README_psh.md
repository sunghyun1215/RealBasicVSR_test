# RealBasicVSR (CVPR 2022)

\[[Paper](https://arxiv.org/pdf/2111.12704.pdf)\]

This is the official repository of "Investigating Tradeoffs in Real-World Video Super-Resolution, arXiv". This repository contains *codes*, *colab*, *video demos* of our work.

**Authors**: [Kelvin C.K. Chan](https://ckkelvinchan.github.io/), [Shangchen Zhou](https://shangchenzhou.com/), [Xiangyu Xu](https://sites.google.com/view/xiangyuxu), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/), *Nanyang Technological University*

**Acknowedgement**: Our work is built upon [MMEditing](https://github.com/open-mmlab/mmediting). The code will also appear in MMEditing soon. Please follow and star this repository and MMEditing!

**Feel free to ask questions. I am currently working on some other stuff but will try my best to reply. If you are also interested in [BasicVSR++](https://github.com/ckkelvinchan/BasicVSR_PlusPlus), which is also accepted to CVPR 2022, please don't hesitate to star!** 




## News
- 23.06.27 2x model train
  -- configs.py file : train_pipeline -> dict of type 'FixedCrop' -> crop_size change from (256, 256) to (128, 128)
  -- .local/lib/python3.8/site-packages/mmedit/models/restorers/real_basicvsr.py:
     --- def train_step : gt_clean interpolate parameter change(scale_factor, view factor 0.5)
  -- .local/lib/python3.8/site-packages/mmedit/models/backbones/sr_backbones/basicvsr_net.py:
     --- def __init__ : self.upsample2 layer pixelshuffle scale_factor change from 2 to 1
