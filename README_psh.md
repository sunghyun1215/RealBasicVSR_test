# Update
## 23.06.27 2x model train
### configs.py file : train_pipeline -> dict of type 'FixedCrop' -> crop_size change from (256, 256) to (128, 128)
### .local/lib/python3.8/site-packages/mmedit/models/restorers/real_basicvsr.py:
- def train_step : gt_clean interpolate parameter change(scale_factor 0.25-> 0.5, view factor /4->/2)
### .local/lib/python3.8/site-packages/mmedit/models/backbones/sr_backbones/basicvsr_net.py:
- def __init__ : self.upsample2 layer pixelshuffle scale_factor change from 2 to 1
- def __init__ : self.img_upsample layer Upsample scale_factor from 4 to 2

## 23.06.28 2x model eval
### Validation data replacement
- BIx4 --> BIx2 (add bicubic-downsampling image-set for GT )

## 23.08.22 realtime inference
### 웹캠 연결 테스트 코드 개발 시작
- 호스트 pc에서는 mmedit import 시 cv2.imshow 가 실행이 안돼서 docker 가상 환경에서 테스트 중
```
xhost +
sudo nvidia-docker run -it --gpus all --privileged -v /dev/video0:/dev/video0 --net=host --ipc=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $XAUTHORITY:/tmp/.XAuthority -e XAUTHORITY=/tmp/.XAuthority --env="QT_X11_NO_MITSHM=1" --name="vsr" supervisely/mmcv-full:1.3.14 /bin/bash
```
- docker 환경 내부 mmedit 위치 : /usr/local/lib/python3.8/dist-packages/mmedit
- realtime_inference.py --> 되긴 하는데 320x240 을 640x480 변환 시 5Hz 정도 출력 됨

# Requirement
## mmcv 설치 시 python 버전 3.9 이상으로 설치해야하는 것 같음
