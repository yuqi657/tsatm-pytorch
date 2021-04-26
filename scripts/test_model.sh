###
 # @Author: your name
 # @Date: 2021-04-24 15:53:25
 # @LastEditTime: 2021-04-24 15:54:04
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /tsn-pytorch/test_model.sh
### 
python test_models.py streetdance245 RGB '/data7/lyq/data/streetdance245/streetdance245_val_frame.txt'  '../workdirs/tsn_1_rgb/streetdance245_bninception__rgb_model_best.pth.tar' \
    --arch resnet50 --workers 2 --test_segments 1
