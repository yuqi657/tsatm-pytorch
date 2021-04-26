###
 # @Author: your name
 # @Date: 2021-04-24 15:52:50
 # @LastEditTime: 2021-04-24 15:53:12
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /tsn-pytorch/train_model.sh
### 
python main.py streetdance245 RGB '/data7/lyq/data/streetdance245/streetdance245_train_frame.txt' '/data7/lyq/data/streetdance245/streetdance245_val_frame.txt' \
   --arch resnet50 --num_segments 3 \
   --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 \
   -b 16 -j 8 --dropout 0.8 \
   --snapshot_pref streetdance245_bninception_
