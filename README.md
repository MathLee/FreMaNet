# FreMaNet
[TGRS2026] [FreMaNet] Lightweight ORSI Salient Object Detection via Frequency and Mutual Assistance Attention [Homepage](https://mathlee.github.io/)

# Network Architecture
   <div align=center>
   <img src="https://github.com/MathLee/FreMaNet/blob/main/image/FreMaNet.png">
   </div>
   
   
# Requirements
   python 3.8 + pytorch 1.13.1


# Saliency maps
   We provide saliency maps of [our FreMaNet](https://github.com/MathLee/FreMaNet/tree/main/SaliencyMap), [lightweight methods](https://pan.baidu.com/s/1r_4FeeOJ4h-klBshg89oCA) (code: frem), and [normal-size methods](https://pan.baidu.com/s/13FVHe5kLilEKtBvC-ZwBxw) (code: frem) on the ORSSD, EORSSD, and ORSI-4199 datasets.


      
   ![Image](https://github.com/MathLee/FreMaNet/blob/main/image/table.png)

   
# Training
   We use data_aug.m for data augmentation. 
   
   Modify paths of datasets, then run train_FreMaNet.py.

Note: Our main model is under './model/GeleNet_models.py'. Our code is built on GeleNet. So in this code, GeleNet refers to our FreMaNet.



# Pre-trained model and testing
1. We provide the pre-trained models in './models/'.

2. Modify paths of pre-trained models and datasets.

3. Run test_FreMaNet.py.

   
# Evaluation Tool
   You can use the [evaluation tool (MATLAB version)](https://github.com/MathLee/MatlabEvaluationTools) to evaluate the above saliency maps.


# [ORSI-SOD_Summary](https://github.com/MathLee/ORSI-SOD_Summary)
   
# Citation
        @ARTICLE{Li_2026_SeaNet,
                author = {Gongyang Li and Shixiang Shi and Yong Wu and Weisi Lin and Zhen Bai},
                title = {Lightweight ORSI Salient Object Detection via Frequency and Mutual Assistance Attention},
                journal = {IEEE Transactions on Geoscience and Remote Sensing},
                volume = {},
                year = {2026},
                }
                
                
If you encounter any problems with the code, want to report bugs, etc.

Please contact me at lllmiemie@163.com or ligongyang@shu.edu.cn.


