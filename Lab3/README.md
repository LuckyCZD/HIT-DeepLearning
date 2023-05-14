对于VGG，如果要运行测试，请将Kaggle上下载的数据集的`test`文件夹分别复制到文件夹`VGG11`和`RepVGG`下。其中：

+ VGG11：可使用main函数进行测试

+ RepVGG：可使用test.py文件进行测试

对于ResNet，如果要运行测试，请将Kaggle上下载的数据集的`test`文件夹复制到`ResNet/plant-seedlings-classification`文件夹下。

+ ResNet：可使用main函数进行测试。如果要训练，则需要同时复制数据集的`train`文件夹至同上路径

​		测试结果保留对应文件夹的predicted.csv文件中

​		VGG11和ResNet的最佳模型过大，可在对应文件夹中的txt文件中的百度网盘链接提取，运行测试时放至对应路径即可。