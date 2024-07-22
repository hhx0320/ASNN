# ASNN
Adaptive Spiking Neural Networks with Hybrid Coding


此代码使用LeNet，来展示具体细节，在transformer和resnet使用相同的修改即可


This code uses LeNet to display specific details, and the same modifications can be used in transformer and ResNet

在进行时间编码时，可以先使用layernorm与sigmoid将输入数据分布进行转换，
再使用util中的fenge函数生成用于划分输入数据的数列。

When performing time encoding, the input data distribution can be transformed using layernorm and sigmoid first,
Use the 'fenge' function in 'util' to generate a sequence for dividing the input data.
