# pytorch converter for ncnn and caffe

only support resnet and sequential net.

uses forward graph, because pytorch1.0 backward function usually gives NoneType! 

for models trained with pytorch version lower than 1.0, using the original converter may be better.


