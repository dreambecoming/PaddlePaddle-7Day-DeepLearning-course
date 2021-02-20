# 百度飞桨深度学习7日打卡营 课程总结1
![paddlepaddle](https://paddlepaddle-org-cn.cdn.bcebos.com/paddle-site-front/favicon-128.png  "百度logo")

[课程链接](https://aistudio.baidu.com/aistudio/course/introduce/6771)	`https://aistudio.baidu.com/aistudio/course/introduce/7073`  
[飞桨官网](https://www.paddlepaddle.org.cn/)	`https://www.paddlepaddle.org.cn/`   
[课程案例合集](https://aistudio.baidu.com/aistudio/projectdetail/1505799?channelType=0&channel=0)	`https://aistudio.baidu.com/aistudio/projectdetail/1505799?channelType=0&channel=0`

****
## 目录
* [深度学习基本概念](#深度学习基本概念)
* [数学基础知识](#数学基础知识)
* [Python基础](#Python基础)
* [预习作业](#预习作业)
    * [客观题](#客观题)
    * [安装飞桨](#安装飞桨)

# 课节0：解决深度学习任务的必备知识及工具
（课前预习内容）
## 深度学习基本概念
* Turing Testing (图灵测试)。
图灵测试是人工智能是否真正能够成功的一个标准，图灵在1950年的论文《机器会思考吗》中提出了图灵测试的概念。即把一个人和一台计算机分别放在两个隔离的房间中，房间外的一个人同时询问人和计算机相同的问题，如果房间外的人无法分别哪个是人，哪个是计算机，就能够说明计算机具有人工智能。
* 医学上的发现  
人的视觉系统处理信息是分级的。高层的特征是低层特征的组合，从低层到高层的特征表达越来越抽象和概念化，也即越来越能表现语义或者意图。  
边缘特征 —–> 基本形状和目标的局部特征——>整个目标
* 深度学习，恰恰就是通过组合低层特征形成更加抽象的高层特征（或属性类别）。
	低层次特征 - - - - (组合) - - ->抽象的高层特征
* 机器学习
	机器学习就是通过算法，使得机器能从大量历史数据中学习规律，从而对新的样本做智能识别或对未来做预测。
* 人工智能
机器学习是实现人工智能的一种手段，深度学习是一种机器学习方法

    * 弱人工智能，也被称为狭义人工智能，是一种为特定的任务而设计和训练的人工智能系统。弱人工智能的形式之一是虚拟个人助理，比如苹果公司的Siri。
    * 强人工智能，又称人工通用智能，是一种具有人类普遍认知能力的人工智能系统。当计算机遇到不熟悉的任务时，它具有足够的智能去寻找解决方案。
* 监督式学习与非监督式学习
监督式学习需要使用有输入和预期输出标记的数据集。
非监督式学习是利用既不分类也不标记的信息进行机器学习，并允许算法在没有指导的情况下对这些信息进行操作。
* 神经网络
神经网络是一组粗略模仿人类大脑，用于模式识别的算法。
神经元之间的每个连接都有一个权重。这个权重表示输入值的重要性。
每个神经元都有一个激活函数。它主要是一个根据输入传递输出的函数。  
神经元分为三种不同类型的层次：
    * 输入层：接收输入数据。在我们的例子中，输入层有四个神经元:出发站、目的地站、出发日期和巴士公司。输入层会将输入数据传递给第一个隐藏层。
    * 隐藏层：对输入数据进行数学计算。创建神经网络的挑战之一是决定隐藏层的数量，以及每一层中的神经元的数量。
    * 输出层：神经元的最后一层，主要作用是为此程序产生给定的输出。

### 示例1. 手写数字识别

将手写数字的灰度图像（28 像素×28 像素）划分到 10 个类别 中（0-9）。

MNIST 数据集是机器学习领域的一个经典数据集，其历史几乎和这个领域一样长，而且已被人们深入研究。这个数据集包含 60000 张训练图像和 10000 张测试图像，由美国国家标准与技术研究院（ National Institute of Standards and Technology，即 NIST 中的 NIST ）在20世纪80年代收集得到。

#### 步骤1：准备数据
1.MINIST数据集包含60000个训练集和10000测试数据集。分为图片和标签，图片是28*28的像素矩阵，标签为0~9共10个数字。
2.使用飞桨内置数据集 paddle.vision,datasets.MNIST 定义 MNIST 数据集的 train_dataset 和 test_dataset 。
3.使用 Normalize 接口对图片进行归一化。

```python
import paddle
from paddle.vision.transforms import Normalize

transform = Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')
# 使用transform对数据集做归一化
print('download training data and load training data')
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
print('load finished')
```
输出：
```
download training data and load training data
load finished
```
取一条数据，观察一下mnist数据集
```python
import numpy as np
import matplotlib.pyplot as plt
train_data0, train_label_0 = train_dataset[0][0],train_dataset[0][1]
train_data0 = train_data0.reshape([28,28])
plt.figure(figsize=(2,2))
plt.imshow(train_data0, cmap=plt.cm.binary)
print('train_data0 label is: ' + str(train_label_0))
```
输出：
```
train_data0 label is: [5]
 
<Figure size 144x144 with 1 Axes>
```
#### 步骤2：配置网络
定义一个简单的多层感知器，一共有三层，两个大小为100的隐层和一个大小为10的输出层。MNIST数据集是手写0到9的灰度图像，类别有10个，所以最后的输出大小是10。最后输出层的激活函数是Softmax，所以最后的输出层相当于一个分类器。加上一个输入层的话，多层感知器的结构是：输入层-->>隐层-->>隐层-->>输出层。
```python
# 定义多层感知机
class MultilayerPerceptron(paddle.nn.Layer):
    def __init__(self, in_features):
        super(MultilayerPerceptron, self).__init__()
        # 形状变换，将数据形状从 [] 变为 []
        self.flatten = paddle.nn.Flatten()
        # 第一个全连接层
        self.linear1 = paddle.nn.Linear(in_features=in_features, out_features=100)
        # 使用ReLU激活函数
        self.act1 = paddle.nn.ReLU()
        # 第二个全连接层
        self.linear2 = paddle.nn.Linear(in_features=100, out_features=100)
        # 使用ReLU激活函数
        self.act2 = paddle.nn.ReLU()
        # 第三个全连接层
        self.linear3 = paddle.nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        # x = x.reshape((-1, 1, 28, 28))
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.linear3(x)
        return x
```

```python
# 使用 paddle.Model 封装 MultilayerPerceptron
model = paddle.Model(MultilayerPerceptron(in_features=784))
# 使用 summary 打印模型结构
model.summary((-1, 1, 28, 28))
```
输出：
```python
---------------------------------------------------------------------------
 Layer (type)       Input Shape          Output Shape         Param #    
===========================================================================
   Flatten-8      [[1, 1, 28, 28]]         [1, 784]              0       
   Linear-16         [[1, 784]]            [1, 100]           78,500     
    ReLU-11          [[1, 100]]            [1, 100]              0       
   Linear-17         [[1, 100]]            [1, 100]           10,100     
    ReLU-12          [[1, 100]]            [1, 100]              0       
   Linear-18         [[1, 100]]            [1, 10]             1,010     
===========================================================================
Total params: 89,610
Trainable params: 89,610
Non-trainable params: 0
---------------------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.01
Params size (MB): 0.34
Estimated Total Size (MB): 0.35
---------------------------------------------------------------------------

{'total_params': 89610, 'trainable_params': 89610}
```
	接着是配置模型，在这一步，我们需要指定模型训练时所使用的优化算法与损失函数，此外，这里我们也可以定义计算精度相关的API。
```python
# 配置模型
model.prepare(paddle.optimizer.Adam(parameters=model.parameters()),  # 使用Adam算法进行优化
              paddle.nn.CrossEntropyLoss(), # 使用CrossEntropyLoss 计算损失
              paddle.metric.Accuracy()) # 使用Accuracy 计算精度
```
#### 步骤3:模型训练
	使用飞桨高层API，可以很快的完成模型训练的部分，只需要在prepare配置好模型训练的相关算法后，调用fit接口，指定训练的数据集，训练的轮数以及数据batch_size，就可以完成模型的训练。
```python
# 开始模型训练
model.fit(train_dataset, # 设置训练数据集
          epochs=5,      # 设置训练轮数
          batch_size=64, # 设置 batch_size
          verbose=1)     # 设置日志打印格式
```
输出：
```python
The loss value printed in the log is the current step, and the metric is the average value of previous step.
Epoch 1/5
step 938/938 [==============================] - loss: 0.2002 - acc: 0.9759 - 8ms/step        
Epoch 2/5
step 938/938 [==============================] - loss: 0.0052 - acc: 0.9786 - 8ms/step         
Epoch 3/5
step 938/938 [==============================] - loss: 0.0147 - acc: 0.9798 - 8ms/step        
Epoch 4/5
step 938/938 [==============================] - loss: 0.0449 - acc: 0.9807 - 8ms/step         
Epoch 5/5
step 938/938 [==============================] - loss: 0.1223 - acc: 0.9840 - 9ms/step         
```
#### 步骤4: 模型评估
	使用飞桨高层API完成模型评估也非常的简单，只需要调用 evaluate 接口并传入验证集即可。这里我们使用测试集作为验证集。
```python
model.evaluate(test_dataset, verbose=1)
```
```python
Eval begin...
The loss value printed in the log is the current batch, and the metric is the average value of previous step.
step 10000/10000 [==============================] - loss: 0.0000e+00 - acc: 0.9743 - 2ms/step         
Eval samples: 10000
{'loss': [0.0], 'acc': 0.9743}
```
#### 步骤5:模型预测
使用飞桨高层API完成模型预测也非常的简单，只需要调用 predict 接口并传入测试集即可。
```python
results = model.predict(test_dataset)
```
输出：
```python
Predict begin...
step 10000/10000 [==============================] - 2ms/step        
Predict samples: 10000
```
```python
# 获取概率最大的label
lab = np.argsort(results)                               #argsort函数返回的是result数组值从小到大的索引值
# print(lab)
```
输出：
```python
print("该图片的预测结果的label为: %d" % lab[0][0][-1][0])  #-1代表读取数组中倒数第一列  
该图片的预测结果的label为: 6
```

## 数学基础知识
高等数学、线性代数、概率论和数理统计  
详见：https://aistudio.baidu.com/aistudio/projectdetail/1497905 或 ![math.md](summary/math.md)


## Python基础
Python是一门解释型、面向对象的高级编程语言。


## 预习作业

### 客观题

一. 单选题（共1题，共20分）

1. 机器学习，深度学习，人工智能 三个概念的关系是？（20分）

>A.1>2>3

>B.1<2<3

>C.2<1<3

>D.1=2=3

>答案：C

二. 多选题（共1题，共20分）

1. 使用飞桨的建模程序包括哪几个部分？（20分）

>A.数据处理

>B.模型设置

>C.训练配置

>D.训练过程

>E.保存模型

>答案：A,B,C,D,E

三. 判断题（共3题，共60分）

1. 我已经掌握了使用Github查找知识的方法？https://guides.github.com/activities/hello-world/（20分）

>A.对

>B.错


>答案：A

2. 此链接为开源社区Github中飞桨高层API板块？https://github.com/PaddlePaddle/hapi（20分）

>A.对

>B.错

>答案：A

3. 我已经掌握了使用飞桨官方文档查找知识的方法？https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/index_cn.html（20分）

>A.对

>B.错

>答案：A

### 安装飞桨

作业请在本地完成，不是在AIStudio进行操作
作业：飞桨本地测试代码运行成功截图

`####请在下面cell中上传飞桨安装成功的截图####`

    飞桨安装文档：https://paddlepaddle.org.cn/install/quick
    
    提示：使用 python 进入python解释器，输入import paddle.fluid ，再输入 paddle.fluid.install_check.run_check()。 如果出现 Your Paddle Fluid is installed successfully!，说明您已成功安装。
    
    本地安装PaddlePaddle的常见错误：https://aistudio.baidu.com/aistudio/projectdetail/697227
    
    手把手教你 win10 安装Paddlepaddle-GPU：https://aistudio.baidu.com/aistudio/projectdetail/696822
    
   
![截图](/materials/screenshot.jpg)
