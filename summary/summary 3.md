# 百度飞桨深度学习7日打卡营 课程总结3
![paddlepaddle](https://paddlepaddle-org-cn.cdn.bcebos.com/paddle-site-front/favicon-128.png  "百度logo")

[课程链接](https://aistudio.baidu.com/aistudio/course/introduce/6771)	`https://aistudio.baidu.com/aistudio/course/introduce/7073`  
[飞桨官网](https://www.paddlepaddle.org.cn/)	`https://www.paddlepaddle.org.cn/`   
[课程案例合集](https://aistudio.baidu.com/aistudio/projectdetail/1505799?channelType=0&channel=0)	`https://aistudio.baidu.com/aistudio/projectdetail/1505799?channelType=0&channel=0`

****
## 目录
* [图像分类](#图像分类)
* [基础知识](#基础知识)
* [『深度学习7日打卡营』 初识卷积神经网络](#深度学习7日打卡营-初识卷积神经网络)
* [作业二](#作业二)
    * [客观题](#客观题)
    * [代码实践](#代码实践)

# 课节2：十二生肖分类
## 图像分类
* 二分类、多分类、多标签

* 影响因素：
    * 视角变化
    * 大小变化
    * 形变
    * 遮挡
    * 光照条件
    * 背景干扰
    * 类内差异

## LeNet-5
 * 卷积神经网络（CNN）的开山鼻祖
 * 1989年提出
 * 定义了CNN的基本结构：卷积层、池化层、全连接层。
 
## 基础知识
[基础知识](https://aistudio.baidu.com/aistudio/projectdetail/1507732)：https://aistudio.baidu.com/aistudio/projectdetail/1507732

## 『深度学习7日打卡营』 初识卷积神经网络

### 问题定义

图像分类，使用LeNet-5网络完成手写数字识别图片的分类。

```python
import paddle
import numpy as np
import matplotlib.pyplot as plt

paddle.__version__
'2.0.0'
```
### 数据准备
继续应用框架中封装好的手写数字识别数据集。

2.1 数据集加载和预处理
```python
# 数据预处理
import paddle.vision.transforms as T

# 数据预处理，TODO：找一下提出的原论文看一下
transform = T.Normalize(mean=[127.5], std=[127.5])

# 训练数据集
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)

# 验证数据集
eval_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)

print('训练样本量：{}，测试样本量：{}'.format(len(train_dataset), len(eval_dataset)))
```
2.2 数据查看
```python
print('图片：')
print(type(train_dataset[0][0]))
print(train_dataset[0][0])
print('标签：')
print(type(train_dataset[0][1]))
print(train_dataset[0][1])

# 可视化展示
plt.figure()
plt.imshow(train_dataset[0][0].reshape([28,28]), cmap=plt.cm.binary)
plt.show()
```
### 模型选择和开发
我们选用LeNet-5网络结构。

LeNet-5模型源于论文“LeCun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition[J]. Proceedings of the IEEE, 1998, 86(11): 2278-2324.”，

论文地址：https://ieeexplore.ieee.org/document/726791

3.1 网络结构定义
3.1.1 模型介绍
![](materials/LeNet-5模型.png)

3.1.2 网络结构代码实现1
理解原论文进行的复现实现，因为模型论文出现在1998年，很多技术还不是最新。

```python
import paddle.nn as nn

network = nn.Sequential(
    nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),  # C1 卷积层
    nn.Tanh(),
    nn.AvgPool2D(kernel_size=2, stride=2),  # S2 平局池化层
    nn.Sigmoid(),   # Sigmoid激活函数
    nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),  # C3 卷积层
    nn.Tanh(),
    nn.AvgPool2D(kernel_size=2, stride=2),  # S4 平均池化层
    nn.Sigmoid(),  # Sigmoid激活函数
    nn.Conv2D(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0), # C5 卷积层
    nn.Tanh(),
    nn.Flatten(),
    nn.Linear(in_features=120, out_features=84), # F6 全连接层
    nn.Tanh(),
    nn.Linear(in_features=84, out_features=10) # OUTPUT 全连接层
)
```
模型可视化
```python
paddle.summary(network, (1, 1, 32, 32))
```
3.1.3 网络结构代码实现2
应用了截止到现在为止新的技术点实现后的模型，用Sequential写法。

```python
import paddle.nn as nn

network_2 = nn.Sequential(
    nn.Conv2D(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2D(kernel_size=2, stride=2),
    nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
    nn.ReLU(),
    nn.MaxPool2D(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(in_features=400, out_features=120),  # 400 = 5x5x16，输入形状为32x32， 输入形状为28x28时调整为256
    nn.Linear(in_features=120, out_features=84),
    nn.Linear(in_features=84, out_features=10)
)
```
模型可视化
```python
paddle.summary(network_2, (1, 1, 28, 28))
```
3.1.4 网络结构代码实现3
应用了截止到现在为止新的技术点实现后的模型，模型结构和【网络结构代码实现2】一致，用Sub Class写法。
```python
class LeNet(nn.Layer):
    """
    继承paddle.nn.Layer定义网络结构
    """

    def __init__(self, num_classes=10):
        """
        初始化函数
        """
        super(LeNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1),  # 第一层卷积
            nn.ReLU(), # 激活函数
            nn.MaxPool2D(kernel_size=2, stride=2),  # 最大池化，下采样
            nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0), # 第二层卷积
            nn.ReLU(), # 激活函数
            nn.MaxPool2D(kernel_size=2, stride=2) # 最大池化，下采样
        )

        self.fc = nn.Sequential(
            nn.Linear(400, 120),  # 全连接
            nn.Linear(120, 84),   # 全连接
            nn.Linear(84, num_classes) # 输出层
        )

    def forward(self, inputs):
        """
        前向计算
        """
        y = self.features(inputs)
        y = paddle.flatten(y, 1)
        out = self.fc(y)

        return out

network_3 = LeNet()
```
模型可视化
```python
paddle.summary(network_3, (1, 1, 28, 28))
```
3.1.4 网络结构代码实现4
直接应用高层API中封装好的LeNet网络接口。
```python
network_4 = paddle.vision.models.LeNet(num_classes=10)
```
模型可视化
通过summary接口来查看搭建的网络结构，查看输入和输出形状，以及需要训练的参数信息。
```python
paddle.summary(network_4, (1, 1, 28, 28))
```
### 模型训练和优化
模型配置  
* 优化器：SGD  
* 损失函数：交叉熵（cross entropy）  
* 评估指标：Accuracy  
```python
# 模型封装
model = paddle.Model(network_4)

# 模型配置
model.prepare(paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()), # 优化器
              paddle.nn.CrossEntropyLoss(), # 损失函数
              paddle.metric.Accuracy()) # 评估指标

# 启动全流程训练
model.fit(train_dataset,  # 训练数据集
          eval_dataset,   # 评估数据集
          epochs=5,       # 训练轮次
          batch_size=64,  # 单次计算数据样本量
          verbose=1)      # 日志展示形式
```
### 模型评估
5.1 模型评估
```python
result = model.evaluate(eval_dataset, verbose=1)

print(result)
```
5.2 模型预测
5.2.1 批量预测
使用model.predict接口来完成对大量数据集的批量预测。
```python
# 进行预测操作
result = model.predict(eval_dataset)
Predict begin...
step 10000/10000 [==============================] - 2ms/step        
Predict samples: 10000
In [28]
# 定义画图方法
def show_img(img, predict):
    plt.figure()
    plt.title('predict: {}'.format(predict))
    plt.imshow(img.reshape([28, 28]), cmap=plt.cm.binary)
    plt.show()

# 抽样展示
indexs = [2, 15, 38, 211]

for idx in indexs:
    show_img(eval_dataset[idx][0], np.argmax(result[0][idx]))
```
### 部署上线
6.1 保存模型
```python
model.save('finetuning/mnist')
```
6.2 继续调优训练
```python
from paddle.static import InputSpec

network = paddle.vision.models.LeNet(num_classes=10)
# 模型封装，为了后面保存预测模型，这里传入了inputs参数
model_2 = paddle.Model(network, inputs=[InputSpec(shape=[-1, 1, 28, 28], dtype='float32', name='image')])

# 加载之前保存的阶段训练模型
model_2.load('finetuning/mnist')

# 模型配置
model_2.prepare(paddle.optimizer.Adam(learning_rate=0.0001, parameters=network.parameters()),  # 优化器
                paddle.nn.CrossEntropyLoss(), # 损失函数
                paddle.metric.Accuracy()) # 评估函数

# 模型全流程训练
model_2.fit(train_dataset,  # 训练数据集
            eval_dataset,   # 评估数据集
            epochs=2,       # 训练轮次
            batch_size=64,  # 单次计算数据样本量
            verbose=1)      # 日志展示形式
```
6.3 保存预测模型
```python
# 保存用于后续推理部署的模型
model_2.save('infer/mnist', training=False)
```

## 作业二
### 客观题
一. 单选题（共7题，共70分）

1. 今晚代码实践属于以下哪类任务？（10分）

>A.图像分割

>B.图像分类

>C.图像检测

>D.图像生成

>答案：B

2. 用摄像头对垃圾进行分类是图像分类任务吗？（10分）

>A.是

>B.不是

>答案：A

3. 猫狗识别属于以下哪种任务？（10分）

>A.多分类任务

>B.单分类任务

>答案：B

4. paddle.nn.Conv2D接口是用来搭建卷积神经网络中的哪个部分？（10分）

>A.池化层

>B.激活函数

>C.卷积层

>D.归一化层

>答案：C

5. 卷积神经网络的开山鼻祖是哪个？（10分）

>A.ResNet

>B.Inception V3

>C.LeNet

>D.VGG

>答案：C

6. 经过15个[3, 3, 3]卷积核操作过后会得到多少通道的特征图？（10分）

>A.18

>B.17

>C.16

>D.15

>答案：D

7. ResNet中增加了一个什么特殊模块？（10分）

>A.差异模块

>B.残差模块

>C.残疾模块

>答案：B

二. 多选题（共3题，共30分）

1. 有哪些问题会对用于提取特征的模型造成考验？（10分）

>A.视角变化

>B.光照条件

>C.背景干扰

>D.遮挡

>答案：A,B,C,D

2. 池化层包含哪几种？（10分）

>A.平均池化

>B.最大池化

>答案：A,B

3. 残差模块的意义包含哪些？（10分）

>A.防止梯度爆炸

>B.防止梯度消失

>答案：A,B

### 代码实践

『深度学习7日打卡营』12生肖分类

#### 问题定义
十二生肖分类的本质是图像分类任务，我们采用CNN网络结构进行相关实践。

#### 数据准备
2.1 解压缩数据集
我们将网上获取的数据集以压缩包的方式上传到aistudio数据集中，并加载到我们的项目内。

在使用之前我们进行数据集压缩包的一个解压。

```python
!unzip -q -o data/data68755/signs.zip
```
2.2 数据标注
我们先看一下解压缩后的数据集长成什么样子。
```
.
├── test
│   ├── dog
│   ├── dragon
│   ├── goat
│   ├── horse
│   ├── monkey
│   ├── ox
│   ├── pig
│   ├── rabbit
│   ├── ratt
│   ├── rooster
│   ├── snake
│   └── tiger
├── train
│   ├── dog
│   ├── dragon
│   ├── goat
│   ├── horse
│   ├── monkey
│   ├── ox
│   ├── pig
│   ├── rabbit
│   ├── ratt
│   ├── rooster
│   ├── snake
│   └── tiger
└── valid
    ├── dog
    ├── dragon
    ├── goat
    ├── horse
    ├── monkey
    ├── ox
    ├── pig
    ├── rabbit
    ├── ratt
    ├── rooster
    ├── snake
    └── tiger
```
数据集分为train、valid、test三个文件夹，每个文件夹内包含12个分类文件夹，每个分类文件夹内是具体的样本图片。

我们对这些样本进行一个标注处理，最终生成train.txt/valid.txt/test.txt三个数据标注文件。

```python
import io
import os
from PIL import Image
from config import get

# 数据集根目录
DATA_ROOT = 'signs'

# 标签List
LABEL_MAP = get('LABEL_MAP')

# 标注生成函数
def generate_annotation(mode):
    # 建立标注文件
    with open('{}/{}.txt'.format(DATA_ROOT, mode), 'w') as f:
        # 对应每个用途的数据文件夹，train/valid/test
        train_dir = '{}/{}'.format(DATA_ROOT, mode)

        # 遍历文件夹，获取里面的分类文件夹
        for path in os.listdir(train_dir):
            # 标签对应的数字索引，实际标注的时候直接使用数字索引
            label_index = LABEL_MAP.index(path)

            # 图像样本所在的路径
            image_path = '{}/{}'.format(train_dir, path)

            # 遍历所有图像
            for image in os.listdir(image_path):
                # 图像完整路径和名称
                image_file = '{}/{}'.format(image_path, image)
                
                try:
                    # 验证图片格式是否ok
                    with open(image_file, 'rb') as f_img:
                        image = Image.open(io.BytesIO(f_img.read()))
                        image.load()
                        
                        if image.mode == 'RGB':
                            f.write('{}\t{}\n'.format(image_file, label_index))
                except:
                    continue


generate_annotation('train')  # 生成训练集标注文件
generate_annotation('valid')  # 生成验证集标注文件
generate_annotation('test')   # 生成测试集标注文件
```
2.3 数据集定义
接下来我们使用标注好的文件进行数据集类的定义，方便后续模型训练使用。

2.3.1 导入相关库
```python
import paddle
import numpy as np
from config import get

paddle.__version__
```
2.3.2 导入数据集的定义实现
我们数据集的代码实现是在dataset.py中。

```python
from dataset import ZodiacDataset
```
2.3.3 实例化数据集类
根据所使用的数据集需求实例化数据集类，并查看总样本量。

```python
train_dataset = ZodiacDataset(mode='train')
valid_dataset = ZodiacDataset(mode='valid')

print('训练数据集：{}张；验证数据集：{}张'.format(len(train_dataset), len(valid_dataset)))
```
#### 模型选择和开发
3.1 网络构建
本次我们使用ResNet50网络来完成我们的案例实践。
```python
network = paddle.vision.models.resnet50(num_classes=get('num_classes'), pretrained=True)
```
```python
model = paddle.Model(network)
model.summary((-1, ) + tuple(get('image_shape')))
```
#### 模型训练和优化
```python
EPOCHS = get('epochs')
BATCH_SIZE = get('batch_size')

def create_optim(parameters):
    step_each_epoch = get('total_images') // get('batch_size')
    lr = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=get('LEARNING_RATE.params.lr'),
                                                  T_max=step_each_epoch * EPOCHS)

    return paddle.optimizer.Momentum(learning_rate=lr,
                                     parameters=parameters,
                                     weight_decay=paddle.regularizer.L2Decay(get('OPTIMIZER.regularizer.factor')))


# 模型训练配置
model.prepare(create_optim(network.parameters()),  # 优化器
              paddle.nn.CrossEntropyLoss(),        # 损失函数
              paddle.metric.Accuracy(topk=(1, 5))) # 评估指标

# 训练可视化VisualDL工具的回调函数
visualdl = paddle.callbacks.VisualDL(log_dir='visualdl_log')

# 启动模型全流程训练
model.fit(train_dataset,            # 训练数据集
          valid_dataset,            # 评估数据集
          epochs=EPOCHS,            # 总的训练轮次
          batch_size=BATCH_SIZE,    # 批次计算的样本量大小
          shuffle=True,             # 是否打乱样本集
          verbose=1,                # 日志展示格式
          save_dir='./chk_points/', # 分阶段的训练模型存储路径
          callbacks=[visualdl])     # 回调函数使用
```
#### 模型存储
将我们训练得到的模型进行保存，以便后续评估和测试使用。

```python
model.save(get('model_save_dir'))
```
#### 模型评估和测试
5.1 批量预测测试
5.1.1 测试数据集
```python
predict_dataset = ZodiacDataset(mode='test')
print('测试数据集样本量：{}'.format(len(predict_dataset)))
```
5.1.2 执行预测
```python
from paddle.static import InputSpec

# 网络结构示例化
network = paddle.vision.models.resnet50(num_classes=get('num_classes'))

# 模型封装
model_2 = paddle.Model(network, inputs=[InputSpec(shape=[-1] + get('image_shape'), dtype='float32', name='image')])

# 训练好的模型加载
model_2.load(get('model_save_dir'))

# 模型配置
model_2.prepare()

# 执行预测
result = model_2.predict(predict_dataset)
```
```python
# 样本映射
LABEL_MAP = get('LABEL_MAP')

# 随机取样本展示
indexs = [2, 38, 56, 92, 100, 303]

for idx in indexs:
    predict_label = np.argmax(result[0][idx])
    real_label = predict_dataset[idx][1]

    print('样本ID：{}, 真实标签：{}, 预测值：{}'.format(idx, LABEL_MAP[real_label], LABEL_MAP[predict_label]))
```
#### 模型部署
```python
model_2.save('infer/zodiac', training=False)
```
