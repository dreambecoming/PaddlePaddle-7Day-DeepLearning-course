# 百度飞桨深度学习7日打卡营 课程总结2
![paddlepaddle](https://paddlepaddle-org-cn.cdn.bcebos.com/paddle-site-front/favicon-128.png  "百度logo")

[课程链接](https://aistudio.baidu.com/aistudio/course/introduce/6771)	`https://aistudio.baidu.com/aistudio/course/introduce/7073`  
[飞桨官网](https://www.paddlepaddle.org.cn/)	`https://www.paddlepaddle.org.cn/`   

****
## 目录
* [课程介绍](#课程介绍)
* [人工智能](#人工智能)
* [机器学习](#机器学习)
* [百度飞桨介绍](#百度飞桨介绍)
* [深度学习案例：手写数字识别](#深度学习案例手写数字识别)
   * [问题定义](#问题定义)
   * [数据准备](#数据准备)
   * [模型选择和开发](#模型选择和开发)
   * [模型训练和调优](#模型训练和调优)
   * [模型评估测试](#模型评估测试)
   * [部署上线](#部署上线)
* [作业一](#作业一)
   

# 课节2：走进深度学习与飞桨高层API
（正式第一课）
## 课程介绍
《深度学习7日打卡营》
* 课程大纲  

    1. 深度学习入门：    深度学习简介；深度学习大致的原理；快速上手实践。  
    2. CV+NLP：  实践图像分类、人脸关键点检测、情感分析、文本生成。  
    3. 部署上手：    带你将模型部署到开发板，完成端到端的学习。  
* 课程收获  

    -深度学习相关的基础概念  
    -计算机视觉领域一些基础和算法  
    -自然语言处理领域一些基础和算法  
    -端到端的案例代码实践  
    -PaddlePaddle深度学习框架（高层APl）   
    -模型部署中的一种方法  
    -自己能应用PaddlePaddle高层API完整搭建并训练一个深度学习网络  
* 使用工具  

    -编程语言：Python  
    -深度学习框架：PaddlePaddle  
    -其他Python包：NumPy、Pillow  
    -深度学习算法：Linear、LeNet、ResNet、LSTM、Seq2Seq 等等  
* 课程形式  

    在线授课：作业实践：项目实践 = 2:1:1  

## 人工智能
* 人工智能是新一轮科技革命和产业变革核心驱动力量    

    * 第一次工业革命：机械技术  
    * 第二次工业革命：电气技术  
    * 第三次工业革命：信息技术  
    * 第四次工业革命：人工智能  

* 人工智能、机器学习和深度学习三者关系  

    * 人工智能：人类想要达成的目标。
    * 机器学习：达成人工智能的一个手段。
    * 深度学习：机器学习中的一种方法。

## 机器学习
* 机器学习是仿人的一套归纳和演绎过程  

  ![pic1](/materials/pic1.jpg)
  
* 机器学习算法构成的三要素  

    1. 假设空间：模型的假设或表示。  
    2. 优化目标：评价或损失函数（Loss）。  
    3. 寻解算法：优化/求解算法 。 
    
* 神经网络
   * 神经元：神经网络中的每个节点称为神经元，由加权和非线性变换（激活函数）组成。    
   * 神经网络：大量的神经元按照不同的层次排布，形成多层的结构连接起来，即称为神经网络。  
   * 前向计算和反向传播：网络的输出计算和参数更新。   
   
## 深度学习 
* 2010年后，深度学习的条件成熟：大数据时代到来、硬件的发展和成熟、算法优化。
* 推动人工智能进入工业大生产阶段，算法的通用性导致标准化、自动化和模块化的框架产生。实现了人工智能研发模式从手工作坊到工业化变革。
* 关键环节：数据、模型、训练和预测。
* 一般流程：  

   1. 问题定义：对现实问题进行分析。直接影响算法的选择、模型评估标准，投入的时间。  
   
   2. 数据准备：  
         > 数据范围定义：适配任务的所需信息。  
         > 数据获取：下载、清洗。  
         > 数据的预处理：预处理、增强。  
         > 数据集定义和切分：训练、评估、测试。    
         
   3. 模型选择和开发：对应问题选用合适的模型，寻找或编写对应的模型代码。 
    
   4. 模型训练和调优：使用数据集启动对模型的训练，围绕业务所需的模型目标进行模型调优。  
   
   5. 模型评估测试：对训练好的模型进行评估测试，验证模型是否达到业务需求。  
   
   6. 部署上线：模型存储、导出、推理服务部署线上系统对接，指标监控。  
   
## 百度飞桨介绍

* 飞桨 （PaddlePaddle，PArallel Distributed Deep Learning）2.0版本   
      闻说双飞桨，翩然下广津。——朱熹  
      取义为：快船。百度希望飞桨助推人工智能走的更快更远！
* 官网：https://www.paddlepaddle.org.cn/
* Github:	https://github.com/PaddlePaddle/Paddle


## 深度学习案例：手写数字识别

### 问题定义
   把灰度的，高为28，宽为28的手写的阿拉伯数字图像分类到0-9的类别当中  
   
   流程：  

   1. 问题定义：对现实问题进行分析。直接影响算法的选择、模型评估标准，投入的时间。  
   
   2. 数据准备：    
   
      数据集定义：paddle.io.Dataset、数据多进程异步加载：paddle.io.Dataloader  
      30+个计算机视觉领域数据预处理工具：paddle.vision.transforms.*  
      14种计算机视觉和文本处理领域常见数据集封装：paddle.vision.datasets/paddle.text.datasets    
         
      数据集名称：   MNIST数据集  
      官网：   http:/yann.lecun.com/exdb/mnist/  
      训练样本量：   60，000张  
      验证样本量：   10，000张  
      单个样本形状：  （28，28）  
      加载使用方式：   paddle.vision.datasets.MNIST  
      
      
   3. 模型选择和开发：
   
      12个计算机视觉领域常用模型封装：paddle.vision.models.*  
      网络结构可视化：model.summary() / paddle.summary()  
      
      飞桨框架：  
      * 方式1：Sequential   
      * 方式2：Subclass  
      * 方式3：内置网络  
    
   4. 模型训练和调优：
   
      模型封装和执行：model.prepare()、model.fit()  
      
      * Model封装
      * 指定优化器
      * 指定Loss计算方法
      * 指定评估指标
      * 按照训练的轮次和数据批次迭代训练
   
   5. 模型评估测试：
   
      模型评估：model.evaluate()  
      模型测试：model.predict()  
  
      * 基于验证样本对模型进行评估验证
      * 得到Loss和评价指标值
   
   6. 部署上线：模型存储、导出、推理服务部署线上系统对接，指标监控。  
   
      预测部署模型存储：model.save（path，training=False）  
   
      * 存储模型  
      * 使用预测引擎部署 →  PaddleSlim  →  Paddlelnference、PaddleLite、PaddleJs



### 数据准备
2.1 数据加载和预处理

导入相关库
```python
import paddle
import numpy as np
import matplotlib.pyplot as plt

paddle.__version__
```
输出：
```python
'2.0.0'
```
```python
import paddle.vision.transforms as T

# 数据的加载和预处理
transform = T.Normalize(mean=[127.5], std=[127.5])

# 训练数据集
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)

# 评估数据集
eval_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)

print('训练集样本量: {}，验证集样本量: {}'.format(len(train_dataset), len(eval_dataset)))
```
```python
训练集样本量: 60000，验证集样本量: 10000
```
2.2 数据集查看
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
输出：
```python
图片：
<class 'numpy.ndarray'>
[[[-1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -0.9764706  -0.85882354 -0.85882354
   -0.85882354 -0.01176471  0.06666667  0.37254903 -0.79607844
    0.3019608   1.          0.9372549  -0.00392157 -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -0.7647059  -0.7176471
   -0.2627451   0.20784314  0.33333334  0.9843137   0.9843137
    0.9843137   0.9843137   0.9843137   0.7647059   0.34901962
    0.9843137   0.8980392   0.5294118  -0.49803922 -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.         -1.
   -1.         -1.         -0.6156863   0.8666667   0.9843137
    0.9843137   0.9843137   0.9843137   0.9843137   0.9843137
    0.9843137   0.9843137   0.96862745 -0.27058825 -0.35686275
   -0.35686275 -0.56078434 -0.69411767 -1.         -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.         -1.
   -1.         -1.         -0.85882354  0.7176471   0.9843137
    0.9843137   0.9843137   0.9843137   0.9843137   0.5529412
    0.42745098  0.9372549   0.8901961  -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -0.37254903  0.22352941
   -0.16078432  0.9843137   0.9843137   0.60784316 -0.9137255
   -1.         -0.6627451   0.20784314 -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -0.8901961
   -0.99215686  0.20784314  0.9843137  -0.29411766 -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.          0.09019608  0.9843137   0.49019608 -0.9843137
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -0.9137255   0.49019608  0.9843137  -0.4509804
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -0.7254902   0.8901961   0.7647059
    0.25490198 -0.15294118 -0.99215686 -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -0.3647059   0.88235295
    0.9843137   0.9843137  -0.06666667 -0.8039216  -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -0.64705884
    0.45882353  0.9843137   0.9843137   0.1764706  -0.7882353
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -0.8745098  -0.27058825  0.9764706   0.9843137   0.46666667
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.          0.9529412   0.9843137   0.9529412
   -0.49803922 -1.         -1.         -1.         -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -0.6392157
    0.01960784  0.43529412  0.9843137   0.9843137   0.62352943
   -0.9843137  -1.         -1.         -1.         -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -0.69411767  0.16078432  0.79607844
    0.9843137   0.9843137   0.9843137   0.9607843   0.42745098
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -0.8117647  -0.10588235  0.73333335  0.9843137   0.9843137
    0.9843137   0.9843137   0.5764706  -0.3882353  -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -0.81960785 -0.48235294
    0.67058825  0.9843137   0.9843137   0.9843137   0.9843137
    0.5529412  -0.3647059  -0.9843137  -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.         -1.
   -1.         -0.85882354  0.34117648  0.7176471   0.9843137
    0.9843137   0.9843137   0.9843137   0.5294118  -0.37254903
   -0.92941177 -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.         -0.5686275
    0.34901962  0.77254903  0.9843137   0.9843137   0.9843137
    0.9843137   0.9137255   0.04313726 -0.9137255  -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.          0.06666667
    0.9843137   0.9843137   0.9843137   0.6627451   0.05882353
    0.03529412 -0.8745098  -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.        ]
  [-1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.         -1.         -1.
   -1.         -1.         -1.        ]]]
标签：
<class 'numpy.ndarray'>
[5]
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:2349: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
  if isinstance(obj, collections.Iterator):
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:2366: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
  return list(data) if isinstance(data, collections.MappingView) else data
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/numpy/lib/type_check.py:546: DeprecationWarning: np.asscalar(a) is deprecated since NumPy v1.16, use a.item() instead
  'a.item() instead', DeprecationWarning, stacklevel=1)

```
### 模型选择和开发
3.1 模型组网

```python
# 模型网络结构搭建
network = paddle.nn.Sequential(
    paddle.nn.Flatten(),           # 拉平，将 (28, 28) => (784)
    paddle.nn.Linear(784, 512),    # 隐层：线性变换层
    paddle.nn.ReLU(),              # 激活函数
    paddle.nn.Linear(512, 10)      # 输出层
)
```
3.2 模型网络结构可视化
```python
# 模型封装
model = paddle.Model(network)

# 模型可视化
model.summary((1, 28, 28))
```
输出：
```python
---------------------------------------------------------------------------
 Layer (type)       Input Shape          Output Shape         Param #    
===========================================================================
   Flatten-1       [[1, 28, 28]]           [1, 784]              0       
   Linear-1          [[1, 784]]            [1, 512]           401,920    
    ReLU-1           [[1, 512]]            [1, 512]              0       
   Linear-2          [[1, 512]]            [1, 10]             5,130     
===========================================================================
Total params: 407,050
Trainable params: 407,050
Non-trainable params: 0
---------------------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.01
Params size (MB): 1.55
Estimated Total Size (MB): 1.57
---------------------------------------------------------------------------

{'total_params': 407050, 'trainable_params': 407050}
```
### 模型训练和调优
```python
# 配置优化器、损失函数、评估指标
model.prepare(paddle.optimizer.Adam(learning_rate=0.001, parameters=network.parameters()),
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy())
              
# 启动模型全流程训练
model.fit(train_dataset,  # 训练数据集
          eval_dataset,   # 评估数据集
          epochs=5,       # 训练的总轮次
          batch_size=64,  # 训练使用的批大小
          verbose=1)      # 日志展示形式
```
输出：
```python
The loss value printed in the log is the current step, and the metric is the average value of previous step.
Epoch 1/5
step  20/938 [..............................] - loss: 0.6420 - acc: 0.5969 - ETA: 12s - 13ms/ste
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:77: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
  return (isinstance(seq, collections.Sequence) and

step  30/938 [..............................] - loss: 0.5265 - acc: 0.6609 - ETA: 11s - 13ms/step

step  40/938 [>.............................] - loss: 0.5089 - acc: 0.7086 - ETA: 10s - 12ms/step
step 938/938 [==============================] - loss: 0.2390 - acc: 0.9125 - 11ms/step         
Eval begin...
The loss value printed in the log is the current batch, and the metric is the average value of previous step.
step 157/157 [==============================] - loss: 0.0227 - acc: 0.9540 - 8ms/step         
Eval samples: 10000
Epoch 2/5
step 938/938 [==============================] - loss: 0.0706 - acc: 0.9590 - 18ms/step        
Eval begin...
The loss value printed in the log is the current batch, and the metric is the average value of previous step.
step 157/157 [==============================] - loss: 0.0045 - acc: 0.9582 - 8ms/step         
Eval samples: 10000
Epoch 3/5
step 938/938 [==============================] - loss: 0.0185 - acc: 0.9688 - 20ms/step        
Eval begin...
The loss value printed in the log is the current batch, and the metric is the average value of previous step.
step 157/157 [==============================] - loss: 0.0101 - acc: 0.9685 - 8ms/step         
Eval samples: 10000
Epoch 4/5
step 938/938 [==============================] - loss: 0.0045 - acc: 0.9744 - 19ms/step        
Eval begin...
The loss value printed in the log is the current batch, and the metric is the average value of previous step.
step 157/157 [==============================] - loss: 0.0058 - acc: 0.9711 - 8ms/step         
Eval samples: 10000
Epoch 5/5
step 938/938 [==============================] - loss: 0.0657 - acc: 0.9757 - 19ms/step        
Eval begin...
The loss value printed in the log is the current batch, and the metric is the average value of previous step.
step 157/157 [==============================] - loss: 0.0013 - acc: 0.9735 - 8ms/step         
Eval samples: 10000
```
### 模型评估测试
5.1 模型评估
```python
# 模型评估，根据prepare接口配置的loss和metric进行返回
result = model.evaluate(eval_dataset, verbose=1)

print(result)
Eval begin...
The loss value printed in the log is the current batch, and the metric is the average value of previous step.
step 10000/10000 [==============================] - loss: 0.0000e+00 - acc: 0.9795 - 2ms/step         
Eval samples: 10000
{'loss': [0.0], 'acc': 0.9795}
```
5.2 模型预测
5.2.1 批量预测
使用model.predict接口来完成对大量数据集的批量预测。

```python
# 进行预测操作
result = model.predict(eval_dataset)

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
输出：
```python
Predict begin...
step 10000/10000 [==============================] - 2ms/step        
Predict samples: 10000
```
5.2.2 单张图片预测
采用model.predict_batch来进行单张或少量多张图片的预测。

```python
# 读取单张图片
image = eval_dataset[501][0]

# 单张图片预测
result = model.predict_batch([image])

# 可视化结果
show_img(image, np.argmax(result))
```
输出：
```python
图片预测为“9”
<Figure size 432x288 with 1 Axes>
```
### 部署上线
6.1 保存模型
```python
# 保存用于后续继续调优训练的模型
model.save('finetuning/mnist')
```
6.2 继续调优训练
```python
from paddle.static import InputSpec


# 模型封装，为了后面保存预测模型，这里传入了inputs参数
model_2 = paddle.Model(network, inputs=[InputSpec(shape=[-1, 28, 28], dtype='float32', name='image')])

# 加载之前保存的阶段训练模型
model_2.load('finetuning/mnist')

# 模型配置
model_2.prepare(paddle.optimizer.Adam(learning_rate=0.001, parameters=network.parameters()),
                paddle.nn.CrossEntropyLoss(),
                paddle.metric.Accuracy())

# 模型全流程训练
model_2.fit(train_dataset, 
            eval_dataset,
            epochs=2,
            batch_size=64,
            verbose=1)
```
输出：
```python
The loss value printed in the log is the current step, and the metric is the average value of previous step.
Epoch 1/2
step 938/938 [==============================] - loss: 0.0205 - acc: 0.9778 - 9ms/step        
Eval begin...
The loss value printed in the log is the current batch, and the metric is the average value of previous step.
step 157/157 [==============================] - loss: 0.0044 - acc: 0.9731 - 8ms/step         
Eval samples: 10000
Epoch 2/2
step 938/938 [==============================] - loss: 0.0120 - acc: 0.9824 - 12ms/step        
Eval begin...
The loss value printed in the log is the current batch, and the metric is the average value of previous step.
step 157/157 [==============================] - loss: 6.8675e-04 - acc: 0.9759 - 8ms/step     
Eval samples: 10000
```
6.3 保存预测模型
```python
# 保存用于后续推理部署的模型
model_2.save('infer/mnist', training=False)
```


## 作业一

一. 单选题（共6题，共60分）
1. 什么是人工智能、机器学习和深度学习？（10分）

A.人工智能是方法，机器学习是路径，深度学习是实践

B.人工智能是目标，机器学习是人工智能实现的手段，深度学习是机器学习其中的一种方法

C.人工智能是前提，机器学习是实践，深度学习是机器学习之上的一种能力

答案：B
2. 深度学习任务实施的万能公式是哪个？（10分）

A.数据准备、问题定义、模型选择和开发、模型训练和调优、模型评估测试、部署上线

B.问题定义、数据准备、模型选择和开发、模型训练和调优、模型评估测试、部署上线

C.问题定义、模型选择和开发、数据准备、模型训练和调优、模型评估测试、部署上线

答案：B
3. model.prepare接口是用于做什么的？（10分）

A.模型封装

B.模型训练

C.模型配置

D.模型预测

答案：C
4. paddle.Model接口是用于做什么的？（10分）

A.模型封装

B.模型训练

C.模型配置

D.模型预测

答案：A
5. model.fit中哪个参数是用于指定训练日志展示格式的？（10分）


A.train_data

B.epochs

C.batch_size

D.verbose

答案：D
6. model.predict接口是用于做什么的？（10分）

A.模型封装

B.模型训练

C.模型配置

D.模型预测

答案：D
二. 多选题（共2题，共40分）
1. 深度学习中的基础概念包含哪几个？（20分）

A.神经元

B.神经网络

C.前向计算

D.反向传播

答案：A,B,C,D
2. 飞桨（PaddlePaddle）能用于做什么？（20分）

A.机器学习算法实现

B.深度学习任务开发

C.大规模分布式训练

答案：A,B,C
