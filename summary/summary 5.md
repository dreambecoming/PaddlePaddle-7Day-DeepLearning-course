# 百度飞桨深度学习7日打卡营 课程总结5
![paddlepaddle](https://paddlepaddle-org-cn.cdn.bcebos.com/paddle-site-front/favicon-128.png  "百度logo")

[课程链接](https://aistudio.baidu.com/aistudio/course/introduce/6771)	`https://aistudio.baidu.com/aistudio/course/introduce/7073`  
[飞桨官网](https://www.paddlepaddle.org.cn/)	`https://www.paddlepaddle.org.cn/`   
[课程案例合集](https://aistudio.baidu.com/aistudio/projectdetail/1505799?channelType=0&channel=0)	`https://aistudio.baidu.com/aistudio/projectdetail/1505799?channelType=0&channel=0`

****
## 目录
* [自然语言处理](#自然语言处理)
* [『PaddleNLP』利用情感分析选择年夜饭](#PaddleNLP利用情感分析选择年夜饭)
* [作业四](#作业四)
    * [客观题](#客观题)
    * [代码实践](#代码实践)

# 课节4：情感倾向性分析
## 自然语言处理
* NLP（Natural Language Processing）
* NLP 任务粒度：字、词语、句子、篇章
* 情感分析，是文本分类（Text categorization）任务的经典场景：
    -输入：一个自然语言句子。  
    -输出：输出这个句子的情感分类。  
* 文本分类的通用步骤：text → Embedding method → Vector representation → Downstream classification task
* 循环神经网络（Recurrent Neural Network, RNN）-处理序列信息
    输入：一个序列信息   
    运行：从左到右逐词处理，不断调用一个相同的网络单元  
    * 长短期记忆网络（Long Short-Term Memory networks，LSTM）
##『PaddleNLP』 利用情感分析选择年夜饭

环境介绍:

* PaddlePaddle框架，AI Studio平台已经默认安装最新版2.0。
        
* PaddleNLP，深度兼容框架2.0，是飞桨框架2.0在NLP领域的最佳实践。
        
* 这里使用的是beta版本，马上也会发布rc版哦。AI Studio平台后续会默认安装PaddleNLP，在此之前可使用如下命令安装。
#### 问题定义

人脸关键点检测，是输入一张人脸图片，模型会返回人脸关键点的一系列坐标，从而定位到人脸的关键信息。
```python
# 下载paddlenlp
!pip install --upgrade paddlenlp==2.0.0b4 -i https://pypi.org/simple
```
```python
import paddle
import paddlenlp

print(paddle.__version__, paddlenlp.__version__)
```
PaddleNLP和Paddle框架的关系
        
* Paddle框架是基础底座，提供深度学习任务全流程API。PaddleNLP基于Paddle框架开发，适用于NLP任务。
        
* PaddleNLP中数据处理、数据集、组网单元等API未来会沉淀到框架paddle.text中。
        
* 代码中继承 class TSVDataset(paddle.io.Dataset)



```python
import numpy as np
from functools import partial

import paddle.nn as nn
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.datasets import MapDatasetWrapper

from utils import load_vocab, convert_example
```
#### 数据准备
自定义数据集

映射式(map-style)数据集需要继承paddle.io.Dataset
        
* __getitem__: 根据给定索引获取数据集中指定样本，在 paddle.io.DataLoader 中需要使用此函数通过下标获取样本。
        
* __len__: 返回数据集样本个数， paddle.io.BatchSampler 中需要样本个数生成下标序列。
```python
class SelfDefinedDataset(paddle.io.Dataset):
    def __init__(self, data):
        super(SelfDefinedDataset, self).__init__()
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
        
    def get_labels(self):
        return ["0", "1", "2"]

def txt_to_list(file_name):
    res_list = []
    for line in open(file_name):
        res_list.append(line.strip().split('\t'))
    return res_list

trainlst = txt_to_list('train.txt')
devlst = txt_to_list('dev.txt')
testlst = txt_to_list('test.txt')

# 通过get_datasets()函数，将list数据转换为dataset。
# get_datasets()可接收[list]参数，或[str]参数，根据自定义数据集的写法自由选择。

train_ds, dev_ds, test_ds = SelfDefinedDataset.get_datasets([trainlst, devlst, testlst])
```
看看数据长什么样

```python
label_list = train_ds.get_labels()
print(label_list)

for i in range(10):
    print (train_ds[i])
```
数据处理

为了将原始数据处理成模型可以读入的格式，本项目将对数据作以下处理：

* 首先使用jieba切词，之后将jieba切完后的单词映射词表中单词id。

* 使用paddle.io.DataLoader接口多线程异步加载数据。

其中用到了PaddleNLP中关于数据处理的API。PaddleNLP提供了许多关于NLP任务中构建有效的数据pipeline的常用API

| API| 简介| 
|:-----------:| :-------------:|
| paddlenlp.data.Stack      |堆叠N个具有相同shape的输入数据来构建一个batch，它的输入必须具有相同的shape，输出便是这些输入的堆叠组成的batch数据|
| paddlenlp.data.Pad      |	堆叠N个输入数据来构建一个batch，每个输入数据将会被padding到N个输入数据中最大的长度|
| paddlenlp.data.Tuple      |将多个组batch的函数包装在一起|


更多数据处理操作详见： https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/data.md
```python
# 下载词汇表文件word_dict.txt，用于构造词-id映射关系。
!wget https://paddlenlp.bj.bcebos.com/data/senta_word_dict.txt

# 加载词表
vocab = load_vocab('./senta_word_dict.txt')

for k, v in vocab.items():
    print(k, v)
    break
```
构造dataloder

下面的create_data_loader函数用于创建运行和预测时所需要的DataLoader对象。

* paddle.io.DataLoader返回一个迭代器，该迭代器根据batch_sampler指定的顺序迭代返回dataset数据。异步加载数据。

* batch_sampler：DataLoader通过 batch_sampler 产生的mini-batch索引列表来 dataset 中索引样本并组成mini-batch

* collate_fn：指定如何将样本列表组合为mini-batch数据。传给它参数需要是一个callable对象，需要实现对组建的batch的处理逻辑，并返回每个batch的数据。在这里传入的是prepare_input函数，对产生的数据进行pad操作，并返回实际长度等。
```python
# Reads data and generates mini-batches.
def create_dataloader(dataset,
                      trans_function=None,
                      mode='train',
                      batch_size=1,
                      pad_token_id=0,
                      batchify_fn=None):
    if trans_function:
        dataset = dataset.apply(trans_function, lazy=True)

    # return_list 数据是否以list形式返回
    # collate_fn  指定如何将样本列表组合为mini-batch数据。传给它参数需要是一个callable对象，需要实现对组建的batch的处理逻辑，并返回每个batch的数据。在这里传入的是`prepare_input`函数，对产生的数据进行pad操作，并返回实际长度等。
    dataloader = paddle.io.DataLoader(
        dataset,
        return_list=True,
        batch_size=batch_size,
        collate_fn=batchify_fn)
        
    return dataloader

# python中的偏函数partial，把一个函数的某些参数固定住（也就是设置默认值），返回一个新的函数，调用这个新函数会更简单。
trans_function = partial(
    convert_example,
    vocab=vocab,
    unk_token_id=vocab.get('[UNK]', 1),
    is_test=False)

# 将读入的数据batch化处理，便于模型batch化运算。
# batch中的每个句子将会padding到这个batch中的文本最大长度batch_max_seq_len。
# 当文本长度大于batch_max_seq时，将会截断到batch_max_seq_len；当文本长度小于batch_max_seq时，将会padding补齐到batch_max_seq_len.
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=vocab['[PAD]']),  # input_ids
    Stack(dtype="int64"),  # seq len
    Stack(dtype="int64")  # label
): [data for data in fn(samples)]


train_loader = create_dataloader(
    train_ds,
    trans_function=trans_function,
    batch_size=128,
    mode='train',
    batchify_fn=batchify_fn)
dev_loader = create_dataloader(
    dev_ds,
    trans_function=trans_function,
    batch_size=128,
    mode='validation',
    batchify_fn=batchify_fn)
test_loader = create_dataloader(
    test_ds,
    trans_function=trans_function,
    batch_size=128,
    mode='test',
    batchify_fn=batchify_fn)
```
#### 模型组建
使用LSTMencoder搭建一个BiLSTM模型用于进行句子建模，得到句子的向量表示。

然后接一个线性变换层，完成二分类任务。

* paddle.nn.Embedding组建word-embedding层
* ppnlp.seq2vec.LSTMEncoder组建句子建模层
* paddle.nn.Linear构造二分类器
* 除LSTM外，seq2vec还提供了许多语义表征方法，详细可参考：[seq2vec介绍](https://aistudio.baidu.com/aistudio/projectdetail/1283423)
```python
class LSTMModel(nn.Layer):
    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 lstm_hidden_size=198,
                 direction='forward',
                 lstm_layers=1,
                 dropout_rate=0,
                 pooling_type=None,
                 fc_hidden_size=96):
        super().__init__()

        # 首先将输入word id 查表后映射成 word embedding
        self.embedder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=padding_idx)

        # 将word embedding经过LSTMEncoder变换到文本语义表征空间中
        self.lstm_encoder = ppnlp.seq2vec.LSTMEncoder(
            emb_dim,
            lstm_hidden_size,
            num_layers=lstm_layers,
            direction=direction,
            dropout=dropout_rate,
            pooling_type=pooling_type)

        # LSTMEncoder.get_output_dim()方法可以获取经过encoder之后的文本表示hidden_size
        self.fc = nn.Linear(self.lstm_encoder.get_output_dim(), fc_hidden_size)

        # 最后的分类器
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text, seq_len):
        # text shape: (batch_size, num_tokens)
        # print('input :', text.shape)
        
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # print('after word-embeding:', embedded_text.shape)

        # Shape: (batch_size, num_tokens, num_directions*lstm_hidden_size)
        # num_directions = 2 if direction is 'bidirectional' else 1
        text_repr = self.lstm_encoder(embedded_text, sequence_length=seq_len)
        # print('after lstm:', text_repr.shape)


        # Shape: (batch_size, fc_hidden_size)
        fc_out = paddle.tanh(self.fc(text_repr))
        # print('after Linear classifier:', fc_out.shape)

        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        # print('output:', logits.shape)
        
        # probs 分类概率值
        probs = F.softmax(logits, axis=-1)
        # print('output probability:', probs.shape)
        return probs

model= LSTMModel(
        len(vocab),
        len(label_list),
        direction='bidirectional',
        padding_idx=vocab['[PAD]'])
model = paddle.Model(model)
```
#### 模型配置和训练
模型配置
```python
optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=5e-5)

loss = paddle.nn.CrossEntropyLoss()
metric = paddle.metric.Accuracy()

model.prepare(optimizer, loss, metric)
```     
```python
# 设置visualdl路径
log_dir = './visualdl'
callback = paddle.callbacks.VisualDL(log_dir=log_dir)
```
模型训练
训练过程中会输出loss、acc等信息。这里设置了10个epoch，在训练集上准确率约97%。

```python
model.fit(train_loader, dev_loader, epochs=10, save_dir='./checkpoints', save_freq=5, callbacks=callback)
```
启动VisualDL查看训练过程可视化结果

启动步骤：

1、切换到本界面左侧「可视化」  
2、日志文件路径选择 'visualdl'  
3、点击「启动VisualDL」后点击「打开VisualDL」，即可查看可视化结果   
```python
results = model.evaluate(dev_loader)
print("Finally test acc: %.5f" % results['acc'])
```
#### 模型预测
```python
label_map = {0: 'negative', 1: 'positive'}
results = model.predict(test_loader, batch_size=128)[0]
predictions = []

for batch_probs in results:
    # 映射分类label
    idx = np.argmax(batch_probs, axis=-1)
    idx = idx.tolist()
    labels = [label_map[i] for i in idx]
    predictions.extend(labels)

# 看看预测数据前5个样例分类结果
for idx, data in enumerate(test_ds.data[:10]):
    print('Data: {} \t Label: {}'.format(data[0], predictions[idx]))
    """
    展示图像，预测关键点
    Args：
        test_images：裁剪后的图像 [224, 224, 3]
        test_outputs: 模型的输出
        batch_size: 批大小
        h: 展示的图像高
        w: 展示的图像宽
    """
```
## 作业四
### 客观题

一. 单选题（共8题，共80分）

1. 在自然语言处理任务中，将字词、句子、篇章转换为id序列是必要步骤吗？（10分）

>A.是的

>B.不是

>答案：A

2. 在循环神经网络（RNN）里，处理句子里各个词时都会调用相同的网络单元吗？（10分）

>A.是的

>B.不是

>答案：A

3. 在循环神经网络（RNN）里，一般情况，句子里各个词是同时输入网络里，还是有序依次输入？（10分）

>A.同时输入

>B.有序依次输入

>答案：B

4. 交叉熵损失函数可以用于回归任务吗？（10分）

>A.可以

>B.不可以

>答案：B

5. 使用PaddleNLP哪个API，方便将数据统一成相同长度，组成批数据?（10分）

>A.paddlenlp.Dataset

>B.paddlenlp.data.Tuple

>C.paddlenlp.data.Pad

>答案：C

6. 使用Paddle哪个API，能够对词向量做初始化？（10分）

>A.paddle.nn.embedding

>B.paddle.nn.Linear

>C.paddle.io.DataLoader

>答案：C

7. 使用Paddle哪个API，能够对网络进行预测？（10分）

>A.model.fit

>B.model.evaluate

>C.model.predict

>D.model.prepare

>答案：C

8. paddle.nn.Linear的作用是？（10分）

>A.构建线性变换层

>B.配置激活函数

>C.配置损失函数

>D.配置优化器

>答案：A

二. 多选题（共2题，共20分）

1. 以下哪些场景涉及自然语言处理？（10分）

>A.对话机器人

>B.垃圾邮件识别

>C.智能写作

>D.搜索引擎

>答案：A、B、C、D

2. 如果你的确学到了一两个知识点，欢迎通过各种方式支持我们。包括哪些方式呢？（10分）


>A.去GitHub上找到我们的作品https://github.com/paddlepaddle/paddle  ，star、star、star～～～

>B.去GitHub上找到NLP板块https://github.com/PaddlePaddle/PaddleNLP ，star、star、star～～～

>C.在AI Studio上fork你喜欢的项目，疯狂实践。

>D.认真做作业、弹幕、QQ群多多交流！

>答案：A、B、C、D

### 代码实践
『PaddleNLP』利用情感分析选择年夜饭

题目：

* 将lstm网络替换成其他网络。可参考[seq2vec介绍](https://aistudio.baidu.com/aistudio/projectdetail/1283423)
            
* 提示位置：self.lstm_encoder = ppnlp.seq2vec.LSTMEncoder()

附加题：

1、改成三分类

2、更换paddlenlp内置数据集
***
参考课上案例，只用修改部分代码。

三分类数据在materials文件内

自定义数据集类中 设置成三分类
```python
def get_labels(self):
        return ["0", "1", "2"]
```
模型构建改成GRU模型
```python
class GRU_Model(nn.Layer):
    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 gru_hidden_size=198,
                 direction='forward',
                 gru_layers=1,
                 dropout_rate=0,
                 pooling_type=None,
                 fc_hidden_size=96):
        super().__init__()

        # 首先将输入word id 查表后映射成 word embedding
        self.embedder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=padding_idx)

        # 将word embedding经过LSTMEncoder变换到文本语义表征空间中
        # self.lstm_encoder = ppnlp.seq2vec.LSTMEncoder(
        #     emb_dim,
        #     lstm_hidden_size,
        #     num_layers=lstm_layers,
        #     direction=direction,
        #     dropout=dropout_rate,
        #     pooling_type=pooling_type)

        self.GRU_encoder = ppnlp.seq2vec.GRUEncoder(
            emb_dim,
            gru_hidden_size,
            num_layers=gru_layers,
            direction=direction,
            dropout=dropout_rate,
            pooling_type=pooling_type)

        # LSTMEncoder.get_output_dim()方法可以获取经过encoder之后的文本表示hidden_size
        # self.fc = nn.Linear(self.lstm_encoder.get_output_dim(), fc_hidden_size)
        #GRU
        self.fc = nn.Linear(self.GRU_encoder.get_output_dim(), fc_hidden_size)

        # 最后的分类器
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text, seq_len):
        #text shape: (batch_size, num_tokens)
        #print('input :', text.shape)
        
        #Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        #print('after word-embeding:', embedded_text.shape)

        #Shape: (batch_size, num_tokens, num_directions*lstm_hidden_size)
        #num_directions = 2 if direction is 'bidirectional' else 1
        text_repr = self.GRU_encoder(embedded_text, sequence_length=seq_len)
        #print('after lstm:', text_repr.shape)


        #Shape: (batch_size, fc_hidden_size)
        fc_out = paddle.tanh(self.fc(text_repr))
        #print('after Linear classifier:', fc_out.shape)

        #Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        #print('output:', logits.shape)
        
        # probs 分类概率值
        probs = F.softmax(logits, axis=-1)
        #print('output probability:', probs.shape)
        return probs

model= GRU_Model(
        len(vocab),
        len(label_list),
        direction='bidirectional',
        padding_idx=vocab['[PAD]'])
model = paddle.Model(model)
```
