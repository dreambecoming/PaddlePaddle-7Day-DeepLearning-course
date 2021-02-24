# 百度飞桨深度学习7日打卡营 课程总结6
![paddlepaddle](https://paddlepaddle-org-cn.cdn.bcebos.com/paddle-site-front/favicon-128.png  "百度logo")

[课程链接](https://aistudio.baidu.com/aistudio/course/introduce/6771)	`https://aistudio.baidu.com/aistudio/course/introduce/7073`  
[飞桨官网](https://www.paddlepaddle.org.cn/)	`https://www.paddlepaddle.org.cn/`   
[课程案例合集](https://aistudio.baidu.com/aistudio/projectdetail/1505799?channelType=0&channel=0)	`https://aistudio.baidu.com/aistudio/projectdetail/1505799?channelType=0&channel=0`

****
## 目录
* [Seq2Seq 模型](#Seq2Seq-模型)
* [『PaddleNLP』新年到，飞桨带你对对联](#PaddleNLP新年到飞桨带你对对联)
* [作业五](#作业五)
    * [客观题](#客观题)
    * [代码实践](#代码实践)

# 课节5：对对联，根据上联，对下联
## Seq2Seq 模型
* Seq2Seq (sequence to sequence)序列到序列的建模
* encoder-decoder框架
    * -将一个任意长度的源序列转换成另一个任意长度的目标序列。
    * -将源序列输入encoder网络，编码成一个向量encoder vector。
    * -将encoder vector送入decoder网络，decoder根据输入的向量信息，输出预测的目标序列。
    * -编码器和解码器内部通常采用RNN单元。
    * 普通的encoder-decode的缺点是：无法充分利用输入序列的信息。
* Attention注意力机制

## 『PaddleNLP』新年到，飞桨带你对对联
基于seq2seq的对联生成
### 问题定义
对联，是汉族传统文化之一，是写在纸、布上或刻在竹子、木头、柱子上的对偶语句。对联对仗工整，平仄协调，是一字一音的汉语独特的艺术形式，是中国传统文化瑰宝。

这里，我们将根据上联，自动写下联。这是一个典型的序列到序列(sequence2sequence, seq2seq）建模的场景，编码器-解码器（Encoder-Decoder）框架是解决seq2seq问题的经典方法，它能够将一个任意长度的源序列转换成另一个任意长度的目标序列：编码阶段将整个源序列编码成一个向量，解码阶段通过最大化预测序列概率，从中解码出整个目标序列。编码和解码的过程通常都使用RNN实现。
* Encoder采用LSTM，Decoder采用带有attention机制的LSTM。
* 以对联的上联作为Encoder的输出，下联作为Decoder的输入，训练模型。

 Studio平台后续会默认安装PaddleNLP，在此之前可使用如下命令安装。
 
```python
 !pip install --upgrade paddlenlp>=2.0.0b -i https://pypi.org/simple
```
```python
import paddlenlp
paddlenlp.__version__
```
```python
import io
import os

from functools import partial

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.data import Vocab, Pad
from paddlenlp.metrics import Perplexity
from paddlenlp.datasets import CoupletDataset
```
### 数据准备
#### 数据集介绍

采用开源的对联数据集[couplet-clean-dataset](https://github.com/v-zich/couplet-clean-dataset)，该数据集过滤了 [couplet-dataset](https://github.com/wb14123/couplet-dataset)中的低俗、敏感内容。

这个数据集包含70w多条训练样本，1000条验证样本和1000条测试样本。

下面列出一些训练集中对联样例：

上联：晚风摇树树还挺 下联：晨露润花花更红

上联：愿景天成无墨迹 下联：万方乐奏有于阗

上联：丹枫江冷人初去 下联：绿柳堤新燕复来

上联：闲来野钓人稀处 下联：兴起高歌酒醉中

#### 加载数据集

paddlenlp.datasets中内置了多个常见数据集，包括这里的对联数据集CoupletDataset。

paddlenlp.datasets均继承paddle.io.Dataset，支持paddle.io.Dataset的所有功能：

* 通过len()函数返回数据集长度，即样本数量。
* 下标索引：通过下标索引[n]获取第n条样本。
* 遍历数据集，获取所有样本。

此外，paddlenlp.datasets，还支持如下操作：

* 调用get_datasets()函数，传入list或者string，获取相对应的train_dataset、development_dataset、test_dataset等。其中train为训练集，用于模型训练； development为开发集，也称验证集validation_dataset，用于模型参数调优；test为测试集，用于评估算法的性能，但不会根据测试集上的表现再去调整模型或参数。

* 调用apply()函数，对数据集进行指定操作。

这里的CoupletDataset数据集继承TranslationDataset，继承自paddlenlp.datasets，除以上通用用法外，还有一些个性设计：

* 在CoupletDataset class中，还定义了transform函数，用于在每个句子的前后加上起始符<s>和结束符</s>，并将原始数据映射成id序列。
```python
train_ds, dev_ds, test_ds = CoupletDataset.get_datasets(['train', 'dev', 'test'])
```
看看数据长什么样

```python
print (len(train_ds), len(test_ds), len(dev_ds))
for i in range(5):
    print (train_ds[i])

print ('\n')
for i in range(5):
    print (test_ds[i])
```
```python
vocab, _ = CoupletDataset.get_vocab()
trg_idx2word = vocab.idx_to_token
vocab_size = len(vocab)

pad_id = vocab[CoupletDataset.EOS_TOKEN]
bos_id = vocab[CoupletDataset.BOS_TOKEN]
eos_id = vocab[CoupletDataset.EOS_TOKEN]
print (pad_id, bos_id, eos_id)
```
#### 构造dataloder

使用paddle.io.DataLoader来创建训练和预测时所需要的DataLoader对象。

paddle.io.DataLoader返回一个迭代器，该迭代器根据batch_sampler指定的顺序迭代返回dataset数据。支持单进程或多进程加载数据，快！

接收如下重要参数:

* batch_sampler：DataLoader通过 batch_sampler 产生的mini-batch索引列表来 dataset 中索引样本并组成mini-batch

* collate_fn：指定如何将样本列表组合为mini-batch数据。传给它参数需要是一个callable对象，需要实现对组建的batch的处理逻辑，并返回每个batch的数据。在这里传入的是prepare_input函数，对产生的数据进行pad操作，并返回实际长度等。

PaddleNLP提供了许多关于NLP任务中构建有效的数据pipeline的常用API

| API| 简介| 
|:-----------:| :-------------:|
| paddlenlp.data.Stack      |堆叠N个具有相同shape的输入数据来构建一个batch，它的输入必须具有相同的shape，输出便是这些输入的堆叠组成的batch数据|
| paddlenlp.data.Pad      |	堆叠N个输入数据来构建一个batch，每个输入数据将会被padding到N个输入数据中最大的长度|
| paddlenlp.data.Tuple      |将多个组batch的函数包装在一起|


更多数据处理操作详见： https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/data.md
```python
def create_data_loader(dataset):
    data_loader = paddle.io.DataLoader(
        dataset,
        batch_sampler=None,
        batch_size = batch_size,
        collate_fn=partial(prepare_input, pad_id=pad_id))
    return data_loader

def prepare_input(insts, pad_id):
    src, src_length = Pad(pad_val=pad_id, ret_length=True)([inst[0] for inst in insts])
    tgt, tgt_length = Pad(pad_val=pad_id, ret_length=True)([inst[1] for inst in insts])
    tgt_mask = (tgt[:, :-1] != pad_id).astype(paddle.get_default_dtype())
    return src, src_length, tgt[:, :-1], tgt[:, 1:, np.newaxis], tgt_mask
```
```python
use_gpu = True
device = paddle.set_device("gpu" if use_gpu else "cpu")

batch_size = 128
num_layers = 2
dropout = 0.2
hidden_size =256
max_grad_norm = 5.0
learning_rate = 0.001
max_epoch = 20
model_path = './couplet_models'
log_freq = 200

# Define dataloader
train_loader = create_data_loader(train_ds)
test_loader = create_data_loader(test_ds)

print(len(train_ds), len(train_loader), batch_size)
# 702594 5490 128  共5490个batch

for i in train_loader:
    print (len(i))
    for ind, each in enumerate(i):
        print (ind, each.shape)
    break
```
### 模型组建
#### 定义Encoder

Encoder部分非常简单，可以直接利用PaddlePaddle2.0提供的RNN系列API的nn.LSTM。

1. nn.Embedding：该接口用于构建 Embedding 的一个可调用对象，根据输入的size (vocab_size, embedding_dim)自动构造一个二维embedding矩阵，用于table-lookup。查表过程如下：
2. nn.LSTM：提供序列，得到encoder_output和encoder_state。  
参数：
* input_size (int) 输入的大小。
* hidden_size (int) - 隐藏状态大小。
* num_layers (int，可选) - 网络层数。默认为1。
* direction (str，可选) - 网络迭代方向，可设置为forward或bidirect（或bidirectional）。默认为forward。
* time_major (bool，可选) - 指定input的第一个维度是否是time steps。默认为False。
* dropout (float，可选) - dropout概率，指的是出第一层外每层输入时的dropout概率。默认为0。

https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/layer/rnn/LSTM_cn.html

输出:

* outputs (Tensor) - 输出，由前向和后向cell的输出拼接得到。如果time_major为True，则Tensor的形状为[time_steps,batch_size,num_directions * hidden_size]，如果time_major为False，则Tensor的形状为[batch_size,time_steps,num_directions * hidden_size]，当direction设置为bidirectional时，num_directions等于2，否则等于1。

* final_states (tuple) - 最终状态,一个包含h和c的元组。形状为[num_lauers * num_directions, batch_size, hidden_size],当direction设置为bidirectional时，num_directions等于2，否则等于1。
```python
class Seq2SeqEncoder(nn.Layer):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers):
        super(Seq2SeqEncoder, self).__init__()
        self.embedder = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2 if num_layers > 1 else 0.)

    def forward(self, sequence, sequence_length):
        inputs = self.embedder(sequence)
        encoder_output, encoder_state = self.lstm(
            inputs, sequence_length=sequence_length)
        
        # encoder_output [128, 18, 256]  [batch_size,time_steps,hidden_size]
        # encoder_state (tuple) - 最终状态,一个包含h和c的元组。 [2, 128, 256] [2, 128, 256] [num_lauers * num_directions, batch_size, hidden_size]
        return encoder_output, encoder_state
```
#### 定义Decoder

##### 定义AttentionLayer
1. nn.Linear线性变换层传入2个参数
* in_features (int) – 线性变换层输入单元的数目。
* out_features (int) – 线性变换层输出单元的数目。

2. paddle.matmul用于计算两个Tensor的乘积，遵循完整的广播规则，关于广播规则，请参考广播 (broadcasting) 。 并且其行为与 numpy.matmul 一致。
* x (Tensor) : 输入变量，类型为 Tensor，数据类型为float32， float64。
* y (Tensor) : 输入变量，类型为 Tensor，数据类型为float32， float64。
* transpose_x (bool，可选) : 相乘前是否转置 x，默认值为False。
* transpose_y (bool，可选) : 相乘前是否转置 y，默认值为False。

3. paddle.unsqueeze用于向输入Tensor的Shape中一个或多个位置（axis）插入尺寸为1的维度

4. paddle.add逐元素相加算子，输入 x 与输入 y 逐元素相加，并将各个位置的输出元素保存到返回结果中。

输入 x 与输入 y 必须可以广播为相同形状。
```python
class AttentionLayer(nn.Layer):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.input_proj = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size + hidden_size, hidden_size)

    def forward(self, hidden, encoder_output, encoder_padding_mask):
        encoder_output = self.input_proj(encoder_output)
        attn_scores = paddle.matmul(
            paddle.unsqueeze(hidden, [1]), encoder_output, transpose_y=True)
        # print('attention score', attn_scores.shape) #[128, 1, 18]

        if encoder_padding_mask is not None:
            attn_scores = paddle.add(attn_scores, encoder_padding_mask)

        attn_scores = F.softmax(attn_scores)
        attn_out = paddle.squeeze(
            paddle.matmul(attn_scores, encoder_output), [1])
        # print('1 attn_out', attn_out.shape) #[128, 256]

        attn_out = paddle.concat([attn_out, hidden], 1)
        # print('2 attn_out', attn_out.shape) #[128, 512]

        attn_out = self.output_proj(attn_out)
        # print('3 attn_out', attn_out.shape) #[128, 256]
        return attn_out
```
#### 定义Seq2SeqDecoderCell
由于Decoder部分是带有attention的LSTM，我们不能复用nn.LSTM，所以需要定义Seq2SeqDecoderCell

1. nn.LayerList 用于保存子层列表，它包含的子层将被正确地注册和添加。列表中的子层可以像常规python列表一样被索引。这里添加了num_layers=2层lstm。
```python
class Seq2SeqDecoderCell(nn.RNNCellBase):
    def __init__(self, num_layers, input_size, hidden_size):
        super(Seq2SeqDecoderCell, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.lstm_cells = nn.LayerList([
            nn.LSTMCell(
                input_size=input_size + hidden_size if i == 0 else hidden_size,
                hidden_size=hidden_size) for i in range(num_layers)
        ])

        self.attention_layer = AttentionLayer(hidden_size)
    
    def forward(self,
                step_input,
                states,
                encoder_output,
                encoder_padding_mask=None):
        lstm_states, input_feed = states
        new_lstm_states = []
        step_input = paddle.concat([step_input, input_feed], 1)
        for i, lstm_cell in enumerate(self.lstm_cells):
            out, new_lstm_state = lstm_cell(step_input, lstm_states[i])
            step_input = self.dropout(out)
            new_lstm_states.append(new_lstm_state)
        out = self.attention_layer(step_input, encoder_output,
                                   encoder_padding_mask)
        return out, [new_lstm_states, out]
```
#### 定义Seq2SeqDecoder
有了Seq2SeqDecoderCell，就可以构建Seq2SeqDecoder了


1. paddle.nn.RNN 该OP是循环神经网络（RNN）的封装，将输入的Cell封装为一个循环神经网络。它能够重复执行 cell.forward() 直到遍历完input中的所有Tensor。
* cell (RNNCellBase) - RNNCellBase类的一个实例。
```python
class Seq2SeqDecoder(nn.Layer):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers):
        super(Seq2SeqDecoder, self).__init__()
        self.embedder = nn.Embedding(vocab_size, embed_dim)
        self.lstm_attention = nn.RNN(
            Seq2SeqDecoderCell(num_layers, embed_dim, hidden_size))
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, trg, decoder_initial_states, encoder_output,
                encoder_padding_mask):
        inputs = self.embedder(trg)

        decoder_output, _ = self.lstm_attention(
            inputs,
            initial_states=decoder_initial_states,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask)
        predict = self.output_layer(decoder_output)

        return predict
```
#### 构建主网络Seq2SeqAttnModel
Encoder和Decoder定义好之后，网络就可以构建起来了

```python
class Seq2SeqAttnModel(nn.Layer):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers,
                 eos_id=1):
        super(Seq2SeqAttnModel, self).__init__()
        self.hidden_size = hidden_size
        self.eos_id = eos_id
        self.num_layers = num_layers
        self.INF = 1e9
        self.encoder = Seq2SeqEncoder(vocab_size, embed_dim, hidden_size,
                                      num_layers)
        self.decoder = Seq2SeqDecoder(vocab_size, embed_dim, hidden_size,
                                      num_layers)

    def forward(self, src, src_length, trg):
        # encoder_output 各时刻的输出h
        # encoder_final_state 最后时刻的输出h，和记忆信号c
        encoder_output, encoder_final_state = self.encoder(src, src_length)
        print('encoder_output shape', encoder_output.shape)  #  [128, 18, 256]  [batch_size,time_steps,hidden_size]
        print('encoder_final_states shape', encoder_final_state[0].shape, encoder_final_state[1].shape) #[2, 128, 256] [2, 128, 256] [num_lauers * num_directions, batch_size, hidden_size]

        # Transfer shape of encoder_final_states to [num_layers, 2, batch_size, hidden_size]？？？
        encoder_final_states = [
            (encoder_final_state[0][i], encoder_final_state[1][i])
            for i in range(self.num_layers)
        ]
        print('encoder_final_states shape', encoder_final_states[0][0].shape, encoder_final_states[0][1].shape) #[128, 256] [128, 256]


        # Construct decoder initial states: use input_feed and the shape is
        # [[h,c] * num_layers, input_feed], consistent with Seq2SeqDecoderCell.states
        decoder_initial_states = [
            encoder_final_states,
            self.decoder.lstm_attention.cell.get_initial_states(
                batch_ref=encoder_output, shape=[self.hidden_size])
        ]

        # Build attention mask to avoid paying attention on padddings
        src_mask = (src != self.eos_id).astype(paddle.get_default_dtype())
        print ('src_mask shape', src_mask.shape)  #[128, 18]
        print(src_mask[0, :])

        encoder_padding_mask = (src_mask - 1.0) * self.INF
        print ('encoder_padding_mask', encoder_padding_mask.shape)  #[128, 18]
        print(encoder_padding_mask[0, :])

        encoder_padding_mask = paddle.unsqueeze(encoder_padding_mask, [1])
        print('encoder_padding_mask', encoder_padding_mask.shape)  #[128, 1, 18]

        predict = self.decoder(trg, decoder_initial_states, encoder_output,
                               encoder_padding_mask)
        print('predict', predict.shape)   #[128, 17, 7931]

        return predict

```
#### 定义损失函数
这里使用的是交叉熵损失函数，我们需要将padding位置的loss置为0，因此需要在损失函数中引入trg_mask参数，由于PaddlePaddle框架提供的paddle.nn.CrossEntropyLoss不能接受trg_mask参数，因此在这里需要重新定义：
```python
class CrossEntropyCriterion(nn.Layer):
    def __init__(self):
        super(CrossEntropyCriterion, self).__init__()

    def forward(self, predict, label, trg_mask):
        cost = F.softmax_with_cross_entropy(
            logits=predict, label=label, soft_label=False)
        cost = paddle.squeeze(cost, axis=[2])
        masked_cost = cost * trg_mask
        batch_mean_cost = paddle.mean(masked_cost, axis=[0])
        seq_cost = paddle.sum(batch_mean_cost)

        return seq_cost
```

#### 模型训练
使用高层API执行训练，需要调用prepare和fit函数。

在prepare函数中，配置优化器、损失函数，以及评价指标。其中评价指标使用的是PaddleNLP提供的困惑度计算API paddlenlp.metrics.Perplexity。

如果你安装了VisualDL，可以在fit中添加一个callbacks参数使用VisualDL观测你的训练过程，如下：
```python
model.fit(train_data=train_loader,
            epochs=max_epoch,
            eval_freq=1,
            save_freq=1,
            save_dir=model_path,
            log_freq=log_freq,
            callbacks=[paddle.callbacks.VisualDL('./log')])
```
在这里，由于对联生成任务没有明确的评价指标，因此，可以在保存的多个模型中，通过人工评判生成结果选择最好的模型。

本项目中，为了便于演示，已经将训练好的模型参数载入模型，并省略了训练过程。读者自己实验的时候，可以尝试自行修改超参数，调用下面被注释掉的fit函数，重新进行训练。

如果读者想要在更短的时间内得到效果不错的模型，可以使用预训练模型技术，例如[《预训练模型ERNIE-GEN自动写诗》](https://aistudio.baidu.com/aistudio/projectdetail/1339888)项目为大家展示了如何利用预训练的生成模型进行训练。
```python
model = paddle.Model(
    Seq2SeqAttnModel(vocab_size, hidden_size, hidden_size,
                        num_layers, pad_id))

optimizer = paddle.optimizer.Adam(
    learning_rate=learning_rate, parameters=model.parameters())
ppl_metric = Perplexity()
model.prepare(optimizer, CrossEntropyCriterion(), ppl_metric)

model.fit(train_data=train_loader,
            epochs=max_epoch,
            eval_freq=1,
            save_freq=1,
            save_dir=model_path,
            log_freq=log_freq)
```     
### 模型预测
#### 定义预测网络Seq2SeqAttnInferModel  

预测网络继承上面的主网络Seq2SeqAttnModel，定义子类Seq2SeqAttnInferModel
```python
class Seq2SeqAttnInferModel(Seq2SeqAttnModel):
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 bos_id=0,
                 eos_id=1,
                 beam_size=4,
                 max_out_len=256):
        self.bos_id = bos_id
        self.beam_size = beam_size
        self.max_out_len = max_out_len
        self.num_layers = num_layers
        super(Seq2SeqAttnInferModel, self).__init__(
            vocab_size, embed_dim, hidden_size, num_layers, eos_id)

        # Dynamic decoder for inference
        self.beam_search_decoder = nn.BeamSearchDecoder(
            self.decoder.lstm_attention.cell,
            start_token=bos_id,
            end_token=eos_id,
            beam_size=beam_size,
            embedding_fn=self.decoder.embedder,
            output_fn=self.decoder.output_layer)

    def forward(self, src, src_length):
        encoder_output, encoder_final_state = self.encoder(src, src_length)

        encoder_final_state = [
            (encoder_final_state[0][i], encoder_final_state[1][i])
            for i in range(self.num_layers)
        ]

        # Initial decoder initial states
        decoder_initial_states = [
            encoder_final_state,
            self.decoder.lstm_attention.cell.get_initial_states(
                batch_ref=encoder_output, shape=[self.hidden_size])
        ]
        # Build attention mask to avoid paying attention on paddings
        src_mask = (src != self.eos_id).astype(paddle.get_default_dtype())

        encoder_padding_mask = (src_mask - 1.0) * self.INF
        encoder_padding_mask = paddle.unsqueeze(encoder_padding_mask, [1])

        # Tile the batch dimension with beam_size
        encoder_output = nn.BeamSearchDecoder.tile_beam_merge_with_batch(
            encoder_output, self.beam_size)
        encoder_padding_mask = nn.BeamSearchDecoder.tile_beam_merge_with_batch(
            encoder_padding_mask, self.beam_size)

        # Dynamic decoding with beam search
        seq_output, _ = nn.dynamic_decode(
            decoder=self.beam_search_decoder,
            inits=decoder_initial_states,
            max_step_num=self.max_out_len,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask)
        return seq_output
```
#### 解码部分
接下来对我们的任务选择beam search解码方式，可以指定beam_size为10。

```python
def post_process_seq(seq, bos_idx, eos_idx, output_bos=False, output_eos=False):
    """
    Post-process the decoded sequence.
    """
    eos_pos = len(seq) - 1
    for i, idx in enumerate(seq):
        if idx == eos_idx:
            eos_pos = i
            break
    seq = [
        idx for idx in seq[:eos_pos + 1]
        if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)
    ]
    return seq
```
```python
beam_size = 10
# init_from_ckpt = './couplet_models/0' # for test
# infer_output_file = './infer_output.txt'

# test_loader, vocab_size, pad_id, bos_id, eos_id = create_data_loader(test_ds, batch_size)
# vocab, _ = CoupletDataset.get_vocab()
# trg_idx2word = vocab.idx_to_token

model = paddle.Model(
    Seq2SeqAttnInferModel(
        vocab_size,
        hidden_size,
        hidden_size,
        num_layers,
        bos_id=bos_id,
        eos_id=eos_id,
        beam_size=beam_size,
        max_out_len=256))

model.prepare()
```
在预测之前，我们需要将训练好的模型参数load进预测网络，之后我们就可以根据对联的上联，生成对联的下联啦！
```python
model.load('couplet_models/model_18')
```
```python
test_ds = CoupletDataset.get_datasets(['test'])
idx = 0
for data in test_loader():
    inputs = data[:2]
    finished_seq = model.predict_batch(inputs=list(inputs))[0]
    finished_seq = finished_seq[:, :, np.newaxis] if len(
        finished_seq.shape) == 2 else finished_seq
    finished_seq = np.transpose(finished_seq, [0, 2, 1])
    for ins in finished_seq:
        for beam in ins:
            id_list = post_process_seq(beam, bos_id, eos_id)
            word_list_l = [trg_idx2word[id] for id in test_ds[idx][0]][1:-1]
            word_list_r = [trg_idx2word[id] for id in id_list]
            sequence = "上联: "+" ".join(word_list_l)+"\t下联: "+" ".join(word_list_r) + "\n"
            print(sequence)
            idx += 1
            break
```
## 作业五
### 客观题

一. 单选题（共8题，共80分）

1. paddlenlp.datasets类和paddle.io.Dataset的关系，下列哪种描述最恰当？（10分）

>A.毫无关系

>B.paddlenlp.datasets继承自paddle.io.Dataset

>答案：B

2. 在循环神经网络（RNN）里，处理句子里各个词时都会调用相同的网络单元吗？（10分）

>A.是的

>B.不是

>答案：A

3. 在循环神经网络（RNN）里，一般情况，句子里各个词是同时输入网络里，还是有序依次输入？（10分）

>A.同时输入

>B.有序依次输入

>答案：B

4. 使用Paddle哪个API，能够加载模型？（10分）

>A.model.load

>B.model.download

>答案：A

5. encoder-decoder只能解决输入序列和输出序列等长的问题吗？（10分）

>A.不是

>B.是的

>答案：A

6. 使用Paddle哪个API，能实现线性变换层？（10分）

>A.paddle.nn.embedding

>B.paddle.nn.Linear

>C.paddle.io.DataLoader

>答案：B

7. 使用Paddle哪个API，能够对网络进行训练？（10分）

>A.model.fit

>B.model.evaluate

>C.model.predict

>D.model.prepare

>答案：A

8. 关于Attention机制，下列哪种说法是错误的？（10分）

>A.Attention机制是一系列注意力分配系数，也就是一系列权重参数。

>B.Attention机制常用于encoder-decoder

>C.Attention机制是一类优化器

>答案：C

二. 多选题（共2题，共20分）

1. 以下哪些场景涉及文本生成？（10分）

>A.机器翻译

>B.垃圾邮件识别

>C.智能写作

>D.搜索引擎

>答案：A,C,D

2. 如果你的确学到了一两个知识点，欢迎通过各种方式支持我们。包括哪些方式呢？（10分）

>A.去GitHub上找到我们的作品https://github.com/paddlepaddle/paddle    ，star、star、star～～～

>B.去GitHub上找到NLP板块https://github.com/PaddlePaddle/PaddleNLP    ，star、star、star～～～

>C.在AI Studio上fork你喜欢的项目，疯狂实践。

>D.认真做作业、弹幕、QQ群多多交流！

>答案：A,B,C,D

### 代码实践
『PaddleNLP』新年到，飞桨带你对对联

**选做**

运行案例得出结果


上联: 众 佛 群 灵 光 圣 地	下联: 众 生 一 念 证 菩 提

上联: 乡 愁 何 处 解	下联: 故 事 几 时 休

上联: 清 池 荷 试 墨	下联: 碧 水 柳 含 情

上联: 既 近 浅 流 安 笔 砚	下联: 欲 将 直 气 定 乾 坤

上联: 日 丽 萱 闱 祝 无 量 寿	下联: 月 明 桂 殿 祝 有 余 龄

上联: 一 地 残 红 风 拾 起	下联: 半 窗 疏 影 月 窥 来

上联: 白 塔 有 情 泪 弹 翠 岛 三 生 梦	下联: 红 尘 无 恙 心 系 苍 生 一 片 心

上联: 霜 华 浓 似 雪	下联: 冰 雪 冷 如 冰

上联: 小 子 听 之 濯 足 濯 缨 借 自 取	下联: 高 僧 悟 也 修 身 养 性 即 如 来

上联: 踏 雪 寻 梅 句	下联: 寻 春 觅 柳 诗

上联: 无 水 不 清 一 鉴 尽 传 云 外 意	下联: 有 山 皆 洗 千 秋 不 染 世 间 尘

上联: 拜 竹 为 师 一 生 常 对 虚 心 客	下联: 寻 梅 作 伴 半 世 每 逢 知 己 人

上联: 眄 晓 日 朝 霞 祥 光 万 里 苍 茫 外	下联: 对 清 风 明 月 明 月 千 秋 寂 寞 中

上联: 忠 孝 节 义 萃 于 一 门 间 披 南 宋 伤 心 史	下联: 忠 孝 忠 孝 昭 其 千 载 后 继 东 方 继 往 开

上联: 一 枕 云 山 观 自 在	下联: 半 窗 竹 影 任 逍 遥

上联: 秀 丽 堂 皇 延 好 客	下联: 和 谐 社 会 庆 新 春

上联: 配 偶	下联: 交 朋

上联: 故 事 含 章 吐 曜 清 明 一 曲 千 秋 醉	下联: 春 风 得 意 流 光 溢 彩 千 秋 万 代 兴

上联: 仁 和 信 义 安 民 清 心 自 律 查 风 纪	下联: 诚 信 文 明 治 国 富 国 强 民 奔 小 康

上联: 一 言 难 尽 同 窗 梦	下联: 半 世 不 知 两 鬓 秋

上联: 酒 高 好 看 东 篱 菊	下联: 月 老 难 寻 北 斗 星

上联: 百 味 溢 琴 弦 几 许 青 春 流 淌	下联: 千 秋 留 笔 墨 千 秋 翰 墨 芬 芳

上联: 惜 有 花 开 人 去 后	下联: 愁 无 月 落 梦 来 时

上联: 满 月 可 观 仙 子 泪	下联: 清 风 不 负 故 人 情

上联: 月 遮 白 雪 夜 织 女	下联: 风 过 泸 州 带 酒 香

上联: 兰 舟 野 渡 逍 遥 梦	下联: 明 月 清 风 自 在 诗

上联: 带 俏 含 羞 梅 弄 雪	下联: 乘 风 破 浪 浪 淘 沙

上联: 何 去 何 从 须 看 红 包 厚 度	下联: 不 来 不 见 莫 言 白 发 多 情

上联: 梅 花 三 弄 风 和 雨	下联: 柳 絮 一 飞 雨 顺 风

上联: 绿 蚁 杯 中 物	下联: 红 尘 梦 里 人

上联: 胜 友 如 云 赞 壮 丽 名 楼 重 光 故 郡	下联: 高 朋 似 海 赞 英 雄 伟 业 永 耀 神 州

上联: 画 栋 雕 梁 门 启 千 般 风 物 秀	下联: 琼 楼 玉 宇 梦 圆 万 里 画 图 新

上联: 牛 车 行 马 路	下联: 骏 马 跃 龙 门

上联: 半 夏 当 归 熟 地 总 比 生 地 好	下联: 千 秋 不 老 高 山 流 水 上 天 高

上联: 走 兽	下联: 飞 禽

上联: 青 山 不 改 三 贤 赋	下联: 碧 水 长 流 四 海 歌

上联: 拈 雁 字 韵 山 风 敢 问 醉 处	下联: 把 酒 诗 情 月 色 不 知 愁 时

上联: 栈 道 连 云 引 无 数 英 雄 豪 杰 登 天 摘 月	下联: 春 风 化 雨 催 万 千 豪 杰 英 雄 壮 志 凌 云

上联: 梦 里 情 人 情 里 梦	下联: 杯 中 月 色 梦 中 人

上联: 文 明 古 国 励 精 图 治 新 崛 起	下联: 和 谐 社 会 勤 劳 致 富 大 腾 飞

上联: 话 旧 老 翁 漫 忆 当 年 骑 马 马	下联: 情 归 故 里 遥 思 往 日 驾 龙 舟

上联: 花 香 常 绕 笔	下联: 鸟 语 总 关 情

上联: 凭 本 领 冲 冠	下联: 靠 科 学 创 新

上联: 无 欲 自 然 心 似 水	下联: 有 情 何 必 梦 如 烟

上联: 脉 脉 人 千 里 念 两 处 风 情 万 重 烟 水	下联: 幽 幽 梦 一 帘 思 一 帘 梦 梦 几 度 春 秋

上联: 千 朵 红 莲 三 里 水	下联: 一 轮 皓 月 一 轮 星

上联: 尽 夜 观 灯 夜 夜 夜 灯 灯 不 夜	下联: 临 晨 听 雨 声 声 声 鼓 鼓 长 鸣

上联: 嘉 施 利 根 动 力 再 接 再 厉	下联: 展 宏 图 展 宏 图 如 画 如 诗

上联: 绸 缎 满 天 风 带 走	下联: 琴 弦 一 曲 月 携 来

上联: 鞠 松 陶 令 宅	下联: 垂 柳 侍 郎 家

上联: 感 恩 观 世 界	下联: 济 世 度 人 生

上联: 淇 淋 藏 雪 柜	下联: 趵 突 淌 冰 泉

上联: 科 学 绘 就 民 生 景	下联: 勤 俭 浇 开 幸 福 花

上联: 佛 道 参 茶 参 造 化	下联: 禅 心 悟 道 悟 禅 机

上联: 清 风 远 播 凤 城 爱	下联: 明 月 长 留 天 井 红

上联: 菡 萏 开 花 吐 艳	下联: 蜻 蜓 点 水 含 情

上联: 读 书 需 用 意	下联: 处 世 要 修 身

上联: 无 聊 友	下联: 有 瘾 人

上联: 春 花 桃 叶 渡	下联: 秋 月 桂 花 香

上联: 一 钩 小 月 斜 檐 角	下联: 两 袖 清 风 入 画 中

上联: 李 李 桃 桃 香 一 苑	下联: 梅 兰 竹 菊 韵 千 秋

上联: 谁 人 能 解 落 花 梦	下联: 哪 个 可 知 流 水 情

上联: 善 报 恶 报 迟 报 速 报 终 须 有 报	下联: 天 知 地 知 我 知 我 知 何 谓 无 知

上联: 记 得 与 君 花 下 别	下联: 不 知 何 处 水 中 央

上联: 春 风 送 暖 催 花 艳	下联: 旭 日 迎 新 映 日 红

上联: 林 深 路 险 人 难 越	下联: 海 阔 天 空 志 不 移

上联: 兽 王 不 敌 群 虫 搏	下联: 牛 鬼 何 须 众 鸟 鸣

上联: 敞 大 关 门 平 垭 口	下联: 开 新 路 路 上 层 楼

上联: 过 隧 道 不 得 超 车	下联: 过 关 关 关 关 关 关

上联: 老 来 渐 得 湖 山 味	下联: 老 去 方 知 岁 月 情

上联: 老 了 还 难 说 真 话	下联: 新 来 未 必 见 真 情

上联: 天 人 合 一 新 城 市	下联: 日 月 同 辉 大 地 天

上联: 树 发 孙 枝 方 茂 盛	下联: 花 开 果 果 更 繁 荣

上联: 缘 枝 摘 果 和 风 伴	下联: 梦 笔 生 花 细 雨 随

上联: 血 肉 筑 长 城 八 年 悲 壮 铭 千 古	下联: 锤 镰 开 盛 世 万 里 江 山 耀 九 州

上联: 一 径 飞 花 寻 旧 梦	下联: 两 行 雁 字 寄 新 愁

上联: 与 尔 同 销 万 古	下联: 同 君 共 享 千 秋

上联: 明 文 传 素 志	下联: 大 笔 写 春 秋

上联: 苍 苔 路 熟 僧 归 寺	下联: 红 叶 楼 高 月 满 楼

上联: 一 帘 卷 走 清 风 夜	下联: 两 袖 清 来 明 月 天

上联: 蓝 天 绿 地 山 川 美	下联: 绿 水 青 山 日 月 新

上联: 清 思 似 水 洞 察 秋 毫 策 已 决	下联: 正 气 如 山 气 冲 霄 汉 展 雄 姿

上联: 宝 篆 焚 香 留 睡 鸭	下联: 清 风 拂 槛 送 归 鸿

上联: 飞 花 一 径 随 风 袅	下联: 落 叶 千 山 伴 月 眠

上联: 动 守 其 时 静 随 其 势 不 倚 不 偏 循 本 位	下联: 安 居 此 处 安 得 安 居 安 居 乐 业 享 安 康

上联: 弹 琴 又 为 相 思 梦	下联: 把 酒 还 吟 寂 寞 诗

上联: 郑 知 县 描 竹 临 傲 骨	下联: 陈 美 人 泼 墨 写 春 秋

上联: 妙 质 因 风 剪	下联: 真 情 似 水 流

上联: 登 梅 喜 鹊 开 春 运	下联: 踏 雪 梅 花 报 福 音

上联: 金 人 汉 满 皆 兄 弟	下联: 天 下 人 和 是 弟 兄

上联: 巴 人 兴 吃 火 锅 鸭	下联: 老 鬼 出 山 山 水 鸡

上联: 文 艺 迎 春 春 风 激 发 正 能 量	下联: 楹 联 贺 岁 喜 气 盈 门 新 画 图

上联: 奥 运 福 娃 喜 迎 天 下 客	下联: 神 州 奥 运 喜 报 世 间 春

上联: 四 面 晴 光 对 屏 障	下联: 一 江 春 水 向 东 流

上联: 春 雨 读 花 信	下联: 秋 风 扫 草 原

上联: 称 胡 师 督 辫 军 光 绪 班 班 可 仿	下联: 夺 冠 军 营 销 战 功 勋 处 处 如 何

上联: 口 技 演 员 说 鸟 话	下联: 神 州 大 业 展 鸿 图

上联: 庙 堂 雨 露 何 关 我	下联: 山 水 风 流 自 在 人

上联: 走 金 光 道 擎 特 色 旗 鹿 洼 崛 起 前 程 远	下联: 奔 富 路 路 拓 新 程 路 龙 马 腾 飞 骏 业 兴

上联: 秋 风 琴 瑟 弹 高 调	下联: 冬 雪 梅 花 伴 暗 香

上联: 久 坐 深 窗 听 花 跌 落	下联: 曾 经 沧 海 看 月 沉 浮

上联: 社 稷 言 称 将 相	下联: 江 山 气 壮 神 州

上联: 有 热 情 何 须 三 把 火	下联: 无 杂 念 不 必 一 身 轻

上联: 大 梁 凌 霄 云 浩 荡	下联: 高 山 仰 岳 日 辉 煌

上联: 钟 声 嘹 亮 感 恩 亿 万 勤 劳 客	下联: 瑞 气 氤 氲 喜 气 千 千 幸 福 人

上联: 员 树 出 天	下联: 天 地 成 地

上联: 拂 镜 羞 温 峤	下联: 垂 帘 静 倚 窗

上联: 松 翠 自 洁 梅 雅 无 争 百 里 悬 冰 春 在 望	下联: 花 香 扑 面 花 香 溢 彩 千 年 流 韵 梦 生 香

上联: 滚 动 新 闻 反 复 念	下联: 风 流 旧 事 总 关 情

上联: 今 年 逢 狗	下联: 昨 日 逢 猪

上联: 雾 里 群 峰 如 有 约	下联: 云 中 一 月 似 无 眠

上联: 收 缩 天 地 穿 越 时 空 千 载 戏 文 千 载 梦	下联: 传 递 古 今 传 承 古 今 万 年 功 业 万 年 春

上联: 同 仇 抗 日 歌 才 子 且 为 战 士	下联: 克 己 捐 躯 颂 英 雄 还 是 英 雄

上联: 墨 香 如 酒 千 层 浪	下联: 墨 韵 似 诗 万 卷 诗

上联: 半 世 情 怀 敲 案 问	下联: 一 蓑 烟 雨 任 平 生

上联: 瞻 大 贵 大 雄 诵 大 悲 顿 生 大 觉	下联: 念 慈 悲 慈 善 念 慈 念 普 度 众 生

上联: 二 三 鸟 语 消 春 困	下联: 一 片 冰 心 在 玉 壶

上联: 风 梳 柳 髻 青 丝 乱	下联: 雨 润 桃 腮 粉 面 娇

上联: 成 败 不 由 人 惟 求 尽 力	下联: 死 生 皆 自 己 但 愿 无 忧

上联: 爱 国 重 家 君 子 义	下联: 修 身 养 性 圣 贤 心

上联: 云 移 月 伴 难 留 步	下联: 日 丽 风 和 不 动 心

上联: 归 山 已 绝 沧 桑 泪	下联: 逝 水 难 留 岁 月 痕

上联: 万 里 海 涛 千 卷 画	下联: 一 江 春 水 一 湖 诗

上联: 百 年 紫 木 固 根 基 回 望 陈 门 进 士	下联: 万 里 青 山 如 画 本 展 望 锦 绣 前 程

上联: 风 华 减 去 谁 留 住	下联: 岁 月 添 来 我 自 来

上联: 望 赤 壁 流 丹 问 谁 挥 梦 笔 皴 红 苗 寨	下联: 看 青 山 焕 彩 看 我 挥 毫 书 写 绿 诗 篇

上联: 福 娃 盛 邀 五 洲 客	下联: 宝 鸡 欢 唱 万 户 春

上联: 久 知 鹄 自 飞 新 得	下联: 常 见 龙 能 起 大 来

上联: 我 曾 置 盏 邀 明 月	下联: 谁 与 拈 花 问 落 花

上联: 彰 一 代 君 臣 表 率	下联: 仰 千 秋 俎 豆 馨 香

上联: 枝 繁 叶 茂 拔 萃 精 英 拟 决 策	下联: 国 富 民 强 扬 鞭 骏 马 奋 腾 飞

上联: 勤 志 有 为 千 般 陶 冶 终 成 器	下联: 勤 劳 无 限 万 种 芬 芳 总 是 春

上联: 亦 静 不 哗 听 涓 滴 天 来 甘 露	下联: 亦 真 亦 幻 见 沧 桑 世 外 桃 源

上联: 打 胡 说	下联: 刮 肚 搜

上联: 长 街 扫 落 三 秋 叶	下联: 短 笛 吹 开 一 剪 梅

上联: 不 懂 古 文 编 白 话	下联: 常 闻 新 曲 奏 清 音

上联: 往 来 不 少 响 各 客	下联: 生 死 何 妨 说 是 非

上联: 风 云 乍 向 怀 中 起	下联: 岁 月 常 从 梦 里 来

上联: 摇 橹 荡 舟 寻 美 境	下联: 挥 毫 泼 墨 写 华 章

上联: 漫 天 瑞 雪 千 山 秀	下联: 遍 地 春 风 万 里 香

上联: 改 革 革 出 万 亩 粮 田 田 聚 宝	下联: 科 学 发 展 千 家 事 业 业 增 辉

上联: 劈 疯 癫 营 造 精 神 蓝 天 绿 地	下联: 治 病 病 保 持 保 障 绿 水 青 山

上联: 小 妹 东 坡 留 佛 印	下联: 高 僧 西 子 悟 禅 机

上联: 赞 荷 文 创 基 业 枝 荣 本 固 昌 万 代	下联: 赞 桃 李 育 桃 李 果 硕 果 丰 誉 千 秋

上联: 酒 烈 风 高 山 路 远 望 君 珍 重	下联: 月 圆 花 好 月 光 高 照 我 婵 娟

上联: 沐 八 载 春 风 艺 花 吐 艳 今 朝 多 异 彩	下联: 兴 千 秋 伟 业 联 苑 增 辉 此 日 尽 奇 香

上联: 入 口 已 然 年 味 道	下联: 回 头 不 见 旧 风 情

上联: 围 炉 夜 话 新 醅 酒	下联: 把 盏 晨 钟 旧 鼓 声

上联: 春 风 一 顾 山 花 笑	下联: 秋 雨 几 时 柳 絮 飞

上联: 雾 花 水 月 清 心 逸	下联: 冰 雪 冰 霜 白 骨 精

上联: 飞 雪 片 片 凝 瑞	下联: 落 花 声 声 唤 春

上联: 东 风 乍 喜 还 沧 海	下联: 紫 燕 初 裁 又 剪 春

上联: 桃 花 窥 镜 珠 江 羞 红 两 岸	下联: 燕 子 裁 春 锦 绣 喜 绿 千 畴

上联: 高 者 顶 天 不 霸 道	下联: 大 夫 拔 地 莫 欺 天

上联: 草 木 蒙 茸 露 湿 青 皋 闲 数 鹭 峰 泛 舟 待 月	下联: 山 峦 叠 翠 风 来 碧 水 静 听 渔 歌 唱 晚 听 涛

上联: 瑞 兆 千 秋 骏 业	下联: 春 回 万 里 春 光

上联: 月 半 举 杯 圆 月 下	下联: 风 中 吹 笛 落 花 间

上联: 庠 序 百 年 为 大 本	下联: 英 雄 千 古 仰 高 风

上联: 推 杯 换 盏 频 交 手	下联: 拍 马 溜 须 总 动 心

上联: 悲 歌 动 地	下联: 喜 气 盈 门

上联: 随 水 逝 泪 成 行 落 花 空 散 去	下联: 随 风 飘 梦 入 梦 流 水 自 生 来

上联: 牛 肚	下联: 马 蹄

上联: 狡 诈	下联: 愚 公

上联: 植 竹 培 兰 修 心 养 性	下联: 栽 花 种 竹 养 性 修 身

上联: 天 地 间 日 星 河 岳 正 气	下联: 山 河 里 日 月 日 月 光 华

上联: 常 常 喝 常 常 醉 常 常 喝 醉	下联: 多 多 多 多 少 多 多 多 少 多

上联: 似 水 流 年 闲 愁 万 种 何 人 会	下联: 如 烟 往 事 旧 梦 千 重 哪 个 知

上联: 一 起 同 尊 泽 后 土 苍 天 共 弘 道 脉	下联: 百 年 共 仰 光 前 天 大 地 同 仰 灵 光

上联: 承 厚 重 人 文 千 古 徽 风 排 闼 入	下联: 展 宏 图 气 象 万 家 春 色 入 怀 来

上联: 强 弓 射 透 百 步 杨	下联: 大 笔 写 出 千 年 文

上联: 常 思 灯 下 老 妈 影	下联: 不 见 人 间 老 子 心

上联: 春 风 绿 染 千 家 树	下联: 旭 日 红 燃 万 户 门

上联: 香 阁 春 回 风 送 暖	下联: 寒 窗 夜 静 月 生 凉

上联: 一 竿 竹 影 横 窗 乱	下联: 十 里 荷 香 扑 面 香

上联: 冲 一 盏 清 茶 细 品 前 尘 往 事	下联: 看 千 年 老 酒 闲 斟 往 事 沧 桑

上联: 梦 筑 中 华 全 民 追 梦	下联: 春 回 大 地 大 地 飞 歌

上联: 精 做 郇 阳 地 道 家 常 思 乡 菜	下联: 闲 游 宝 岛 人 间 客 醉 醉 乡 人

上联: 风 调 雨 顺 百 花 艳	下联: 国 泰 民 安 万 事 兴

上联: 庭 竹 不 收 帘 影 去	下联: 梅 花 犹 带 雪 香 来

上联: 近 梅 已 是 三 分 雅	下联: 临 水 方 知 一 脉 香

上联: 城 开 山 日 早	下联: 鸟 啭 鸟 声 甜

上联: 春 夜 疏 帘 邀 月 影	下联: 秋 风 小 院 惹 花 香

上联: 一 生 老 病 死	下联: 千 古 古 风 流

上联: 水 白 如 冰 粉 藕 如 弯 饼 圆 如 月	下联: 山 青 似 画 图 山 似 画 图 画 似 诗

上联: 欢 天 喜 地 笑 迎 乖 兔 宝 贝	下联: 乐 地 乐 天 乐 奏 好 猫 哆 哩

上联: 杜 门 远 万 马 兵 尘 海 上 引 年 仙 枣 大	下联: 孔 子 长 千 年 俎 豆 山 中 留 俎 豆 馨 香

上联: 红 丝 穿 露 珠 帘 冷	下联: 紫 燕 穿 云 玉 指 凉

上联: 访 山 问 水 知 音 渺	下联: 踏 雪 寻 梅 雅 韵 悠

上联: 万 里 长 江 华 夏 文 明 扬 海 外	下联: 千 秋 大 业 神 州 伟 业 耀 中 华

上联: 兄 及 弟 矣 式 相 好 矣 无 相 尤 矣	下联: 夫 之 子 也 有 所 谓 乎 有 所 谓 乎

上联: 韬 光 养 晦 风 云 独 秀 潇 湘 竹	下联: 反 腐 倡 廉 日 月 长 辉 舜 尧 天

上联: 风 清 月 白 闲 鸥 鹭	下联: 日 丽 花 红 醉 蝶 蜂

上联: 千 里 目	下联: 万 年 春

上联: 燕 唱 山 歌 催 嫩 绿	下联: 莺 歌 燕 舞 唤 春 红

上联: 猪 八 戒 扮 姑 娘 好 歹 不 像	下联: 鼠 百 年 谋 大 计 输 赢 难 得

上联: 石 洞 流 甘 露	下联: 山 泉 洗 俗 尘

上联: 山 能 醉 意 常 流 韵	下联: 水 可 怡 情 总 动 情

上联: 新 朋 老 友 夸 廉 吏	下联: 美 酒 佳 肴 赞 美 人

上联: 并 蒂 花 开 连 理 树	下联: 并 蒂 蒂 开 并 蒂 花

上联: 秋 河 曙 耿 耿	下联: 夜 雨 夜 凄 凄

上联: 云 连 瀑 布 悬 岩 秀	下联: 风 过 泸 州 带 酒 香

上联: 济 水 自 清 河 自 浊	下联: 江 山 如 画 水 长 流

上联: 利 害 当 头 操 守 见	下联: 风 云 在 上 任 行 行

上联: 道 路 畅 通 车 如 流 水	下联: 佛 光 普 照 佛 即 菩 提

上联: 檐 雨 连 珠 难 剪 断	下联: 秋 风 入 梦 不 回 头

上联: 统 一 齐 民 心 民 心 连 两 岸	下联: 同 心 协 力 力 国 力 振 千 秋

上联: 民 常 有 德 犹 多 福	下联: 国 是 无 忧 不 少 忧

上联: 宴 罢 兰 亭 留 墨 宝	下联: 风 来 柳 榭 醉 花 香

上联: 搅 出 新 纹 鱼 共 舞	下联: 搅 浑 浊 水 月 相 随

上联: 五 岳 德 高 儒 释 道	下联: 千 秋 功 伟 武 功 勋

上联: 打 草 寻 山 勤 动 手	下联: 乘 风 破 浪 正 当 头

上联: 千 年 余 庆 千 秋 颂	下联: 万 里 长 城 万 代 兴

上联: 最 喜 老 头 无 赖	下联: 不 知 大 梦 难 圆

上联: 满 心 眷 恋 情 荡 荡	下联: 一 梦 缠 绵 梦 悠 悠

上联: 社 会 和 谐 担 于 肩 上	下联: 人 民 富 裕 乐 在 心 中

上联: 宝 墨 寄 闲 情 紫 洞 清 歌 听 一 曲 鱼 游 春 水	下联: 清 风 传 雅 韵 清 风 明 月 看 千 年 龙 跃 龙 门

上联: 八 一 高 歌 万 里 长 城 担 日 月	下联: 千 秋 大 业 千 秋 伟 业 铸 辉 煌

上联: 对 月 吟 须 浮 白 醉	下联: 临 风 把 酒 醉 红 尘

上联: 应 法 闻 鸡 起 舞	下联: 为 民 跃 马 扬 鞭

上联: 笛 弄 梅 花 五 岭 三 山 飘 瑞 雪	下联: 风 吹 柳 絮 一 江 一 水 荡 春 潮

上联: 与 月 交 心 知 照 顾	下联: 随 风 入 梦 觉 相 思

上联: 鹏 程 万 里 身 须 健	下联: 骏 业 千 秋 气 自 雄

上联: 古 道 经 年 无 信 使	下联: 高 山 流 水 有 知 音

上联: 穿 云 海 两 肩 担 水	下联: 破 雾 天 一 手 遮 天

上联: 愁 情 由 曲 解	下联: 爱 意 自 天 然

上联: 平 天 湖 上 赏 明 月	下联: 杏 花 村 里 品 清 香

上联: 烟 雨 雾 绕 江 心 岛	下联: 日 月 星 辉 海 角 天

上联: 一 生 功 过 盖 棺 定	下联: 半 世 沧 桑 世 事 空

上联: 负 手 云 烟 流 水 高 山 闲 自 在	下联: 放 怀 风 月 清 风 明 月 乐 逍 遥

上联: 每 感 言 轻 难 表 意	下联: 常 思 正 直 不 关 心

上联: 平 天 湖 上 嫦 娥 舒 广 袖	下联: 杏 花 村 里 蝴 蝶 舞 春 风

上联: 敬 业 乐 群 疏 导 源 头 活 水	下联: 求 真 务 实 弘 扬 正 气 清 风

上联: 无 所 不 通 无 所 事	下联: 不 知 难 得 有 知 音

上联: 风 烛 何 堪 空 落 泪	下联: 梅 花 未 必 枉 凝 眉

上联: 浇 科 学 水 挥 智 慧 锄 韶 华 七 秩 同 文 曲	下联: 育 李 培 桃 育 桃 李 育 李 李 千 秋 大 业 篇

上联: 欲 学 狂 人 发 酵 芝 麻 捕 风 捉 影 求 轰 动	下联: 不 知 老 鬼 出 山 山 水 擒 虎 捉 刀 富 富 强

上联: 采 一 篮 童 年 往 事	下联: 赊 几 分 月 色 闲 情

上联: 灵 地 棉 湖 镇	下联: 青 山 碧 水 环

上联: 山 巅 茶 场 散 清 馨 松 竹 瀑 泉 美	下联: 门 外 春 风 化 雨 露 桃 花 春 意 浓

上联: 春 蚕 又 吐 新 茧 把 人 生 画 卷 织 成 美 锦	下联: 蜡 炬 将 燃 旧 情 将 梦 想 文 章 铸 就 丰 碑

上联: 客 来 把 酒 花 间 醉	下联: 风 过 泸 州 梦 里 香

上联: 枕 上 轻 寒 窗 外 雨	下联: 枕 边 寂 寞 梦 中 人

上联: 闹 井	下联: 喧 天

上联: 马 蹄 奋 起 追 欧 美	下联: 羊 角 顶 开 揽 月 光

上联: 字 到 无 锋 称 极 品	下联: 情 于 有 意 是 真 情

上联: 执 掌 三 铡 刀 诚 彰 正 义	下联: 挥 毫 一 蜕 纸 不 负 春 秋

上联: 英 雄 不 问 出 处	下联: 儿 女 难 得 糊 涂

上联: 见 也 难 别 也 难 形 如 彩 凤 双 飞 翼	下联: 闻 之 乐 乐 之 乐 天 下 神 龙 一 啸 风

上联: 勤 俭 生 富 贵	下联: 勤 劳 致 富 强

上联: 慢 赏 好 书 如 品 酒	下联: 闲 吟 佳 句 似 吟 诗

上联: 伴 随 经 史 同 游 古	下联: 共 赏 春 秋 共 读 书

上联: 梅 子 流 酸 怜 苦 李	下联: 梅 花 傲 雪 傲 寒 梅

上联: 岁 前 事 真 仿 佛	下联: 天 上 人 不 如 神

上联: 归 去 来 兮 见 放 魂 萦 三 户 地	下联: 归 来 去 矣 何 堪 梦 断 一 江 秋

上联: 岳 色 河 声 韵 满 京 畿 盛 象 宏 开 中 国 画	下联: 山 光 水 色 花 香 燕 赵 春 风 浩 荡 大 江 潮

上联: 凝 心 总 把 锤 镰 举	下联: 放 眼 常 将 梦 想 飞

上联: 乡 村 六 秩 恍 如 隔 世	下联: 岁 月 千 秋 恰 似 当 时

上联: 喜 庆 三 春 红 我 容 颜 莫 过 新 天 美 酒	下联: 欣 逢 五 福 喜 咱 梦 想 能 成 大 地 欢 歌

上联: 德 政 归 心 燕 舞 秦 川 祥 气 绕	下联: 春 风 得 意 莺 歌 燕 岭 紫 气 腾

上联: 四 朝 长 乐 老	下联: 千 古 古 风 流

上联: 古 月 光 明 新 世 界	下联: 春 风 浩 荡 旧 乾 坤

上联: 古 域 春 光 六 旬 伟 业 逾 千 载	下联: 中 华 骏 业 万 里 宏 图 壮 九 州

上联: 溪 声 流 入 砚	下联: 月 色 碾 成 诗

上联: 鸟 雀 交 鸣 诗 有 韵	下联: 莺 莺 婉 转 曲 无 声

上联: 老 了 流 年 无 悲 无 喜	下联: 时 来 往 事 有 梦 有 香

上联: 门 通 小 径 连 芳 草	下联: 风 过 泸 州 带 酒 香

上联: 嫁 得 潘 家 郎 有 水 有 田 有 米	下联: 迎 来 女 子 女 无 男 无 女 无 男

上联: 往 事 淘 空 将 身 独 许 这 般 月	下联: 相 思 寄 尽 把 酒 相 邀 那 段 情

上联: 走 基 层 听 民 声 向 下 扎 根 接 地 气	下联: 创 大 业 兴 国 运 中 华 崛 起 展 雄 风

上联: 嘉 树 来 西 域 凤 翥 龙 翔 十 里 市 城 添 秀 色	下联: 祥 云 起 东 风 龙 腾 虎 跃 九 州 大 地 焕 新 颜

上联: 盛 世 抒 怀 笔 舞 云 龙 同 作 赋	下联: 春 风 得 意 诗 吟 雅 韵 共 吟 诗

上联: 杨 柳 小 腰 难 把 握	下联: 桃 花 依 旧 笑 春 风

上联: 衣 沾 月 色 如 霜 洗	下联: 袖 拂 清 风 似 水 流

上联: 中 秋 醉 也 二 婆 挥 泪 砸 三 少	下联: 大 地 悲 哉 一 女 伤 心 泣 几 人

上联: 松 桧 老 依 云 外 地	下联: 楼 台 深 锁 洞 中 天

上联: 时 绎 古 书 以 明 古 义	下联: 人 登 高 阁 而 上 高 楼

上联: 论 语 无 平 调	下联: 文 章 有 古 风

上联: 柳 色 常 随 春 意 绿	下联: 梅 香 总 伴 腊 梅 香

上联: 人 兴 财 旺 年 年 好	下联: 日 丽 风 和 处 处 春

上联: 文 化 添 新 拳 石 喷 泉 堪 驻 足	下联: 人 文 焕 彩 心 潮 逐 浪 更 扬 帆

上联: 夜 色 难 留 君 去 意	下联: 春 风 不 度 我 归 心

上联: 乡 书 一 字 千 金 贵	下联: 老 酒 三 杯 万 事 休

上联: 星 斗 满 天 难 比 月	下联: 风 云 一 路 不 如 云

上联: 归 处 不 知 风 几 度	下联: 归 时 已 觉 月 三 更

上联: 忍 性 吞 气 茹 苦 饮 痛	下联: 披 肝 沥 胆 忠 肝 义 胆

上联: 各 有 千 秋 兄 弟 三 人 相 继 去	下联: 相 期 一 日 江 山 一 统 共 争 先

上联: 瘦 月 斜 眯 千 涧 水	下联: 清 风 漫 卷 万 山 云

上联: 知 春 雨 滴 藏 金 地	下联: 晓 夏 风 吹 绽 玉 花

上联: 秋 灯 千 点 雨	下联: 雁 字 几 行 诗

上联: 明 花 许 我 三 分 色	下联: 明 月 谁 人 一 段 情

上联: 韵 流 青 史 丹 樨 在	下联: 风 过 泸 州 带 酒 香

上联: 苍 海 扬 波 高 歌 始 祖 恩 深 似 海	下联: 青 山 耸 翠 大 展 宏 图 志 壮 如 山

上联: 烟 柳 斜 阳 归 去 东 南 余 半 壁	下联: 桃 花 流 水 归 来 春 夏 第 一 枝

上联: 一 朝 飞 雪 满 天 字	下联: 几 度 流 云 遍 地 诗

上联: 杨 柳 多 情 笑 问 春 风 桃 李 事	下联: 桃 花 依 旧 笑 看 明 月 水 云 心

上联: 新 月 如 钩 钩 起 心 田 多 少 事	下联: 春 风 似 剪 裁 成 梦 幻 几 分 愁

上联: 龙 陪 旭 日 观 奇 景	下联: 蛇 伴 春 风 入 画 图

上联: 一 亭 诗 韵 夸 儿 孝	下联: 两 袖 清 风 赞 子 孙

上联: 灵 羊 奉 献 兴 邦 硕 果	下联: 骏 马 奔 腾 富 国 宏 图

上联: 卓 越 超 群 德 达 三 江 通 四 海	下联: 达 观 致 远 文 明 四 海 耀 千 秋

上联: 抚 石 可 辨 文 章 影	下联: 挥 笔 能 书 翰 墨 香

上联: 红 桃 笑 破 胭 脂 口	下联: 绿 柳 轻 摇 翡 翠 裙

上联: 室 陋 几 曾 输 月 色	下联: 人 闲 何 处 觅 花 香

上联: 舍 经 难 顿 悟 未 上 此 楼 永 与 佛 天 离 咫 尺	下联: 修 道 可 修 行 无 边 彼 岸 长 随 法 雨 洗 尘 心

上联: 春 风 已 过 斜 阳 道	下联: 明 月 还 临 明 月 楼

上联: 数 遍 千 官 无 所 事	下联: 几 多 万 事 有 知 音

上联: 四 面 荷 花 三 面 柳 于 泉 城 润 色	下联: 万 家 灯 火 万 家 灯 照 梦 想 成 真

上联: 嫁 女 婚 男 百 年 合	下联: 结 婚 婚 姻 四 海 同

上联: 人 在 征 途 满 园 春 色 谁 关 注	下联: 情 牵 故 里 一 片 冰 心 我 自 由

上联: 厨 艺 高 超 工 境 界	下联: 匠 心 巧 妙 妙 文 章

上联: 踏 浪 归 来 风 雨 江 湖 犹 识 我	下联: 登 楼 望 远 江 山 山 水 不 知 年

上联: 落 纸 云 烟 凝 往 事	下联: 挥 毫 笔 墨 写 新 篇

上联: 艺 苑 腾 蛟 圣 域 方 收 硕 果	下联: 联 坛 跃 马 神 州 更 上 层 楼

上联: 歧 路	下联: 长 江

上联: 鹊 飞 疑 是 银 河 渡	下联: 雁 去 疑 为 玉 簟 秋

上联: 创 业 创 新 天 下 道 源 新 故 事	下联: 承 前 启 后 世 间 遗 产 大 文 章

上联: 红 尘 信 似 云 霞 聚	下联: 青 史 犹 如 日 月 长

上联: 燕 舞 新 年 喜	下联: 莺 歌 盛 世 春

上联: 红 绿 青 黄 黑 白 皆 都 有	下联: 红 红 绿 绿 红 红 各 样 红

上联: 祥 光 烁 破 千 生 病	下联: 瑞 气 迎 来 万 里 春

上联: 灵 似 甘 霖 解 口 渴	下联: 心 如 明 月 照 人 心

上联: 清 风 拂 面 迷 人 意	下联: 明 月 盈 怀 醉 客 心

上联: 欲 邀 我 佛 来 执 意 催 山 起 祥 云 莲 开 宝 座	下联: 不 染 尘 嚣 去 修 身 养 性 修 善 果 我 悟 禅 机

上联: 千 篇 一 律 无 新 意	下联: 万 古 千 秋 有 古 风

上联: 法 乃 国 纲 行 廉 反 腐 一 身 胆	下联: 德 为 民 本 治 国 安 邦 两 袖 风

上联: 书 中 便 晓 人 间 事	下联: 笔 下 方 知 世 上 情

上联: 坐 拥 湖 山 淡 看 利 名 鹤 子 梅 妻 仙 是 我	下联: 卧 听 风 月 闲 听 风 雨 松 风 竹 韵 月 为 邻

上联: 寂 寞 溪 边 客	下联: 相 思 陌 上 人

上联: 湖 中 玉 镜 邀 明 月	下联: 岭 上 青 山 伴 夕 阳

上联: 两 井 水 何 奇 活 人 济 世 苍 天 鉴	下联: 九 州 山 不 老 壮 志 凌 云 白 鹤 飞

上联: 拾 级 上 天 都 赏 景 健 身 臂 展 青 松 迎 奥 运	下联: 登 峰 登 绝 顶 登 高 极 目 胸 怀 赤 县 展 雄 风

上联: 三 门 组 稿 紫 燕 编 排 采 撷 春 光 镶 版 面	下联: 四 海 蜚 声 金 鸡 报 晓 迎 来 喜 气 满 人 间

上联: 窑 火 旺 千 秋 光 耀 中 华 美 誉 长 随 丝 路 远	下联: 城 乡 兴 百 业 兴 隆 大 业 宏 图 大 展 画 图 新

上联: 竹 篙 桂 楫 飞 如 箭	下联: 铢 襻 香 腰 巧 似 珠

上联: 新 岁 新 景 新 气 象	下联: 好 风 好 雨 好 风 光

上联: 促 寿	下联: 延 年

上联: 青 春 无 价 没 人 卖	下联: 岁 月 有 情 有 酒 浇

上联: 知 我 法 随 缘 不 生 执 念	下联: 悟 禅 心 自 悟 即 是 修 行

上联: 墨 笔 狂 书 龙 飞 凤 舞 出 神 韵	下联: 联 花 怒 放 蝶 舞 蜂 飞 入 画 图

上联: 茂 星 环 月 天 重 笑	下联: 明 月 清 风 夜 未 央

上联: 一 片 春 云 凝 紫 气	下联: 满 园 秋 色 染 红 霞

上联: 葡 萄 架 下 斟 新 酒	下联: 橄 榄 枝 头 唱 大 风

上联: 清 江 一 盏 多 情 月	下联: 明 月 千 秋 有 意 人

上联: 贺 新 春 换 新 居 心 花 怒 放	下联: 迎 盛 世 兴 骏 业 骏 业 兴 隆

上联: 股 逢 牛 市 套 方 解	下联: 脚 踏 马 鞍 车 上 行

上联: 死 生 今 忽 异	下联: 死 死 自 生 悲

上联: 云 锁 山 门 关 俗 客	下联: 月 临 水 榭 洗 尘 心

上联: 花 香 难 果 腹	下联: 月 色 不 关 心

上联: 淡 了 红 尘 淡 了 往 事 我 心 谁 洗	下联: 抛 开 俗 念 抛 却 尘 缘 梦 幻 我 来

上联: 平 安 竹 报 全 家 庆	下联: 幸 福 花 开 满 院 春

上联: 盛 世 添 筹 娱 晚 景	下联: 高 山 流 水 遇 知 音

上联: 纵 览 古 今 中 外 千 年 史	下联: 纵 观 天 地 人 间 一 片 天

上联: 润 物 滋 春 甘 作 雨	下联: 修 身 养 性 乐 为 天

上联: 苦 口 婆 心 是 佛	下联: 清 风 明 月 为 人

上联: 长 大 知 才 须 有 用	下联: 少 年 学 问 要 无 求

上联: 及 此 春 初 因 时 游 目	下联: 与 时 俱 进 自 有 知 音

上联: 求 真 务 实 锤 炼 工 匠 精 神 助 力 调 转 促	下联: 利 国 利 民 利 民 家 庭 幸 福 安 居 乐 升 平

上联: 日 丽 鲜 花 含 淑 气	下联: 风 和 紫 燕 舞 春 风

上联: 时 时 灵 气 时 时 趣	下联: 处 处 风 情 处 处 春

上联: 鹏 起 抗 英 捐 热 血	下联: 龙 腾 盛 世 展 雄 风

上联: 眼 皮 打 架 不 如 去	下联: 眉 眼 看 穿 无 奈 何

上联: 拜 佛 许 下 千 钧 愿	下联: 读 书 读 书 一 卷 书

上联: 乐 有 诗 书 墙 上 挂	下联: 乐 无 日 月 水 中 捞

上联: 清 廉 简 约 护 州 政	下联: 正 气 清 廉 民 族 魂

上联: 钟 高 密 地 灵 官 居 著 作	下联: 鼓 高 高 天 地 帝 业 传 承

上联: 野 火 烧 不 尽	下联: 春 风 吹 得 来

上联: 杯 水 可 容 天 上 月	下联: 春 风 不 度 世 间 花

上联: 宝 鸡 报 晓 十 五 年 看 创 新 绘 出 和 谐 画	下联: 金 凤 朝 阳 万 千 里 喜 发 展 绘 成 富 裕 图

上联: 生 日 星 光 晃 晃	下联: 死 年 月 色 朦 胧

上联: 香 雪 无 声 自 好 色	下联: 春 风 有 意 总 关 情

上联: 上 海 自 来 水 来 自 海 上	下联: 南 山 东 阿 阿 阿 阿 胶 胶

上联: 立 志 增 才 饱 含 昌 珏 千 滴 汗	下联: 修 身 致 富 勤 奋 骅 骝 一 片 天

上联: 今 生 缘 尽 无 相 欠	下联: 来 世 相 逢 不 再 来

上联: 螺 尖 蚌 扁 鳖 甲 圆 满 盘 皆 壳	下联: 凤 尾 凰 雏 凤 翅 飞 一 翅 齐 毛

上联: 未 把 萤 虫 当 蜡 烛	下联: 常 将 雁 字 作 文 章

上联: 得 意 春 风 莫 忘 我	下联: 无 心 明 月 不 知 年

上联: 长 天 待 我 舒 鹏 翼	下联: 大 地 回 春 入 马 蹄

上联: 凭 栏 望 月 佳 人 泪 尽 关 山 远	下联: 把 酒 临 风 故 友 情 深 岁 月 长

上联: 秋 翻 枫 页 谁 为 画	下联: 月 落 荷 塘 我 作 诗

上联: 一 夜 春 风 千 岭 翠	下联: 满 山 秋 色 满 山 红

上联: 泄 柳	下联: 舂 薪

上联: 周 报 传 八 卦	下联: 文 章 耀 九 州

上联: 一 亭 风 月 冷	下联: 两 岸 水 云 闲

上联: 桃 红 残 雪 尽	下联: 柳 绿 暖 风 来

上联: 江 清 随 月 老	下联: 海 阔 任 风 狂

上联: 千 秋 扬 正 气	下联: 万 里 荡 春 风

上联: 一 亭 风 雨 过	下联: 千 古 古 今 传

上联: 科 技 兴 邦 旧 岁 已 添 三 道 喜	下联: 勤 劳 致 富 新 年 更 上 一 层 楼

上联: 年 来 惆 怅 还 依 旧	下联: 风 过 泸 州 带 酒 香

上联: 一 叶 归 帆 天 际 远	下联: 千 帆 破 浪 水 中 央

上联: 道 德 文 章 都 入 理	下联: 文 章 笔 墨 总 关 情

上联: 一 宵 大 雨 蛙 声 乱	下联: 半 夜 清 风 燕 影 斜

上联: 叽 哩 咕 噜 怪 话	下联: 汪 汪 汪 汪 真 情

上联: 细 雨 梦 回 鸡 塞 远	下联: 春 风 春 暖 马 蹄 香

上联: 春 风 不 解 花 间 语	下联: 明 月 难 知 梦 里 人

上联: 上 下 五 千 年 龙 族 问 天 华 夏 煌 煌 谁 肇 造	下联: 纵 横 八 万 里 人 文 蔚 起 神 州 熠 熠 我 争 先

上联: 先 锋 奋 击 除 倭 寇	下联: 大 业 宏 开 振 国 威

上联: 龙 荣 半 世 千 秋 贵	下联: 蛇 舞 三 春 万 里 香

上联: 解 粽 筵 开 老 翁 半 日 斟 蒲 酒	下联: 寻 梅 酒 醉 稚 子 一 庭 摘 菊 诗

上联: 门 迎 紫 气 铺 天 福	下联: 户 纳 春 风 入 画 图

上联: 云 开 雾 散	下联: 电 闪 雷 鸣

上联: 藏 书 集 画 展 艺 施 才 巧 手 掀 开 新 世 界	下联: 泼 墨 挥 毫 挥 毫 泼 墨 丹 心 写 就 大 文 章

上联: 浮 云 窥 海 月	下联: 明 月 照 天 心

上联: 寸 阴 宁 越 度	下联: 寸 草 不 争 荣

上联: 莫 让 光 阴 如 水 逝	下联: 休 将 岁 月 似 风 流

上联: 水 秀 山 青 花 容 月 貌	下联: 花 香 鸟 语 鸟 语 花 香

上联: 我 劝 天 公 重 抖 擞	下联: 谁 怜 月 老 再 团 圆

上联: 日 序 班 头 如 候 补	下联: 春 风 拂 面 似 花 开

上联: 喜 逢 盛 世 频 增 寿	下联: 欢 庆 新 年 喜 报 春

上联: 玉 笛 约 春 峰 染 翠	下联: 金 樽 邀 月 酒 飘 香

上联: 云 梦 八 千 里	下联: 风 情 一 万 年

上联: 圣 代 即 今 多 雨 露	下联: 名 山 何 处 有 烟 霞

上联: 花 影 云 拖 地	下联: 月 光 月 照 天

上联: 家 中 自 有 春 秋 韵	下联: 笔 下 常 留 岁 月 痕

上联: 乌 有	下联: 黄 无

上联: 心 头 苇 荡 笔 底 牧 歌 风 华 十 秩 弘 联 韵	下联: 笔 下 风 流 笔 端 墨 韵 锦 绣 千 秋 绘 画 图

上联: 欲 静 憎 蝉 噪	下联: 方 知 喜 鹊 鸣

上联: 春 信 千 家 传 紫 燕	下联: 花 香 万 里 绽 红 花

上联: 槐 花 落 尽 无 心 事	下联: 柳 絮 飞 空 有 意 思

上联: 赏 松 风 格 学 竹 虚 心 藏 梅 傲 骨 真 君 子	下联: 敬 业 精 神 培 桃 育 李 继 往 开 来 大 丈 夫

上联: 涑 水 欢 歌 桃 醉 杜 康 三 千 岁 月	下联: 瑶 山 焕 彩 燕 飞 燕 赵 一 派 春 光

上联: 欲 穷 千 里 宏 观 目	下联: 不 负 一 生 大 爱 心

上联: 饮 酒 花 间 酒 散 花 间 花 亦 醉	下联: 吟 诗 月 下 风 流 月 下 月 犹 寒

上联: 磊 成 三 块 石	下联: 风 过 一 重 山

上联: 栉 风 沐 雨 人 生 路	下联: 戴 月 披 星 世 界 观

上联: 山 中 宰 相 南 朝 典	下联: 门 外 仙 人 北 国 风

上联: 前 浪 陈 江 缺 后 浪	下联: 高 山 流 水 遇 知 音

上联: 心 能 转 物	下联: 志 可 凌 云

上联: 古 狗	下联: 新 人

上联: 姥 姥	下联: 公 公

上联: 书 中 自 出 千 钟 粟 取 之 有 道	下联: 笔 下 难 求 万 卷 书 得 者 无 忧

上联: 桃 红 李 白 妃 子 醉	下联: 柳 绿 花 红 美 人 娇

上联: 对 月 梳 妆 青 丝 钩 角 挂	下联: 临 风 把 盏 红 袖 画 眉 开

上联: 信 手 拈 来 神 笔 已 臻 诗 境 妙	下联: 随 心 放 下 神 州 更 上 画 堂 高

上联: 一 杯 相 送 花 辞 去	下联: 两 袖 清 风 月 照 来

上联: 无 缝 对 接 华 阳 喜 逢 日 月 同 辉	下联: 万 象 更 新 大 地 春 到 春 夏 秋 冬

上联: 先 声 传 喜 讯 有 幸 居 楼 多 年 愿 望 一 朝 遂	下联: 大 业 展 宏 图 无 忧 处 世 万 里 欣 逢 百 业 兴

上联: 岩 乃 山 下 石	下联: 山 如 水 中 山

上联: 廉 启 新 章 正 气 敢 教 风 气 爽	下联: 德 扬 正 气 清 风 不 让 党 旗 红

上联: 风 定 秋 千 闲 卧 月	下联: 月 明 夜 半 静 敲 诗

上联: 笔 有 深 情 耕 月 夜	下联: 心 无 俗 虑 洗 尘 心

上联: 端 午 万 村 包 粽 子	下联: 元 宵 一 夜 挂 灯 花

上联: 花 落 一 窗 诗 串 起	下联: 月 圆 半 夜 梦 随 来

上联: 刘 剑 胆	下联: 李 文 心

上联: 口 齿 不 清 常 跑 调	下联: 嘴 巴 巴 肉 好 揩 油

上联: 鸟 语 花 香 有 声 有 色 三 春 景	下联: 花 香 鸟 语 无 语 无 声 四 季 歌

上联: 一 对 手 纹 启 一 宗 始 祖	下联: 两 行 文 字 传 万 代 宗 师

上联: 伏 枥 雄 心 在	下联: 扬 鞭 壮 志 坚

上联: 云 边 路 绕 秋 山 色	下联: 月 下 花 开 春 水 香

上联: 星 淡 月 明 天 气 派	下联: 风 和 日 丽 地 精 神

上联: 翠 竹 黄 花 皆 佛 性	下联: 清 风 明 月 是 禅 心

上联: 按 经 济 规 律 发 展 经 济	下联: 以 经 济 经 济 和 谐 社 会

上联: 寒 梅 已 作 东 风 信	下联: 明 月 还 为 北 斗 星

上联: 大 火 焚 林 真 够 呛	下联: 小 桥 流 水 不 糊 涂

上联: 星 斗 满 天 无 月 亮	下联: 春 风 遍 地 有 花 香

上联: 依 诗 咏 对 须 遵 格	下联: 对 酒 当 歌 不 厌 烦

上联: 林 间 鸟 奏 笙 簧 月	下联: 陌 上 花 开 蝶 恋 花

上联: 济 世 良 方 除 病 患	下联: 修 身 正 道 济 民 生

上联: 锐 气 千 年 盈 北 海	下联: 雄 心 万 里 壮 南 疆

上联: 寨 外 涓 溪 流 我 泪	下联: 门 前 紫 燕 舞 春 风

上联: 风 一 动 千 般 变 化	下联: 月 半 轮 万 里 乾 坤

上联: 青 山 昨 夜 风 吹 过	下联: 碧 水 今 宵 月 照 来

上联: 作 后 学 津 梁 藜 照 年 年 旧 梦 未 抛 天 禄 阁	下联: 为 先 生 典 范 读 书 代 代 高 风 不 负 栋 梁 材

上联: 闲 云 拂 岫 时 舒 展	下联: 明 月 临 窗 每 觊 觎

上联: 挥 笔 如 剑 倚 麓 山 豪 气 干 云 揽 月 去	下联: 挥 毫 似 刀 挥 铁 马 豪 情 壮 志 踏 云 来

上联: 执 镫 引 缰 扶 农 跃 上 千 里 马	下联: 乘 风 破 浪 破 浪 冲 开 一 片 天

上联: 桃 呈 福 寿 李 呈 喜	下联: 梅 吐 芬 芳 李 贺 春

上联: 应 虎 步 春 声 和 风 带 雨 谁 施 泽	下联: 为 龙 腾 盛 世 盛 世 扬 帆 我 领 航

上联: 独 览 群 峰 高 低 不 论 皆 平 视	下联: 常 思 大 道 远 近 无 争 尽 远 观

上联: 塔 影 梅 香 山 色 千 般 秀	下联: 山 光 水 色 湖 光 万 里 春

上联: 春 雨 依 然 有 意 绿	下联: 桃 花 依 旧 笑 颜 红

上联: 双 龙 曾 献 瑞	下联: 百 鸟 不 争 春

上联: 表 面	下联: 开 心

上联: 业 内 富 者 者 又 者 者 者 富 者	下联: 家 中 家 家 家 家 家 家 家 家 家

上联: 心 为 业 本	下联: 德 是 人 才

上联: 扇 转 非 为 己	下联: 杯 空 不 是 人

上联: 云 石 足 勾 留 稔 知 清 气 源 山 野	下联: 江 山 须 造 就 明 察 古 今 论 古 今

上联: 卓 荦 岂 共 红 尘 误	下联: 达 人 何 必 白 云 归

上联: 民 安 国 泰 三 春 送 暖 三 农 活	下联: 政 善 人 和 万 象 更 新 百 业 兴

上联: 睡 前 添 一 笔	下联: 醉 后 醉 三 杯

上联: 环 球 通 讯	下联: 大 地 回 春

上联: 登 宝 塔 品 中 山 松 醪 庆 贺 民 强 国 富	下联: 登 高 楼 看 大 地 春 色 迎 来 国 泰 民 安

上联: 鸟 语 空 山 惊 寂 寞	下联: 花 香 小 院 惹 相 思

上联: 生 发 灵 可 医 疗 毛 病	下联: 死 生 命 能 医 疗 病 灾

上联: 对 连 错	下联: 联 对 联

上联: 松 下 下 棋 寻 子 路	下联: 月 中 中 箭 有 嫦 娥

上联: 柳 枕 湖 边 鱼 戏 月	下联: 梅 开 岭 上 鸟 争 春

上联: 山 色 泉 声 涵 静 照	下联: 松 风 竹 韵 入 清 流

上联: 清 明 尘 世 清 明 景	下联: 雨 露 风 情 淡 雅 情

上联: 魔 日 逆 天 曾 使 樱 花 泣 血	下联: 春 风 得 意 不 教 柳 絮 伤 心

上联: 纵 横 笔 下 三 千 意	下联: 俯 仰 胸 中 万 仞 峰

上联: 盛 世 春 临 小 康 在 望 喜 听 农 家 储 百 万	下联: 小 康 福 满 大 有 无 边 欣 看 大 业 耀 千 秋

上联: 源 浊 难 见 流 清 水	下联: 望 重 方 知 望 远 山

上联: 一 波 情 浪 涌 平 湖	下联: 满 目 春 光 入 画 图

上联: 箬 笠 红 尘 外	下联: 瑶 琴 绿 绮 中

上联: 望 路 烟 霞 外	下联: 登 山 日 月 中

上联: 奇 文 共 欣 赏 偶 尔 重 温 陶 令 句	下联: 佳 句 同 吟 咏 何 须 再 诵 谢 公 诗

上联: 花 香 一 盏 收 浓 淡	下联: 月 色 半 窗 照 浅 深

上联: 新 月 初 悬 没 线 银 钩 能 钓 海	下联: 清 风 乍 起 无 声 玉 笛 可 吹 箫
