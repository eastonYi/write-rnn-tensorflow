
# Generative Handwriting Demo using TensorFlow

## reference
See this blog post at [blog.otoro.net](http://blog.otoro.net/2015/12/12/handwriting-generation-demo-in-tensorflow) for more information.

## theory
An attempt to implement the random handwriting generation portion of Alex Graves' [paper](http://arxiv.org/abs/1308.0850).


## data
### 增量坐标数据选择的合理性
人在写字时, 都是以开始落笔的坐标作为起点, 一般不会考虑全局坐标. 所以如果写字的训练数据都是从左上角开始的, 那么如果要求从其他地方开始进行采样, 就会缺乏信息. 因为模型对绝对坐标敏感.

batch_size=50, seq_length=300

训练数据和验证数据就是错开了一个时刻,这和训练language model是一样的:
```
x_batch.append(np.copy(data[idx:idx+self.seq_length]))
y_batch.append(np.copy(data[idx+1:idx+self.seq_length+1]))
```
![](assets/微信截图_20170817140533.png)

取validation data(仅取一个batch)
![](assets/微信截图_20170817152851.png)

取训练时的batch
![](assets/微信截图_20170817152838.png)
`self.pointer`用来标记取sequence作为训练数据的指针. 这个指针在当前sequence data中取完一个长度为`seq_length`的sequence作为batch中的一个后, 指针可能是指向下一个sequence data, 也可能原地不动, 原地概率与sequence data的长度成正比. 当指针指到sequence data底时,返回第一个.


## Model
模型分两种状态: 一个是训练状态,一个是采样状态(训练状态`infer =False`).
使用多层LSTM作为RNN的cell

```
outputs, state_out = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, self.state_in, cell, loop_function=None, scope='rnnlm')
output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, args.rnn_size])
```
### RNN Model Output
```
outputs, state_out =tf.nn.bidirectional_dynamic_rnn(...)
output = tf.reshape(tf.concat(axis=2, values=outputs), [-1, 2 *args.rnn_size])
output = tf.nn.xw_plus_b(output, output_w, output_b)
```
最终的输出是`[(batch_size*seq_length) * NOUT]`

### Distribution Parameters Output
RNN模型的输出需要经过非线性映射才能得到合法的参数:
```
o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos =
    get_mixture_coef(output)
```
*因为模型输出是分布/分布参数, 不能直接与target比较, 损失函数是target输出看做是一个采样(分布空间中的一个坐标)对应于模型输出分布中的概值的大小. 这个概率值越大越好.*
对于Mixed Gaussian Distribution, 求输入点在多有成分中的概率, 然后加权求和.
```
result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
epsilon = 1e-20
result1 = tf.multiply(result0, z_pi)
result1 = tf.reduce_sum(result1, 1, keep_dims=True)
```

因为分布是连续的, 一点处的概率值为无穷小. 把坐标代入分布函数得到的值并不是概率值, 不过可以作为概率值大小比较的参考. 为了简便, 直接把此值作为概率值的代表:
```
tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
```
这个函数就是取(x1, x2)处的分布函数值
有时这个值会突破$p<1$的限制(还不是偶尔), 因为论文中使用的是负对数函数, 损失值会突破$C>0$的限制. 在保留损失函数的情况下, 需要将分布函数值整体限制在$\forall \textbf{X}, f(\textbf{X})<1$中;

```
lossfunc = get_lossfunc(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos, x1_data, x2_data, eos_data)
```
进行的是一个batch的loss.
另一个思路是将连续概率分布离散化,通过把输入空间进行分割把每一个区域的概率值算出来, 然后判断输入属于哪一个区域, 返回概率值.


```
flat_target_data = tf.reshape(self.target_data,[-1, 3])
[x1_data, x2_data, eos_data] = tf.split(axis=1, num_or_size_splits=3, value=flat_target_data)
```
注意这里的target: `x1_data, x2_data, eos_data`都是2d Tensor, 其中axis =1 的长度为1. 后面`get_lossfunc`中`result1`在`reduce_sum`时保留了维度(`result1 = tf.reduce_sum(result1, 1, keep_dims=True)`); `result2`的`z_eos`也和`eos_data`都是2d(`z_eos = output[:, 0:1]`, `[x1_data, x2_data, eos_data] = tf.split(axis=1, num_or_size_splits=3, value=flat_target_data)`), 于是`result1`和`result2`都是2d向 Tensor, 一次可以合法地对应元素相加.
## Train

You will need permission from [these wonderful people](http://www.iam.unibe.ch/fki/databases/iam-on-line-handwriting-database) people to get the IAM On-Line Handwriting data.  Unzip `lineStrokes-all.tar.gz` into the data subdirectory, so that you end up with `data/lineStrokes/a01`, `data/lineStrokes/a02`, etc.  Afterwards, running `python train.py` will start the training process.

A number of flags can be set for training if you wish to experiment with the parameters.  The default values are in `train.py`

## Sample
采样时, model的batch参数设为1(只生成一个sequence), seq_length也设为1(因为下一时刻的输入得靠当前时刻的输出采样作为输入, 没办法预先提供给model)
首先每个分量的范围不同, 类型也不同. 神经网络的输出是连续空间$R$(??)上的值, 如果实际参数是一定取值范围内的连续值, 可以经过非线性映射处理, 采样时直接采经过处理后的数; 如果是bool变量, 实际取值是$\{0,1\}$, 那么将网络输出映射到$(0, 1)$, 采样时再生成一个$(0,1)$之间的随机数映射后的数进行对比,决定输出0或1.
对于MGD, 采样时先平均随机选取一个Gaussian成分, 在这个成分的基础上进行采样(这样处理的好处是可以套用现成的Gaussian分布采样程序). 序列的结束标志是一个(0,1)区间的数, 也采用随机的方式:
```
eos = 1 if random.random() < o_eos[0][0] else 0
```
确定bool变量的采样结果.


### IPython interactive session.

If you wish to experiment with this code interactively, just run `%run -i sample.py` in an IPython console, and then the following code is an example on how to generate samples and show them inside IPython.

```
[strokes, params] = model.sample(sess, 800)
draw_strokes(strokes, factor=8, svg_filename = 'sample.normal.svg')
draw_strokes_random_color(strokes, factor=8, svg_filename = 'sample.color.svg')
draw_strokes_random_color(strokes, factor=8, per_stroke_mode = False, svg_filename = 'sample.multi_color.svg')
draw_strokes_eos_weighted(strokes, params, factor=8, svg_filename = 'sample.eos.svg')
draw_strokes_pdf(strokes, params, factor=8, svg_filename = 'sample.pdf.svg')

```

# data processing
![](assets/微信截图_20170817140533.png)
原始数据:
`stroke_data`: 一段时间的数据, 看作是一个sequence, 模型建模的就是sequence变量的之间的依赖关系. 长度任意.
`strokes`: 多个时间片段的数据, 看作是多个独立的sequence组成的list, 个数任意.

处理数据:
1. 筛选长度大于`seq_length`的sequence
```
for data in self.raw_data:
    if len(data) > (self.seq_length+2):
```
2. 异常值限制
```
data = np.minimum(data, self.limit)
data = np.maximum(data, -self.limit)
```
3. 确定数据类型+放缩
```
data = np.array(data,dtype=np.float32)
data[:,0:2] /= self.scale_factor
```
4. 分训练集与测试集
```
cur_data_counter = cur_data_counter + 1
if cur_data_counter % 20 == 0:
    self.valid_data.append(data)
else:
    self.data.append(data)
```
5. batch选取
为了高效, 模型需要批训练, 每一批的sequence必须是等长的(超参数`seq_length`). 为了简单起见, 模型只建模`seq_length`以内的序列关系. 设计这样取batch的规则: 一个sequence可以取多个`seq_length`长度的sequence放入batch中, 具体取多少个,与sequence的长度概率上成正比, 另外取`seq_length`的位置是随机的. 直观上这样处理比将sequence硬切分成`seq_length`效果要好.
![](assets/微信截图_20170817152838.png)
训练数据和验证数据就是错开了一个时刻,这和训练language model是一样的:
```
x_batch.append(np.copy(data[idx:idx+self.seq_length]))
y_batch.append(np.copy(data[idx+1:idx+self.seq_length+1]))
```
# training
和语言模型一样的训练. 取一段sequence, 向后一个时刻作为target. rnn网络中cell每个时刻的输入都是这个sequence的每个时刻, 而不需要前一时刻输出的条件概率采样(需要前一时刻cell的内部状态).
因为模型输出是分布/分布参数, 不能直接与target比较,需要对分布进行采样后才能进行比较.
# Sampling
采样是另一个与训练等同重要的环节. 在训练过程中用不到采样, 是网络训练结束后, 生成序列的过程才需要.
下面是代码实现时的一些常规做法, 还没有找到有理论支撑的更好的做法.
训练时cell的初始化是全0, 采样时cell的初始化也是全0. 另外设置cell的0时刻输入.
## 混合成分的采样
首先每个分量的范围不同, 类型也不同. 神经网络的输出是连续空间$R$(??)上的值, 如果实际参数是一定取值范围内的连续值, 可以经过非线性映射处理, 采样时直接采经过处理后的数; 如果是bool变量, 实际取值是$\{0,1\}$, 那么将网络输出映射到$(0, 1)$, 表示取某值的概率. 再经过采样确定实际取值.
对于MGD, 采样时先平均随机选取一个Gaussian成分, 在这个成分的基础上进行采样(这样处理的好处是可以套用现成的Gaussian分布采样程序). 序列的结束标志是一个(0,1)区间的数, 也采用随机的方式:
```
eos = 1 if random.random() < o_eos[0][0] else 0
```
确定bool变量的采样结果.

在生成过程中每一时刻模型输出分布参数, 直接按照分布进行采样比较困难,
## 拟合
