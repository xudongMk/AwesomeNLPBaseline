## 文本分类

这里首先介绍一篇基于深度学习的文本分类综述，《Deep Learning Based Text Classification: A Comprehensive Review》，论文来源：https://arxiv.org/abs/2004.03705

**文本分类简介**：

文本分类是NLP中一个非常经典任务（对给定的句子、查询、段落或者文档打上相应的类别标签）。其应用包括机器问答、垃圾邮件识别、情感分析、新闻分类、用户意图识别等。文本数据的来源也十分的广泛，比如网页数据、邮件内容、聊天记录、社交媒体、用户评论等。

**文本分类三大方法**：

1. Rule-based methods：使用预定义的规则进行分类，需要很强的领域知识而且系统很难维护
2. ML (data-driven) based methods：经典的机器学习方法使用特征提取（Bow词袋等）来提取特征，再使用朴素贝叶斯、SVM、HMM、Gradien Boosting Tree和随机森林等方法进行分类。深度学习方法通常使用的是end2end形式，比如Transformer、Bert等。
3. Hybrid methods：基于规则和基于机器学习（深度学习）方法的混合

**文本分类任务**：

1. 情感分析（Sentiment Analysis）：给定文本，分析用户的观点并且抽取出他们的主要观点。可以是二分类，也可以是多分类任务
2. 新闻分类（News Categorization）：识别新闻主题，并给用户推荐相关的新闻。主要应用于推荐系统
3. 主题分析（Topic Analysis）：给定文本，抽取出其文本的一个或者多个主题
4. 机器问答（Question Answering）：提取式（extractive），给定问题和一堆候选答案，从中识别出正确答案；生成式（generative），给定问题，然后生成答案。（NL2SQL？）
5. 自然语言推理（Natural Language Inference）：文本蕴含任务，预测一个文本是否可以从另一个文本中推断出。一般包括entailment、contradiction和neutral三种关系类型

**文本分类模型（深度学习）**：

1. 基于前馈神经网络（Feed-Forward Neural Networks）
2. 基于循环神经网络（RNN）
3. 基于卷积神经网络（CNN）
4. 基于胶囊高神经网络（Capsule networks）
5. 基于Attention机制
6. 基于记忆增强网络（Memory-augmented networks）
7. 基于Transformer机制
8. 基于图神经网络
9. 基于孪生神经网络（Siamese Neural Network）
10. 混合神经网络（Hybrid models）

详解见https://blog.csdn.net/u013963380/article/details/106957420（只详细描述了前4种深度学习模型）。

## 文本分类数据集

Deep Learning Based Text Classification: A Comprehensive Review一文中提到了很多的文本分类的数据集，大多数是英文的。

下面列出一些中文文本分类数据集：

| 数据集   | 说明                                                         | 链接                                                         |
| :------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| THUCNews | THUCNews是根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成。<br />包含财经、彩票、房产、股票、家居、教育等14个类别。<br />原始数据集见：[链接](http://thuctc.thunlp.org/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews) | [下载地址](http://thuctc.thunlp.org/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews) |
| 今日头条 | 来源于今日头条，为短文本分类任务，数据包含15个类别           | [下载地址](https://storage.googleapis.com/cluebenchmark/tasks/tnews_public.zip) |
| IFLYTEK  | 1.7万多条关于app应用描述的长文本标注数据，包含和日常生活相关的各类应用主题，共119个类别 | [下载地址](https://storage.googleapis.com/cluebenchmark/tasks/iflytek_public.zip) |
| 新闻标题 | 数据集来源于Kesci平台，为新闻标题领域短文本分类任务。<br />内容大多为短文本标题(length<50)，数据包含15个类别，共38w条样本 | [下载地址](https://pan.baidu.com/s/1vyGSIycsan3YWHEjBod9pw) |
| 复大文本 | 数据集来源于复旦大学，为短文本分类任务，数据包含20个类别，共9804篇文档 | [下载地址](https://pan.baidu.com/s/1vyGSIycsan3YWHEjBod9pw) |
| OCNLI    | 中文原版自然语言推理，是第一个非翻译的、使用原生汉语的大型中文自然语言推理数据集<br />详细见https://github.com/CLUEbenchmark/OCNLI | [下载地址](https://storage.googleapis.com/cluebenchmark/tasks/ocnli_public.zip) |
| 情感分析 | OCEMOTION–中文情感分类，对应文章https://www.aclweb.org/anthology/L16-1291.pdf<br />原始数据集未找到，只有一部分数据 | [下载地址](https://pan.baidu.com/s/1vyGSIycsan3YWHEjBod9pw) |
| 更新ing  | ...                                                          | ...                                                          |

还有一些其他的中文文本数据集，可以在CLUE上搜索，CLUE地址：https://www.cluebenchmarks.com/ ，但是下载需要注册账号，有的链接失效，有的限制日下载次数，这里放到百度网盘供下载学习使用，提取码：lrmv。（请勿用于商业目的）

## 文本分类Baseline算法实现

使用Tensorflow1.x版本Estimator高阶api实现常见文本分类算法，主要包括前馈神经网络（all 全连接层）模型、双向LSTM模型、文本卷积网络（TextCnn）、Transformer。

环境信息：

tensorflow==1.13.1

python==3.7

**数据预处理**

要求训练集和测试集分开存储（提供划分数据集方法），另外需要对文本进行分词，数据EDA部分可以见示例中的tnews_data_eda.ipynb文件。

在训练模型前，需要先运行preprocess.py文件进行数据预处理，将数据处理成id形式并保存为pkl形式，另外中间过程产生的词表也会保存为vocab.txt文件。

**文件结构**

- data_path：数据集存放的位置
- data_utils：数据处理相关的工具类存放位置
- model_ckpt：模型checkpoint保存的位置
- model_pb：pb形式的模型保存的位置
- models：文本分类baseline模型存放的位置，包括BiLstm、TextCnn等
- train_main.py：模型训练主入口
- preprocess.py：数据预处理代码，包括划分数据集、转换文本为id等
- tf_metrics.py：tensorflow1.x版本不支持多分类的指标函数，这里使用的是Guillaume Genthial编写的多分类指标函数，[github地址](https://github.com/guillaumegenthial/tf_metrics)
- inference.py：推理主入口

**模型训练过程**

- 首先准备好数据集，放在data_path下，然后运行preprocess.py文件
- 运行train_main.py，具体的模型参数可以在ARGS里面设置，也可以使用python train_main.py --train_path='./data_path/emotion_data.pkl'的形式

**模型推理**

- 推理代码在inference.py中

## 示例

下面使用中文任务测评基准(CLUE benchmark)的头条新闻分类数据来进行demo示例演示：

数据集下载地址：https://github.com/CLUEbenchmark/CLUE 中的[TNEWS'数据集下载](https://storage.googleapis.com/cluebenchmark/tasks/tnews_public.zip)

该数据集来自今日头条新闻版块，共15个类别的新闻，包括旅游、教育、金融、军事等。

```
数据量：训练集(53,360)，验证集(10,000)，测试集(10,000)
例子：
{"label": "102", "label_des": "news_entertainment", "sentence": "江疏影甜甜圈自拍，迷之角度竟这么好看，美吸引一切事物"}
每一条数据有三个属性，从前往后分别是 分类ID，分类名称，新闻字符串（仅含标题）。
```

**1.数据EDA**

数据EDA部分见tnews_data_eda.ipynb，主要是简单分析一下数据集的文本的长度分布、类别标签的数量比。然后对文本进行分词，这里使用的jieba分词软件。分词后将数据集保存到data_path目录下。

```
# 各种类别标签的数量分布
109    5955
104    5200
102    4976
113    4851
107    4118
101    4081
103    3991
110    3632
108    3437
116    3390
112    3368
115    2886
106    2107
100    1111
114     257
```

**2.设置训练参数**

参数设置的时候有几个参数需要根据自己的数据分布来设置：

- vocab_size：这里的大小，一般需要根据自己生成的vocab.txt中词表的大小来设置
- num_label：类别标签的数量
- train_path/eval_path：数据集的路径
- weights权重设置：根据数据EDA中的类别标签分布，设置weights=[0.9,0.9,0.9,0.9,1,1,1,1,1,1,1,1,1,1.2,1.5]，后面几个类别的数量明显很少，权重设置大一点。具体数值自己根据个人分析来定义

其他的参数视个人情况而定

**3.模型训练并保存模型**

这里使用的是BiLstm模型。

代码中保存了两种模型形式，一种是checkpoint，另一种是pb格式

**4.开始预测并提交结果**

预测代码见inferece.py，最后在CLUE上提交的结果是50.92（[ALBERT-xxlarge](https://github.com/google-research/albert) ：59.46，目前[UER-ensemble](https://github.com/dbiir/UER-py)：72.20）

## 中文文本分类比赛OR评测

1.[零基础入门NLP-新闻文本分类](https://tianchi.aliyun.com/competition/entrance/531810/introduction?spm=5176.12281973.1005.4.3dd52448KQuWQe)（DataWhale和天池举办的学习赛）

2.[中文CLUE的各种分类任务的评测](https://www.cluebenchmarks.com/)