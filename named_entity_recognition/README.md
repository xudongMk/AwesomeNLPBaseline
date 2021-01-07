## 命名实体识别（Named Entity Recognition）

这里首先介绍一篇基于深度学习的命名实体识别综述，《A Survey on Deep Learning for Named Entity Recognition》，论文来源：https://arxiv.org/abs/1812.09449（2020年3月份发表在TKDE）

**1.命名实体识别简介**

命名实体识别（Named Entity Recognition，NER）旨在给定的文本中识别出属于预定义的类别片段（如人名、位置、组织等）。NER一直是很多自然语言应用的基础，如机器问答、文本摘要和机器翻译。

NER任务最早是由第六届语义理解会议（Sixth Message Understanding Conference）提出，但当时仅定义一些通用的实体类型，如组织、人名和地点。

**2.命名实体识别常用方法**

- 基于规则的方法（Rule-based Approaches）：不需要标注数据，依赖人工规则，特定领域需要专家知识
- 无监督学习方法（Unsupervised Learning Approaches）：不需要标注数据，依赖于无监督学习方法，如聚类算法
- 基于特征的有监督学习方法（Feature-based Supervised Learning Approaches）：将NER当作一个多分类问题或序列标签分类任务，依赖于特征工程
- 基于深度学习的方法（DL-based Approaches）：后面详细介绍

论文简单介绍了前三种方法，这里也不在赘述，感兴趣的可以看论文。

**3.基于深度学习的方法**

文章中将NER任务拆解成三个结构：

- 输入的分布式表示（Distributed Representations for Input）
- 上下文编码（Context Encoder Architectures）
- 标签解码（Tag Decoder Architectures）

这里不在展开描述具体的内容（有兴趣的可以去翻论文），下表总结了基于神经网络的NER模型的工作，并展示了每个NER模型在各类数据集上的表现。

![image](https://github.com/xudongMk/AwesomeNLPBaseline/blob/main/named_entity_recognition/pics/%E5%91%BD%E5%90%8D%E5%AE%9E%E4%BD%93%E8%AF%86%E5%88%AB%E7%9A%84%E6%A8%A1%E5%9E%8B%E6%80%BB%E7%BB%93%E5%9B%BE.png)

总结：BiLstm+CRF是使用深度学习的NER最常见的体系结构，以Cloze风格使用预训练双向Transformer在CoNLL03数据集上达到了SOTA效果（93.5%），另外Bert+Dice Loss在OntoNotes5.0数据集上达到了SOTA效果（92.07%）。

**4.评测指标**

文中将NER的评测指标Precision、Recall和F1分成了两类。

- Exact match：严格匹配方法，需要识别的边界和类别都正确
- Relaxed match：宽松匹配方法，实体位置区间重叠、位置正确类别错误等都视为正确



## 命名实体识别数据集

命名实体识别数据集一般是BIO或者BIOES模式标注。

- BIO模式：具体指B-begin、I-inside、O-outside
- BIOES模式：具体指B-begin、I-inside、O-outside、E-end、S-single

首先是综述中提到的几个数据集，见下表，具体的就不介绍了。

![image](https://github.com/xudongMk/AwesomeNLPBaseline/blob/main/named_entity_recognition/pics/%E5%91%BD%E5%90%8D%E5%AE%9E%E4%BD%93%E8%AF%86%E5%88%AB%E6%95%B0%E6%8D%AE%E5%9B%BE.png)



下面介绍一个中文的命名实体识别数据集，**CLUENER 细粒度命名实体识别**，地址：https://github.com/CLUEbenchmark/CLUENER2020

- 数据类别：10个，地址、书名、公司、游戏、政府、电影、姓名、组织、职位和景点
- 数据分布：训练集10748，测试集1343，具体类别分布见原文
- 数据来源：在THUCTC文本分类数据集基础上，选出部分数据进行细粒度实体标注



## 命名实体识别Baseline算法实现

使用Tensorflow1.x版本Estimator高阶api实现常见的命名实体识别算法，主要包括BiLstm+CRF、Bert、Bert+CRF。

（当前只在本目录下实现了BiLstm+CRF，至于BERT的在bert_downstream目录下暂未实现）

环境信息：

tensorflow==1.13.1

python==3.7

**数据预处理**

要求训练集和测试集分开存储，要求数据集格式为BIO形式。

在训练模型前，需要先运行preprocess.py文件进行数据预处理，将数据处理成id形式并保存为pkl形式，另外中间过程产生的词表也会保存为vocab.txt文件。

**文件结构**

- data_path：数据集存放的位置
- data_utils：数据处理相关的工具类存放位置
- model_ckpt：chekpoint模型保存的位置
- model_pb：pb形式的模型保存为位置
- models：ner基本的算法存放位置，如BiLstm等
- preprocess.py：数据预处理代码
- ner_main.py：训练主入口

**模型训练**

- 首先准备好数据集，放在data_path下，然后运行preprocess.py文件
- 运行ner_main.py，具体的模型参数可以在ARGS里面设置，也可以使用python ner_main.py --train_path='./data_path/clue_data.pkl'的形式

**模型推理**

- 推理代码在inference.py中



## 示例

下面使用中文任务测评基准(CLUE benchmark)的CLUENER数据进行demo示例演示：

数据集下载地址[[CLUENER细粒度命名实体识别](https://github.com/CLUEbenchmark/CLUENER2020)]，该数据由CLUEBenchMark整理，数据分为10个标签类别分别为: 地址（address），书名（book），公司（company），游戏（game），政府（government），电影（movie），姓名（name），组织机构（organization），职位（position），景点（scene）

数据集分布：

```
训练集：10748
验证集集：1343

按照不同标签类别统计，训练集数据分布如下（注：一条数据中出现的所有实体都进行标注，如果一条数据出现两个地址（address）实体，那么统计地址（address）类别数据的时候，算两条数据）：
【训练集】标签数据分布如下：
地址（address）:2829
书名（book）:1131
公司（company）:2897
游戏（game）:2325
政府（government）:1797
电影（movie）:1109
姓名（name）:3661
组织机构（organization）:3075
职位（position）:3052
景点（scene）:1462

【验证集】标签数据分布如下：
地址（address）:364
书名（book）:152
公司（company）:366
游戏（game）:287
政府（government）:244
电影（movie）:150
姓名（name）:451
组织机构（organization）:344
职位（position）:425
景点（scene）:199
```

**1.数据EDA：**

省略，需要的可以自己分析一下数据集的分布情况

**2.数据预处理：**

转换BIO形式，具体conver_bio.py，将CLUE提供的数据集转换为BIO标注形式；运行preprocess.py将数据集转换为id形式并保存为pkl形式。

**3.模型训练：**

代码见ner_main.py，参数设置的时候有几个参数需要根据自己的数据分布来设置：

- vocab_size：这里的大小，一般需要根据自己生成的vocab.txt中词表的大小来设置
- num_tags：类别标签的数量，算上O，这里是21类
- train_path/eval_path：数据集的路径

其他的参数视个人情况而定

**4.开始预测并提交结果**

预测代码见inference.py

#todo next 只完成了一部分，写入文件的部分暂时未完成。因为其提交的文件格式有点难受....太细化了...



## NER的比赛

1.天池的比赛 https://tianchi.aliyun.com/competition/entrance/531824/introduction

2.CLUE的评测 https://www.cluebenchmarks.com/introduce.html



## 扩展

- 美团搜索中NER技术的探索和实践：https://tech.meituan.com/2020/07/23/ner-in-meituan-nlp.html




