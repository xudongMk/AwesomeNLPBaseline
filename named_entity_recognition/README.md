## 命名实体识别（Named Entity Recognition）

这里首先介绍一篇基于深度学习的命名实体识别综述，《A Survey on Deep Learning for Named Entity Recognition》，论文来源：https://arxiv.org/abs/1812.09449（2020年3月份发表在TKDE）

https://zhuanlan.zhihu.com/p/141088583

https://github.com/luopeixiang/named_entity_recognition

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

![](E:\AwesomeNLPBaseLine\命名实体识别的模型总结图.png)

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

![image-20201214234022320](E:\AwesomeNLPBaseLine\命名实体识别数据图)



下面介绍一个中文的命名实体识别数据集，**CLUENER 细粒度命名实体识别**，地址：https://github.com/CLUEbenchmark/CLUENER2020

- 数据类别：10个，地址、书名、公司、游戏、政府、电影、姓名、组织、职位和景点
- 数据分布：训练集10748，测试集1343，具体类别分布见原文
- 数据来源：在THUCTC文本分类数据集基础上，选出部分数据进行细粒度实体标注



## 命名实体识别Baseline算法实现

使用Tensorflow1.x版本Estimator高阶api实现常见的命名实体识别算法，主要包括BiLstm+CRF、Bert、Bert+CRF。

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

**模型参数解释**

- train_path/test_path：训练集/测试集文件路径
- model_ckpt_dir：模型checkpoint的保存路径
- model_pb_dir：pb形式的模型保存路径
- vocab_size：词表大小
- emb_size：embedding层的大小
- hidden_dim：rnn的隐层大小
- num_tags：标签的数量
- drop_out：dropout rate
- batch_size：批处理样本大小
- epoch：全样本迭代的次数
- type：使用具体的模型来文本分类，有lstm、cnn等
- lr：学习率大小



#todo 直接加入 数据集的结果对比



## 示例



NER的应用：

https://tech.meituan.com/2020/07/23/ner-in-meituan-nlp.html

美团搜索中NER技术的探索和实践



## NER的比赛

1.达观数据的

2.天池的比赛

3.CLUE的评测

