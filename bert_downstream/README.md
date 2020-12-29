## BERT介绍

**简介**

BERT是谷歌于2018年10月公开的一种预训练模型。该模型一经发布，就引起了学术界以及工业界的广泛关注。在效果方面，BERT刷新了11个NLP任务的当前最优效果，该方法也被评为2018年NLP的重大进展以及NAACL 2019的best paper。BERT和早前OpenAI发布的GPT方法技术路线基本一致，只是在技术细节上存在略微差异。两个工作的主要贡献在于使用预训练+微调的思路来解决自然语言处理问题。以BERT为例，模型应用包括2个环节：

- 预训练（Pre-training），该环节在大量通用语料上学习网络参数，通用语料包括Wikipedia、Book Corpus，这些语料包含了大量的文本，能够提供丰富的语言相关现象。
- 微调（Fine-tuning），该环节使用“任务相关”的标注数据对网络参数进行微调，不需要再为目标任务设计Task-specific网络从头训练。

模型的详细信息可见论文，原文 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805v1)，这是BERT在2018年10月发布的版本，与2019年5月版本[v2](https://arxiv.org/abs/1810.04805v2)有稍许不同。

英文不好的可以参考大佬的论文翻译：[BERT论文中文翻译](https://github.com/yuanxiaosc/BERT_Paper_Chinese_Translation)

**各类BERT预训练模型**：

- 官网BERT：https://github.com/google-research/bert
- Transformers：https://github.com/huggingface/transformers
- 哈工大讯飞：https://github.com/ymcui/Chinese-BERT-wwm
- Brightmart：https://github.com/brightmart/roberta_zh
- CLUEPretrainedModels：https://github.com/CLUEbenchmark/CLUEPretrainedModels

**BERT下游任务**

随着预训练模型的提出，大大减少了我们对NLP任务设计特定结构的需求，我们只需要在BERT等预训练模型之后再接一些简单的网络，即可完成我们的NLP任务，而且效果非常好。

原因也非常简单，BERT等预训练模型通过大量语料的无监督学习，已经将语料中的知识迁移进了预训练模型的Embedding中，为此我们只需在针对特定任务增加结构来进行微调，即可适应当前任务，这也是迁移学习的魔力所在。

下面介绍几类下游任务：

- 句子对分类任务，如自然语言推理（NLI），其数据集一般有MNLI、QNLI、STS-B、MRPC等
- 单句子分类任务，如文本分类（Text-classification），其数据集一般有SST-2、CoLA等
- 问答任务，数据集一般有SQuAD v1.1等
- 单句子token标注任务，如命名实体识别（NER），其数据集一般有CoNLL-2003等



## BERT下游任务代码

下面基于官方BERT的fine-tune代码来实现文本分类、命名实体识别和多任务学习的Baseline模型。

（下面所有任务的预训练模型都是基于哈工大讯飞实验室的**`BERT-wwm-ext, Chinese`**模型，模型下载地址见上述链接）

1.文本分类

```
数据集来源：情感分类，是包含7个分类的细粒度情感性分析数据集，NLP中文预训练模型泛化能力挑战赛的数据集
运行脚本见train_classifier.py
```

2.命名实体识别

```
数据集来源：CLUENER 细粒度命名实体识别,数据分为10个标签类别，详细信息见：https://github.com/CLUEbenchmark/CLUENER2020
运行脚本见train_ner.py
```

3.多任务学习

```
数据集来源：NLP中文预训练模型泛化能力挑战赛，https://tianchi.aliyun.com/competition/entrance/531841/introduction
运行脚本见trian_multi_learning.py

3个epoch没做任何tricks，当前的score是0.5717
```



## 拓展

下面推荐两篇有意思的文章

1.How to Fine-tune bert for Text-classification

2.few sample bert fine-tune