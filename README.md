## AwesomeNLPBaseline

本项目是NLP领域一些任务的基准模型实现，包括文本分类、命名实体识别、实体关系抽取、NL2SQL、CKBQA以及BERT的各种下游任务应用。

主要使用Tensorflow1.x
（话说Tensorflow1.0版本在实现某些任务的时候是真心的不如torch，实力劝退。不要问既然这么难用为什么不用torch？问就是正因为难用才要用，而且在公司部署项目的时候，TF的as-server模式真香。）

**任务介绍**

- 文本分类
- 命名实体识别
- bert下游任务
- 实体关系抽取
- nl2sql
- ckbqa
- doing（持续更新）

**目录结构如下**：

* text_classification: 文本分类
* named_entity_recognition: 命名实体识别
* entity_relation_extraction: 实体关系抽取
* ckbqa: 中文知识问答
* nl2sql: 自然语言到Sql语句
* bert_downstream: 基于bert进行fine-tune下游任务以及bert相关研究

Tip：当前只实现了文本分类，bert下游任务，命名实体识别三个任务，其他的等有空了再补上。


**声明**

本项目是作者平时学习和工作中遇到的NLP任务积累，仅供学习交流。欢迎提issue和pr。

