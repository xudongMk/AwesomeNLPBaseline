## KBQA简介

基于知识库的问答（Knowledge Based Question Answering，KBQA）是自然语言处理（NLP）领域的热门研究方向。知识库（知识图谱， Knowledge Based/Knowledge Graph）是知识的结构化表示，一般是由一组SPO三元组（主语Subject，谓语Predicate，宾语Object）形式构成（也称实体，关系，属性三元组），表示实体和实体间存在的语义关系。例如，中国的首都是北京，可以表示为：[中国，首都，北京]。

基于知识库的问答主要步骤是接收一个自然语言问句，识别出句子中的实体，理解问句的语义关系，构建有关实体和关系的查询语句，进而从知识库中检索出答案。

目前基于知识库的问答主要方法有：

- 基于语义解析/规则的方法
- 基于信息检索/信息抽取的方法

这里有一篇2019年KGQA的综述：Introduction to Neural Network Based Approaches for Question Answering over Knowledge Graphs。这篇文章将KGQA/KBQA当作语义解析的任务来对待，然后介绍了几种语义解析方法，如Classification、Ranking、Translation等。这里不做介绍，感兴趣的可以去翻原文。

基于中文知识库问答（**Chinese Knowledge Based Question Answering，CKBQA**）相比英文的KBQA，中文知识库包含关系多，数据集难以覆盖所有关系，另外中文语言的特点，有居多的挑战。

**基于语义解析/规则的方法：**

该类方法使用字典、规则和机器学习，直接从问题中解析出实体、关系和逻辑组合。这里介绍两篇论文，一篇是 The APVA-TURBO Approach to Question Answering in Knowledge Base，文章使用序列标注模型解析问题中的实体，利用端到端模型解析问题中的关系序列。另一篇 A State-transition Framework to Answer Complex
Questions over Knowledge Base，文章中提出了一种状态转移框架并结合卷积神经网络等方法。（上述方法均基于英文数据集）

基于语义解析/规则的方法一般步骤：

- 实体识别：使用领域词表，相似度等（也可以使用深度学习模型，如BiLstm+CRF，BERT等）
- 属性关系识别：词表规则，或使用分类模型
- 答案查询：基于前两个步骤，更加规则模板转换SPARQL等查询语言进行查询

基于语义解析/规则的方法比较简单，当前Github上很多KBQA的项目都是基于这种模式。

这里推荐几个基于语义解析/规则的 KBQA项目：

- 豆瓣的电影知识图谱问答：https://github.com/weizhixiaoyi/DouBan-KGQA
- 基于NLPCC数据的KBQA：https://zhuanlan.zhihu.com/p/62946533

**基于信息检索/信息抽取的方法：**

该类方法首先根据问题得到若干个候选实体，根据预定义的逻辑形式，从知识库中抽取与候选实体相连的关系作为候选查询路径，再使用文本匹配模型，选择出与问题相似度最高的候选查询路径，到知识库中检索答案。这里介绍一种增强路径匹配的方法： Improved neural relation detection for knowledge base question answering。

当前CKBQA任务上，大多采用的是基于信息检索/信息抽取的方法，一般的步骤：

- 实体与关系识别
- 路径匹配
- 答案检索

在CCKS的KBQA比赛中这种方法非常常见，CCKS官网网站上有每一年的评测论文，下面推荐几个最新的：

- 2019年CCKS的KBQA任务第四名方案：DUTIR中文开放域知识问答评测报告
- 2020年CCKS的KBQA任务第一名方案：基于特征融合的中文知识库问答方法

具体内容可见官网的评测论文，这里附件上传，见ckbqa目录下两个pdf文件。

## 中英文数据集

英文数据集：

- FREE917:第一个大规模的KBQA数据集，于2013年提出，包含917 个问题，同时提供相应逻辑查询，覆盖600多种freebase上的关系。
- Webquestions：数据集中有6642个问题答案对，数据集规模虽然较FREE917提高了不少，但有两个突出的缺陷：没有提供对应的查询，不利于基于逻辑表达式模型的训练；另外webquestions中简单问句多而复杂问句少。
- WebQSP：是WEBQUESTIONS的子集，问题都是需要多跳才能回答，属于multi-relation KBQA dataset，另外补全了对应的查询句。
- Complexquestion、GRAPHQUESTIONS：在问句的结构和表达多样性等方面进一步增强了WEBQUESTIONSP，，包括类型约束，显\隐式的时间约束，聚合操作。
- SimpleQuestions：数据规模较大，共100K，数据形式为(quesition，knowledge base fact)，均为简单问题，只需KB中的一个三元组即可回答,即single-relation dataset。

英文数据集较多，这里只列举几个常见的。详细的数据集可见北航的[KBQA调研](https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/KBQA%E8%B0%83%E7%A0%94-%E5%AD%A6%E6%9C%AF%E7%95%8C.md#13-%E6%95%B0%E6%8D%AE%E9%9B%86)

中文数据集：

- NLPCC开放领域知识图谱问答的数据集：简单问题（单跳问题），14609条训练数据，9870条验证和测试数据，数据集下载。
- CCKS开放领域知识图谱问答的数据集：包含简单问题和复杂问答，2298条训练数据，766的验证和测试数据，数据集下载。

除了上述两个中文数据集（提取码均是），CLUE上还提供了一些问答的数据集，可以见[CLUE的数据集搜索](https://www.cluebenchmarks.com/dataSet_search_modify.html?keywords=QA)。

## KBQA的实现

下面基于CCKS的数据集来实现2019年第四名方案和2020年第一名方案。

CCKS的数据集，百度网盘下载地址：链接：https://pan.baidu.com/s/1NI9VrhuvOgyTFk1tGjlZIw   提取码：l7pm 

todo list（等有空实现了就补上）：

- 使用tensorflow实现2019年第四名方案
- 使用tensorflow实现2020年第一名方案

另外，附上2019年第四名方案的开源地址，流程还算完整，但想端到端完整运行有点困难，而且很多数据的处理过程都耦合在模型中。需要花一定的时间去整理。



## 扩展

- 美团大脑：知识图谱的建模方法及其应用：https://tech.meituan.com/2018/11/01/meituan-ai-nlp.html
- 百度大脑UNIT3.0详解之知识图谱与对话：https://baijiahao.baidu.com/s?id=1643915882369765998&wfr=spider&for=pc
- 更新ing
