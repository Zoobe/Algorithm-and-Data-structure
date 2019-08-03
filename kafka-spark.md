寻找leader的时候输入是topic和partition(Int),返回是leader的主机和端口号组成的tuple，步骤：

1. 用topic创建一个TopicMetadataRequest
2. 通过SimpleConsumer发送出去获得TopicMetadataResponse含有Topic的源信息的响应
3. 在topicsMetadata元数据中找到和输入topic相等的topicsMetadata，从TopicMetadata中拿到分区元数据partitionsMetadata
4. 再过滤出分区数和输入分区数相等的分区元数据，从每个分区元数据中找到leader对应的host和port
5. 最后得到的是leader信息

通过topic的集合，获得topic,partition封装而成的对象TopicAndPartition

1. 通过topic获得每个TopicMetadata元数据，流程大概如上。
2. 对于每个TopicMetadata获得对应分区元数据PartitionMetadata，将对应的topic和partitionId构造为TopicAndPartition返回集合



项目文档

这个工具将帮助你使用spark-streaming从kafka拉取消息并且在处理kafka处理偏移量方面表现更好，并且支持失败处理

该消费者使用kafka Consumer API实现了一个稳定的接收器从kafka拉取数据，并存储接收到的文件到Spark BlockManager。这一套操作逻辑将自动检测topic的分区数量bing基于kafka配置生成Receiver的数量。每个Receiver 可以从一个或多个kafka分区中拉取数据。比如：某个topic有100个分区，如果Spark Consumer配置20个Receivers，每个Receiver将处理5个分区

一旦Spark Streaming批次完成，consumer将提交offset

在spark driver代码中，调用**ReceiverLauncher.launch**方法将启动一个Receiver

## Kafka-Spark-Consumer的显著特点

- 使用最新的Kafka Consumer API.支持**kafka安全**
- 使用Zookeeper为每个kafak分区存储偏移量，以便于失败的时候恢复
- 该Consumer 不使用WAL机制处理Driver 或者 Executor失败的问题。因为该Consumer有能力在每个batch间隔存储当前偏移量，在任何失败的时候都可以从正确的偏移量恢复
- 该Consumer实现了PID Rate Controller来控制背压
- 该consumer能够使用**Message** 拦截器在写入Spark Block Manager之前对kafka消息进行预处理
- 支持**Consumer** 延迟检查器（类似与ConsumerOffsetChecker）来寻找Consumer 延迟



与Spark 开箱即用的kafka Consumers有什么不同

这个Consumer 是一个稳定容错的消费者接受器。能从任何潜在的失败中恢复并且不需要WAL机制







