# 项目
## 批处理


## 流处理
分为以下几步：
### 一、	构建SparkStreaming上下文
传入的第一个参数，和之前的spark上下文一样，也是SparkConf对象；第二个参数则不太一样，是实时处理batch的interval </br>
spark streaming，每隔一小段时间，会去收集一次数据源（kafka）中的数据，做成一个batch,每次都是处理一个batch中的数据</br>
咱们这里项目中，就设置5秒钟的batch interval,
每隔5秒钟，咱们的spark streaming作业就会收集最近5秒内的数据源接收过来的数据
~~~
    val sparkConf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("AdClickRealTimeStatSpark")

    val ssc = new StreamingContext(sparkConf, Seconds(5))
~~~
### 二、	获得Kafka数据源
主要要放置的就是，你要连接的kafka集群的地址（broker集群的地址列表）
~~~
    val kafkaParams = Map[String, String](
      "metadata.broker.list"->ConfigurationManager.getProperty(Constants.KAFKA_METADATA_BROKER_LIST)
    )

    // 构建topic set
    val kafkaTopics = ConfigurationManager.getProperty(Constants.KAFKA_TOPICS)
    val kafkaTopicsSplited = kafkaTopics.split(",")
    val topics = kafkaTopicsSplited.toSet

    // 基于kafka direct api模式，构建出了针对kafka集群中指定topic的输入DStream
    // 两个值，val1，val2；val1没有什么特殊的意义；val2中包含了kafka topic中的一条一条的实时日志数据
    val adRealTimeLogDStream = KafkaUtils.createDirectStream[String, String, StringDecoder, StringDecoder](
      ssc, kafkaParams, topics)
~~~
#### Kafka的两种模式direct方式读取和receiver方式读取的区别
Spark-Streaming获取kafka数据的两种方式-Receiver与Direct的方式，可以从代码中简单理解成Receiver方式是通过zookeeper来连接kafka队列，Direct方式是直接连接到kafka的节点上获取数据了。

##### 一、基于Receiver的方式

这种方式使用Receiver来获取数据。Receiver是使用Kafka的高层次Consumer API来实现的。receiver从Kafka中获取的数据都是存储在Spark Executor的内存中的，然后Spark Streaming启动的job会去处理那些数据。

然而，在默认的配置下，这种方式可能会因为底层的失败而丢失数据。如果要启用高可靠机制，让数据零丢失，就必须启用Spark Streaming的预写日志机制（Write Ahead Log，WAL）。该机制会同步地将接收到的Kafka数据写入分布式文件系统（比如HDFS）上的预写日志中。所以，即使底层节点出现了失败，也可以使用预写日志中的数据进行恢复。
需要注意的要点

1. Kafka中的topic的partition，与Spark中的RDD的partition是没有关系的。所以，在KafkaUtils.createStream()中，提高partition的数量，只会增加一个Receiver中，读取partition的线程的数量。不会增加Spark处理数据的并行度。

2. 可以创建多个Kafka输入DStream，使用不同的consumer group和topic，来通过多个receiver并行接收数据。

3. 如果基于容错的文件系统，比如HDFS，启用了预写日志机制，接收到的数据都会被复制一份到预写日志中。因此，在KafkaUtils.createStream()中，设置的持久化级别是StorageLevel.MEMORY_AND_DISK_SER。

##### 二、基于Direct的方式

这种新的不基于Receiver的直接方式，是在Spark 1.3中引入的，从而能够确保更加健壮的机制。替代掉使用Receiver来接收数据后，这种方式会周期性地查询Kafka，来获得每个topic+partition的最新的offset，从而定义每个batch的offset的范围。当处理数据的job启动时，就会使用Kafka的简单consumer api来获取Kafka指定offset范围的数据。

这种方式有如下优点：

1. 简化并行读取：如果要读取多个partition，不需要创建多个输入DStream然后对它们进行union操作。Spark会创建跟Kafka partition一样多的RDD partition，并且会并行从Kafka中读取数据。所以在Kafka partition和RDD partition之间，有一个一对一的映射关系。

2. 高性能：如果要保证零数据丢失，在基于receiver的方式中，需要开启WAL机制。这种方式其实效率低下，因为数据实际上被复制了两份，Kafka自己本身就有高可靠的机制，会对数据复制一份，而这里又会复制一份到WAL中。而基于direct的方式，不依赖Receiver，不需要开启WAL机制，只要Kafka中作了数据的复制，那么就可以通过Kafka的副本进行恢复。

3. 一次且仅一次的事务机制：

基于receiver的方式，是使用Kafka的高阶API来在ZooKeeper中保存消费过的offset的。这是消费Kafka数据的传统方式。这种方式配合着WAL机制可以保证数据零丢失的高可靠性，但是却无法保证数据被处理一次且仅一次，可能会处理两次。因为Spark和ZooKeeper之间可能是不同步的。

4. 降低资源。Direct不需要Receivers，其申请的Executors全部参与到计算任务中；而Receiver-based则需要专门的Receivers来读取Kafka数据且不参与计算。因此相同的资源申请，Direct 能够支持更大的业务。
5. 降低内存。Receiver-based的Receiver与其他Exectuor是异步的，并持续不断接收数据，对于小业务量的场景还好，如果遇到大业务量时，需要提高Receiver的内存，但是参与计算的Executor并无需那么多的内存。而Direct 因为没有Receiver，而是在计算时读取数据，然后直接计算，所以对内存的要求很低。实际应用中我们可以把原先的10G降至现在的2-4G左右。

6. 鲁棒性更好。
Receiver-based方法需要Receivers来异步持续不断的读取数据，因此遇到网络、存储负载等因素，导致实时任务出现堆积，但Receivers却还在持续读取数据，此种情况很容易导致计算崩溃。Direct 则没有这种顾虑，其Driver在触发batch 计算任务时，才会读取数据并计算。队列出现堆积并不会引起程序的失败。</br>
基于direct的方式，使用kafka的简单api，Spark Streaming自己就负责追踪消费的offset，并保存在checkpoint中。Spark自己一定是同步的，因此可以保证数据是消费一次且仅消费一次。

### 三、根据动态黑名单进行数据过滤
~~~
    val filteredAdRealTimeLogDStream = filterByBlacklist(adRealTimeLogDStream)
~~~
使用transform算子（将dstream中的每个batch RDD进行处理，转换为任意的其他RDD，功能很强大）
0. 原始数据格式`timestamp province city userid adid`,将原始数据rdd映射成<userid, tuple2<string, string>>
1. 首先，从mysql中查询所有黑名单用户，将其转换为一个rdd
查询黑名单mysql得到的是一个List，将其映射为`tuple <userid,true>`的格式
~~~
val tuples = adBlacklists.map(ad => (ad.getUserid, true))
~~~
2. 通过当前transform遍历的rdd拿到sparkContext对象,采用`异步线程`的方式将黑名单转换为RDD，方便后面join
~~~
val sc = rdd.context
val blacklistRDD = sc.parallelize(tuples)
~~~
3. 将原始数据rdd映射成`<userid, tuple<string, string>>`的格式
4. 将原始日志数据rdd，与黑名单rdd，进行左外连接,为什么不用inner join，内连接，会导致数据丢失，
如果说原始日志的userid，没有在对应的黑名单中，join不到，左外连接不会出现这样的情况，最后还要将数据变回原来的格式的
~~~
val joinedRDD = mappedRDD.leftOuterJoin(blacklistRDD)
~~~
5. 过滤掉那些为join之后为true的数据`userid, tuple<string, string>,true`的数据
6. 再将数据map回去`userid, tuple<string, string>`
- 注意：感觉这里会问异步查询和join的方式的区别和底层
### 四、生成动态黑名单
- 原始数据格式`timestamp province city userid adid`---分别对应`某个时间点 某个省份 某个城市 某个用户 某个广告`
1. 将原始日志的格式处理成<yyyyMMdd_userid_adid, 1L>格式，每天每个用户对每个广告的点击量
2. 由于5秒的的点击还会精确到秒，所以要将所有的数据加起来，针对处理后的日志格式，执行reduceByKey算子
3. 此时得到每个5s的batch中每个用户对每支广告的点击次数`<yyyyMMdd_userid_adid, clickCount>`
4. 每5s的数据和点击次数写入库中，注意这里采用的是对每个分区写库，而且是数据库连接池的方式，所以是高性能的
5. 查询MySQL数据库中当指定日期，指定广告，指定用户的点击次数，注意这个日期一般是一天，次数大于100就怀疑刷单，过滤掉
还要查看当前5s内刷单是否次数过多
6. 对数据去重之后就写入数据库中，因为可能多个用户对不同商品由刷单行为，最终只用存储用户id

### 五、计算广告点击流量实时统计结果（yyyyMMdd_province_city_adid,clickCount）
计算每天各省各城市各广告的点击量，这份数据，实时不断地更新到mysql中的，J2EE系统，是提供实时报表给用户查看的。
j2ee系统每隔几秒钟，就从mysql中搂一次最新数据，每次都可能不一样。
将数据处理成`<date_province_city_adid,count>`的格式，在spark集群中保留一份；在mysql中，也保留一份
`updateStateByKey`算子,维护一份key的全局状态
1. 将原始DStream数据`date province city userid adid`映射成`<date_province_city_adid,1>`
### 六、实时统计每天每个省份top3热门广告
### 七、实时统计每天每个广告在最近1小时的滑动窗口内的点击趋势（每分钟的点击量）
### 八、项目调优和HA要点
