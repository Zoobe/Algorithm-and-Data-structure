# 项目
## 批处理
用户访问session分析Spark作业，接收用户创建的分析任务自动化执行数据处理过程
用户可能指定的条件如下：
~~~
 * 1、时间范围：起始日期~结束日期
 * 2、性别：男或女
 * 3、年龄范围
 * 4、职业：多选
 * 5、城市：多选
 * 6、搜索词：多个搜索词，只要某个session中的任何一个action搜索过指定的关键词，那么session就符合条件
 * 7、点击品类：多个品类，只要某个session中的任何一个action点击过某个品类，那么session就符合条件
~~~
### 任务执行的方式
J2EE平台在接收用户创建任务的请求之后，会将任务信息插入MySQL的task表中，任务参数以JSON格式封装在task_param字段中
- 接着J2EE平台会执行我们的spark-submit shell脚本，并将taskid作为参数传递给spark-submit shell脚本
- spark-submit shell脚本，在执行时，是可以接收参数的，并且会将接收的参数，传递给Spark作业的main函数
- 参数就封装在main函数的args数组中

### 一、任务参数解析
1. 通过DAO组件查询Task表，拿到Json格式的任务参数
2. 使用FastJson工具解析任务参数
### 二、按照session粒度进行数据聚合
1. 首先要从`user_visit_action`表中，查询出来指定日期范围内的行为数据
2. 得到actionRDD，就是一个公共RDD，后面还要使用，将其persist持久化
    如果是persist(StorageLevel.MEMORY_ONLY())，纯内存，无序列化，那么就可以用cache()方法来替代
		 * StorageLevel.MEMORY_ONLY_SER()，第二选择
		 * StorageLevel.MEMORY_AND_DISK()，第三选择
		 * StorageLevel.MEMORY_AND_DISK_SER()，第四选择
		 * StorageLevel.DISK_ONLY()，第五选择
		 * 
		 * 如果内存充足，要使用双副本高可靠机制
		 * 选择后缀带_2的策略
		 * StorageLevel.MEMORY_ONLY_2()
3. 首先，将行为数据，按照session_id进行groupByKey分组，再将分组内每一个sessionid的搜索词，点击品类等字符串拼接，计算每个session访问时长，访问步长
4. 将上面的数据拼接为一个大的`<userid,大字符串>`的格式，再和user表查到的用户信息join，**注意此处可以采用map join数据倾斜的解决方法**
5. 最后将数据映射为`<sessionid,(sessionid,searchKeywords,clickCategoryIds,age,professional,city,sex)`的格式
### 四、按照筛选参数对数据进行过滤并自定义accumulator对访问行为进行统计
1. 按照之前解析到的任务的其他参数对数据进行过滤，因为过滤也是一条一条的处理，所以可以同时计算访问比例
2. 注册自定义的accumulator为一个访问时长各个区间占比，步长各个区间占比的字符串，使用之前要注册
~~~
spark.sparkContext.register(accumulator,"SessionAggrStatAccumulator")
~~~
3. 最后获得通过筛选的明细数据
### 五、用户随机抽取
1. 计算出每天每小时的session数量
### 六、获得Top10热门商品
### 六、获得Top10热门session

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
2. 对刚刚的DStream调用updateStateByKey算子，首先判断这个key是否已经存在，如果存在，就将这个key对应的值累加，并将key，value返回
3. 将返回的值也就是统计出来的值写入到MySQL中
### 六、实时统计每天每个省份top3热门广告
1. 将上一步得到的DStream的格式`<yyyyMMdd_province_city_adid, clickCount>`转换为`<yyyyMMdd_province_adid, clickCount>`的格式
2. 调用reduceByKey算子，计算出每天各省份各广告的点击量
3. 目的是计算出每天各个省份的Top3广告，将上一步得到的RDD转换为Dataframe，注册为一张临时表，使用Spark SQL，通过开窗函数，获取到各省份的top3热门广告
    - 将RDD映射为`Row(date, province, adid, clickCount)`格式
    - 通过RDD拿到SparkSession单例对象，注册一张临时表
      ~~~
      val spark = SparkSessionSingleton.getInstance(rdd.sparkContext.getConf)
      val dailyAdClickCountByProvinceDF = spark.createDataFrame(rowsRDD,schema)
      ~~~
    - 使用Spark SQL执行SQL语句，配合开窗函数，统计出各身份top3热门的广告
      ~~~
      val provinceTop3AdDF = spark.sql(
        "SELECT "
          + "date,"
          + "province,"
          + "ad_id,"
          + "click_count "
          + "FROM ( "
            + "SELECT "
                + "date,"
                + "province,"
                + "ad_id,"
                + "click_count,"
              + "ROW_NUMBER() OVER(PARTITION BY province ORDER BY click_count DESC) rank "  //ROW_NUMBER函数按照click_count倒序加上行号
            + "FROM tmp_daily_ad_click_count_by_prov "
          + ") t "
          + "WHERE rank>=3" //得到rank大于3的
      )
      ~~~
4. 将其中的数据批量更新到MySQL中
### 七、实时统计每天每个广告在最近1小时的滑动窗口内的点击趋势（每分钟的点击量）
采用滑动窗口的方式，设定窗宽为60分钟也就是一个小时，滑动间隔为10秒，对原始数据统计
1. 因为只和时间和广告数目有关，所以将数据映射为`<yyyyMMddHHMM_adid,1L>`，每次出来一个新的batch，都要获取最近1小时内的所有的batch
2. 然后根据key进行reduceByKey操作，调用`reduceByKeyAndWindow`算子，统计出来最近一小时内的各分钟各广告的点击次数
3. 写入数据库中，注意以上的所有写库操作都是先进行查询，分成两组，如果查询结果存在，加入更新组`update`，如果查询结果不存在加入插入组`insert`
### 八、项目调优和HA要点
#### HA高可用性：
High Availability，如果有些数据丢失，或者节点挂掉；那么不能让你的实时计算程序挂了；必须做一些数据上的冗余副本，保证你的实时计算程序可以7 * 24小时的运转。
1. updateStateByKey、window等有状态的操作，自动进行checkpoint，必须设置checkpoint目录
checkpoint目录：容错的文件系统的目录，比如说，常用的是HDFS

SparkStreaming.checkpoint("hdfs://192.168.1.105:9090/checkpoint")

设置完这个基本的checkpoint目录之后，有些会自动进行checkpoint操作的DStream，就实现了HA高可用性；checkpoint，相当于是会把数据保留一份在容错的文件系统中，一旦内存中的数据丢失掉；那么就可以直接从文件系统中读取数据；不需要重新进行计算
2. Driver高可用性
第一次在创建和启动StreamingContext的时候，那么将持续不断地将实时计算程序的元数据（比如说，有些dstream或者job执行到了哪个步骤），如果后面，不幸，因为某些原因导致driver节点挂掉了；那么可以让spark集群帮助我们自动重启driver，然后继续运行时候计算程序，并且是接着之前的作业继续执行；没有中断，没有数据丢失
第一次在创建和启动StreamingContext的时候，将元数据写入容错的文件系统（比如hdfs）；spark-submit脚本中加一些参数；保证在driver挂掉之后，spark集群可以自己将driver重新启动起来；而且driver在启动的时候，不会重新创建一个streaming context，而是从容错文件系统（比如hdfs）中读取之前的元数据信息，包括job的执行进度，继续接着之前的进度，继续执行。通过SparkStreaming工厂创建一个SparkStreaming对象，再次启动的时候会通过工厂找上次的SparkStreaming对象
要求：
+ --deploy-mode cluster  保证运行在cluster模式，这样Driver运行再Worker上面，由Worker监控，但是这种模式不方便我们调试程序
+ --supervise   cluster模式中Driver运行在某个worker上，整个参数是由worker监控driver，挂了之后再重启driver。
~~~
JavaStreamingContextFactory contextFactory = new JavaStreamingContextFactory() {
  @Override 
  public JavaStreamingContext create() {
    JavaStreamingContext jssc = new JavaStreamingContext(...);  
    JavaDStream<String> lines = jssc.socketTextStream(...);     
    jssc.checkpoint(checkpointDirectory);                       
    return jssc;
  }
};

JavaStreamingContext context = JavaStreamingContext.getOrCreate(checkpointDirectory, contextFactory);
context.start();
context.awaitTermination();
~~~
3. 实现RDD高可用性：启动WAL预写日志机制
spark streaming，从原理上来说，是通过receiver来进行数据接收的；接收到的数据，会被划分成一个一个的block；block会被组合成一个batch；针对一个batch，会创建一个rdd；启动一个job来执行我们定义的算子操作。

receiver主要接收到数据，那么就会立即将数据写入一份到容错文件系统（比如hdfs）上的checkpoint目录中的，一份磁盘文件中去；作为数据的冗余副本。

无论你的程序怎么挂掉，或者是数据丢失，那么数据都不肯能会永久性的丢失；因为肯定有副本。

WAL（Write-Ahead Log）预写日志机制
spark.streaming.receiver.writeAheadLog.enable true

#### 实时计算程序性能调优：
1. 并行化数据接收：处理多个topic的数据时比较有效**只适用于receiver模式**
~~~
int numStreams = 5;
List<JavaPairDStream<String, String>> kafkaStreams = new ArrayList<JavaPairDStream<String, String>>(numStreams);
for (int i = 0; i < numStreams; i++) {
  kafkaStreams.add(KafkaUtils.createStream(...));
}
JavaPairDStream<String, String> unifiedStream = streamingContext.union(kafkaStreams.get(0), kafkaStreams.subList(1, kafkaStreams.size()));
unifiedStream.print();
~~~

2. spark.streaming.blockInterval：增加block数量，增加每个batch rdd的partition数量，增加处理并行度，**只适用于receiver模式**

receiver从数据源源源不断地获取到数据；首先是会按照block interval，将指定时间间隔的数据，收集为一个block；默认时间是200ms，官方推荐不要小于50ms；接着呢，会将指定batch interval时间间隔内的block，合并为一个batch；创建为一个rdd，然后启动一个job，去处理这个batch rdd中的数据

batch rdd，它的partition数量是多少呢？一个batch有多少个block，就有多少个partition；就意味着并行度是多少；就意味着每个batch rdd有多少个task会并行计算和处理。

当然是希望可以比默认的task数量和并行度再多一些了；可以手动调节block interval；减少block interval；每个batch可以包含更多的block；有更多的partition；也就有更多的task并行处理每个batch rdd。

定死了，初始的rdd过来，直接就是固定的partition数量了

3. inputStream.repartition(<number of partitions>)：重分区，增加每个batch rdd的partition数量

有些时候，希望对某些dstream中的rdd进行定制化的分区
对dstream中的rdd进行重分区，去重分区成指定数量的分区，这样也可以提高指定dstream的rdd的计算并行度

4. 调节并行度
spark.default.parallelism
reduceByKey(numPartitions)

5. 使用Kryo序列化机制：

spark streaming，也是有不少序列化的场景的
提高序列化task发送到executor上执行的性能，如果task很多的时候，task序列化和反序列化的性能开销也比较可观
默认输入数据的存储级别是StorageLevel.MEMORY_AND_DISK_SER_2，receiver接收到数据，默认就会进行持久化操作；首先序列化数据，存储到内存中；如果内存资源不够大，那么就写入磁盘；而且，还会写一份冗余副本到其他executor的block manager中，进行数据冗余。

6. batch interval：每个的处理时间必须小于batch interval(防止出现数据积压）**只适用于receiver模式**

实际上你的spark streaming跑起来以后，其实都是可以在spark ui上观察它的运行情况的；可以看到batch的处理时间；
如果发现batch的处理时间大于batch interval，就必须调节batch interval
尽量不要让batch处理时间大于batch interval
比如你的batch每隔5秒生成一次；你的batch处理时间要达到6秒；就会出现，batch在你的内存中日积月累，一直囤积着，没法及时计算掉，释放内存空间；而且对内存空间的占用越来越大，那么此时会导致内存空间快速消耗

如果发现batch处理时间比batch interval要大，就尽量将batch interval调节大一些




