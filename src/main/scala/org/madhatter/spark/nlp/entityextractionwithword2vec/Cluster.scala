package org.madhatter.spark.nlp.entityextractionwithword2vec

import java.util.concurrent.atomic.AtomicInteger

import akka.actor.{ActorRef, ActorSystem, PoisonPill, Props}
import breeze.linalg.{DenseVector, Vector, squaredDistance}
import org.apache.log4j.Logger
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by madhatter on 7/24/2016.
  */
object Cluster extends App {

  val log = Logger.getLogger(this.getClass.getName)
  private val conf = new SparkConf(true)
    .setMaster("local[*]")
    .setAppName("Word2Vec_clustering")
    .set("spark.executor.memory", "8g")
    .set("spark.driver.memory", "8g")
  private val sc = new SparkContext("local", "Word Count", conf)
  val sqlContext = new org.apache.spark.sql.SQLContext(sc)

  val datasetPath = "./data/text8/text8"
  val pretraindModelPath = "./model/data/text8word2vec"
  val clusteringOutputPath = "./clustering/"

  trainAndTest()

  def trainAndTest(): Unit = {
    val vectorModel: Word2VecModel = getModel()
    log.info("Clustering...")
    try {
      val dataset: DataFrame = sqlContext.createDataFrame(
        vectorModel.getVectors.toSeq.map { case (word: String, vectorArr: Array[Float]) =>
          (word, Vectors.dense(vectorArr.map(_.toDouble)))
        }).toDF("id", "features")

      // Trains a k-means model
      val kmeans = new KMeans()
        .setK(500)
        .setMaxIter(2)
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
      val kmeansModel = kmeans.fit(dataset)
      // Shows the resulting centers
      println("Final Centers: ")
      //print words with their respective clusters
      val clusteringOutputSystem = ActorSystem("ClusteringOutputSystem")
      val actors: Array[ActorRef] = kmeansModel.clusterCenters.zipWithIndex.map { case (center: Vector[Double], id: Int) =>
        log.debug(s"created write buffer actor for cluster #$id")
        clusteringOutputSystem.actorOf(Props(new ClusterPrinterActor(id, clusteringOutputPath, center.toJson)))
      }
      val wordCounter = new AtomicInteger()
      val centers: Array[Vector[Double]] = kmeansModel.clusterCenters.map(center => DenseVector(center.toArray))
      vectorModel.getVectors.toSeq.par.foreach {
        case (word, wordVector: Array[Float]) =>
          val v: Vector[Double] = DenseVector(wordVector.map(_.toDouble))
          actors(closestPoint(v, centers)) ! AppendWord(word)
          val w = wordCounter.incrementAndGet()
          if (w % 100 == 0) log.debug(s"words: $w")
      }
      actors.foreach(_ ! PoisonPill)
      clusteringOutputSystem.shutdown()

    } catch {
      case exp: Exception => log.error("An error has occurred while clustering...", exp)
    }
  }

  def getModel(): Word2VecModel = {

    var model: Word2VecModel = null
    try {
      log.info("Loading...")
      model = Word2VecModel.load(sc, pretraindModelPath)
      log.info("Loaded...")
    } catch {
      case exp: Exception =>
        log.warn("No pretrained model has been found,training... Exception: " + exp.toString)
        log.info("Reading the dataset")
        //.persist(StorageLevel.DISK_ONLY)
        val input = sc.textFile(datasetPath).map(line => line.split(" ").toSeq)

        val word2vec = new Word2Vec()
        //word2vec.setVectorSize(300)
        //word2vec.setWindowSize(50)
        //word2vec.setMinCount(5)
        log.info("Training...")
        model = word2vec.fit(input)

        // Save model
        model.save(sc, pretraindModelPath)

    }
    model
  }

  def closestPoint(p: Vector[Double], centers: Array[Vector[Double]]): Int = {
    var bestIndex = 0
    var closest = Double.PositiveInfinity
    for (i <- 0 until centers.length) {
      val tempDist = squaredDistance(p, centers(i))
      if (tempDist < closest) {
        closest = tempDist
        bestIndex = i
      }
    }
    bestIndex
  }
}
