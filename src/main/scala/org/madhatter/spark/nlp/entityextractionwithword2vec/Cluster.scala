package org.madhatter.spark.nlp.entityextractionwithword2vec


import akka.actor.{ActorRef, ActorSystem, Props}
import breeze.linalg.{DenseVector, Vector, squaredDistance}
import org.apache.log4j.Logger
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.apache.spark.{SparkConf, SparkContext}

import scala.io.Source

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

  //paths
  val datasetPath = "./data/text8/text8"
  val pretraindModelPath = "./model/data/text8word2vec"
  val clusteringOutputPath = "./clustering/"
  val clusteringModelOutputPath = "./clustering/kmeans_model"
  //parameters
  val clusterSize = 500
  val kmeansMaxIter = 10
  trainAndTest()

  def trainAndTest(): Unit = {
    val vectorModel: Word2VecModel = getModel()
    log.info("Clustering...")
    try {
      val dataset: DataFrame = sqlContext.createDataFrame(
        vectorModel.getVectors.toSeq.map { case (word: String, vectorArr: Array[Float]) =>
          (word, Vectors.dense(vectorArr.map(_.toDouble)))
        }).toDF("id", "features")


      val kmeansModel = loadKmeans(dataset)

      // Shows the resulting centers
      println("Final Centers: ")
      kmeansModel.clusterCenters.foreach(println)
      //print words with their respective clusters
      val clusteringOutputSystem = ActorSystem("ClusteringOutputSystem")
      val actors: Array[ActorRef] = kmeansModel.clusterCenters.zipWithIndex.map {
        case (center, id) =>
          log.debug(s"creating write buffer actor for cluster #$id")
          clusteringOutputSystem.actorOf(Props(new ClusterPrinterActor(id, clusteringOutputPath, center.toString)))
      }
      val centers: Array[Vector[Double]] = kmeansModel.clusterCenters.map(center => DenseVector(center.toArray))
      vectorModel.getVectors.toSeq.par.foreach {
        case (word, wordVector: Array[Float]) =>
          val v: Vector[Double] = DenseVector(wordVector.map(_.toDouble))
          actors(closestPoint(v, centers)) ! AppendWord(word)
      }
      actors.foreach(_ ! Close)
      clusteringOutputSystem.shutdown()
      println("-- SEARCH --")
      while (true) {
        val line = Source.stdin.getLines().take(1).next().trim
        println(s"searching $line")
        val words = line.split(" ")
        if (words.isEmpty) println("no input")
        else if (words.length > 1) {
          //if a phrase is given as an input, avg of all the vectors that we can find will be used
          val zeroVector = vectorModel.getVectors.take(1).map { case (x, y) => y.map(x => 0d) }.toList(0)
          var failedSearches: Int = 0
          def getVec(w: String): Vector[Double] = vectorModel.getVectors.get(w) match {
            case vec: Some[Array[Float]] =>
              DenseVector(vec.get.map(_.toDouble))
            case _ => println(s"Can not find the word $w")
              failedSearches = failedSearches + 1
              DenseVector(zeroVector)
          }
          val total = words.map(w => getVec(w)).reduce(_ + _) / (words.length - failedSearches)
          if (failedSearches == words.length) println(s"Can not find $line")
          else println(s"cluster ${closestPoint(total, centers)}")
        } else {
          vectorModel.getVectors.get(words(0)) match {
            case vec: Some[Array[Float]] =>
              try {
                val v: Vector[Double] = DenseVector(vec.get.map(_.toDouble))
                println(s"cluster ${closestPoint(v, centers)}")
              } catch {
                case exp: Exception => println(exp)
              }
            case None => println(s"Can not find the word ${words(0)}")
          }
        }
      }

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

  // Trains a k-means model
  def loadKmeans(dataFrame: DataFrame): KMeansModel = {
    try {
      KMeansModel.load(clusteringModelOutputPath)
    } catch {
      case exp: Exception =>
        log.error("Can not load the kmeans model", exp)
        log.info("Training a new kmeans model")

        val kmeans = new KMeans()
          .setK(clusterSize)
          .setMaxIter(kmeansMaxIter)
          .setFeaturesCol("features")
          .setPredictionCol("prediction")
        val model = kmeans.fit(dataFrame)
        model.save(clusteringModelOutputPath)
        model
    }


  }
}
