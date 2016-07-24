package org.madhatter.spark.nlp.twitterstreamanalysis

import org.apache.log4j.Logger
import org.apache.spark.streaming.twitter.TwitterUtils
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by madhatter on 7/24/2016.
  */
object StreamAnalysis extends App {

  val log = Logger.getLogger(this.getClass.getName)
  private val conf = new SparkConf(true)
    .setMaster("local[*]")
    .setAppName("Word2Vec_clustering")
    .set("spark.executor.memory", "8g")
    .set("spark.driver.memory", "8g")
  private val sc = new SparkContext("local", "Word Count", conf)
  val sqlContext = new org.apache.spark.sql.SQLContext(sc)
  val streamingContext = new StreamingContext(sc, Seconds(5))

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
    try {
      System.setProperty("twitter4j.oauth.consumerKey", "xxx")
      System.setProperty("twitter4j.oauth.consumerSecret", "xxx")
      System.setProperty("twitter4j.oauth.accessToken", "xxx")
      System.setProperty("twitter4j.oauth.accessTokenSecret", "xxx")

      val stream = TwitterUtils.createStream(streamingContext, None)
      stream.start()
      val tags = stream.flatMap { status =>
        status.getHashtagEntities.map(_.getText)
      }
      tags.countByValue()
        .foreachRDD { rdd =>
          val now = org.joda.time.DateTime.now()
          rdd
            .sortBy(_._2)
            .map(x => (x, now))
            .saveAsTextFile(s"~/twitter/$now")
        }
    } catch {
      case exp: Exception => log.error("An error has occurred while clustering...", exp)
    }
  }

}
