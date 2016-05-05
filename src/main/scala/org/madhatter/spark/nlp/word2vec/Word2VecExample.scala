package org.madhatter.spark.nlp.word2vec

import org.apache.log4j.Logger
import org.apache.spark._
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.storage.StorageLevel

/**
  * Created by cenkcorapci on 15/04/16.
  * -Xmx2048M -d64
  */
object Word2VecExample extends App {

  val log = Logger.getLogger(this.getClass.getName)
  private val conf = new SparkConf(true)
    .setMaster("local[*]")
    .setAppName("Word2Vec_tryout")
    .set("spark.executor.memory", "4g")
    .set("spark.driver.memory", "4g")
  private val sc = new SparkContext("local", "Word Count", conf)

  val datasetPath = "/Users/madhatter/Documents/data-science/data/text8.txt"
  val pretraindModelPath = "./model/data/text8word2vec"

  trainAndTest()

  def trainAndTest(): Unit = {
    var model: Word2VecModel = null
    try {
      log.info("Loading...")
      model = Word2VecModel.load(sc, pretraindModelPath)
      log.info("Loaded...")
    } catch {
      case exp: Exception =>
        log.warn("No pretrained model has been found,training... Exception: " + exp.toString)
        log.info("Reading the dataset")
        val input = sc.textFile(datasetPath).persist(StorageLevel.DISK_ONLY).map(line => line.split(" ").toSeq)

        val word2vec = new Word2Vec()
        log.info("Training...")
        model = word2vec.fit(input)

        // Save model
        model.save(sc, pretraindModelPath)
    }
    log.info("Testing...")
    val synonyms = model.findSynonyms("china", 40)

    for ((synonym, cosineSimilarity) <- synonyms) {
      println(s"$synonym $cosineSimilarity")
    }
    model.getVectors.get("king") match {
      case vecKing: Some[Array[Float]] =>
        model.getVectors.get("woman") match {
          case vecWoman: Some[Array[Float]] =>
            model.getVectors.get("man") match {
              case vecMan: Some[Array[Float]] =>
                log.info("Man,king and queen exists, doing the example...")
                val res = addVectors(substractVectors(vecKing.get, vecMan.get), vecWoman.get)
                var sim: String = ""
                var mostSim: Float = 0f
                var initiated: Boolean = false
                model.getVectors.foreach {
                  case (k, v) =>

                    if (!initiated) {
                      initiated = true
                      mostSim = nearnessScore(v, res)
                      sim = k
                    } else if (nearnessScore(v, res) < mostSim) {
                      mostSim = nearnessScore(v, res)
                      sim = k
                    }
                }
                println(s"King - Man + Woman: $sim")
            }

        }
    }
  }


  def addVectors(vec1: Array[Float], vec2: Array[Float]): Array[Float] = {
    var res: Array[Float] = Array.ofDim(vec1.length)
    for (i <- vec1.indices) {
      res(i) = vec1(i) + vec2(i)
    }
    res
  }

  def substractVectors(vec1: Array[Float], vec2: Array[Float]): Array[Float] = {
    var res: Array[Float] = Array.ofDim(vec1.length)
    for (i <- vec1.indices) {
      res(i) = vec1(i) - vec2(i)
    }
    res
  }

  def nearnessScore(vec1: Array[Float], vec2: Array[Float]): Float = {
    var res: Float = 0f
    for (i <- 0 until vec1.length) {
      res += math.abs(vec1(i) - vec2(i))
    }
    res
  }
}
