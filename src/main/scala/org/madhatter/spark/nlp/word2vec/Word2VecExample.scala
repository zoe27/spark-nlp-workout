package org.madhatter.spark.nlp.word2vec

import org.apache.log4j.Logger
import org.apache.spark._
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}

/**
  * Created by cenkcorapci on 15/04/16.
  * -Xmx6144M -d64
  */
object Word2VecExample extends App {
  val log = Logger.getLogger(this.getClass.getName)
  val sc = new SparkContext("local", "Word Count", "/usr/local/Cellar/apache-spark/1.6.1",
    List("target/spark-nlp-workout-1.0.jar"))
  val datasetPath = "/Users/cenkcorapci/Documents/data-science/data/text8"
  val pretraindModelPath = "/Users/cenkcorapci/Documents/data-science/projects/spark-nlp-workout/model/text8word2vec"
  val synonyms = model.findSynonyms("china", 40)
  try {
    model = Word2VecModel.load(sc, pretraindModelPath)
  } catch {
    case exp: Exception =>
      log.warn("No pretrained model has been found,training... Exception: " + exp.toString)
      log.info("Reading the dataset")
      val input = sc.textFile(datasetPath).map(line => line.split(" ").toSeq)

      val word2vec = new Word2Vec()
      log.info("Training...")
      model = word2vec.fit(input)

      // Save model
      model.save(sc, pretraindModelPath)
  }
  var model: Word2VecModel = null

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
