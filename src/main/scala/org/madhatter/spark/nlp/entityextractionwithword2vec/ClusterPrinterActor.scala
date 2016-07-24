package org.madhatter.spark.nlp.entityextractionwithword2vec

import java.io.{BufferedWriter, File, FileWriter}

import akka.actor.{Actor, ActorLogging, PoisonPill}

/**
  * Created by madhatter on 7/24/2016.
  */
class ClusterPrinterActor(clusterId: Int, path: String, center: String) extends Actor with ActorLogging {
  private val file = new File(path + clusterId + ".cluster")
  private val bw = new BufferedWriter(new FileWriter(file))
  bw.write(center)
  for (i <- 0 until 5) bw.newLine()
  bw.write("--------------words-----------------")

  def receive = {
    case word: AppendWord =>
      try {
        bw.write(word.word)
        bw.newLine()
      } catch {
        case exp: Exception => log.error(s"Cant write $word to cluster file #$clusterId", exp)
      }
    case PoisonPill =>
      bw.close()
  }

}

case class AppendWord(word: String)
