package org.madhatter.spark.nlp.entityextractionwithword2vec

import java.io.{BufferedWriter, File, FileWriter}
import java.util.concurrent.atomic.AtomicInteger

import akka.actor.{Actor, ActorLogging, PoisonPill}

/**
  * Created by madhatter on 7/24/2016.
  */
class ClusterPrinterActor(clusterId: Int, path: String, center: String) extends Actor with ActorLogging {
  private val file = new File(path + clusterId + ".cluster")
  private val bw = new BufferedWriter(new FileWriter(file))
  private val wordCounter = new AtomicInteger()

  bw.write(center)
  for (i <- 0 until 5) bw.newLine()
  bw.write("--------------words-----------------")

  def receive = {
    case word: AppendWord =>
      try {
        bw.write(word.word)
        bw.newLine()
        val w = wordCounter.incrementAndGet()
        if (w % 100 == 0) log.debug(s"cluster #$clusterId words: $w")
      } catch {
        case exp: Exception => log.error(s"Cant write $word to cluster file #$clusterId", exp)
      }
    case Close =>
      bw.close()
      self ! PoisonPill
  }

}

case class AppendWord(word: String)

case object Close