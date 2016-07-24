name := "spark-nlp-workout"

version := "1.0"

scalaVersion := "2.11.7"


assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs@_*) => MergeStrategy.discard
  case x => MergeStrategy.first
}

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.6.1" //% "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.6.1" //% "provided"
//libraryDependencies += "com.typesafe.akka" %% "akka-actor" % "2.4.4"
//libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "0.4-rc3.8"
//libraryDependencies += "org.deeplearning4j" % "deeplearning4j-nlp" % "0.4-rc3.8"
//libraryDependencies += "org.nd4j" % "nd4j-x86" % "0.4-rc3.8"

