name := "spark-nlp-workout"

version := "1.0"

scalaVersion := "2.11.7"


assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs@_*) => MergeStrategy.discard
  case x => MergeStrategy.first
}

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.6.1" //% "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.6.1" //% "provided"