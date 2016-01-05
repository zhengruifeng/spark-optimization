package com.foxmail.ruifengz.optimization

import org.apache.spark.rdd.RDD
import breeze.linalg.{Vector, DenseVector}

import scala.reflect.ClassTag

/**
  * Created by zrf on 12/4/15.
  */
abstract class Optimizer extends Serializable {

  /**
    * Solve the provided convex optimization problem.
    */
  def optimize(data: RDD[(Double, Vector[Double])], initialWeights: Vector[Double]): Vector[Double]
}