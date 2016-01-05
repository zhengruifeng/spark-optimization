package com.foxmail.ruifengz.optimization


import breeze.linalg.{norm, Vector, DenseVector}

/**
  * Created by zrf on 12/4/15.
  */

abstract class Regularizer extends Serializable {


  /**
    *
    * @param weights current weights vector
    * @param grad gradient WITHOUT regularization, will be added by regularization's gradient
    * @return regularization value
    */
  def compute(weights: Vector[Double], grad: Vector[Double]): Double

}

class EmptyRegularizer extends Regularizer {
  /**
    *
    * @param weights current weights vector
    * @param grad gradient WITHOUT regularization, will be added by regularization's gradient
    * @return regularization value
    */
  override def compute(weights: Vector[Double], grad: Vector[Double]): Double = 0.0
}


class L2Regularizer(val regParam: Double) extends Regularizer {

  /**
    *
    * @param weights current weights vector
    * @param grad gradient WITHOUT regularization, will be added by regularization's gradient
    * @return regularization value
    */
  override def compute(weights: Vector[Double], grad: Vector[Double]): Double = {
    grad :-= weights * regParam

    weights.dot(weights) * regParam / 2.0
  }
}

