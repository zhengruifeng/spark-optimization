package com.foxmail.ruifengz.optimization

import breeze.linalg.{Vector, DenseVector}

/**
  * Created by zrf on 12/4/15.
  */

abstract class Gradient extends Serializable {

  /**
    * Compute the gradient and loss given the features of a single data point,
    * add the gradient to a provided vector to avoid creating new objects, and return loss.
    *
    * @param label label for this data point
    * @param data features for one data point
    * @param weights weights/coefficients corresponding to features
    * @param cumGradient the computed gradient will be added to this vector
    *
    * @return loss
    */
  def compute(label: Double,
              data: Vector[Double],
              weights: Vector[Double],
              cumGradient: Vector[Double]): Double

}
