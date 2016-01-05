package com.foxmail.ruifengz.classification


import com.foxmail.ruifengz.optimization._
import com.foxmail.ruifengz.util.VectorUtil

import breeze.linalg.{Vector, DenseMatrix, DenseVector, sum}

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.util.Random

/**
  * Created by zrf on 12/4/15.
  */

class FMModel(val task: Int,
              val w0: Double,
              val w: Option[Vector[Double]],
              val v: DenseMatrix[Double]) extends Serializable {

  if (w.isDefined)
    require(w.get.length == v.cols)

  def predict(test: Vector[Double]): Double = {

    require(test.size == v.cols)

    var pred = w0

    if (w.isDefined) {
      pred += test.dot(w.get)
    }

    for (f <- 0 until v.rows) {
      val c = test :* v(f, ::).t
      val s = sum(c)
      val s2 = c.dot(c)
      pred += (s * s - s2) / 2
    }

    task match {
      case 0 =>
        pred
      case 1 =>
        1.0 / (1.0 + Math.exp(-pred))
    }
  }
}


class FMGradient(val task: Int,
                 val dim: (Boolean, Boolean, Int),
                 val numFeatures: Int,
                 val minLabel: Double,
                 val maxLabel: Double) extends Gradient {

  val k0 = dim._1
  val k1 = dim._2
  val k2 = dim._3

  def predict(data: Vector[Double],
              weights: Vector[Double]): (Double, Array[Double]) = {

    val n = weights.length

    var pred = if (k0) weights(n - 1) else 0.0

    if (k1) {
      val w = weights(numFeatures * k2 until numFeatures * k2 + numFeatures)
      pred += data.dot(w)
    }

    val sums = Array.fill(k2)(0.0)
    for (f <- 0 until k2) {
      val vf = weights(f until numFeatures * k2 by k2)
      val c = data :* vf
      sums(f) = sum(c)
      val sum2 = c.dot(c)
      pred += (sums(f) * sums(f) - sum2) / 2
    }

    if (task == 0) {
      pred = Math.max(pred, minLabel)
      pred = Math.min(pred, maxLabel)
    }

    (pred, sums)
  }


  override def compute(label: Double,
                       data: Vector[Double],
                       weights: Vector[Double],
                       cumGradient: Vector[Double]): Double = {

    require(weights.length == cumGradient.length)

    val n = weights.length

    val (pred, sums) = predict(data, weights)

    val mult = task match {
      case 0 =>
        pred - label
      case 1 =>
        -label * (1.0 - 1.0 / (1.0 + Math.exp(-label * pred)))
    }

    if (k0) {
      cumGradient(n - 1) += mult
    }

    if (k1) {
      val gradW = cumGradient(numFeatures * k2 until numFeatures * k2 + numFeatures)
      gradW :+= (data * mult)
    }

    for (f <- 0 until k2) {
      val gradVf = cumGradient(f until numFeatures * k2 by k2)
      val diff = data :* (sums(f) - (gradVf :* data))
      gradVf :+= (diff * mult)
    }

    task match {
      case 0 =>
        (pred - label) * (pred - label)
      case 1 =>
        1 - Math.signum(pred * label)
    }
  }
}


class FMRegularizer(val dim: (Boolean, Boolean, Int),
                    val regParam: (Double, Double, Double),
                    val numFeatures: Int) extends Regularizer {

  val k0 = dim._1
  val k1 = dim._2
  val k2 = dim._3
  val r0 = regParam._1
  val r1 = regParam._2
  val r2 = regParam._3

  /**
    *
    * @param weights current weights vector
    * @param grad gradient WITHOUT regularization, will be added by regularization's gradient
    * @return regularization value
    */
  override def compute(weights: Vector[Double],
                       grad: Vector[Double]): Double = {

    require(weights.length == grad.length)

    val n = weights.length

    var regVal = 0.0

    if (k0) {
      grad(n - 1) += weights(n - 1) * r0
      regVal += weights(n - 1) * weights(n - 1) * r0
    }

    if (k1) {
      val w = weights(numFeatures * k2 until numFeatures * k2 + numFeatures)
      val gradW = grad(numFeatures * k2 until numFeatures * k2 + numFeatures)
      gradW :+= (w * r1)
      regVal += (w.dot(w) * r1)
    }


    val v = weights(0 until numFeatures * k2)
    val gradV = grad(0 until numFeatures * k2)
    gradV :+= (v * r2)
    regVal += (v.dot(v) * r2)

    regVal / 2.0
  }
}


object FMWithSGD {

  def train(input: RDD[LabeledPoint],
            task: Int = 0,
            miniBatchFraction: Double = 0.01,
            numIterations: Int = 1000,
            learningRate: Double = 0.01,
            momentum: Double = 0.9,
            dim: (Boolean, Boolean, Int) = (true, true, 4),
            regParam: (Double, Double, Double) = (0, 1e-2, 1e-6),
            initStd: Double = 0.1): FMModel = {

    val numFeatures = input.first().features.size

    val k0 = dim._1
    val k1 = dim._2
    val k2 = dim._3

    val initialWeights =
      (k0, k1) match {
        case (true, true) =>
          DenseVector(Array.fill(numFeatures * k2)(Random.nextGaussian() * initStd) ++
            Array.fill(numFeatures + 1)(0.0))

        case (true, false) =>
          DenseVector(Array.fill(numFeatures * k2)(Random.nextGaussian() * initStd) ++
            Array(0.0))

        case (false, true) =>
          DenseVector(Array.fill(numFeatures * k2)(Random.nextGaussian() * initStd) ++
            Array.fill(numFeatures)(0.0))

        case (false, false) =>
          DenseVector(Array.fill(numFeatures * k2)(Random.nextGaussian() * initStd))
      }

    val data = task match {
      case 0 =>
        input.map(l => (l.label, VectorUtil.transform(l.features))).persist()
      case 1 =>
        input.map(l => (if (l.label > 0) 1.0 else -1.0, VectorUtil.transform(l.features))).persist()
    }

    var minLabel = Double.MaxValue
    var maxLabel = Double.MinValue

    if (task == 0) {
      val (minT, maxT) = data.map(_._1).aggregate[(Double, Double)]((Double.MaxValue, Double.MinValue))({
        case ((min, max), v) =>
          (Math.min(min, v), Math.max(max, v))
      }, {
        case ((min1, max1), (min2, max2)) =>
          (Math.min(min1, min2), Math.max(max1, max2))
      })

      minLabel = minT
      maxLabel = maxT
    }

    val gradient = new FMGradient(task, dim, numFeatures, minLabel, maxLabel)
    val regularizer = new FMRegularizer(dim, regParam, numFeatures)
    val updater = new SGDUpdater(learningRate, momentum)
    val criteria = new AlwaysFalseCriteria

    val (weights, lossRecords) = GradientDescent.run(data, miniBatchFraction, numIterations,
      gradient, regularizer, updater, criteria, initialWeights)

    val v = weights(0 until numFeatures * k2).toDenseVector.toDenseMatrix.reshape(k2, numFeatures)

    val w = if (k1) Some(weights(numFeatures * k2 until numFeatures * k2 + numFeatures)) else None

    val w0 = if (k0) weights(weights.length - 1) else 0.0

    new FMModel(task, w0, w, v)
  }
}


