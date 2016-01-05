package com.foxmail.ruifengz.optimization


import breeze.linalg.{DenseVector, Vector}
import breeze.numerics.{signum, sqrt}

/**
  * Created by zrf on 12/4/15.
  */
abstract class Updater extends Serializable {

  /**
    *
    * @param grad gradient
    * @param weightsRecords weights history
    * @param weights current weights, will be updated
    */
  def compute(iter: Int,
              grad: Vector[Double],
              weightsRecords: Seq[Vector[Double]],
              weights: Vector[Double]): Unit

  /**
    *
    * @param weights
    */
  def postprocess(weights: Vector[Double]): Unit = {}


  def maxRecodes: Int = 0
}


class SGDUpdater(val learningRate: Double = 0.01,
                 val momentum: Double = 0.9) extends Updater {

  require(learningRate > 0)
  require(momentum >= 0)

  /**
    *
    * @param grad gradient
    * @param weightsRecords weights history
    * @param weights current weights, will be updated
    */
  override def compute(iter: Int,
                       grad: Vector[Double],
                       weightsRecords: Seq[Vector[Double]],
                       weights: Vector[Double]): Unit = {

    if (momentum > 0) {
      if (iter == 0) {
        weights -= (grad * learningRate)
      } else {
        val weights1 = weightsRecords.last
        weights :*= (1 + momentum)
        weights :-= weights1 * momentum
        weights :-= grad * learningRate
      }
    } else {
      weights -= grad * learningRate
    }
  }


  override def maxRecodes: Int =
    if (momentum > 0) 1 else 0
}


class AdaGradUpdater(val learningRate: Double = 0.1,
                     val epsilon: Double = 1e-8) extends Updater {

  require(learningRate > 0)
  require(epsilon > 0)

  private var cumGrad2: DenseVector[Double] = null

  /**
    *
    * @param grad gradient
    * @param weightsRecords weights history
    * @param weights current weights, will be updated
    */
  override def compute(iter: Int,
                       grad: Vector[Double],
                       weightsRecords: Seq[Vector[Double]],
                       weights: Vector[Double]): Unit = {

    if (cumGrad2 == null)
      cumGrad2 = DenseVector.fill[Double](weights.length, epsilon)

    cumGrad2 += grad :* grad

    weights -= (grad :/ sqrt(cumGrad2)) * learningRate
  }
}


class AdaDeltaUpdater(val decayRate: Double = 0.95,
                      val epsilon: Double = 1e-8) extends Updater {

  require(decayRate > 0 && decayRate < 1)
  require(epsilon > 0)

  private var cumGrad: DenseVector[Double] = null
  private var cumUpdate: DenseVector[Double] = null

  /**
    *
    * @param grad gradient
    * @param weightsRecords weights history
    * @param weights current weights, will be updated
    */
  override def compute(iter: Int,
                       grad: Vector[Double],
                       weightsRecords: Seq[Vector[Double]],
                       weights: Vector[Double]): Unit = {


    if (cumGrad == null)
      cumGrad = DenseVector.zeros[Double](weights.length)

    if (cumUpdate == null)
      cumUpdate = DenseVector.zeros[Double](weights.length)


    cumGrad :*= decayRate
    cumGrad :+= grad :* grad * (1 - decayRate)

    val update = grad :* sqrt(cumUpdate + epsilon) :/ sqrt(cumGrad + epsilon)

    cumUpdate :*= decayRate
    cumUpdate :+= update :* update * (1 - decayRate)

    weights :-= update
  }
}


class AdamUpdater(val learningRate: Double,
                  val beta1: Double = 0.9,
                  val beta2: Double = 0.999,
                  val epsilon: Double = 1e-8) extends Updater {

  require(learningRate > 0)
  require(beta1 >= 0 && beta1 < 1)
  require(beta2 >= 0 && beta2 < 1)
  require(epsilon > 0)

  private var mom1: DenseVector[Double] = null
  private var mom2: DenseVector[Double] = null
  private var t: Int = 0

  /**
    *
    * @param grad gradient
    * @param weightsRecords weights history
    * @param weights current weights, will be updated
    */
  override def compute(iter: Int,
                       grad: Vector[Double],
                       weightsRecords: Seq[Vector[Double]],
                       weights: Vector[Double]): Unit = {

    if (mom1 == null)
      mom1 = DenseVector.zeros[Double](weights.length)

    if (mom2 == null)
      mom2 = DenseVector.zeros[Double](weights.length)

    mom1 :*= beta1
    mom1 :+= grad * (1 - beta1)

    mom2 :*= beta2
    mom2 :+= grad :* grad * (1 - beta2)

    val b1 = 1 - Math.pow(beta1, t)
    val b2 = 1 - Math.pow(beta2, t)

    weights :-= (mom1 :/ sqrt(mom2 + b2 * epsilon)) * (Math.sqrt(b2) / b1 * learningRate)

    t += 1
  }
}


class NAGUpdater(val learningRate: Double,
                 val momentum: Double) extends Updater {

  require(learningRate > 0)
  require(momentum >= 0)

  private var v: DenseVector[Double] = null

  /**
    *
    * @param grad gradient
    * @param weightsRecords weights history
    * @param weights current weights, will be updated
    */
  override def compute(iter: Int,
                       grad: Vector[Double],
                       weightsRecords: Seq[Vector[Double]],
                       weights: Vector[Double]): Unit = {

    if (v == null)
      v = DenseVector.zeros[Double](weights.length)

    weights :+= v * (momentum * momentum)
    weights :-= grad * ((1 + momentum) * learningRate)

    v :*= momentum
    v :-= grad * learningRate
  }


  override def postprocess(weights: Vector[Double]): Unit = {
    if (v != null)
      weights :-= v * momentum
  }
}


class RMSpropUpdater(val learningRate: Double,
                     val rmsDecayRate: Double = 0.02) extends Updater {

  require(learningRate > 0)
  require(rmsDecayRate > 0)

  private var v: DenseVector[Double] = null
  private var g: DenseVector[Double] = null

  /**
    *
    * @param grad gradient
    * @param weightsRecords weights history
    * @param weights current weights, will be updated
    */
  override def compute(iter: Int,
                       grad: Vector[Double],
                       weightsRecords: Seq[Vector[Double]],
                       weights: Vector[Double]): Unit = {

    if (v == null)
      v = DenseVector.zeros[Double](weights.length)

    if (g == null)
      g = DenseVector.zeros[Double](weights.length)

    val weights1 = weightsRecords.last

    val positive = (signum(g :* grad) + 1.0) / 2.0

    v :+= positive * rmsDecayRate

    v :*= (positive * rmsDecayRate - (rmsDecayRate - 1))

    g := grad

    weights :-= v * learningRate
  }
}


