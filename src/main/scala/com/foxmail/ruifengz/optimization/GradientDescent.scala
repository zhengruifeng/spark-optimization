package com.foxmail.ruifengz.optimization

import org.apache.spark.Logging
import org.apache.spark.rdd.RDD

import breeze.linalg.{Vector, DenseVector}

import scala.collection.mutable.ArrayBuffer


/**
  * Created by zrf on 12/4/15.
  */
class GradientDescent(
                       private var gradient: Gradient,
                       private var regularizer: Regularizer,
                       private var updater: Updater,
                       private var criteria: Criteria)
  extends Optimizer with Logging {

  private var miniBatchFraction: Double = 0.1
  private var numIterations: Int = 100

  /**
    * :: Experimental ::
    * Set fraction of data to be used for each iteration.
    * Default 0.1 (corresponding to deterministic/classical gradient descent)
    */
  def setNumBatches(miniBatchFraction: Int): this.type = {
    require(miniBatchFraction > 0)
    this.miniBatchFraction = miniBatchFraction
    this
  }

  /**
    * Set the number of iteration for training. Default 100.
    */
  def setNumEpochs(numIterations: Int): this.type = {
    require(numIterations > 0)
    this.numIterations = numIterations
    this
  }

  /**
    * Set the gradient function (of the loss function of one single data example)
    * to be used for Gradient Descent.
    */
  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }

  /**
    * Set the regularizer function (of the loss function of one single data example)
    * to be used for Gradient Descent.
    */
  def setRegularizer(regularizer: Regularizer): this.type = {
    this.regularizer = regularizer
    this
  }

  /**
    * Set the updater function to actually perform a gradient step in a given direction.
    * The updater is responsible to perform the update from the regularization term as well,
    * and therefore determines what kind or regularization is used, if any.
    */
  def setUpdater(updater: Updater): this.type = {
    this.updater = updater
    this
  }

  /**
    * Set the criteria function (of the loss function of one single data example)
    * to be used for Gradient Descent.
    */
  def setCriteria(criteria: Criteria): this.type = {
    this.criteria = criteria
    this
  }


  override def optimize(data: RDD[(Double, Vector[Double])], initialWeights: Vector[Double]): Vector[Double] = {
    val (weights, _) = GradientDescent.run(
      data,
      miniBatchFraction,
      numIterations,
      gradient,
      regularizer,
      updater,
      criteria,
      initialWeights)
    weights
  }

}


object GradientDescent extends Logging {

  def run(data: RDD[(Double, Vector[Double])],
          miniBatchFraction: Double,
          numIterations: Int,
          gradient: Gradient,
          regularizer: Regularizer,
          updater: Updater,
          criteria: Criteria,
          initialWeights: Vector[Double]): (Vector[Double], Array[Double]) = {


    val maxRecodes = Math.max(updater.maxRecodes, criteria.maxRecodes)

    val lossRecodes = ArrayBuffer[Double]()

    val numExamples = data.count()

    logWarning(s"numExamples $numExamples.")

    // if no data, return initial weights to avoid NaNs
    if (numExamples == 0) {
      logWarning("GradientDescent.run returning initial weights, no data found")
      return (initialWeights, lossRecodes.toArray)
    }

    val n = initialWeights.length
    var weightsRecords = ArrayBuffer[Vector[Double]]()
    if (maxRecodes > 0) {
      weightsRecords.append(initialWeights)
    }

    val weights = initialWeights.copy.toDenseVector

    var iter = 0
    var converged = false

    while (iter < numIterations && !converged) {

      val bcWeights = data.context.broadcast(weights)

      val (grad: DenseVector[Double], error: Double, batchSize: Long) =
        data.sample(false, miniBatchFraction, 42 + iter)
          .treeAggregate((DenseVector.zeros[Double](n), 0.0, 0l))(

            seqOp = (c: (DenseVector[Double], Double, Long), v: (Double, Vector[Double])) => {
              // c: (grad, loss, count), v: (label, features)
              val l = gradient.compute(v._1, v._2, bcWeights.value, c._1)
              (c._1, c._2 + l, c._3 + 1)
            },

            combOp = (c1: (DenseVector[Double], Double, Long), c2: (DenseVector[Double], Double, Long)) => {
              (c1._1 += c2._1, c1._2 + c2._2, c1._3 + c2._3)
            }
          )


      if (batchSize > 0) {
        grad :/= batchSize.toDouble

        val regVal = regularizer.compute(weights, grad)
        val loss = error / batchSize + regVal
        lossRecodes.append(loss)

        logInfo(s"Iteration $iter. BatchSize $batchSize, Loss $loss, Error ${error / batchSize}, Reg $regVal.")

        updater.compute(iter, grad, weightsRecords, weights)

        if (maxRecodes > 0) {
          if (weightsRecords.length >= maxRecodes) {
            weightsRecords = weightsRecords.takeRight(maxRecodes - 1)
          }
          weightsRecords.append(weights)
        }

        converged = criteria.compute(weightsRecords, lossRecodes)
      } else {
        logWarning(s"Iteration $iter. BatchSize $batchSize.")
      }
      iter += 1
    }

    updater.postprocess(weights)

    (weights, lossRecodes.toArray)
  }
}









