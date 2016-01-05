package com.foxmail.ruifengz.util

import breeze.linalg.{SparseVector, DenseVector, Vector}

import org.apache.spark.mllib.linalg.{Vector => SparkVector, DenseVector => SparkDenseVector, SparseVector => SparkSparseVector}


/**
  * Created by zrf on 12/4/15.
  */
object VectorUtil {

  def transform(vector: SparkVector): Vector[Double] = {
    vector match {
      case vec: SparkDenseVector =>
        new DenseVector(vec.values)

      case vec: SparkSparseVector =>
        new SparseVector(vec.indices, vec.values, vec.size)
    }
  }

}
