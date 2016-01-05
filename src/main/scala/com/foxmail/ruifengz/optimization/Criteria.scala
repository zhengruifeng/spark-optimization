package com.foxmail.ruifengz.optimization

import breeze.linalg.Vector

/**
  * Created by zrf on 12/4/15.
  */
abstract class Criteria extends Serializable {

  /**
    *
    * @param weightsRecords
    * @param lossRecords
    * @return
    */
  def compute(weightsRecords: Seq[Vector[Double]],
              lossRecords: Seq[Double]): Boolean


  // maximum number of weights records
  def maxRecodes: Int = 1
}


class AlwaysFalseCriteria extends Criteria {

  override def compute(weightsRecords: Seq[Vector[Double]],
                       lossRecords: Seq[Double]): Boolean = false

  override def maxRecodes: Int = 0
}