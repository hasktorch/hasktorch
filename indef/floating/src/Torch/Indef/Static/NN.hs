module Torch.Indef.Static.NN where

import Torch.Dimensions

import qualified Torch.Class.NN as Dynamic
import qualified Torch.Class.NN.Static as Class

import Torch.Indef.Types
import Torch.Indef.Dynamic.NN ()
import Torch.Indef.Static.Tensor ()
import Torch.Indef.Static.NN.Abs ()

instance Class.AbsCriterion Tensor where
  _absCriterion_updateOutput i t o = Dynamic.absCriterion_updateOutput (asDynamic i) (asDynamic t) (asDynamic o)
  _absCriterion_updateGradInput i t go gi = Dynamic.absCriterion_updateGradInput (asDynamic i) (asDynamic t) (asDynamic go) (asDynamic gi)

instance Class.BCECriterion Tensor where
  _bCECriterion_updateOutput i t o b w = Dynamic.bCECriterion_updateOutput (asDynamic i) (asDynamic t) (asDynamic o) b (asDynamic w)
  _bCECriterion_updateGradInput i t go gi b w = Dynamic.bCECriterion_updateGradInput (asDynamic i) (asDynamic t) (asDynamic go) (asDynamic gi) b (asDynamic w)

instance Class.BatchNormalization Tensor where
  _batchNormalization_updateOutput t0 t1 t2 t3 t4 t5 t6 t7 = Dynamic.batchNormalization_updateOutput
    (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4)
    (asDynamic t5) (asDynamic t6) (asDynamic t7)
  _batchNormalization_backward t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 = Dynamic.batchNormalization_backward
    (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4)
    (asDynamic t5) (asDynamic t6) (asDynamic t7) (asDynamic t8) (asDynamic t9)

instance Class.Col2Im Tensor where
  _col2Im_updateOutput t0 t1 = Dynamic.col2Im_updateOutput (asDynamic t0) (asDynamic t1)
  _col2Im_updateGradInput g0 g1 = Dynamic.col2Im_updateGradInput (asDynamic g0) (asDynamic g1)

instance Class.NN Tensor where

