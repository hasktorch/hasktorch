module TensorTests where

import Tensor

-- |basic test of garbage collected tensor
testGCTensor = do
  let t0 = tensorNew_ (D2 8 4)
      t1 = t0
  fillMutate_ 3.0 t1
  let t2 = fillCopy_ 6.0 t1
  disp_ t0 -- should be matrix of 3.0
  disp_ t1 -- should be matrix of 3.0
  disp_ t2 -- should be matrix of 6.0

testOps = do
  disp_ $ neg $ addConst (tensorNew_ (D2 2 2)) 3
  disp_ $ sigmoid $ neg $ addConst (tensorNew_ (D2 2 2)) 3
  disp_ $ sigmoid $ addConst (tensorNew_ (D2 2 2)) 3
