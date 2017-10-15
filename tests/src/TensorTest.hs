module Main where

import TensorDouble
import TensorDoubleMath
import TensorDoubleRandom
import Random
import TensorTypes
import TensorUtils

-- TODO : move raw tests elsewhere?
import TensorRaw
import THDoubleTensorMath

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

  let foo = fillCopy_ 3.0 $ tensorNew_ (D1 5)
  print $ 3.0 * 3.0 * 5
  print $ dot foo foo

  disp_ $ (tensorNew_ (D1 5)) ^+ 2.0
  disp_ $ ((tensorNew_ (D1 5)) ^+ 2.0) ^/ 4.0

-- TODO : move raw test elsewhere?
rawTest = do
  x <- tensorRaw (D1 5) 2.0
  y <- tensorRaw (D1 5) 3.0
  z <- tensorRaw (D1 5) 4.0
  disp x
  -- cadd = z <- y + scalar * x, z value discarded
  print $ 2.0 * 4.4 + 3.0
  c_THDoubleTensor_cadd z y 4.4 x
  disp z

testCadd = do
  let foo = fillCopy_ 5.0 $ tensorNew_ (D1 5)
  let bar = fillCopy_ 2.0 $ tensorNew_ (D1 5)
  print $ 5 + 3 * 2
  disp_ $ cadd foo 3.0 bar

testCopy :: IO ()
testCopy = do
  let foo = fillCopy_ 5.0 $ tensorNew_ (D2 3 3)
  let bar = newWithTensor foo
  disp_ foo
  disp_ bar
  let baz = foo ^+ 2.0
  let fob = bar ^- 2.0
  disp_ foo
  disp_ bar
  disp_ baz
  disp_ fob
  pure ()


matrixMultTest = do
  gen <- newRNG
  mapM_ (\_ -> go gen) [1..10]
  where
    go gen = do
      let mat = (tensorDoubleInit (D2 10 7) 2.2)
      let vec = (tensorDoubleInit (D1 7) 1.0)
      mat <- uniformT mat gen (-10.0) (10.0)
      vec <- uniformT vec gen (-10.0) (10.0)
      disp_ mat
      disp_ vec
      disp_ $ mat !* vec

main = do
  testGCTensor
  testOps
  rawTest
  testCadd
  testCopy
  matrixMultTest
