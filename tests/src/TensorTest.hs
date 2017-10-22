module Main where

import TensorDouble
import TensorDoubleLapack
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
  let t0 = tdNew (D2 8 4)
      t1 = t0
  fillMutate_ 3.0 t1
  let t2 = fillCopy_ 6.0 t1
  disp t0 -- should be matrix of 3.0
  disp t1 -- should be matrix of 3.0
  disp t2 -- should be matrix of 6.0

testOps = do
  disp $ neg $ addConst (tdNew (D2 2 2)) 3
  disp $ sigmoid $ neg $ addConst (tdNew (D2 2 2)) 3
  disp $ sigmoid $ addConst (tdNew (D2 2 2)) 3

  let foo = fillCopy_ 3.0 $ tdNew (D1 5)
  print $ 3.0 * 3.0 * 5
  print $ dot foo foo

  disp $ (tdNew (D1 5)) ^+ 2.0
  disp $ ((tdNew (D1 5)) ^+ 2.0) ^/ 4.0

-- TODO : move raw test elsewhere?
rawTest = do
  x <- tensorRaw (D1 5) 2.0
  y <- tensorRaw (D1 5) 3.0
  z <- tensorRaw (D1 5) 4.0
  dispRaw x
  -- cadd = z <- y + scalar * x, z value discarded
  print $ 2.0 * 4.4 + 3.0
  c_THDoubleTensor_cadd z y 4.4 x
  dispRaw z

testCadd = do
  let foo = fillCopy_ 5.0 $ tdNew (D1 5)
  let bar = fillCopy_ 2.0 $ tdNew (D1 5)
  print $ 5 + 3 * 2
  disp $ cadd foo 3.0 bar

testCopy :: IO ()
testCopy = do
  let foo = fillCopy_ 5.0 $ tdNew (D2 3 3)
  let bar = td_newWithTensor foo
  disp foo
  disp bar
  let baz = foo ^+ 2.0
  let fob = bar ^- 2.0
  disp foo
  disp bar
  disp baz
  disp fob
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
      disp mat
      disp vec
      disp $ mat !* vec

testLapack = do
  rng <- newRNG
  let rnd = tdNew (D2 2 2)
  t <- uniformT rnd rng (-1.0) 1.0

  let b = tensorDoubleInit (D1 2) 1.0
  let (resA, resB) = gesv t b
  disp resA
  disp resB

  let (resQ, resR) = qr t
  disp resQ
  disp resR
  pure ()

main = do
  testGCTensor
  testOps
  rawTest
  testCadd
  testCopy
  testLapack
  matrixMultTest

