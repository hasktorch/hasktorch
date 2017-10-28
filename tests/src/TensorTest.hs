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
  let t0 = td_new (D2 8 4)
      t1 = t0
  td_fill_ 3.0 t1
  let t2 = td_fill 6.0 t1
  disp t0 -- should be matrix of 3.0
  disp t1 -- should be matrix of 3.0
  disp t2 -- should be matrix of 6.0

testOps = do
  disp $ td_neg $ td_addConst (td_new (D2 2 2)) 3
  disp $ td_sigmoid $ td_neg $ td_addConst (td_new (D2 2 2)) 3
  disp $ td_sigmoid $ td_addConst (td_new (D2 2 2)) 3

  let foo = td_fill 3.0 $ td_new (D1 5)
  print $ 3.0 * 3.0 * 5
  print $ td_dot foo foo

  disp $ (td_new (D1 5)) ^+ 2.0
  disp $ ((td_new (D1 5)) ^+ 2.0) ^/ 4.0

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
  let foo = td_fill 5.0 $ td_new (D1 5)
  let bar = td_fill 2.0 $ td_new (D1 5)
  print $ 5 + 3 * 2
  disp $ td_cadd foo 3.0 bar

testCopy :: IO ()
testCopy = do
  let foo = td_fill 5.0 $ td_new (D2 3 3)
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
      let mat = (td_init (D2 10 7) 2.2)
      let vec = (td_init (D1 7) 1.0)
      mat <- td_uniform mat gen (-10.0) (10.0)
      vec <- td_uniform vec gen (-10.0) (10.0)
      disp mat
      disp vec
      disp $ mat !* vec

testLapack = do
  rng <- newRNG
  let rnd = td_new (D2 2 2)
  t <- td_uniform rnd rng (-1.0) 1.0

  let b = td_init (D1 2) 1.0
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

