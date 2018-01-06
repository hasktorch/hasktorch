{-# LANGUAGE DataKinds #-}
module GarbageCollectionSpec (spec) where

import Torch.Core.Tensor.Dynamic.Double
import Torch.Core.Tensor.Dynamic.DoubleMath
import Torch.Core.Tensor.Dynamic.DoubleRandom
import Torch.Core.Tensor.Dynamic.GenericLapack
import Torch.Core.Tensor.Types
import Torch.Core.Tensor.Dim

-- TODO : move raw tests elsewhere?
import Torch.Raw.Tensor.Generic
import THDoubleTensorMath

import Lens.Micro ()
import Torch.Prelude.Extras

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  it "runs scenario: testGCTensor" testGCTensor
  it "runs scenario: testOps" testOps
  it "runs scenario: rawTest" rawTest
  it "runs scenario: testCadd" testCadd
  it "runs scenario: testCopy" testCopy
  it "runs scenario: testLapack" testLapack
  it "runs scenario: matrixMultTest" matrixMultTest

-- | basic test of garbage collected tensor
testGCTensor :: Property
testGCTensor = monadicIO . run $ do
  let t0 = td_new (dim :: Dim '[8, 4])
      t1 = t0
  td_fill_ 3 t1
  let t2 = td_fill 6 t1
  printTensor t0 -- should be matrix of 3.0
  printTensor t1 -- should be matrix of 3.0
  printTensor t2 -- should be matrix of 6.0

testOps :: IO ()
testOps = do
  printTensor $ td_neg $ td_addConst (td_new (dim :: Dim '[2, 2])) 3
  printTensor $ td_sigmoid $ td_neg $ td_addConst (td_new (dim :: Dim '[2,2])) 3
  printTensor $ td_sigmoid $ td_addConst (td_new (dim :: Dim '[2, 2])) 3

  let foo = td_fill 3 $ td_new (dim :: Dim '[5])
  print $ (3.0 * 3.0 * 5 :: Double)
  print $ td_dot foo foo

  printTensor $ (td_new (dim :: Dim '[5])) ^+ 2.0
  printTensor $ ((td_new (dim :: Dim '[5])) ^+ 2.0) ^/ 4.0

-- TODO : move raw test elsewhere?
rawTest = do
  x <- constant (dim :: Dim '[5]) 2.0
  y <- constant (dim :: Dim '[5]) 3.0
  z <- constant (dim :: Dim '[5]) 4.0
  dispRaw x
  -- cadd = z <- y + scalar * x, z value discarded
  print $ (2.0 * 4.4 + 3.0 :: Double)
  c_THDoubleTensor_cadd z y 4.4 x
  dispRaw z

testCadd = do
  let foo = td_fill 5.0 $ td_new (dim :: Dim '[5])
  let bar = td_fill 2.0 $ td_new (dim :: Dim '[5])
  print $ 5 + 3 * 2
  printTensor $ td_cadd foo 3.0 bar

testCopy :: IO ()
testCopy = do
  let foo = td_fill 5.0 $ td_new (dim :: Dim '[3, 3])
  let bar = td_newWithTensor foo
  printTensor foo
  printTensor bar
  let baz = foo ^+ 2.0
  let fob = bar ^- 2.0
  printTensor foo
  printTensor bar
  printTensor baz
  printTensor fob
  pure ()

matrixMultTest :: IO ()
matrixMultTest = do
  gen <- newRNG
  mapM_ (\_ -> go gen) [1..10]
  where
    go gen = do
      let mat = (td_init (dim :: Dim '[10, 7]) 2.2)
      let vec = (td_init (dim :: Dim '[7]) 1.0)
      mat <- td_uniform mat gen (-10.0) (10.0)
      vec <- td_uniform vec gen (-10.0) (10.0)
      printTensor mat
      printTensor vec
      -- printTensor $ mat !* vec

testLapack :: IO ()
testLapack = do
  rng <- newRNG
  let rnd = td_new (dim :: Dim '[2, 2])
  t <- td_uniform rnd rng (-1.0) 1.0

  let b = td_init (dim :: Dim '[2]) 1.0
  let (resA, resB) = td_gesv t b
  printTensor resA
  printTensor resB

  let (resQ, resR) = td_qr t
  printTensor resQ
  printTensor resR
  pure ()

