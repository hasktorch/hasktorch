module GarbageCollectionSpec (spec) where

import Torch.Core.Tensor.Dynamic.Double
import Torch.Core.Tensor.Dynamic.DoubleLapack
import Torch.Core.Tensor.Dynamic.DoubleMath
import Torch.Core.Tensor.Dynamic.DoubleRandom
import Torch.Core.Tensor.Types

-- TODO : move raw tests elsewhere?
import Torch.Core.Tensor.Raw
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
testGCTensor = monadicIO $ do
  let t0 = td_new (D2 (8, 4))
      t1 = t0
  run $ td_fill_ (3.0 :: Double) t1
  let t2 = td_fill (6.0 :: Double) t1
  run $ td_p t0 -- should be matrix of 3.0
  run $ td_p t1 -- should be matrix of 3.0
  run $ td_p t2 -- should be matrix of 6.0

testOps :: IO ()
testOps = do
  td_p $ td_neg $ td_addConst (td_new (D2 (2, 2))) 3
  td_p $ td_sigmoid $ td_neg $ td_addConst (td_new $ D2 (2,2)) 3
  td_p $ td_sigmoid $ td_addConst (td_new $ D2 (2, 2)) 3

  let foo = td_fill (3.0 :: Double) $ td_new (D1 5)
  print $ (3.0 * 3.0 * 5 :: Double)
  print $ td_dot foo foo

  td_p $ (td_new (D1 5)) ^+ 2.0
  td_p $ ((td_new (D1 5)) ^+ 2.0) ^/ 4.0

-- TODO : move raw test elsewhere?
rawTest = do
  x <- tensorRaw (D1 5) 2.0
  y <- tensorRaw (D1 5) 3.0
  z <- tensorRaw (D1 5) 4.0
  dispRaw x
  -- cadd = z <- y + scalar * x, z value discarded
  print $ (2.0 * 4.4 + 3.0 :: Double)
  c_THDoubleTensor_cadd z y 4.4 x
  dispRaw z

testCadd = do
  let foo = td_fill 5.0 $ td_new (D1 5)
  let bar = td_fill 2.0 $ td_new (D1 5)
  print $ 5 + 3 * 2
  td_p $ td_cadd foo 3.0 bar

testCopy :: IO ()
testCopy = do
  let foo = td_fill 5.0 $ td_new $ D2 (3, 3)
  let bar = td_newWithTensor foo
  td_p foo
  td_p bar
  let baz = foo ^+ 2.0
  let fob = bar ^- 2.0
  td_p foo
  td_p bar
  td_p baz
  td_p fob
  pure ()

matrixMultTest :: IO ()
matrixMultTest = do
  gen <- newRNG
  mapM_ (\_ -> go gen) [1..10]
  where
    go gen = do
      let mat = (td_init (D2 (10, 7)) 2.2)
      let vec = (td_init (D1 7) 1.0)
      mat <- td_uniform mat gen (-10.0) (10.0)
      vec <- td_uniform vec gen (-10.0) (10.0)
      td_p mat
      td_p vec
      -- td_p $ mat !* vec

testLapack :: IO ()
testLapack = do
  rng <- newRNG
  let rnd = td_new (D2 (2, 2))
  t <- td_uniform rnd rng (-1.0) 1.0

  let b = td_init (D1 2) 1.0
  let (resA, resB) = td_gesv t b
  td_p resA
  td_p resB

  let (resQ, resR) = td_qr t
  td_p resQ
  td_p resR
  pure ()

