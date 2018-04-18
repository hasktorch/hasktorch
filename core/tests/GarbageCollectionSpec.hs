{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
module GarbageCollectionSpec (spec) where

import Torch.Dynamic as Math
import qualified Torch.Core.Random as R (new)

import Torch.Prelude.Extras

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  -- it "runs scenario: testGCTensor" testGCTensor
  -- it "runs scenario: testOps" testOps
  it "runs scenario: rawTest" rawTest
  it "runs scenario: testCadd" testCadd
  it "runs scenario: testCopy" testCopy
  it "runs scenario: testLapack" testLapack
  it "runs scenario: matrixMultTest" matrixMultTest

{-
-- | basic test of garbage collected tensor
testGCTensor :: Property
testGCTensor = monadicIO . run $ do
  let t0 = _new (dim :: Dim '[8, 4])
      t1 = t0
  _fill 3 t1
  let t2 = _fill 6 t1
  printTensor t0 -- should be matrix of 3.0
  printTensor t1 -- should be matrix of 3.0
  printTensor t2 -- should be matrix of 6.0

testOps :: IO ()
testOps = do
  printTensor $ _neg $ _addConst (_new (dim :: Dim '[2, 2])) 3
  printTensor $ _sigmoid $ _neg $ _addConst (_new (dim :: Dim '[2,2])) 3
  new (dim :: Dim '[2, 2]) >>= addConst () 3 >>= sigmoid >>= \(r::DoubleDynamic) -> printTensor r

  foo :: FloatDynamic <- constant (dim :: Dim '[5]) 3
  print (3.0 * 3.0 * 5 :: Double)
  dot foo foo >>= print

  new (dim :: Dim '[5]) >>= (`add` 2) >>= \(r::DoubleDynamic) -> printTensor r
  new (dim :: Dim '[5]) >>= (`add` 2) >>= (`Math.div` 4) >>= \(r::DoubleDynamic) -> printTensor r
-}

-- TODO : move raw test elsewhere?
rawTest = do
  x :: FloatDynamic <- constant (dim :: Dim '[5]) 2.0
  y <- constant (dim :: Dim '[5]) 3.0
  z <- constant (dim :: Dim '[5]) 4.0
  printTensor x
  -- cadd = z <- y + scalar * x, z value discarded
  print (2.0 * 4.4 + 3.0 :: Double)
  _cadd z y 4.4 x
  printTensor z

testCadd = do
  foo :: FloatDynamic <- constant (dim :: Dim '[5]) 5
  bar :: FloatDynamic <- constant (dim :: Dim '[5]) 2
  print $ 5 + 3 * 2
  cadd foo 3.0 bar >>= printTensor

testCopy :: IO ()
testCopy = do
  foo :: FloatDynamic <- new (dim :: Dim '[3, 3])
  _fill foo 5
  bar <- newWithTensor foo
  printTensor foo
  printTensor bar
  baz <- add foo 2.0
  fob <- sub bar 2.0
  printTensor foo
  printTensor bar
  printTensor baz
  printTensor fob
  pure ()

matrixMultTest :: IO ()
matrixMultTest = do
  gen <- R.new
  mapM_ (\_ -> go gen) [1..10]
  where
    go gen = do
      mat' :: DoubleDynamic <- uniform (dim :: Dim '[10, 7]) gen (-10) 10
      vec' :: DoubleDynamic <- uniform (dim :: Dim '[7])     gen (-10) 10
      printTensor mat'
      printTensor vec'
      -- printTensor $ mat !* vec

testLapack :: IO ()
testLapack = do
  rng <- R.new
  t :: DoubleDynamic <- uniform (dim :: Dim '[2, 2]) rng (-1.0) 1.0

  b <- constant (dim :: Dim '[2]) 1.0
  resA <- constant (dim :: Dim '[2, 2]) 0
  resB <- constant (dim :: Dim '[2, 2]) 0
  _gesv resA resB t b
  printTensor resA
  printTensor resB

  resQ <- constant (dim :: Dim '[2, 2]) 0
  resR <- constant (dim :: Dim '[2, 2]) 0
  _qr resQ resR t
  printTensor resQ
  printTensor resR

