{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
module GarbageCollectionSpec (spec) where

import Numeric.Dimensions
import Torch.Double.Dynamic as Math
import qualified Torch.Core.Random as R (newRNG)

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
  let t0 = _new (dims :: Dims '[8, 4])
      t1 = t0
  _fill 3 t1
  let t2 = _fill 6 t1
  print t0 -- should be matrix of 3.0
  print t1 -- should be matrix of 3.0
  print t2 -- should be matrix of 6.0

testOps :: IO ()
testOps = do
  print $ _neg $ _addConst (_new (dims :: Dims '[2, 2])) 3
  print $ _sigmoid $ _neg $ _addConst (_new (dims :: Dims '[2,2])) 3
  new (dims :: Dims '[2, 2]) >>= addConst () 3 >>= sigmoid >>= \(r::DoubleDynamic) -> print r

  foo :: DoubleDynamic <- constant (dims :: Dims '[5]) 3
  print (3.0 * 3.0 * 5 :: Double)
  dot foo foo >>= print

  new (dims :: Dims '[5]) >>= (`add` 2) >>= \(r::DoubleDynamic) -> print r
  new (dims :: Dims '[5]) >>= (`add` 2) >>= (`Math.div` 4) >>= \(r::DoubleDynamic) -> print r
-}

-- TODO : move raw test elsewhere?
rawTest = do
  let
    x :: DoubleDynamic
    x = constant (dims :: Dims '[5]) 2.0
    y = constant (dims :: Dims '[5]) 3.0
    z = constant (dims :: Dims '[5]) 4.0
  print x
  -- cadd = z <- y + scalar * x, z value discarded
  print (2.0 * 4.4 + 3.0 :: Double)
  cadd_ z 4.4 x
  cadd_ y 4.4 x
  print z
  print y

testCadd = do
  let foo :: DoubleDynamic = constant (dims :: Dims '[5]) 5
      bar :: DoubleDynamic = constant (dims :: Dims '[5]) 2
  print $ 5 + 3 * 2
  cadd foo 3.0 bar >>= print

testCopy :: IO ()
testCopy = do
  foo :: DoubleDynamic <- new (dims :: Dims '[3, 3])
  fill_ foo 5
  bar <- newWithTensor foo
  print foo
  print bar
  baz <- add foo 2.0
  fob <- sub bar 2.0
  print foo
  print bar
  print baz
  print fob
  pure ()

matrixMultTest :: IO ()
matrixMultTest = do
  gen <- R.newRNG
  let Just o10 = ord2Tuple (-10, 10)
  mapM_ (\_ -> go gen o10) [1..10]
  where
    go gen o10 = do
      mat' :: DoubleDynamic <- uniform (dims :: Dims '[10, 7]) gen o10
      vec' :: DoubleDynamic <- uniform (dims :: Dims '[7])     gen o10
      print mat'
      print vec'
      -- print $ mat !* vec

testLapack :: IO ()
testLapack = do
  rng <- R.newRNG
  let Just o1 = ord2Tuple (-1.0, 1.0)
  t :: DoubleDynamic <- uniform (dims :: Dims '[2, 2]) rng o1

  let b  = constant (dims :: Dims '[2, 1]) 1.0
      x  = constant (dims :: Dims '[2, 1]) 0
      lu = constant (dims :: Dims '[2, 2]) 0

  gesv_ (x, lu) b t
  print x
  print lu

  let resQ = constant (dims :: Dims '[2, 2]) 0
      resR = constant (dims :: Dims '[2, 2]) 0
  qr_ (resQ,resR) t
  print resQ
  print resR

