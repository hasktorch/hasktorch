{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Indef.Static.TensorSpec (spec) where

import Control.Monad (replicateM, void)
import Data.Maybe (fromMaybe)
import Numeric.Dimensions (Dim, dim)
import Test.Hspec
import Test.QuickCheck (Property)
import Test.QuickCheck.Monadic (run, monadicIO)
import qualified Data.List as List
import qualified Data.List.NonEmpty as NE

import Torch.Indef.Mask
import Torch.Indef.Static.Tensor
import Torch.Indef.Static.Tensor.Math
import Torch.Indef.Static.Tensor.Math.CompareT
import Torch.Indef.Types (Tensor, HsReal, asDynamic)
import qualified Torch.Indef.Dynamic.Tensor as Dynamic
import qualified Torch.Indef.Storage as Storage


main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  -- describe "Eq instance" eqSpec
  describe "transpose"   transpose2dSpec
  describe "resize"      resizeSpec
  describe "scenario" $
    it "runs this scenario as expected without crashing" testScenario
  describe "expand" $
    it "can expand a vector without crashing" expandScenario
  describe "tensor construction and access" tensorConstructionSpec
  describe "stacking" stackingSpec
  describe "un/squeezing" squeezeUnsqeezeSpec
  describe "transpose2d" transpose2dSpec
  xdescribe "expectations between tensor and storage" tensorStorageExpectationsSpec

testScenario :: Property
testScenario = monadicIO . run $ do
  putStrLn "\n1 - new [2, 2]"
  print (new :: Tensor '[2, 2])
  putStrLn "\n2 - new [2, 4]"
  print (new :: Tensor '[2, 4])
  putStrLn "\n3 - new [2, 2, 2]"
  print (new :: Tensor '[2, 2, 2])
  putStrLn "\n4 - new [8, 4]"
  print (new :: Tensor '[8, 4])
  putStrLn "\n5 - constant [3, 4]"
  print $ asDynamic (constant 2 :: Tensor '[3, 4])
  putStrLn "\n6 - newClone [2, 3]"
  print $ newClone (constant 2 :: Tensor '[2, 3])

eqSpec :: Spec
eqSpec = do
  it "should be True"  . allOf $ (constant 4 :: Tensor '[2,3]) `eqTensor` (constant 4 :: Tensor '[2,3])
  it "should be False" . allOf $ (constant 3 :: Tensor '[2,3]) `eqTensor` (constant 1 :: Tensor '[2,3])

-- transposeSpec :: Spec
-- transposeSpec = do
--   runIO $ print $ trans . trans . trans $ (constant 3 :: Tensor '[3,2])
--   it "should transpose correctly" $ (trans . trans $ (constant 3 :: Tensor '[3,2])) == (constant 3 :: Tensor '[3,2])

resizeSpec :: Spec
resizeSpec = runIO $ do
  Just (vec :: Tensor '[6]) <- fromList [1, 5, 2, 4, 3, 3.5]
  let mtx = resizeAs vec :: Tensor '[3,2]
  print vec
  print mtx
 where

expandScenario :: IO ()
expandScenario = do
  Just (foo :: Tensor '[4]) <- fromList [1,2,3,4]
  let result = expand2d foo :: Tensor '[3, 4]
  print result

tensorConstructionSpec :: Spec
tensorConstructionSpec =
  describe "matrix" $ do
    it "with two rows of 3 elements" $ do
      let (r1, r2) = ([1,2,3],[4,5,6])
      (tensordata <$> (unsafeMatrix [r1, r2] :: IO (Tensor '[2,3])))
        >>= (`shouldBe` r1 ++ r2)

    describe "15x25 elements" $ do
      let ixs :: [(Word, [(Word, Double)])]
          ixs = fmap (\r -> (r, fmap (\c -> (c, fromIntegral $ 12 * r + c)) [0..24])) [0..14]
          xs  = fmap ((fmap snd) . snd) ixs
      ten :: Tensor '[15,25] <- runIO $ unsafeMatrix xs

      it "returns the same values with tensordata" $ do
        tensordata ten `shouldBe` concat xs
        tensordata ten `shouldBe` concat xs

      it "returns the same values with get2d" $ do
        let
          tvals :: [[HsReal]]
          tvals =
            flip fmap ixs $ \(r, ics) ->
              flip fmap ics $ \(c, _) ->
                fromMaybe (error "test is set up incorrectly") $ get2d ten r c
        pure tvals >>= (`shouldBe` xs)

tensorStorageExpectationsSpec :: Spec
tensorStorageExpectationsSpec = do
  let ixs :: [(Word, [(Word, Double)])]
      ixs = fmap (\r -> (r, fmap (\c -> (c, fromIntegral $ 12 * r + c)) [0..24])) [0..14]
      xs  = fmap ((fmap snd) . snd) ixs
  ten :: Tensor '[15,25] <- runIO $ unsafeMatrix xs

  it "prints the same storage values that it gets" $ do
    let std = Storage.storagedata (storage ten)
    let td = tensordata ten
    std `shouldBe` td

  -- forM_ [1,2,3] $ \i -> do
  --   it ("[Attempt " ++ show i ++ "] prints the same storage values as tensorSlices") $ do
  --     SomeDims ds <- getDims ten
  --     let t = asDynamic ten
  --         ds' = (fromIntegral <$> listDims ds)
  --     Dynamic.TenMatricies (m@(r0:|rs):|[]) <- Dynamic.tensorSlices
  --       t
  --       undefined
  --       (Dynamic.get2d t)
  --       ds'
  --     m `shouldBe` (NE.fromList xs)

  it "has the same data after printing (paranoia levels=high)" $ do
    tensordata ten `shouldBe` concat xs

  it "has the same storage data after printing (paranoia levels=high)" $ do
    Storage.storagedata (storage ten) `shouldBe` concat xs


stackingSpec :: Spec
stackingSpec = do
  it "works with two identical tensors" $ do
    let t = constant (-(1/4)) :: Tensor '[6]
    tensordata (stack1d0 t t) `shouldBe` (replicate 12 (-1/4))

  it "works with two differing tensors" $ do
    let t1 = constant (-(1/4)) :: Tensor '[6]
        t2 = constant   (1/4)  :: Tensor '[6]
    tensordata (stack1d0 t1 t2) `shouldBe` (replicate 6 (-1/4) ++ replicate 6 (1/4))

squeezeUnsqeezeSpec :: Spec
squeezeUnsqeezeSpec = do
  let d1 = dim :: Dim 1
  it "unsqueeze1d adds a dimension" $ do
    shape (unsqueeze1d d1 (constant 50 :: Tensor '[3,4,5])) `shouldBe` [3,1,4,5]

  it "unsqueeze1d_ adds a dimension and is unsafe" $ do
    let t = constant 50 :: Tensor '[3,4,5]
    shape t `shouldBe` [3,4,5]
    t' <- unsqueeze1d_ d1 t
    shape t' `shouldBe` [3,1,4,5]
    shape t  `shouldBe` [3,1,4,5]

  it "squeeze removes a dimension" $ do
    let t = (constant 50 :: Tensor '[3,1,4,5])
    shape t `shouldBe` [3,1,4,5]
    shape (squeeze1d d1 t) `shouldBe` [3,4,5]

  it "squeeze1d_ adds a dimension and is unsafe" $ do
    let t = (constant 50 :: Tensor '[3,1,4,5])
    shape t `shouldBe` [3,1,4,5]
    t' <- squeeze1d_ d1 t
    shape t' `shouldBe` [3,4,5]
    shape t  `shouldBe` [3,4,5]

  it "unsqueeze+squeeze forms an identity" $ do
    let t  = constant 50 :: Tensor '[3,4,5]
        t' = squeeze1d d1 (unsqueeze1d d1 t)
    shape t' `shouldBe` [3,4,5]
    nElement t' `shouldBe` product [3,4,5]
    tensordata t' `shouldSatisfy` all (== 50)

transpose2dSpec :: Spec
transpose2dSpec = do
  it "transposes the shape" $ do
    let t = constant 10 :: Tensor '[2,3]
    shape t `shouldBe` [2,3]
    shape (transpose2d t) `shouldBe` [3,2]
    shape t `shouldBe` [2,3]

  it "doesn't change the underlying data" $ do
    let xs = [[1..3],[4..6]]
    t :: Tensor '[2,3] <- unsafeMatrix xs
    tensordata t `shouldBe` tensordata (transpose2d t)
    tensordata t `shouldBe` concat xs


