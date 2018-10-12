{-# LANGUAGE LambdaCase #-}
module Torch.Static.TensorSpec where

import Control.Monad
import GHC.Int
import Test.Hspec
import Torch.Double
import Data.Maybe
import Data.List.NonEmpty (NonEmpty((:|)))
import qualified Torch.Double.Dynamic as Dynamic
import qualified Torch.Double.Storage as Storage
import qualified Data.List as List
import qualified Data.List.NonEmpty as NE

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "tensor construction and access" tensorConstructionSpec
  describe "un/squeezing" $ do
    let d1 = dim :: Dim 1
    it "a dimension" $ do
      shape (unsqueeze1d d1 (constant 50 :: Tensor '[3,4,5])) `shouldBe` [3,1,4,5]

    it "unsqueeze adds a dimension" $ do
      shape (unsqueeze1d d1 (constant 50 :: Tensor '[3,4,5])) `shouldBe` [3,1,4,5]

    it "squeeze removes a dimension2" $ do
      shape (constant 50 :: Tensor '[3,1,4,5]) `shouldBe` [3,1,4,5]

    it "squeeze removes a dimension" $ do
      shape (squeeze1d d1 (constant 50 :: Tensor '[3,1,4,5])) `shouldBe` [3,4,5]

    it "unsqueeze+squeeze forms an identity" $ do
      shape (squeeze1d d1 (unsqueeze1d d1 (constant 50 :: Tensor '[3,4,5]))) `shouldBe` [3,4,5]

  describe "stacking" $ do
    it "works with two identical tensors" $ do
      let t = constant (-(1/4)) :: Tensor '[6]
      tensordata (stack1d0 t t) `shouldBe` replicate 12 (-1/4)

    it "works with two differing tensors" $ do
      let t1 = constant (-(1/4)) :: Tensor '[6]
          t2 = constant   (1/4)  :: Tensor '[6]
      tensordata (stack1d0 t1 t2) `shouldBe` (replicate 6 (-1/4) ++ replicate 6 (1/4))


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

      it "returns the same values with get2d" $ do
        let
          tvals :: [[HsReal]]
          tvals =
            flip fmap ixs $ \(r, ics) ->
              flip fmap ics $ \(c, _) ->
                fromMaybe (error "impossible -- test is set up incorrectly") $ get2d ten r c

        pure tvals >>= (`shouldBe` xs)

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

      it "has the expected storage after print" $
        Storage.storagedata (storage ten) `shouldBe` concat xs


