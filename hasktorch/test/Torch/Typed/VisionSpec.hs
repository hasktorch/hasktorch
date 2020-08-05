{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE TypeApplications #-}

module Torch.Typed.VisionSpec
  ( Torch.Typed.VisionSpec.spec
  )
where

import Prelude hiding (length)

import Test.Hspec (Spec, describe, it, shouldBe)

import Torch       (TensorLike (asValue))
import Torch.Typed


checkAsTensor :: IO ()
checkAsTensor = do
    imagesBS <- decompressFile "test/data" "mnist-sample-images-idx3-ubyte.gz"
    labelsBS <- decompressFile "test/data" "mnist-sample-labels-idx1-ubyte.gz"
    let mnist = MnistData imagesBS labelsBS
    length mnist `shouldBe` 16
    (asValue (toDynamic (getImages @16 mnist [0 ..])) :: [[Float]])
      `shouldBe` asValue (toDynamic (getImages' @16 mnist [0 ..]))

spec :: Spec
spec = describe "Load images" $
  it "Comparison of using asTensor and memcpy" checkAsTensor
