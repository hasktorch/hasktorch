{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DataKinds #-}

module ImageSpec
  ( spec
  )
where

import           Test.Hspec
import           Control.Exception.Safe
import           Control.Monad.State.Strict

import qualified Image                         as I
import           Torch.Typed.Tensor
import qualified Torch.Tensor                  as D
import           GHC.Generics

spec :: Spec
spec = describe "Load images" $
  it "Comparison of using asTensor and memcpy" $ do
    imagesBS <- I.decompressFile "test/data" "mnist-sample-images-idx3-ubyte.gz"
    labelsBS <- I.decompressFile "test/data" "mnist-sample-labels-idx1-ubyte.gz"
    let mnist = I.MnistData imagesBS labelsBS
    I.length mnist `shouldBe` 16
    (D.asValue (toDynamic (I.getImages @16 mnist [0 ..])) :: [[Float]])
      `shouldBe` D.asValue (toDynamic (I.getImages' @16 mnist [0 ..]))
