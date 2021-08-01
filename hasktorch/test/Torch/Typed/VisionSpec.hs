{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DataKinds #-}

module Torch.Typed.VisionSpec
  ( Torch.Typed.VisionSpec.spec
  )
where

import Prelude hiding (length)

import Test.Hspec (Spec, describe, it, shouldBe)

import Torch.Typed
import Torch (asValue, DeviceType ( CPU ))
import qualified Torch as D (DType ( Float ) )


checkAsTensor :: IO ()
checkAsTensor = do
    imagesBS <- decompressFile "test/data" "mnist-sample-images-idx3-ubyte.gz"
    labelsBS <- decompressFile "test/data" "mnist-sample-labels-idx1-ubyte.gz"
    let mnist = MnistData imagesBS labelsBS
    length mnist `shouldBe` 16
    (asValue (toDynamic (getImages @16 mnist [0 ..])) :: [[Float]])
      `shouldBe` asValue (toDynamic (getImages' @16 mnist [0 ..]))

checkImageFolder :: IO ()
checkImageFolder = do
    imagesBS <- decompressFile "test/data" "mnist-sample-images-idx3-ubyte.gz"
    labelsBS <- decompressFile "test/data" "mnist-sample-labels-idx1-ubyte.gz"
    imageFolder <- getImageFolder "test/data/images"
    let mnist = MnistData imagesBS labelsBS
    (asValue (toDynamic (reshape @'[1, 1, 28, 28] $ getImages @1 mnist [0 ..])) :: [[[[Float]]]])
      `shouldBe` asValue (toDynamic $ narrow @1 @0 @1 . getFolderImage @'[1, 3, 28, 28] @'D.Float  @'(CPU, 0) imageFolder $ 0)

spec :: Spec
spec = describe "Load images" $ do
          it "Comparison of using asTensor and memcpy" 
            checkAsTensor
            
          it "Comparison of getFolderImages and asTensor" 
            checkImageFolder
          
