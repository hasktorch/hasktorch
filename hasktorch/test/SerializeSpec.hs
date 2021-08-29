module SerializeSpec (spec) where

import System.Directory (removeFile)
import System.IO
import Test.Hspec
import Torch.Serialize
import Torch.Tensor
import Torch.TensorFactories

spec :: Spec
spec = do
  it "save and load tensor" $ do
    let i =
          [ [0, 1, 1],
            [2, 0, 2]
          ] ::
            [[Int]]
        v = [3, 4, 5] :: [Float]
    save [(asTensor i), (asTensor v)] "test.pt"
    tensors <- load "test.pt"
    removeFile "test.pt"
    length tensors `shouldBe` 2
    let [ii, vv] = tensors
    (asValue ii :: [[Int]]) `shouldBe` i
    (asValue vv :: [Float]) `shouldBe` v
  it "save and load a raw data of numpy" $ do
    let org = zeros' [4]
    -- Following python script generates 'test/data/numpy_rawfile' file
    --
    -- #!/usr/bin/env python
    -- import torch
    -- f = open("test/data/numpy_rawfile","wb")
    -- torch.tensor([1,2,3,4], dtype=torch.float32).numpy().tofile(f)
    --
    new <- System.IO.withFile "test/data/numpy_rawfile" System.IO.ReadMode $
      \h -> loadBinary h org
    (asValue new :: [Float]) `shouldBe` [1, 2, 3, 4]
    System.IO.withFile "numpy_rawfile" System.IO.WriteMode $
      \h -> saveBinary h new
    new' <- System.IO.withFile "numpy_rawfile" System.IO.ReadMode $
      \h -> loadBinary h org
    (asValue new' :: [Float]) `shouldBe` [1, 2, 3, 4]
