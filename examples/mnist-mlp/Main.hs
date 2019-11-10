{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Control.Monad (forM_)
import GHC.Generics

import Torch.Autograd as A
import Torch.Tensor
import Torch.NN

import           ATen.Cast
import qualified Data.ByteString as BS
import qualified Data.ByteString.Internal as BSI
import qualified Torch.DType as D
import qualified Torch.Tensor                  as D
import qualified Torch.TensorOptions           as D
import qualified Foreign.ForeignPtr            as F
import qualified Foreign.Ptr                   as F
import qualified Torch.Managed.Native as LibTorch

import qualified Image as I

data MLPSpec = MLPSpec {
    inputFeatures :: Int,
    outputFeatures :: Int,
    hiddenFeatures0 :: Int,
    hiddenFeatures1 :: Int
    } deriving (Show, Eq)

data MLP = MLP { 
    l0 :: Linear,
    l1 :: Linear,
    l2 :: Linear
    } deriving (Generic, Show)

getLabels' :: Int -> I.MnistData -> [Int] -> Tensor
  
getLabels' n mnist imageIdxs =
  asTensor $ map (I.getLabel mnist) . take n $ imageIdxs

getImages ::
  Int
  -> Int
  -> I.MnistData
  -> [Int]
  -> IO Tensor
getImages n dataDim mnist imageIdxs = do
  let (BSI.PS fptr off len) = I.images mnist
  t <- (cast2 LibTorch.empty_lo :: [Int] -> D.TensorOptions -> IO D.Tensor)
         [n, dataDim]
         (D.withDType D.UInt8 D.defaultOpts)
  D.withTensor t $ \ptr1 -> do
    F.withForeignPtr fptr $ \ptr2 -> do
      forM_ (zip [0..(n-1)] imageIdxs) $ \(i,idx) -> do
        BSI.memcpy
          (F.plusPtr ptr1 (dataDim*i))
          (F.plusPtr ptr2 (off+16+dataDim*idx))
          dataDim
  return $ D.toType D.Float t

instance Parameterized MLP
instance Randomizable MLPSpec MLP where
    sample MLPSpec {..} = 
        MLP
            <$> sample (LinearSpec inputFeatures hiddenFeatures0)
            <*> sample (LinearSpec hiddenFeatures0 hiddenFeatures1)
            <*> sample (LinearSpec hiddenFeatures1 outputFeatures)

train = undefined

main :: IO ()
main = do
    (trainData, testData) <- I.initMnist
    let labels = getLabels' 10 trainData [0..100]
    print labels
    let spec = MLPSpec 768 10 512 256 
    init <- sample spec
    putStrLn "Done"
