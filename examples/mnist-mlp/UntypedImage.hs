{-# LANGUAGE ScopedTypeVariables #-}

module UntypedImage where

import Prelude hiding (min, max)
import qualified Prelude as P

import Control.Monad (forM_)

import           Torch.Internal.Cast
import qualified Data.ByteString as BS
import qualified Data.ByteString.Internal as BSI
import qualified Torch.DType as D
import qualified Torch.Tensor                  as D
import qualified Torch.TensorOptions           as D
import qualified Foreign.ForeignPtr            as F
import qualified Foreign.Ptr                   as F
import qualified Torch.Internal.Managed.TensorFactories as LibTorch

import qualified Image as I
import Torch.Functional hiding (take)
import Torch.Tensor
import Torch.NN

getLabels' :: Int -> I.MnistData -> [Int] -> Tensor
getLabels' n mnist imageIdxs =
  asTensor $ map (I.getLabel mnist) . take n $ imageIdxs

getImages' ::
  Int -- number of observations in minibatch
  -> Int -- dimensionality of the data
  -> I.MnistData -- mnist data representation
  -> [Int] -- indices of the dataset
  -> IO Tensor
getImages' n dataDim mnist imageIdxs = do
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

-- http://paulbourke.net/dataformats/asciiart/
grayScale10 = " .:-=+*#%@"
grayScale70 = reverse "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "

-- Display an MNIST image tensor as ascii text
dispImage :: Tensor -> IO ()
dispImage img = do
    mapM (\row ->
        mapM (\col -> 
            putChar $ grayScale !! (P.floor $ scaled !! row !! col)
            ) [0,downSamp..27] >>  putStrLn ""
        ) [0,downSamp..27]
    pure ()
    where 
        downSamp = 2
        grayScale = grayScale10
        paletteMax = (fromIntegral $ length grayScale) - 1.0
        img' = reshape [28, 28] img
        scaled :: [[Float]] = let (mn, mx) = (min img', max img')  
            in asValue $ (img' - mn) / (mx - mn) * paletteMax
