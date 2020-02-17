{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE LambdaCase #-}

module Main where

import qualified Torch.DType                   as D
import qualified Torch.Tensor                  as D
import qualified Torch.Functional               as D
import qualified Torch.TensorFactories         as D
import qualified Torch.TensorOptions           as D
import           Torch.Utils.Image

import           System.IO                     (FilePath)


-- [batch, channel, height, width] -> [batch, channel, height, width]
conv :: [[[[Float]]]] -> D.Tensor -> D.Tensor
conv weight input = D.toType D.UInt8 $ chw2hwc $ (\i -> D.clamp 0 255 (conv' i)) $ hwc2chw $ D.toType D.Float $ input
  where
    conv' input' = D.conv2d'
                   (D.asTensor weight)
                   (D.ones' [3])
                   (1,1)
                   (0,0)
                   input'

sharpness :: D.Tensor -> D.Tensor
sharpness input = conv weight input
  where
    weight = do
      o <- [0,1,2]
      return $ do
        i <- [0,1,2]
        if o == i then
          return [ [0  , -1 ,  0]
                 , [-1 ,  5 , -1]
                 , [0  , -1 ,  0]
                 ]
        else
          return [ [0  ,  0 ,  0]
                 , [0  ,  0 ,  0]
                 , [0  ,  0 ,  0]
                 ]

lowpass :: D.Tensor -> D.Tensor
lowpass input = conv weight input
  where
    weight = do
      o <- [0,1,2]
      return $ do
        i <- [0,1,2]
        if o == i then
          return [ [1/9  ,  1/9 ,  1/9]
                 , [1/9  ,  1/9 ,  1/9]
                 , [1/9  ,  1/9 ,  1/9]
                 ]
        else
          return [ [0  ,  0 ,  0]
                 , [0  ,  0 ,  0]
                 , [0  ,  0 ,  0]
                 ]

main = do
  readImage "input.bmp" >>= \case
    Right tensor -> do
      writeBitmap "output_sharpness.bmp" $ sharpness tensor
      writeBitmap "output_lowpass.bmp" $ lowpass tensor
      writePng "output_sharpness.png" $ sharpness tensor
      writePng "output_lowpass.png" $ lowpass tensor
    Left err -> print err
  return ()
