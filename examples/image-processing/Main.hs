{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import System.IO (FilePath)
import qualified Torch as T
import Torch.Vision

-- [batch, channel, height, width] -> [batch, channel, height, width]
conv :: [[[[Float]]]] -> T.Tensor -> T.Tensor
conv weight input = T.toType T.UInt8 $ chw2hwc $ (\i -> T.clamp 0 255 (conv' i)) $ hwc2chw $ T.toType T.Float $ input
  where
    conv' input' =
      T.conv2d'
        (T.asTensor weight)
        (T.ones' [3])
        (1, 1)
        (0, 0)
        input'

sharpness :: T.Tensor -> T.Tensor
sharpness input = conv weight input
  where
    weight = do
      o <- [0, 1, 2]
      return $ do
        i <- [0, 1, 2]
        if o == i
          then
            return
              [ [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
              ]
          else
            return
              [ [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
              ]

lowpass :: T.Tensor -> T.Tensor
lowpass input = conv weight input
  where
    weight = do
      o <- [0, 1, 2]
      return $ do
        i <- [0, 1, 2]
        if o == i
          then
            return
              [ [1 / 9, 1 / 9, 1 / 9],
                [1 / 9, 1 / 9, 1 / 9],
                [1 / 9, 1 / 9, 1 / 9]
              ]
          else
            return
              [ [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
              ]

main = do
  readImageAsRGB8 "input.bmp" >>= \case
    Right tensor -> do
      writeBitmap "output_sharpness.bmp" $ sharpness tensor
      writeBitmap "output_lowpass.bmp" $ lowpass tensor
      writePng "output_sharpness.png" $ sharpness tensor
      writePng "output_lowpass.png" $ lowpass tensor
    Left err -> print err
  return ()
