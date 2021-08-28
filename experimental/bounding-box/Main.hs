{-# LANGUAGE LambdaCase #-}

module Main where

import qualified Codec.Picture as I
import Control.Exception.Safe
import Control.Monad (forM_, when)
import Data.List.Split
import System.Environment (getArgs)
import qualified System.Exit
import Torch.DType
import Torch.Functional
import Torch.NN
import Torch.Tensor
import Torch.Vision

main = do
  args <- getArgs
  when (length args /= 4) $ do
    System.Exit.die "Usage: bounding-box label-file annotation-file input-image-file output-image-file"
  let [lfile, bbfile, ifile, ofile] = args

  labels <- lines <$> readFile lfile
  I.readImage ifile >>= \case
    Left err -> print err
    Right input_image' -> do
      let input_image = I.convertRGB8 input_image'
          width = I.imageWidth input_image
          height = I.imageHeight input_image
      str <- readFile bbfile
      let dats = map (splitOn " ") (lines str)
      print $ length dats
      print dats
      forM_ dats $ \(classid' : x' : y' : w' : h' : _) -> do
        let classid = read classid' :: Int
            x = fromIntegral width * read x' :: Float
            y = fromIntegral height * read y' :: Float
            w = fromIntegral width * read w' :: Float
            h = fromIntegral height * read h' :: Float
            x0 = x - w / 2
            y0 = y - h / 2
            x1 = x + w / 2
            y1 = y + h / 2
        drawString (labels !! classid) (round x0 + 1) (round y0 + 1) (255, 255, 255) (0, 0, 0) input_image
        drawRect (round x0) (round y0) (round x1) (round y1) (255, 255, 255) input_image
      I.writePng ofile input_image
