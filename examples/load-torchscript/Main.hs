{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Torch.Script
import Torch.Utils.Image
import qualified Torch.DType                   as D
import qualified Torch.Tensor                  as D
import qualified Torch.Functional              as D

import Control.Exception.Safe (catch,throwIO)
import Language.C.Inline.Cpp.Exceptions (CppException(..))

prettyException :: IO a -> IO a
prettyException func =
  func `catch` \a@(CppStdException message) -> do
    putStrLn message
    throwIO (CppStdException "")

main :: IO ()
main = prettyException $ do
  model <- load "resnet_model.pt"
  mimg <- readImage "elephant.jpg"
  case mimg of
    Left err -> print err
    Right img'-> do
      let img = IVTensor $ D.toType D.Float $ hwc2chw img'
      v <- forward model [img]
      case v of
        IVTensor v' -> print $ D.argmax (D.Dim 1) D.RemoveDim v'
        _ -> print "Return value is not tensor."
