{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE LambdaCase #-}

module Main where

import System.Environment (getArgs)
import Torch.Script
import Torch.Utils.Image
import qualified Torch.DType                   as D
import qualified Torch.Tensor                  as D
import qualified Torch.TensorFactories         as D
import qualified Torch.TensorOptions           as D
import qualified Torch.Functional              as D

import Control.Exception.Safe (catch,throwIO)
import Language.C.Inline.Cpp.Exceptions (CppException(..))
import System.Posix.DynamicLinker

prettyException :: IO a -> IO a
prettyException func =
  func `catch` \a@(CppStdException message) -> do
    putStrLn message
    throwIO (CppStdException "")

main :: IO ()
main = prettyException $ do
  let opt = \case
        [] -> ["resnet","resnet_model.pt","elephant.jpg"]
        a@[mode',model',input'] -> a
        _ -> error $ "Usage: load-torchscript (resnet or maskrcnn) model-file image-file"
  [mode,modelfile,inputfile] <- opt <$> getArgs
  _ <- dlopen "libtorchvision.so" [RTLD_GLOBAL,RTLD_LAZY]
  model <- load modelfile
  mimg <- readImage inputfile
  case (mimg,mode) of
    (Left err,_) -> print err
    (Right img',"resnet")-> do
      let img = IVTensor $ D.toType D.Float $ hwc2chw img'
      v <- forward model [img]
      case v of
        IVTensor v' -> print $ D.argmax (D.Dim 1) D.RemoveDim v'
        _ -> print "Return value is not tensor."
    (Right img',"maskrcnn")-> do
      let img = IVTensorList [D.squeezeAll $ D.toType D.Float $ hwc2chw img']
      v <- forward model [img]
      print v
      case v of
        IVTensor v' -> print $ v'
        _ -> print "Return value is not tensor."
