{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Control.Exception.Safe (catch, throwIO)
import Language.C.Inline.Cpp.Exceptions (CppException (..))
import System.Environment (getArgs)
import System.Posix.DynamicLinker
import qualified Torch.DType as D
import qualified Torch.Functional as D
import Torch.Script
import qualified Torch.Tensor as D
import qualified Torch.TensorFactories as D
import qualified Torch.TensorOptions as D
import Torch.Vision

prettyException :: IO a -> IO a
prettyException func =
  func `catch` \a@(CppStdException message) -> do
    putStrLn message
    throwIO (CppStdException "")

main :: IO ()
main = prettyException $ do
  let opt = \case
        [] -> ["resnet", "resnet_model.pt", "elephant.jpg"]
        a@[mode', model', input'] -> a
        _ -> error $ "Usage: load-torchscript (resnet or maskrcnn) model-file image-file"
  [mode, modelfile, inputfile] <- opt <$> getArgs
  _ <- dlopen "libtorchvision.so" [RTLD_GLOBAL, RTLD_LAZY]
  model <- load WithoutRequiredGrad modelfile
  mimg <- readImageAsRGB8 inputfile
  case (mimg, mode) of
    (Left err, _) -> print err
    (Right img', "resnet") -> do
      let img'' = D.divScalar (255.0 :: Float) $ D.toType D.Float $ hwc2chw img'
          img = IVTensor $ img''
          [[r, g, b]] = D.asValue img'' :: [[[[Float]]]]
      print $ take 10 (head r)
      let v = forward model [img]
      case v of
        IVTensor v' -> print $ D.argmax (D.Dim 1) D.RemoveDim v'
        _ -> print "Return value is not tensor."
    (Right img', "maskrcnn") -> do
      let img'' = D.divScalar (255.0 :: Float) $ D.toType D.Float $ hwc2chw img'
          [[r, g, b]] = D.asValue img'' :: [[[[Float]]]]
      print $ take 10 (head r)
      let img = IVTensorList [D.squeezeAll $ img'']
          v = forward model [img]
      print v
      case v of
        IVTensor v' -> print $ v'
        _ -> print "Return value is not tensor."
