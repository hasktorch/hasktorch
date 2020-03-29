
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE UndecidableInstances #-}



module Torch.Internal.Managed.Type.Module where


import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import Foreign.ForeignPtr.Unsafe
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Unmanaged.Type.Generator
import Torch.Internal.Unmanaged.Type.IntArray
import Torch.Internal.Unmanaged.Type.Scalar
import Torch.Internal.Unmanaged.Type.Storage
import Torch.Internal.Unmanaged.Type.Tensor
import Torch.Internal.Unmanaged.Type.TensorList
import Torch.Internal.Unmanaged.Type.TensorOptions
import Torch.Internal.Unmanaged.Type.Tuple
import Torch.Internal.Unmanaged.Type.StdString
import Torch.Internal.Unmanaged.Type.Dimname
import Torch.Internal.Unmanaged.Type.DimnameList
import Torch.Internal.Unmanaged.Type.IValue
import Torch.Internal.Unmanaged.Type.IValueList
import Torch.Internal.Unmanaged.Type.Module
import Torch.Internal.Unmanaged.Type.C10List

import qualified Torch.Internal.Unmanaged.Type.Module as Unmanaged

newModule :: ForeignPtr StdString -> IO (ForeignPtr Module)
newModule = cast1 Unmanaged.newModule

save :: ForeignPtr Module -> FilePath -> IO ()
save = cast2 Unmanaged.save

load :: FilePath -> IO (ForeignPtr Module)
load = cast1 Unmanaged.load

forward :: ForeignPtr Module -> (ForeignPtr (StdVector IValue)) -> IO (ForeignPtr IValue)
forward = cast2 Unmanaged.forward

registerParameter :: ForeignPtr Module -> ForeignPtr StdString -> ForeignPtr Tensor -> CBool -> IO ()
registerParameter = cast4 Unmanaged.registerParameter

registerModule :: ForeignPtr Module -> ForeignPtr StdString -> ForeignPtr Module -> IO ()
registerModule = cast3 Unmanaged.registerModule

train :: ForeignPtr Module -> CBool -> IO ()
train = cast2 Unmanaged.train

runMethod :: ForeignPtr Module -> ForeignPtr StdString -> ForeignPtr (C10List IValue) -> IO (Ptr IValue)
runMethod = cast3 Unmanaged.runMethod

runMethod1 :: ForeignPtr Module -> ForeignPtr StdString -> ForeignPtr IValue -> IO (Ptr IValue)
runMethod1 = cast3 Unmanaged.runMethod1

getParameters :: ForeignPtr Module -> IO (ForeignPtr TensorList)
getParameters = cast1 Unmanaged.getParameters

setParameters :: ForeignPtr Module -> ForeignPtr TensorList -> IO ()
setParameters = cast2 Unmanaged.setParameters

toDevice :: ForeignPtr Module -> DeviceType -> Int16 -> IO ()
toDevice = cast3 Unmanaged.toDevice

clone :: ForeignPtr Module -> IO (ForeignPtr Module)
clone = cast1 Unmanaged.clone

define :: ForeignPtr Module -> ForeignPtr StdString -> IO ()
define = cast2 Unmanaged.define

-- TODO: Not using unsafeForeignPtrToPtr
trace :: String -> String -> (ForeignPtr TensorList -> IO (ForeignPtr TensorList)) -> ForeignPtr TensorList -> IO (ForeignPtr Module)
trace moduleName functionName func inputs = cast3 (\m f inps -> Unmanaged.trace m f (trans func) inps) moduleName functionName inputs
  where
    trans :: (ForeignPtr TensorList -> IO (ForeignPtr TensorList)) -> Ptr TensorList -> IO (Ptr TensorList)
    trans func inputs = do
      inputs' <- fromPtr inputs
      ret <- func inputs'
      return $ unsafeForeignPtrToPtr ret
