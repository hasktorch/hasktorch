
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

register_parameter :: ForeignPtr Module -> ForeignPtr StdString -> ForeignPtr Tensor -> CBool -> IO ()
register_parameter = cast4 Unmanaged.register_parameter

register_module :: ForeignPtr Module -> ForeignPtr StdString -> ForeignPtr Module -> IO ()
register_module = cast3 Unmanaged.register_module

train :: ForeignPtr Module -> CBool -> IO ()
train = cast2 Unmanaged.train

run_method :: ForeignPtr Module -> ForeignPtr StdString -> ForeignPtr (C10List IValue) -> IO (Ptr IValue)
run_method = cast3 Unmanaged.run_method

run_method1 :: ForeignPtr Module -> ForeignPtr StdString -> ForeignPtr IValue -> IO (Ptr IValue)
run_method1 = cast3 Unmanaged.run_method1

define :: ForeignPtr Module -> ForeignPtr StdString -> IO ()
define = cast2 Unmanaged.define

-- TODO: Not using unsafeForeignPtrToPtr
trace :: (ForeignPtr TensorList -> IO (ForeignPtr TensorList)) -> ForeignPtr TensorList -> IO (ForeignPtr Module)
trace func inputs = cast1 (Unmanaged.trace (trans func)) inputs
  where
    trans :: (ForeignPtr TensorList -> IO (ForeignPtr TensorList)) -> Ptr TensorList -> IO (Ptr TensorList)
    trans func inputs = do
      inputs' <- fromPtr inputs
      ret <- func inputs'
      return $ unsafeForeignPtrToPtr ret
