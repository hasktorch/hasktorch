{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.Internal.Managed.Type.C10Dict where

import Control.Monad (forM)
import Foreign hiding (newForeignPtr)
import Foreign.C.String
import Foreign.C.Types
import Foreign.Concurrent
import Torch.Internal.Cast
import Torch.Internal.Class
import Torch.Internal.Type
import qualified Torch.Internal.Unmanaged.Type.C10Dict as Unmanaged
import Torch.Internal.Unmanaged.Type.Dimname
import Torch.Internal.Unmanaged.Type.DimnameList
import Torch.Internal.Unmanaged.Type.Generator
import Torch.Internal.Unmanaged.Type.IValue
import Torch.Internal.Unmanaged.Type.IntArray
import Torch.Internal.Unmanaged.Type.Scalar
import Torch.Internal.Unmanaged.Type.StdString
import Torch.Internal.Unmanaged.Type.Storage
import Torch.Internal.Unmanaged.Type.Tensor
import Torch.Internal.Unmanaged.Type.TensorList
import Torch.Internal.Unmanaged.Type.TensorOptions
import Torch.Internal.Unmanaged.Type.Tuple

newC10Dict :: ForeignPtr IValue -> ForeignPtr IValue -> IO (ForeignPtr (C10Dict '(IValue, IValue)))
newC10Dict = cast2 Unmanaged.newC10Dict

c10Dict_empty :: ForeignPtr (C10Dict '(IValue, IValue)) -> IO (CBool)
c10Dict_empty = cast1 Unmanaged.c10Dict_empty

c10Dict_size :: ForeignPtr (C10Dict '(IValue, IValue)) -> IO (CSize)
c10Dict_size = cast1 Unmanaged.c10Dict_size

c10Dict_at :: ForeignPtr (C10Dict '(IValue, IValue)) -> ForeignPtr IValue -> IO (ForeignPtr IValue)
c10Dict_at = cast2 Unmanaged.c10Dict_at

c10Dict_insert :: ForeignPtr (C10Dict '(IValue, IValue)) -> ForeignPtr IValue -> ForeignPtr IValue -> IO ()
c10Dict_insert = cast3 Unmanaged.c10Dict_insert

c10Dict_toList :: ForeignPtr (C10Dict '(IValue, IValue)) -> IO [(ForeignPtr IValue, ForeignPtr IValue)]
c10Dict_toList obj = withForeignPtr obj $ \obj' -> do
  v <- Unmanaged.c10Dict_toList obj' :: IO [(Ptr IValue, Ptr IValue)]
  forM v $ \(a, b) -> do
    a' <- uncast a return
    b' <- uncast b return
    return (a', b')
