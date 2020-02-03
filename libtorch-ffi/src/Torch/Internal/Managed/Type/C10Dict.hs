
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Managed.Type.C10Dict where


import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
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

import qualified Torch.Internal.Unmanaged.Type.C10Dict as Unmanaged



-- newC10Dict :: IO (ForeignPtr (C10Dict '(IValue,IValue)))
-- newC10Dict = cast0 Unmanaged.newC10Dict

c10Dict_empty :: ForeignPtr (C10Dict '(IValue,IValue)) -> IO (CBool)
c10Dict_empty = cast1 Unmanaged.c10Dict_empty

c10Dict_size :: ForeignPtr (C10Dict '(IValue,IValue)) -> IO (CSize)
c10Dict_size = cast1 Unmanaged.c10Dict_size

c10Dict_at :: ForeignPtr (C10Dict '(IValue,IValue)) -> ForeignPtr IValue -> IO (ForeignPtr IValue)
c10Dict_at = cast2 Unmanaged.c10Dict_at

c10Dict_insert :: ForeignPtr (C10Dict '(IValue,IValue)) -> ForeignPtr IValue -> ForeignPtr IValue -> IO ()
c10Dict_insert = cast3 Unmanaged.c10Dict_insert
