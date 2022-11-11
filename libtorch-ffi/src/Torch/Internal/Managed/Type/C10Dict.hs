
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
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects

import qualified Torch.Internal.Unmanaged.Type.C10Dict as Unmanaged

import Control.Monad (forM)


newC10Dict :: ForeignPtr IValue -> ForeignPtr IValue -> IO (ForeignPtr (C10Dict '(IValue,IValue)))
newC10Dict = _cast2 Unmanaged.newC10Dict

c10Dict_empty :: ForeignPtr (C10Dict '(IValue,IValue)) -> IO (CBool)
c10Dict_empty = _cast1 Unmanaged.c10Dict_empty

c10Dict_size :: ForeignPtr (C10Dict '(IValue,IValue)) -> IO (CSize)
c10Dict_size = _cast1 Unmanaged.c10Dict_size

c10Dict_at :: ForeignPtr (C10Dict '(IValue,IValue)) -> ForeignPtr IValue -> IO (ForeignPtr IValue)
c10Dict_at = _cast2 Unmanaged.c10Dict_at

c10Dict_insert :: ForeignPtr (C10Dict '(IValue,IValue)) -> ForeignPtr IValue -> ForeignPtr IValue -> IO ()
c10Dict_insert = _cast3 Unmanaged.c10Dict_insert

c10Dict_toList :: ForeignPtr (C10Dict '(IValue,IValue)) -> IO [(ForeignPtr IValue,ForeignPtr IValue)]
c10Dict_toList obj = withForeignPtr obj $ \obj' -> do
  v <- Unmanaged.c10Dict_toList obj' :: IO [(Ptr IValue,Ptr IValue)]
  forM v $ \(a,b) -> do
    a' <- uncast a return 
    b' <- uncast b return
    return (a',b')
