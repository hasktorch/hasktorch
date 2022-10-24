
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Managed.Type.C10List where


import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects

import qualified Torch.Internal.Unmanaged.Type.C10List as Unmanaged



newC10ListIValue :: ForeignPtr IValue -> IO (ForeignPtr (C10List IValue))
newC10ListIValue elem = _cast1 Unmanaged.newC10ListIValue elem

newC10ListTensor :: IO (ForeignPtr (C10List Tensor))
newC10ListTensor = _cast0 Unmanaged.newC10ListTensor

newC10ListOptionalTensor :: IO (ForeignPtr (C10List (C10Optional Tensor)))
newC10ListOptionalTensor = _cast0 Unmanaged.newC10ListOptionalTensor

newC10ListDouble :: IO (ForeignPtr (C10List CDouble))
newC10ListDouble = _cast0 Unmanaged.newC10ListDouble

newC10ListInt :: IO (ForeignPtr (C10List Int64))
newC10ListInt = _cast0 Unmanaged.newC10ListInt

newC10ListBool :: IO (ForeignPtr (C10List CBool))
newC10ListBool = _cast0 Unmanaged.newC10ListBool





c10ListIValue_empty :: ForeignPtr (C10List IValue) -> IO (CBool)
c10ListIValue_empty = _cast1 Unmanaged.c10ListIValue_empty

c10ListTensor_empty :: ForeignPtr (C10List Tensor) -> IO (CBool)
c10ListTensor_empty = _cast1 Unmanaged.c10ListTensor_empty

c10ListOptionalTensor_empty :: ForeignPtr (C10List (C10Optional Tensor)) -> IO (CBool)
c10ListOptionalTensor_empty = _cast1 Unmanaged.c10ListOptionalTensor_empty

c10ListDouble_empty :: ForeignPtr (C10List CDouble) -> IO (CBool)
c10ListDouble_empty = _cast1 Unmanaged.c10ListDouble_empty

c10ListInt_empty :: ForeignPtr (C10List Int64) -> IO (CBool)
c10ListInt_empty = _cast1 Unmanaged.c10ListInt_empty

c10ListBool_empty :: ForeignPtr (C10List CBool) -> IO (CBool)
c10ListBool_empty = _cast1 Unmanaged.c10ListBool_empty

c10ListIValue_size :: ForeignPtr (C10List IValue) -> IO (CSize)
c10ListIValue_size = _cast1 Unmanaged.c10ListIValue_size

c10ListTensor_size :: ForeignPtr (C10List Tensor) -> IO (CSize)
c10ListTensor_size = _cast1 Unmanaged.c10ListTensor_size

c10ListOptionalTensor_size :: ForeignPtr (C10List (C10Optional Tensor)) -> IO (CSize)
c10ListOptionalTensor_size = _cast1 Unmanaged.c10ListOptionalTensor_size

c10ListDouble_size :: ForeignPtr (C10List CDouble) -> IO (CSize)
c10ListDouble_size = _cast1 Unmanaged.c10ListDouble_size

c10ListInt_size :: ForeignPtr (C10List Int64) -> IO (CSize)
c10ListInt_size = _cast1 Unmanaged.c10ListInt_size

c10ListBool_size :: ForeignPtr (C10List CBool) -> IO (CSize)
c10ListBool_size = _cast1 Unmanaged.c10ListBool_size

c10ListIValue_at :: ForeignPtr (C10List IValue) -> CSize -> IO (ForeignPtr IValue)
c10ListIValue_at = _cast2 Unmanaged.c10ListIValue_at

c10ListTensor_at :: ForeignPtr (C10List Tensor) -> CSize -> IO (ForeignPtr Tensor)
c10ListTensor_at = _cast2 Unmanaged.c10ListTensor_at

c10ListOptionalTensor_at :: ForeignPtr (C10List (C10Optional Tensor)) -> CSize -> IO (ForeignPtr Tensor)
c10ListOptionalTensor_at = _cast2 Unmanaged.c10ListOptionalTensor_at

c10ListDouble_at :: ForeignPtr (C10List CDouble) -> CSize -> IO CDouble
c10ListDouble_at = _cast2 Unmanaged.c10ListDouble_at

c10ListInt_at :: ForeignPtr (C10List Int64) -> CSize -> IO Int64
c10ListInt_at = _cast2 Unmanaged.c10ListInt_at

c10ListBool_at :: ForeignPtr (C10List CBool) -> CSize -> IO CBool
c10ListBool_at = _cast2 Unmanaged.c10ListBool_at

c10ListIValue_push_back :: ForeignPtr (C10List IValue) -> ForeignPtr IValue -> IO ()
c10ListIValue_push_back = _cast2 Unmanaged.c10ListIValue_push_back

c10ListTensor_push_back :: ForeignPtr (C10List Tensor) -> ForeignPtr Tensor -> IO ()
c10ListTensor_push_back = _cast2 Unmanaged.c10ListTensor_push_back

c10ListOptionalTensor_push_back :: ForeignPtr (C10List (C10Optional Tensor)) -> ForeignPtr Tensor -> IO ()
c10ListOptionalTensor_push_back = _cast2 Unmanaged.c10ListOptionalTensor_push_back

c10ListDouble_push_back :: ForeignPtr (C10List CDouble) -> CDouble -> IO ()
c10ListDouble_push_back = _cast2 Unmanaged.c10ListDouble_push_back

c10ListInt_push_back :: ForeignPtr (C10List Int64) -> Int64 -> IO ()
c10ListInt_push_back = _cast2 Unmanaged.c10ListInt_push_back

c10ListBool_push_back :: ForeignPtr (C10List CBool) -> CBool -> IO ()
c10ListBool_push_back = _cast2 Unmanaged.c10ListBool_push_back
