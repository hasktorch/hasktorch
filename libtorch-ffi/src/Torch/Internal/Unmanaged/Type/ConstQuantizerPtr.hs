
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Unmanaged.Type.ConstQuantizerPtr where


import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C
import qualified Data.Map as Map
import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import Torch.Internal.Type
import Torch.Internal.Class

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<ATen/ATen.h>"
C.include "<vector>"




-- Quantizer is not available in pytorch-1.2.
deleteConstQuantizerPtr :: Ptr ConstQuantizerPtr -> IO ()
deleteConstQuantizerPtr object = return () --[C.throwBlock| void { delete $(c10::intrusive_ptr<Quantizer>* object);}|]

instance CppObject ConstQuantizerPtr where
  fromPtr ptr = newForeignPtr ptr (deleteConstQuantizerPtr ptr)





