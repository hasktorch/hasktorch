{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.Internal.Unmanaged.Type.IntArray where

import qualified Data.Map as Map
import Foreign hiding (newForeignPtr)
import Foreign.C.String
import Foreign.C.Types
import Foreign.Concurrent
import qualified Language.C.Inline.Context as C
import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Types as C
import Torch.Internal.Class
import Torch.Internal.Type

C.context $ C.cppCtx <> mempty {C.ctxTypesTable = typeTable}

C.include "<ATen/ATen.h>"

C.include "<vector>"

newIntArray ::
  IO (Ptr IntArray)
newIntArray =
  [C.throwBlock| std::vector<int64_t>* { return new std::vector<int64_t>(
    );
  }|]

deleteIntArray :: Ptr IntArray -> IO ()
deleteIntArray object = [C.throwBlock| void { delete $(std::vector<int64_t>* object);}|]

instance CppObject IntArray where
  fromPtr ptr = newForeignPtr ptr (deleteIntArray ptr)

intArray_empty ::
  Ptr IntArray ->
  IO (CBool)
intArray_empty _obj =
  [C.throwBlock| bool { return (*$(std::vector<int64_t>* _obj)).empty(
    );
  }|]

intArray_size ::
  Ptr IntArray ->
  IO (CSize)
intArray_size _obj =
  [C.throwBlock| size_t { return (*$(std::vector<int64_t>* _obj)).size(
    );
  }|]

intArray_at_s ::
  Ptr IntArray ->
  CSize ->
  IO (Int64)
intArray_at_s _obj _s =
  [C.throwBlock| int64_t { return (*$(std::vector<int64_t>* _obj)).at(
    $(size_t _s));
  }|]

intArray_push_back_l ::
  Ptr IntArray ->
  Int64 ->
  IO (())
intArray_push_back_l _obj _v =
  [C.throwBlock| void {  (*$(std::vector<int64_t>* _obj)).push_back(
    $(int64_t _v));
  }|]
