
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module ATen.Unmanaged.Type.Dimname where


import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C
import qualified Data.Map as Map
import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import ATen.Type
import ATen.Class

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<ATen/ATen.h>"
C.include "<vector>"





deleteDimname :: Ptr Dimname -> IO ()
deleteDimname object = [C.throwBlock| void { delete $(at::Dimname* object);}|]

instance CppObject Dimname where
  fromPtr ptr = newForeignPtr ptr (deleteDimname ptr)



dimname_isBasic
  :: Ptr Dimname
  -> IO (CBool)
dimname_isBasic _obj =
  [C.throwBlock| bool { return (*$(at::Dimname* _obj)).isBasic(
    );
  }|]

dimname_isWildcard
  :: Ptr Dimname
  -> IO (CBool)
dimname_isWildcard _obj =
  [C.throwBlock| bool { return (*$(at::Dimname* _obj)).isWildcard(
    );
  }|]

dimname_matches_n
  :: Ptr Dimname
  -> Ptr Dimname
  -> IO (CBool)
dimname_matches_n _obj _other =
  [C.throwBlock| bool { return (*$(at::Dimname* _obj)).matches(
    *$(at::Dimname* _other));
  }|]



