
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Unmanaged.Type.StdString where


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
C.include "<string>"



newStdString
  :: IO (Ptr StdString)
newStdString  =
  [C.throwBlock| std::string* { return new std::string(
    );
  }|]

newStdString_s
  :: String
  -> IO (Ptr StdString)
newStdString_s str =
  withCString str $ \cstr -> [C.throwBlock| std::string* { return new std::string($(char* cstr));}|]

deleteStdString :: Ptr StdString -> IO ()
deleteStdString object = [C.throwBlock| void { delete $(std::string* object);}|]

instance CppObject StdString where
  fromPtr ptr = newForeignPtr ptr (deleteStdString ptr)

string_c_str
  :: Ptr StdString
  -> IO String
string_c_str str = [C.throwBlock| const char* { return (*$(std::string* str)).c_str();}|] >>= peekCString
