
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
import qualified Language.C.Inline.Cpp.Unsafe as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C
import qualified Data.Map as Map
import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type


C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

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

string_c_str
  :: Ptr StdString
  -> IO String
string_c_str str = [C.throwBlock| const char* { return (*$(std::string* str)).c_str();}|] >>= peekCString
