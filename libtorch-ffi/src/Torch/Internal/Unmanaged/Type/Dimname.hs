
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Unmanaged.Type.Dimname where


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

C.include "<ATen/core/Dimname.h>"
C.include "<vector>"



newDimname_n
  :: Ptr Dimname
  -> IO (Ptr Dimname)
newDimname_n _x =
  [C.throwBlock| at::Dimname* { return new at::Dimname(
    *$(at::Dimname* _x));
  }|]




dimname_symbol
  :: Ptr Dimname
  -> IO (Ptr Symbol)
dimname_symbol _obj =
  [C.throwBlock| at::Symbol* { return new at::Symbol((*$(at::Dimname* _obj)).symbol(
    ));
  }|]

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



fromSymbol_s
  :: Ptr Symbol
  -> IO (Ptr Dimname)
fromSymbol_s _name =
  [C.throwBlock| at::Dimname* { return new at::Dimname(at::Dimname::fromSymbol(
    *$(at::Symbol* _name)));
  }|]

wildcard
  :: IO (Ptr Dimname)
wildcard  =
  [C.throwBlock| at::Dimname* { return new at::Dimname(at::Dimname::wildcard(
    ));
  }|]

isValidName_s
  :: Ptr StdString
  -> IO (CBool)
isValidName_s _name =
  [C.throwBlock| bool { return (at::Dimname::isValidName(
    *$(std::string* _name)));
  }|]

