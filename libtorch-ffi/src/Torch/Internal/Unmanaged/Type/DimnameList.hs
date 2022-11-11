
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Unmanaged.Type.DimnameList where


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



newDimnameList
  :: IO (Ptr DimnameList)
newDimnameList  =
  [C.throwBlock| std::vector<at::Dimname>* { return new std::vector<at::Dimname>(
    );
  }|]





dimnameList_empty
  :: Ptr DimnameList
  -> IO (CBool)
dimnameList_empty _obj =
  [C.throwBlock| bool { return (*$(std::vector<at::Dimname>* _obj)).empty(
    );
  }|]

dimnameList_size
  :: Ptr DimnameList
  -> IO (CSize)
dimnameList_size _obj =
  [C.throwBlock| size_t { return (*$(std::vector<at::Dimname>* _obj)).size(
    );
  }|]

dimnameList_at_s
  :: Ptr DimnameList
  -> CSize
  -> IO (Ptr Dimname)
dimnameList_at_s _obj _s =
  [C.throwBlock| at::Dimname* { return new at::Dimname((*$(std::vector<at::Dimname>* _obj)).at(
    $(size_t _s)));
  }|]

dimnameList_push_back_n
  :: Ptr DimnameList
  -> Ptr Dimname
  -> IO (())
dimnameList_push_back_n _obj _v =
  [C.throwBlock| void {  (*$(std::vector<at::Dimname>* _obj)).push_back(
    *$(at::Dimname* _v));
  }|]



