{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleInstances   #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE PolyKinds           #-}
{-# LANGUAGE QuasiQuotes         #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeFamilies        #-}

module Torch.Internal.Unmanaged.Type.IValueList where


import qualified Data.Map                         as Map
import           Foreign
import           Foreign.C.String
import           Foreign.C.Types
import qualified Language.C.Inline.Context        as C
import qualified Language.C.Inline.Cpp            as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Types                 as C
import           Torch.Internal.Type


C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<ATen/core/ivalue.h>"
C.include "<vector>"



newIValueList
  :: IO (Ptr IValueList)
newIValueList  =
  [C.throwBlock| std::vector<at::IValue>* { return new std::vector<at::IValue>(
    );
  }|]




ivalueList_empty
  :: Ptr IValueList
  -> IO (CBool)
ivalueList_empty _obj =
  [C.throwBlock| bool { return (*$(std::vector<at::IValue>* _obj)).empty(
    );
  }|]

ivalueList_size
  :: Ptr IValueList
  -> IO (CSize)
ivalueList_size _obj =
  [C.throwBlock| size_t { return (*$(std::vector<at::IValue>* _obj)).size(
    );
  }|]

ivalueList_at
  :: Ptr IValueList
  -> CSize
  -> IO (Ptr IValue)
ivalueList_at _obj _s =
  [C.throwBlock| at::IValue* { return new at::IValue((*$(std::vector<at::IValue>* _obj)).at(
    $(size_t _s)));
  }|]

ivalueList_push_back
  :: Ptr IValueList
  -> Ptr IValue
  -> IO (())
ivalueList_push_back _obj _v =
  [C.throwBlock| void {  (*$(std::vector<at::IValue>* _obj)).push_back(
    *$(at::IValue* _v));
  }|]



