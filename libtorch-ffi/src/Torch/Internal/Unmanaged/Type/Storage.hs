{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleInstances   #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE PolyKinds           #-}
{-# LANGUAGE QuasiQuotes         #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeFamilies        #-}

module Torch.Internal.Unmanaged.Type.Storage where


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



C.include "<ATen/Storage.h>"
C.include "<vector>"



newStorage
  :: IO (Ptr Storage)
newStorage  =
  [C.throwBlock| at::Storage* { return new at::Storage(
    );
  }|]

