{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE PolyKinds           #-}
{-# LANGUAGE QuasiQuotes         #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}

module Torch.Internal.Unmanaged.Serialize where

import           Foreign.C.String
import           Foreign.Ptr
import qualified Language.C.Inline.Context        as C
import qualified Language.C.Inline.Cpp            as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Types                 as C

import Torch.Internal.Type

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<vector>"
C.include "<torch/serialize.h>"
C.include "<ATen/Tensor.h>"

save :: Ptr TensorList -> FilePath -> IO ()
save inputs file = withCString file $ \cfile -> [C.throwBlock| void {
    torch::save(*$(std::vector<at::Tensor>* inputs),$(char* cfile));
  }|]

load :: FilePath -> IO (Ptr TensorList)
load file = withCString file $ \cfile -> [C.throwBlock| std::vector<at::Tensor>* {
    std::vector<at::Tensor> tensor_vec;
    torch::load(tensor_vec,$(char* cfile));
    return new std::vector<at::Tensor>(tensor_vec);
  }|]
