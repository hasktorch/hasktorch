{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module Torch.Internal.Unmanaged.Serialize where

import Foreign.Ptr
import Foreign.C.String
import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Unsafe as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C

import Torch.Internal.Type

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<vector>"
C.include "<fstream>"
C.include "<torch/serialize.h>"
C.include "<ATen/Tensor.h>"
C.include "<ATen/core/ivalue.h>"

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

pickleSave :: Ptr IValue -> FilePath -> IO ()
pickleSave inputs file = withCString file $ \cfile -> [C.throwBlock| void {
    auto output = torch::pickle_save(*$(at::IValue* inputs));
    auto fout = std::ofstream($(char* cfile), std::ios::out | std::ofstream::binary);
    std::copy(output.begin(), output.end(), std::ostreambuf_iterator<char>(fout));
  }|]

pickleLoad :: FilePath -> IO (Ptr IValue)
pickleLoad file = withCString file $ \cfile -> [C.throwBlock| at::IValue* {
    auto fin = std::ifstream($(char* cfile), std::ios::in | std::ifstream::binary);
    const std::vector<char> input = std::vector<char>(std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>());
    return new at::IValue(torch::pickle_load(input));
  }|]
