
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}


module Torch.Internal.Unmanaged.Type.Module where


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

C.include "<torch/script.h>"
C.include "<vector>"

-- From libtorch/include/torch/csrc/jit/script/module.h

deleteModule :: Ptr Module -> IO ()
deleteModule object = [C.throwBlock| void { delete $(torch::jit::script::Module* object);}|]
instance CppObject Module where
  fromPtr ptr = newForeignPtr ptr (deleteModule ptr)


save :: Ptr Module -> FilePath -> IO ()
save obj file = withCString file $ \cfile -> [C.throwBlock| void {
    $(torch::jit::script::Module* obj)->save($(char* cfile));
  }|]

load :: FilePath -> IO (Ptr Module)
load file = withCString file $ \cfile -> [C.throwBlock| torch::jit::script::Module* {
    return new torch::jit::script::Module(torch::jit::load($(char* cfile)));
  }|]

forward :: Ptr Module -> (Ptr (StdVector IValue)) -> IO (Ptr IValue)
forward obj inputs = [C.throwBlock| at::IValue* {
    return new at::IValue($(torch::jit::script::Module* obj)->forward(*$(std::vector<at::IValue>* inputs)));
  }|]
