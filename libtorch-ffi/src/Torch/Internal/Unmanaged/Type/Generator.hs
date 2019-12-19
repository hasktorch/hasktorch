
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Unmanaged.Type.Generator where


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
C.include "<ATen/CPUGenerator.h>"
C.include "<ATen/CUDAGenerator.h>"
C.include "<vector>"


newCUDAGenerator
  :: Word16
  -> IO (Ptr Generator)
newCUDAGenerator _device_index = getDefaultCUDAGenerator _device_index >>= generator_clone

newCPUGenerator
  :: Word64
  -> IO (Ptr Generator)
newCPUGenerator _seed_in =
  [C.throwBlock| at::Generator* { return new at::CPUGenerator(
    $(uint64_t _seed_in));
  }|]



deleteGenerator :: Ptr Generator -> IO ()
deleteGenerator object = [C.throwBlock| void { delete $(at::Generator* object);}|]

instance CppObject Generator where
  fromPtr ptr = newForeignPtr ptr (deleteGenerator ptr)



generator_set_current_seed
  :: Ptr Generator
  -> Word64
  -> IO (())
generator_set_current_seed _obj _seed =
  [C.throwBlock| void {  (*$(at::Generator* _obj)).set_current_seed(
    $(uint64_t _seed));
  }|]

generator_current_seed
  :: Ptr Generator
  -> IO (Word64)
generator_current_seed _obj =
  [C.throwBlock| uint64_t { return (*$(at::Generator* _obj)).current_seed(
    );
  }|]

generator_seed
  :: Ptr Generator
  -> IO (Word64)
generator_seed _obj =
  [C.throwBlock| uint64_t { return (*$(at::Generator* _obj)).seed(
    );
  }|]

generator_clone
  :: Ptr Generator
  -> IO (Ptr Generator)
generator_clone _obj =
  [C.throwBlock| at::Generator* { return (*$(at::Generator* _obj)).clone(
    ).get();
  }|]



getDefaultCUDAGenerator
  :: Word16
  -> IO (Ptr Generator)
getDefaultCUDAGenerator _device_index =
  [C.throwBlock| at::Generator* { return (at::detail::getCUDAHooks().getDefaultCUDAGenerator(
    $(uint16_t _device_index)));
  }|]

getDefaultCPUGenerator
  :: IO (Ptr Generator)
getDefaultCPUGenerator  =
  [C.throwBlock| at::Generator* { return (at::detail::getDefaultCPUGenerator(
    ));
  }|]
