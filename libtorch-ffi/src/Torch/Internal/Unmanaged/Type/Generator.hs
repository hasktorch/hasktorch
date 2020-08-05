{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleInstances   #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE PolyKinds           #-}
{-# LANGUAGE QuasiQuotes         #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeFamilies        #-}

module Torch.Internal.Unmanaged.Type.Generator where


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

C.include "<ATen/detail/CUDAHooksInterface.h>"
C.include "<ATen/CPUGeneratorImpl.h>"
C.include "<ATen/core/Generator.h>"

C.include "<vector>"


newCUDAGenerator _device_index = getDefaultCUDAGenerator _device_index >>= generator_clone

newCPUGenerator
  :: Word64
  -> IO (Ptr Generator)
newCPUGenerator _seed_in =
  [C.throwBlock| at::Generator* { return new at::Generator(at::detail::createCPUGenerator(
    $(uint64_t _seed_in)));
  }|]



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
  [C.throwBlock| at::Generator* { return new at::Generator((*$(at::Generator* _obj)).clone(
    ));
  }|]



getDefaultCUDAGenerator
  :: Word16
  -> IO (Ptr Generator)
getDefaultCUDAGenerator _device_index =
  [C.throwBlock| at::Generator* { return new at::Generator(at::detail::getCUDAHooks().getDefaultCUDAGenerator(
    $(uint16_t _device_index)));
  }|]

getDefaultCPUGenerator
  :: IO (Ptr Generator)
getDefaultCPUGenerator  =
  [C.throwBlock| at::Generator* { return new at::Generator(at::detail::getDefaultCPUGenerator(
    ));
  }|]
