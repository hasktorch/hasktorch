
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
import qualified Language.C.Inline.Cpp.Unsafe as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C
import qualified Data.Map as Map
import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type


C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<ATen/detail/CUDAHooksInterface.h>"
C.include "<ATen/detail/MPSHooksInterface.h>"
C.include "<ATen/CPUGeneratorImpl.h>"
C.include "<ATen/core/Generator.h>"

C.include "<vector>"


newCUDAGenerator _device_index = getDefaultCUDAGenerator _device_index >>= generator_clone

newMPSGenerator = getDefaultMPSGenerator >>= generator_clone

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

generator_get_device
  :: Ptr Generator
  -> IO Int64
generator_get_device _obj =
  [C.throwBlock| int64_t { return ((*$(at::Generator* _obj)).device().index());
  }|]

generator_is_cpu
  :: Ptr Generator
  -> IO CBool
generator_is_cpu _obj =
  [C.throwBlock| bool { return ((*$(at::Generator* _obj)).device().is_cpu());
  }|]

generator_is_cuda
  :: Ptr Generator
  -> IO CBool
generator_is_cuda _obj =
  [C.throwBlock| bool { return ((*$(at::Generator* _obj)).device().is_cuda());
  }|]

generator_is_hip
  :: Ptr Generator
  -> IO CBool
generator_is_hip _obj =
  [C.throwBlock| bool { return ((*$(at::Generator* _obj)).device().is_hip());
  }|]

generator_is_mps
  :: Ptr Generator
  -> IO CBool
generator_is_mps _obj =
  [C.throwBlock| bool { return ((*$(at::Generator* _obj)).device().is_mps());
  }|]

getDefaultCUDAGenerator
  :: Word16
  -> IO (Ptr Generator)
getDefaultCUDAGenerator _device_index =
  [C.throwBlock| at::Generator* { return new at::Generator(at::detail::getCUDAHooks().getDefaultCUDAGenerator(
    $(uint16_t _device_index)));
  }|]

getDefaultMPSGenerator
  :: IO (Ptr Generator)
getDefaultMPSGenerator =
  [C.throwBlock| at::Generator* { return new at::Generator(at::detail::getMPSHooks().getDefaultMPSGenerator());
  }|]

getDefaultCPUGenerator
  :: IO (Ptr Generator)
getDefaultCPUGenerator  =
  [C.throwBlock| at::Generator* { return new at::Generator(at::detail::getDefaultCPUGenerator(
    ));
  }|]
