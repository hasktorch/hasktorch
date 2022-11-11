
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Unmanaged.Type.TensorOptions where


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



C.include "<ATen/TensorOptions.h>"
C.include "<vector>"



newTensorOptions_s
  :: ScalarType
  -> IO (Ptr TensorOptions)
newTensorOptions_s _d =
  [C.throwBlock| at::TensorOptions* { return new at::TensorOptions(
    $(at::ScalarType _d));
  }|]

tensorOptions_device_D
  :: Ptr TensorOptions
  -> DeviceType
  -> IO (Ptr TensorOptions)
tensorOptions_device_D _obj _device =
  [C.throwBlock| at::TensorOptions* { return new at::TensorOptions((*$(at::TensorOptions* _obj)).device(
    $(at::DeviceType _device)));
  }|]

tensorOptions_device_index_s
  :: Ptr TensorOptions
  -> Int16
  -> IO (Ptr TensorOptions)
tensorOptions_device_index_s _obj _device_index =
  [C.throwBlock| at::TensorOptions* { return new at::TensorOptions((*$(at::TensorOptions* _obj)).device_index(
    $(int16_t _device_index)));
  }|]

tensorOptions_dtype_s
  :: Ptr TensorOptions
  -> ScalarType
  -> IO (Ptr TensorOptions)
tensorOptions_dtype_s _obj _dtype =
  [C.throwBlock| at::TensorOptions* { return new at::TensorOptions((*$(at::TensorOptions* _obj)).dtype(
    $(at::ScalarType _dtype)));
  }|]

tensorOptions_dtype
  :: Ptr TensorOptions
  -> IO (Ptr TensorOptions)
tensorOptions_dtype _obj =
  [C.throwBlock| at::TensorOptions* { return new at::TensorOptions((*$(at::TensorOptions* _obj)).dtype(
    ));
  }|]

tensorOptions_layout_L
  :: Ptr TensorOptions
  -> Layout
  -> IO (Ptr TensorOptions)
tensorOptions_layout_L _obj _layout =
  [C.throwBlock| at::TensorOptions* { return new at::TensorOptions((*$(at::TensorOptions* _obj)).layout(
    $(at::Layout _layout)));
  }|]

tensorOptions_requires_grad_b
  :: Ptr TensorOptions
  -> CBool
  -> IO (Ptr TensorOptions)
tensorOptions_requires_grad_b _obj _requires_grad =
  [C.throwBlock| at::TensorOptions* { return new at::TensorOptions((*$(at::TensorOptions* _obj)).requires_grad(
    $(bool _requires_grad)));
  }|]

tensorOptions_has_device
  :: Ptr TensorOptions
  -> IO (CBool)
tensorOptions_has_device _obj =
  [C.throwBlock| bool { return (*$(at::TensorOptions* _obj)).has_device(
    );
  }|]

tensorOptions_device_index
  :: Ptr TensorOptions
  -> IO (Int32)
tensorOptions_device_index _obj =
  [C.throwBlock| int32_t { return (*$(at::TensorOptions* _obj)).device_index(
    );
  }|]

tensorOptions_has_dtype
  :: Ptr TensorOptions
  -> IO (CBool)
tensorOptions_has_dtype _obj =
  [C.throwBlock| bool { return (*$(at::TensorOptions* _obj)).has_dtype(
    );
  }|]

tensorOptions_layout
  :: Ptr TensorOptions
  -> IO (Layout)
tensorOptions_layout _obj =
  [C.throwBlock| at::Layout { return (*$(at::TensorOptions* _obj)).layout(
    );
  }|]

tensorOptions_has_layout
  :: Ptr TensorOptions
  -> IO (CBool)
tensorOptions_has_layout _obj =
  [C.throwBlock| bool { return (*$(at::TensorOptions* _obj)).has_layout(
    );
  }|]

tensorOptions_requires_grad
  :: Ptr TensorOptions
  -> IO (CBool)
tensorOptions_requires_grad _obj =
  [C.throwBlock| bool { return (*$(at::TensorOptions* _obj)).requires_grad(
    );
  }|]

tensorOptions_has_requires_grad
  :: Ptr TensorOptions
  -> IO (CBool)
tensorOptions_has_requires_grad _obj =
  [C.throwBlock| bool { return (*$(at::TensorOptions* _obj)).has_requires_grad(
    );
  }|]

tensorOptions_backend
  :: Ptr TensorOptions
  -> IO (Backend)
tensorOptions_backend _obj =
  [C.throwBlock| at::Backend { return (*$(at::TensorOptions* _obj)).backend(
    );
  }|]

dtype_s
  :: ScalarType
  -> IO (Ptr TensorOptions)
dtype_s _dtype =
  [C.throwBlock| at::TensorOptions* { return new at::TensorOptions(at::dtype(
    $(at::ScalarType _dtype)));
  }|]

layout_L
  :: Layout
  -> IO (Ptr TensorOptions)
layout_L _layout =
  [C.throwBlock| at::TensorOptions* { return new at::TensorOptions(at::layout(
    $(at::Layout _layout)));
  }|]

device_D
  :: DeviceType
  -> IO (Ptr TensorOptions)
device_D _device =
  [C.throwBlock| at::TensorOptions* { return new at::TensorOptions(at::device(
    $(at::DeviceType _device)));
  }|]

device_index_s
  :: Int16
  -> IO (Ptr TensorOptions)
device_index_s _device_index =
  [C.throwBlock| at::TensorOptions* { return new at::TensorOptions(at::device_index(
    $(int16_t _device_index)));
  }|]

requires_grad_b
  :: CBool
  -> IO (Ptr TensorOptions)
requires_grad_b _requires_grad =
  [C.throwBlock| at::TensorOptions* { return new at::TensorOptions(at::requires_grad(
    $(bool _requires_grad)));
  }|]

