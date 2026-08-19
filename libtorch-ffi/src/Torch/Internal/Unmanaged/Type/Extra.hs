
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Unmanaged.Type.Extra where


import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Unsafe as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Inline.Unsafe as CUnsafe
import qualified Language.C.Types as C
import qualified Data.Map as Map
import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type


C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<ATen/Functions.h>"
C.include "<ATen/Tensor.h>"
C.include "<ATen/TensorOperators.h>"
C.include "<vector>"
C.include "<torch/csrc/autograd/generated/variable_factories.h>"

tensor_assign1_l
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> IO ()
tensor_assign1_l _obj _idx0 _val  =
  [C.throwBlock| void { (*$(at::Tensor* _obj))[$(int64_t _idx0)] = $(int64_t _val); }|]

tensor_assign2_l
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO ()
tensor_assign2_l _obj _idx0 _idx1 _val  =
  [C.throwBlock| void { (*$(at::Tensor* _obj))[$(int64_t _idx0)][$(int64_t _idx1)] = $(int64_t _val); }|]

tensor_assign1_d
  :: Ptr Tensor
  -> Int64
  -> CDouble
  -> IO ()
tensor_assign1_d _obj _idx0 _val  =
  [C.throwBlock| void { (*$(at::Tensor* _obj))[$(int64_t _idx0)] = $(double _val); }|]

tensor_assign2_d
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> CDouble
  -> IO ()
tensor_assign2_d _obj _idx0 _idx1 _val  =
  [C.throwBlock| void { (*$(at::Tensor* _obj))[$(int64_t _idx0)][$(int64_t _idx1)] = $(double _val); }|]


tensor_assign1_t
  :: Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> IO ()
tensor_assign1_t _obj _idx0 _val  =
  [C.throwBlock| void { (*$(at::Tensor* _obj))[$(int64_t _idx0)] = *$(at::Tensor* _val); }|]

tensor_assign2_t
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Ptr Tensor
  -> IO ()
tensor_assign2_t _obj _idx0 _idx1 _val  =
  [C.throwBlock| void { (*$(at::Tensor* _obj))[$(int64_t _idx0)][$(int64_t _idx1)] = *$(at::Tensor* _val); }|]


tensor_names
  :: Ptr Tensor
  -> IO (Ptr DimnameList)
tensor_names _obj =
  [C.throwBlock| std::vector<at::Dimname>* {
      auto ref = (*$(at::Tensor* _obj)).names();
      std::vector<at::Dimname>* vec = new std::vector<at::Dimname>();
      for(int i=0;i<ref.size();i++){
        vec->push_back(ref[i]);
      }
      return vec;
  }|]

tensor_to_device
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tensor_to_device reference input =
  [C.throwBlock| at::Tensor* {
      auto d = (*$(at::Tensor* reference)).device();
      return new at::Tensor((*$(at::Tensor* input)).to(d));
  }|]

new_empty_tensor
  :: [Int]
  -> Ptr TensorOptions
  -> IO (Ptr Tensor)
new_empty_tensor [x] _options = do
  let x' = fromIntegral x
  [C.throwBlock| at::Tensor* {
    return new at::Tensor(torch::empty({$(int x')}, *$(at::TensorOptions* _options)));
  }|]

new_empty_tensor [x,y] _options = do
  let x' = fromIntegral x
      y' = fromIntegral y
  [C.throwBlock| at::Tensor* {
    return new at::Tensor(torch::empty({$(int x'),$(int y')}, *$(at::TensorOptions* _options)));
  }|]

new_empty_tensor [x,y,z] _options = do
  let x' = fromIntegral x
      y' = fromIntegral y
      z' = fromIntegral z
  [C.throwBlock| at::Tensor* {
    return new at::Tensor(torch::empty({$(int x'),$(int y'),$(int z')}, *$(at::TensorOptions* _options)));
  }|]

new_empty_tensor _size _options = do
  let len = fromIntegral $ length _size
  shape <- [C.throwBlock| std::vector<int64_t>* {
    return new std::vector<int64_t>($(int len));
  }|]
  ptr <- [C.throwBlock| int64_t* {
    return $(std::vector<int64_t>* shape)->data();
  }|]
  pokeArray ptr (map fromIntegral _size)

  [C.throwBlock| at::Tensor* {
    auto v = new at::Tensor(torch::empty(*$(std::vector<int64_t>* shape), *$(at::TensorOptions* _options)));
    delete $(std::vector<int64_t>* shape);
    return v;
  }|]

tensor_dim_unsafe
  :: Ptr Tensor
  -> IO (Int64)
tensor_dim_unsafe _obj =
  [C.throwBlock| int64_t { return (*$(at::Tensor* _obj)).dim();
  }|]

tensor_dim_c_unsafe
  :: Ptr Tensor
  -> IO (Int64)
tensor_dim_c_unsafe _obj =
  [CUnsafe.block| int64_t { return (*$(at::Tensor* _obj)).dim();
  }|]
