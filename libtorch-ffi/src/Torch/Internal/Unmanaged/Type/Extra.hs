
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
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Inline.Context as C
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
