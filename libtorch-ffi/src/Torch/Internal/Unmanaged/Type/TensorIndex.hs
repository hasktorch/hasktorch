
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Unmanaged.Type.TensorIndex where


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

C.include "<ATen/TensorIndexing.h>"
C.include "<vector>"

newTensorIndexList :: IO (Ptr (StdVector TensorIndex))
newTensorIndexList = [C.throwBlock| std::vector<at::indexing::TensorIndex>* { return new std::vector<at::indexing::TensorIndex>(); }|]

newTensorIndexWithInt :: CInt -> IO (Ptr TensorIndex)
newTensorIndexWithInt value = [C.throwBlock| at::indexing::TensorIndex* { return new at::indexing::TensorIndex($(int value)); }|]

newTensorIndexWithBool :: CBool -> IO (Ptr TensorIndex)
newTensorIndexWithBool value = [C.throwBlock| at::indexing::TensorIndex* { return new at::indexing::TensorIndex((bool)$(bool value)); }|]

newTensorIndexWithSlice :: CInt -> CInt -> CInt -> IO (Ptr TensorIndex)
newTensorIndexWithSlice start stop step = [C.throwBlock| at::indexing::TensorIndex* { return new at::indexing::TensorIndex(at::indexing::Slice($(int start),$(int stop),$(int step))); }|]

newTensorIndexWithTensor :: Ptr Tensor -> IO (Ptr TensorIndex)
newTensorIndexWithTensor value = [C.throwBlock| at::indexing::TensorIndex* { return new at::indexing::TensorIndex(*$(at::Tensor* value)); }|]

newTensorIndexWithEllipsis :: IO (Ptr TensorIndex)
newTensorIndexWithEllipsis = [C.throwBlock| at::indexing::TensorIndex* { return new at::indexing::TensorIndex("..."); }|]

newTensorIndexWithNone :: IO (Ptr TensorIndex)
newTensorIndexWithNone = [C.throwBlock| at::indexing::TensorIndex* { return new at::indexing::TensorIndex(at::indexing::None); }|]


tensorIndexList_empty :: Ptr (StdVector TensorIndex) -> IO (CBool)
tensorIndexList_empty _obj = [C.throwBlock| bool { return (*$(std::vector<at::indexing::TensorIndex>* _obj)).empty(); }|]

tensorIndexList_size :: Ptr (StdVector TensorIndex) -> IO (CSize)
tensorIndexList_size _obj = [C.throwBlock| size_t { return (*$(std::vector<at::indexing::TensorIndex>* _obj)).size(); }|]

tensorIndexList_push_back :: Ptr (StdVector TensorIndex) -> Ptr TensorIndex -> IO ()
tensorIndexList_push_back _obj _v = [C.throwBlock| void {  (*$(std::vector<at::indexing::TensorIndex>* _obj)).push_back(*$(at::indexing::TensorIndex* _v)); }|]

index :: Ptr Tensor -> Ptr (StdVector TensorIndex) -> IO (Ptr Tensor)
index _obj idx = [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index(*$(std::vector<at::indexing::TensorIndex>* idx))); } |]

index_put_ :: Ptr Tensor -> Ptr (StdVector TensorIndex) -> Ptr Tensor -> IO (Ptr Tensor)
index_put_ _obj idx value = [C.throwBlock| at::Tensor* { return new at::Tensor((*$(at::Tensor* _obj)).index_put_(*$(std::vector<at::indexing::TensorIndex>* idx),*$(at::Tensor * value))); } |]
