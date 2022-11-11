
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Unmanaged.Type.StdVector where


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

C.include "<ATen/Scalar.h>"
C.include "<vector>"

newStdVectorScalar :: IO (Ptr (StdVector Scalar))
newStdVectorScalar = [C.throwBlock| std::vector<at::Scalar>* { return new std::vector<at::Scalar>(); }|]

newStdVectorDouble :: IO (Ptr (StdVector CDouble))
newStdVectorDouble = [C.throwBlock| std::vector<double>* { return new std::vector<double>(); }|]

newStdVectorInt :: IO (Ptr (StdVector Int64))
newStdVectorInt = [C.throwBlock| std::vector<int64_t>* { return new std::vector<int64_t>(); }|]

newStdVectorBool :: IO (Ptr (StdVector CBool))
newStdVectorBool = [C.throwBlock| std::vector<bool>* { return new std::vector<bool>(); }|]

stdVectorScalar_empty :: Ptr (StdVector Scalar) -> IO (CBool)
stdVectorScalar_empty _obj = [C.throwBlock| bool { return (*$(std::vector<at::Scalar>* _obj)).empty(); }|]

stdVectorDouble_empty :: Ptr (StdVector CDouble) -> IO (CBool)
stdVectorDouble_empty _obj = [C.throwBlock| bool { return (*$(std::vector<double>* _obj)).empty(); }|]

stdVectorInt_empty :: Ptr (StdVector Int64) -> IO (CBool)
stdVectorInt_empty _obj = [C.throwBlock| bool { return (*$(std::vector<int64_t>* _obj)).empty(); }|]

stdVectorBool_empty :: Ptr (StdVector CBool) -> IO (CBool)
stdVectorBool_empty _obj = [C.throwBlock| bool { return (*$(std::vector<bool>* _obj)).empty(); }|]

stdVectorDouble_size :: Ptr (StdVector CDouble) -> IO (CSize)
stdVectorDouble_size _obj = [C.throwBlock| size_t { return (*$(std::vector<double>* _obj)).size(); }|]

stdVectorInt_size :: Ptr (StdVector Int64) -> IO (CSize)
stdVectorInt_size _obj = [C.throwBlock| size_t { return (*$(std::vector<int64_t>* _obj)).size(); }|]

stdVectorBool_size :: Ptr (StdVector CBool) -> IO (CSize)
stdVectorBool_size _obj = [C.throwBlock| size_t { return (*$(std::vector<bool>* _obj)).size(); }|]

stdVectorScalar_at :: Ptr (StdVector Scalar) -> CSize -> IO (Ptr Scalar)
stdVectorScalar_at _obj _s = [C.throwBlock| at::Scalar* { return new at::Scalar((*$(std::vector<at::Scalar>* _obj))[$(size_t _s)]); }|]

stdVectorDouble_at :: Ptr (StdVector CDouble) -> CSize -> IO CDouble
stdVectorDouble_at _obj _s = [C.throwBlock| double { return (double)((*$(std::vector<double>* _obj))[$(size_t _s)]); }|]

stdVectorInt_at :: Ptr (StdVector Int64) -> CSize -> IO Int64
stdVectorInt_at _obj _s = [C.throwBlock| int64_t { return (int64_t)((*$(std::vector<int64_t>* _obj))[$(size_t _s)]); }|]

stdVectorBool_at :: Ptr (StdVector CBool) -> CSize -> IO CBool
stdVectorBool_at _obj _s = [C.throwBlock| bool { return ((*$(std::vector<bool>* _obj))[$(size_t _s)]); }|]

stdVectorScalar_push_back :: Ptr (StdVector Scalar) -> Ptr Scalar -> IO ()
stdVectorScalar_push_back _obj _v = [C.throwBlock| void {  (*$(std::vector<at::Scalar>* _obj)).push_back(*$(at::Scalar* _v)); }|]

stdVectorDouble_push_back :: Ptr (StdVector CDouble) -> CDouble -> IO ()
stdVectorDouble_push_back _obj _v = [C.throwBlock| void {  (*$(std::vector<double>* _obj)).push_back($(double _v)); }|]

stdVectorInt_push_back :: Ptr (StdVector Int64) -> Int64 -> IO ()
stdVectorInt_push_back _obj _v = [C.throwBlock| void {  (*$(std::vector<int64_t>* _obj)).push_back($(int64_t _v)); }|]

stdVectorBool_push_back :: Ptr (StdVector CBool) -> CBool -> IO ()
stdVectorBool_push_back _obj _v = [C.throwBlock| void {  (*$(std::vector<bool>* _obj)).push_back($(bool _v)); }|]



