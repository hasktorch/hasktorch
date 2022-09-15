{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Objects where

import Foreign.C.String
import Foreign.C.Types
import Foreign


import Torch.Internal.Type
import Torch.Internal.Class

foreign import ccall unsafe "hasktorch_finalizer.h &delete_c10dict"
  c_delete_c10dict  :: FunPtr( Ptr (C10Dict '(IValue,IValue)) -> IO () )

foreign import ccall unsafe "hasktorch_finalizer.h delete_c10dict"
  c_delete_c10dict'  :: Ptr (C10Dict '(IValue,IValue)) -> IO ()

instance CppObject (C10Dict '(IValue,IValue)) where
  fromPtr ptr = newForeignPtr c_delete_c10dict ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_c10dict' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_c10listivalue"
  c_delete_c10listivalue :: FunPtr ( Ptr (C10List IValue) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_c10listivalue"
  c_delete_c10listivalue' ::  Ptr (C10List IValue) -> IO ()

foreign import ccall unsafe "hasktorch_finalizer.h &delete_c10listtensor"
  c_delete_c10listtensor :: FunPtr ( Ptr (C10List Tensor) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_c10listtensor"
  c_delete_c10listtensor' ::  Ptr (C10List Tensor) -> IO ()

foreign import ccall unsafe "hasktorch_finalizer.h &delete_c10listoptionaltensor"
  c_delete_c10listoptionaltensor :: FunPtr ( Ptr (C10List (C10Optional Tensor)) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_c10listoptionaltensor"
  c_delete_c10listoptionaltensor' ::  Ptr (C10List (C10Optional Tensor)) -> IO ()

foreign import ccall unsafe "hasktorch_finalizer.h &delete_c10listdouble"
  c_delete_c10listdouble :: FunPtr ( Ptr (C10List CDouble) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_c10listdouble"
  c_delete_c10listdouble' ::  Ptr (C10List CDouble) -> IO ()

foreign import ccall unsafe "hasktorch_finalizer.h &delete_c10listint"
  c_delete_c10listint :: FunPtr ( Ptr (C10List Int64) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_c10listint"
  c_delete_c10listint' ::  Ptr (C10List Int64) -> IO ()

foreign import ccall unsafe "hasktorch_finalizer.h &delete_c10listbool"
  c_delete_c10listbool :: FunPtr ( Ptr (C10List CBool) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_c10listbool"
  c_delete_c10listbool' ::  Ptr (C10List CBool) -> IO ()

instance CppObject (C10List IValue) where
  fromPtr ptr = newForeignPtr c_delete_c10listivalue ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_c10listivalue' ptr
  {-# INLINE [1] deletePtr #-}

instance CppObject (C10List Tensor) where
  fromPtr ptr = newForeignPtr c_delete_c10listtensor ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_c10listtensor' ptr
  {-# INLINE [1] deletePtr #-}

instance CppObject (C10List (C10Optional Tensor)) where
  fromPtr ptr = newForeignPtr c_delete_c10listoptionaltensor ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_c10listoptionaltensor' ptr
  {-# INLINE [1] deletePtr #-}

instance CppObject (C10List CDouble) where
  fromPtr ptr = newForeignPtr c_delete_c10listdouble ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_c10listdouble' ptr
  {-# INLINE [1] deletePtr #-}

instance CppObject (C10List Int64) where
  fromPtr ptr = newForeignPtr c_delete_c10listint ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_c10listint' ptr
  {-# INLINE [1] deletePtr #-}

instance CppObject (C10List CBool) where
  fromPtr ptr = newForeignPtr c_delete_c10listbool ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_c10listbool' ptr
  {-# INLINE [1] deletePtr #-}



foreign import ccall unsafe "hasktorch_finalizer.h &delete_c10tuple"
  c_delete_c10tuple :: FunPtr ( Ptr (C10Ptr IVTuple) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_c10tuple"
  c_delete_c10tuple' ::  Ptr (C10Ptr IVTuple) -> IO ()

instance CppObject (C10Ptr IVTuple) where
  fromPtr ptr = newForeignPtr c_delete_c10tuple ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_c10tuple' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_context"
  c_delete_context :: FunPtr ( Ptr Context -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_context"
  c_delete_context' ::  Ptr Context -> IO ()

instance CppObject Context where
  fromPtr ptr = newForeignPtr c_delete_context ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_context' ptr
  {-# INLINE [1] deletePtr #-}


foreign import ccall unsafe "hasktorch_finalizer.h &delete_dimname"
  c_delete_dimname :: FunPtr ( Ptr Dimname -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_dimname"
  c_delete_dimname' ::  Ptr Dimname -> IO ()

instance CppObject Dimname where
  fromPtr ptr = newForeignPtr c_delete_dimname ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_dimname' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_dimnamelist"
  c_delete_dimnamelist :: FunPtr ( Ptr DimnameList -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_dimnamelist"
  c_delete_dimnamelist' ::  Ptr DimnameList -> IO ()

instance CppObject DimnameList where
  fromPtr ptr = newForeignPtr c_delete_dimnamelist ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_dimnamelist' ptr
  {-# INLINE [1] deletePtr #-}


foreign import ccall unsafe "hasktorch_finalizer.h &delete_generator"
  c_delete_generator :: FunPtr ( Ptr Generator -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_generator"
  c_delete_generator' ::  Ptr Generator -> IO ()

instance CppObject Generator where
  fromPtr ptr = newForeignPtr c_delete_generator ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_generator' ptr
  {-# INLINE [1] deletePtr #-}


foreign import ccall unsafe "hasktorch_finalizer.h &delete_ivalue"
  c_delete_ivalue :: FunPtr ( Ptr IValue -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_ivalue"
  c_delete_ivalue' ::  Ptr IValue -> IO ()

instance CppObject IValue where
  fromPtr ptr = newForeignPtr c_delete_ivalue ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_ivalue' ptr
  {-# INLINE [1] deletePtr #-}


foreign import ccall unsafe "hasktorch_finalizer.h &delete_ivaluelist"
  c_delete_ivaluelist :: FunPtr ( Ptr IValueList -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_ivaluelist"
  c_delete_ivaluelist' ::  Ptr IValueList -> IO ()

instance CppObject IValueList where
  fromPtr ptr = newForeignPtr c_delete_ivaluelist ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_ivaluelist' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_intarray"
  c_delete_intarray :: FunPtr ( Ptr IntArray -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_intarray"
  c_delete_intarray' ::  Ptr IntArray -> IO ()

instance CppObject IntArray where
  fromPtr ptr = newForeignPtr c_delete_intarray ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_intarray' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_module"
  c_delete_module :: FunPtr ( Ptr Module -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_module"
  c_delete_module' ::  Ptr Module -> IO ()

instance CppObject Module where
  fromPtr ptr = newForeignPtr c_delete_module ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_module' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_jitgraph"
  c_delete_jitgraph :: FunPtr ( Ptr (SharedPtr JitGraph) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_jitgraph"
  c_delete_jitgraph' ::  Ptr (SharedPtr JitGraph) -> IO ()

instance CppObject (SharedPtr JitGraph) where
  fromPtr ptr = newForeignPtr c_delete_jitgraph ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_jitgraph' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_jitnode"
  c_delete_jitnode :: FunPtr ( Ptr JitNode -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_jitnode"
  c_delete_jitnode' ::  Ptr JitNode -> IO ()

instance CppObject JitNode where
  fromPtr ptr = newForeignPtr c_delete_jitnode ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_jitnode' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_jitvalue"
  c_delete_jitvalue :: FunPtr ( Ptr JitValue -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_jitvalue"
  c_delete_jitvalue' ::  Ptr JitValue -> IO ()

instance CppObject JitValue where
  fromPtr ptr = newForeignPtr c_delete_jitvalue ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_jitvalue' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_scalar"
  c_delete_scalar :: FunPtr ( Ptr Scalar -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_scalar"
  c_delete_scalar' ::  Ptr Scalar -> IO ()

instance CppObject Scalar where
  fromPtr ptr = newForeignPtr c_delete_scalar ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_scalar' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_stdarraybool2"
  c_delete_stdarraybool2 :: FunPtr ( Ptr (StdArray '(CBool,2)) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_stdarraybool2"
  c_delete_stdarraybool2' ::  Ptr (StdArray '(CBool,2)) -> IO ()

foreign import ccall unsafe "hasktorch_finalizer.h &delete_stdarraybool3"
  c_delete_stdarraybool3 :: FunPtr ( Ptr (StdArray '(CBool,3)) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_stdarraybool3"
  c_delete_stdarraybool3' ::  Ptr (StdArray '(CBool,3)) -> IO ()

foreign import ccall unsafe "hasktorch_finalizer.h &delete_stdarraybool4"
  c_delete_stdarraybool4 :: FunPtr ( Ptr (StdArray '(CBool,4)) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_stdarraybool4"
  c_delete_stdarraybool4' ::  Ptr (StdArray '(CBool,4)) -> IO ()

instance CppObject (StdArray '(CBool,2)) where
  fromPtr ptr = newForeignPtr c_delete_stdarraybool2 ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_stdarraybool2' ptr
  {-# INLINE [1] deletePtr #-}

instance CppObject (StdArray '(CBool,3)) where
  fromPtr ptr = newForeignPtr c_delete_stdarraybool3 ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_stdarraybool3' ptr
  {-# INLINE [1] deletePtr #-}

instance CppObject (StdArray '(CBool,4)) where
  fromPtr ptr = newForeignPtr c_delete_stdarraybool4 ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_stdarraybool4' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_stdstring"
  c_delete_stdstring :: FunPtr ( Ptr StdString -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_stdstring"
  c_delete_stdstring' ::  Ptr StdString -> IO ()

instance CppObject StdString where
  fromPtr ptr = newForeignPtr c_delete_stdstring ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_stdstring' ptr
  {-# INLINE [1] deletePtr #-}



foreign import ccall unsafe "hasktorch_finalizer.h &delete_storage"
  c_delete_storage :: FunPtr ( Ptr Storage -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_storage"
  c_delete_storage' ::  Ptr Storage -> IO ()

instance CppObject Storage where
  fromPtr ptr = newForeignPtr c_delete_storage ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_storage' ptr
  {-# INLINE [1] deletePtr #-}



foreign import ccall unsafe "hasktorch_finalizer.h &delete_symbol"
  c_delete_symbol :: FunPtr ( Ptr Symbol -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_symbol"
  c_delete_symbol' ::  Ptr Symbol -> IO ()

instance CppObject Symbol where
  fromPtr ptr = newForeignPtr c_delete_symbol ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_symbol' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_tensor"
  c_delete_tensor  :: FunPtr( Ptr Tensor -> IO () )

foreign import ccall unsafe "hasktorch_finalizer.h delete_tensor"
  c_delete_tensor' :: Ptr Tensor -> IO () 

instance CppObject Tensor where
  fromPtr ptr = newForeignPtr c_delete_tensor ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_tensor' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_tensorindex"
  c_delete_tensorindex :: FunPtr ( Ptr TensorIndex -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_tensorindex"
  c_delete_tensorindex' ::  Ptr TensorIndex -> IO ()

foreign import ccall unsafe "hasktorch_finalizer.h &delete_tensorindexlist"
  c_delete_tensorindexlist :: FunPtr ( Ptr (StdVector TensorIndex) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_tensorindexlist"
  c_delete_tensorindexlist' ::  Ptr (StdVector TensorIndex) -> IO ()

instance CppObject TensorIndex where
  fromPtr ptr = newForeignPtr c_delete_tensorindex ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_tensorindex' ptr
  {-# INLINE [1] deletePtr #-}

instance CppObject (StdVector TensorIndex) where
  fromPtr ptr = newForeignPtr c_delete_tensorindexlist ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_tensorindexlist' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_tensorlist"
  c_delete_tensorlist :: FunPtr ( Ptr TensorList -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_tensorlist"
  c_delete_tensorlist' ::  Ptr TensorList -> IO ()

instance CppObject TensorList where
  fromPtr ptr = newForeignPtr c_delete_tensorlist ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_tensorlist' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_tensoroptions"
  c_delete_tensoroptions :: FunPtr ( Ptr TensorOptions -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_tensoroptions"
  c_delete_tensoroptions' ::  Ptr TensorOptions -> IO ()

instance CppObject TensorOptions where
  fromPtr ptr = newForeignPtr c_delete_tensoroptions ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_tensoroptions' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_tensortensor"
  c_delete_tensortensor :: FunPtr ( Ptr (StdTuple '(Tensor,Tensor)) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_tensortensor"
  c_delete_tensortensor' ::  Ptr (StdTuple '(Tensor,Tensor)) -> IO ()

instance CppObject (StdTuple '(Tensor,Tensor)) where
  fromPtr ptr = newForeignPtr c_delete_tensortensor ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_tensortensor' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_tensortensortensortensortensor"
  c_delete_tensortensortensortensortensor :: FunPtr ( Ptr (StdTuple '(Tensor,Tensor,Tensor,Tensor,Tensor)) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_tensortensortensortensortensor"
  c_delete_tensortensortensortensortensor' ::  Ptr (StdTuple '(Tensor,Tensor,Tensor,Tensor,Tensor)) -> IO ()

instance CppObject (StdTuple '(Tensor,Tensor,Tensor,Tensor,Tensor)) where
  fromPtr ptr = newForeignPtr c_delete_tensortensortensortensortensor ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_tensortensortensortensortensor' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_tensortensortensortensorlist"
  c_delete_tensortensortensortensorlist :: FunPtr ( Ptr (StdTuple '(Tensor,Tensor,Tensor,TensorList)) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_tensortensortensortensorlist"
  c_delete_tensortensortensortensorlist' ::  Ptr (StdTuple '(Tensor,Tensor,Tensor,TensorList)) -> IO ()

instance CppObject (StdTuple '(Tensor,Tensor,Tensor,TensorList)) where
  fromPtr ptr = newForeignPtr c_delete_tensortensortensortensorlist ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_tensortensortensortensorlist' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_tensortensortensortensorint64"
  c_delete_tensortensortensortensorint64 :: FunPtr ( Ptr (StdTuple '(Tensor,Tensor,Tensor,Tensor,Int64)) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_tensortensortensortensorint64"
  c_delete_tensortensortensortensorint64' ::  Ptr (StdTuple '(Tensor,Tensor,Tensor,Tensor,Int64)) -> IO ()

instance CppObject (StdTuple '(Tensor,Tensor,Tensor,Tensor,Int64)) where
  fromPtr ptr = newForeignPtr c_delete_tensortensortensortensorint64 ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_tensortensortensortensorint64' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_tensortensortensor"
  c_delete_tensortensortensor :: FunPtr ( Ptr (StdTuple '(Tensor,Tensor,Tensor)) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_tensortensortensor"
  c_delete_tensortensortensor' ::  Ptr (StdTuple '(Tensor,Tensor,Tensor)) -> IO ()

instance CppObject (StdTuple '(Tensor,Tensor,Tensor)) where
  fromPtr ptr = newForeignPtr c_delete_tensortensortensor ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_tensortensortensor' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_tensortensortensortensor"
  c_delete_tensortensortensortensor :: FunPtr ( Ptr (StdTuple '(Tensor,Tensor,Tensor,Tensor)) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_tensortensortensortensor"
  c_delete_tensortensortensortensor' ::  Ptr (StdTuple '(Tensor,Tensor,Tensor,Tensor)) -> IO ()

instance CppObject (StdTuple '(Tensor,Tensor,Tensor,Tensor)) where
  fromPtr ptr = newForeignPtr c_delete_tensortensortensortensor ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_tensortensortensortensor' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_cdoubleint64"
  c_delete_cdoubleint64 :: FunPtr ( Ptr (StdTuple '(CDouble,Int64)) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_cdoubleint64"
  c_delete_cdoubleint64' ::  Ptr (StdTuple '(CDouble,Int64)) -> IO ()

instance CppObject (StdTuple '(CDouble,Int64)) where
  fromPtr ptr = newForeignPtr c_delete_cdoubleint64 ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_cdoubleint64' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_cdoublecdouble"
  c_delete_cdoublecdouble :: FunPtr ( Ptr (StdTuple '(CDouble,CDouble)) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_cdoublecdouble"
  c_delete_cdoublecdouble' ::  Ptr (StdTuple '(CDouble,CDouble)) -> IO ()

instance CppObject (StdTuple '(CDouble,CDouble)) where
  fromPtr ptr = newForeignPtr c_delete_cdoublecdouble ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_cdoublecdouble' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_tensorgenerator"
  c_delete_tensorgenerator :: FunPtr ( Ptr (StdTuple '(Tensor,Generator)) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_tensorgenerator"
  c_delete_tensorgenerator' ::  Ptr (StdTuple '(Tensor,Generator)) -> IO ()

instance CppObject (StdTuple '(Tensor,Generator)) where
  fromPtr ptr = newForeignPtr c_delete_tensorgenerator ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_tensorgenerator' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_tensortensorcdoubleint64"
  c_delete_tensortensorcdoubleint64 :: FunPtr ( Ptr (StdTuple '(Tensor,Tensor,CDouble,Int64)) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_tensortensorcdoubleint64"
  c_delete_tensortensorcdoubleint64' ::  Ptr (StdTuple '(Tensor,Tensor,CDouble,Int64)) -> IO ()

instance CppObject (StdTuple '(Tensor,Tensor,CDouble,Int64)) where
  fromPtr ptr = newForeignPtr c_delete_tensortensorcdoubleint64 ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_tensortensorcdoubleint64' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_optimizer"
  c_delete_optimizer  :: FunPtr( Ptr Optimizer -> IO () )

foreign import ccall unsafe "hasktorch_finalizer.h delete_optimizer"
  c_delete_optimizer' :: Ptr Optimizer -> IO () 

instance CppObject Optimizer where
  fromPtr ptr = newForeignPtr c_delete_optimizer ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_optimizer' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_stdvectordouble"
  c_delete_stdvectordouble :: FunPtr ( Ptr (StdVector CDouble) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_stdvectordouble"
  c_delete_stdvectordouble' ::  Ptr (StdVector CDouble) -> IO ()

instance CppObject (StdVector CDouble) where
  fromPtr ptr = newForeignPtr c_delete_stdvectordouble ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_stdvectordouble' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_stdvectorint"
  c_delete_stdvectorint :: FunPtr ( Ptr (StdVector CInt) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_stdvectorint"
  c_delete_stdvectorint' ::  Ptr (StdVector CInt) -> IO ()

instance CppObject (StdVector CInt) where
  fromPtr ptr = newForeignPtr c_delete_stdvectorint ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_stdvectorint' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_stdvectorbool"
  c_delete_stdvectorbool :: FunPtr ( Ptr (StdVector CBool) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_stdvectorbool"
  c_delete_stdvectorbool' ::  Ptr (StdVector CBool) -> IO ()

instance CppObject (StdVector CBool) where
  fromPtr ptr = newForeignPtr c_delete_stdvectorbool ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_stdvectorbool' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_stream"
  c_delete_stream :: FunPtr ( Ptr Stream -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_stream"
  c_delete_stream' ::  Ptr Stream -> IO ()

instance CppObject Stream where
  fromPtr ptr = newForeignPtr c_delete_stream ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_stream' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_arrayrefscalar"
  c_delete_arrayrefscalar :: FunPtr ( Ptr (ArrayRef Scalar) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_arrayrefscalar"
  c_delete_arrayrefscalar' ::  Ptr (ArrayRef Scalar) -> IO ()

instance CppObject (ArrayRef Scalar) where
  fromPtr ptr = newForeignPtr c_delete_arrayrefscalar ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_arrayrefscalar' ptr
  {-# INLINE [1] deletePtr #-}

foreign import ccall unsafe "hasktorch_finalizer.h &delete_vectorscalar"
  c_delete_vectorscalar :: FunPtr ( Ptr (StdVector Scalar) -> IO ())

foreign import ccall unsafe "hasktorch_finalizer.h delete_vectorscalar"
  c_delete_vectorscalar' ::  Ptr (StdVector Scalar) -> IO ()

instance CppObject (StdVector Scalar) where
  fromPtr ptr = newForeignPtr c_delete_vectorscalar ptr
  {-# INLINE [1] fromPtr #-}
  deletePtr ptr = c_delete_vectorscalar' ptr
  {-# INLINE [1] deletePtr #-}
