{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Byte.TensorMathCompare
  ( c_ltValue
  , c_gtValue
  , c_leValue
  , c_geValue
  , c_eqValue
  , c_neValue
  , c_ltValueT
  , c_gtValueT
  , c_leValueT
  , c_geValueT
  , c_eqValueT
  , c_neValueT
  , p_ltValue
  , p_gtValue
  , p_leValue
  , p_geValue
  , p_eqValue
  , p_neValue
  , p_ltValueT
  , p_gtValueT
  , p_leValueT
  , p_geValueT
  , p_eqValueT
  , p_neValueT
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_ltValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCByteTensor_ltValue"
  c_ltValue :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaByteTensor -> CUChar -> IO ()

-- | c_gtValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCByteTensor_gtValue"
  c_gtValue :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaByteTensor -> CUChar -> IO ()

-- | c_leValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCByteTensor_leValue"
  c_leValue :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaByteTensor -> CUChar -> IO ()

-- | c_geValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCByteTensor_geValue"
  c_geValue :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaByteTensor -> CUChar -> IO ()

-- | c_eqValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCByteTensor_eqValue"
  c_eqValue :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaByteTensor -> CUChar -> IO ()

-- | c_neValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCByteTensor_neValue"
  c_neValue :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaByteTensor -> CUChar -> IO ()

-- | c_ltValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCByteTensor_ltValueT"
  c_ltValueT :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaByteTensor -> CUChar -> IO ()

-- | c_gtValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCByteTensor_gtValueT"
  c_gtValueT :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaByteTensor -> CUChar -> IO ()

-- | c_leValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCByteTensor_leValueT"
  c_leValueT :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaByteTensor -> CUChar -> IO ()

-- | c_geValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCByteTensor_geValueT"
  c_geValueT :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaByteTensor -> CUChar -> IO ()

-- | c_eqValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCByteTensor_eqValueT"
  c_eqValueT :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaByteTensor -> CUChar -> IO ()

-- | c_neValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCByteTensor_neValueT"
  c_neValueT :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaByteTensor -> CUChar -> IO ()

-- | p_ltValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCByteTensor_ltValue"
  p_ltValue :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaByteTensor -> CUChar -> IO ())

-- | p_gtValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCByteTensor_gtValue"
  p_gtValue :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaByteTensor -> CUChar -> IO ())

-- | p_leValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCByteTensor_leValue"
  p_leValue :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaByteTensor -> CUChar -> IO ())

-- | p_geValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCByteTensor_geValue"
  p_geValue :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaByteTensor -> CUChar -> IO ())

-- | p_eqValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCByteTensor_eqValue"
  p_eqValue :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaByteTensor -> CUChar -> IO ())

-- | p_neValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCByteTensor_neValue"
  p_neValue :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaByteTensor -> CUChar -> IO ())

-- | p_ltValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCByteTensor_ltValueT"
  p_ltValueT :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaByteTensor -> CUChar -> IO ())

-- | p_gtValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCByteTensor_gtValueT"
  p_gtValueT :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaByteTensor -> CUChar -> IO ())

-- | p_leValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCByteTensor_leValueT"
  p_leValueT :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaByteTensor -> CUChar -> IO ())

-- | p_geValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCByteTensor_geValueT"
  p_geValueT :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaByteTensor -> CUChar -> IO ())

-- | p_eqValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCByteTensor_eqValueT"
  p_eqValueT :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaByteTensor -> CUChar -> IO ())

-- | p_neValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCByteTensor_neValueT"
  p_neValueT :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaByteTensor -> CUChar -> IO ())