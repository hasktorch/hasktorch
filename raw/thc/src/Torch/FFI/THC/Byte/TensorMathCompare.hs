{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Byte.TensorMathCompare where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_ltValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCudaByteTensor_ltValue"
  c_ltValue :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ()

-- | c_gtValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCudaByteTensor_gtValue"
  c_gtValue :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ()

-- | c_leValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCudaByteTensor_leValue"
  c_leValue :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ()

-- | c_geValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCudaByteTensor_geValue"
  c_geValue :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ()

-- | c_eqValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCudaByteTensor_eqValue"
  c_eqValue :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ()

-- | c_neValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCudaByteTensor_neValue"
  c_neValue :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ()

-- | c_ltValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCudaByteTensor_ltValueT"
  c_ltValueT :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ()

-- | c_gtValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCudaByteTensor_gtValueT"
  c_gtValueT :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ()

-- | c_leValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCudaByteTensor_leValueT"
  c_leValueT :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ()

-- | c_geValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCudaByteTensor_geValueT"
  c_geValueT :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ()

-- | c_eqValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCudaByteTensor_eqValueT"
  c_eqValueT :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ()

-- | c_neValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCudaByteTensor_neValueT"
  c_neValueT :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ()

-- | p_ltValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCudaByteTensor_ltValue"
  p_ltValue :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ())

-- | p_gtValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCudaByteTensor_gtValue"
  p_gtValue :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ())

-- | p_leValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCudaByteTensor_leValue"
  p_leValue :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ())

-- | p_geValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCudaByteTensor_geValue"
  p_geValue :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ())

-- | p_eqValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCudaByteTensor_eqValue"
  p_eqValue :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ())

-- | p_neValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCudaByteTensor_neValue"
  p_neValue :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ())

-- | p_ltValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCudaByteTensor_ltValueT"
  p_ltValueT :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ())

-- | p_gtValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCudaByteTensor_gtValueT"
  p_gtValueT :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ())

-- | p_leValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCudaByteTensor_leValueT"
  p_leValueT :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ())

-- | p_geValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCudaByteTensor_geValueT"
  p_geValueT :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ())

-- | p_eqValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCudaByteTensor_eqValueT"
  p_eqValueT :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ())

-- | p_neValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCudaByteTensor_neValueT"
  p_neValueT :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ())