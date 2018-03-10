{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Float.TensorMathCompare
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
foreign import ccall "THCTensorMathCompare.h THCFloatTensor_ltValue"
  c_ltValue :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaFloatTensor -> CFloat -> IO ()

-- | c_gtValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCFloatTensor_gtValue"
  c_gtValue :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaFloatTensor -> CFloat -> IO ()

-- | c_leValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCFloatTensor_leValue"
  c_leValue :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaFloatTensor -> CFloat -> IO ()

-- | c_geValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCFloatTensor_geValue"
  c_geValue :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaFloatTensor -> CFloat -> IO ()

-- | c_eqValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCFloatTensor_eqValue"
  c_eqValue :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaFloatTensor -> CFloat -> IO ()

-- | c_neValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCFloatTensor_neValue"
  c_neValue :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaFloatTensor -> CFloat -> IO ()

-- | c_ltValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCFloatTensor_ltValueT"
  c_ltValueT :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CFloat -> IO ()

-- | c_gtValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCFloatTensor_gtValueT"
  c_gtValueT :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CFloat -> IO ()

-- | c_leValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCFloatTensor_leValueT"
  c_leValueT :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CFloat -> IO ()

-- | c_geValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCFloatTensor_geValueT"
  c_geValueT :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CFloat -> IO ()

-- | c_eqValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCFloatTensor_eqValueT"
  c_eqValueT :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CFloat -> IO ()

-- | c_neValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THCFloatTensor_neValueT"
  c_neValueT :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CFloat -> IO ()

-- | p_ltValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCFloatTensor_ltValue"
  p_ltValue :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaFloatTensor -> CFloat -> IO ())

-- | p_gtValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCFloatTensor_gtValue"
  p_gtValue :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaFloatTensor -> CFloat -> IO ())

-- | p_leValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCFloatTensor_leValue"
  p_leValue :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaFloatTensor -> CFloat -> IO ())

-- | p_geValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCFloatTensor_geValue"
  p_geValue :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaFloatTensor -> CFloat -> IO ())

-- | p_eqValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCFloatTensor_eqValue"
  p_eqValue :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaFloatTensor -> CFloat -> IO ())

-- | p_neValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCFloatTensor_neValue"
  p_neValue :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaFloatTensor -> CFloat -> IO ())

-- | p_ltValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCFloatTensor_ltValueT"
  p_ltValueT :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CFloat -> IO ())

-- | p_gtValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCFloatTensor_gtValueT"
  p_gtValueT :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CFloat -> IO ())

-- | p_leValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCFloatTensor_leValueT"
  p_leValueT :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CFloat -> IO ())

-- | p_geValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCFloatTensor_geValueT"
  p_geValueT :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CFloat -> IO ())

-- | p_eqValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCFloatTensor_eqValueT"
  p_eqValueT :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CFloat -> IO ())

-- | p_neValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THCFloatTensor_neValueT"
  p_neValueT :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CFloat -> IO ())