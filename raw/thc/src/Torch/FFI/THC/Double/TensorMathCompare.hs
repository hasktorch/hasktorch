{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Double.TensorMathCompare
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
foreign import ccall "THCTensorMathCompare.h THDoubleTensor_ltValue"
  c_ltValue :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (())

-- | c_gtValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THDoubleTensor_gtValue"
  c_gtValue :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (())

-- | c_leValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THDoubleTensor_leValue"
  c_leValue :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (())

-- | c_geValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THDoubleTensor_geValue"
  c_geValue :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (())

-- | c_eqValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THDoubleTensor_eqValue"
  c_eqValue :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (())

-- | c_neValue :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THDoubleTensor_neValue"
  c_neValue :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (())

-- | c_ltValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THDoubleTensor_ltValueT"
  c_ltValueT :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (())

-- | c_gtValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THDoubleTensor_gtValueT"
  c_gtValueT :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (())

-- | c_leValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THDoubleTensor_leValueT"
  c_leValueT :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (())

-- | c_geValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THDoubleTensor_geValueT"
  c_geValueT :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (())

-- | c_eqValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THDoubleTensor_eqValueT"
  c_eqValueT :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (())

-- | c_neValueT :  state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h THDoubleTensor_neValueT"
  c_neValueT :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (())

-- | p_ltValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THDoubleTensor_ltValue"
  p_ltValue :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (()))

-- | p_gtValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THDoubleTensor_gtValue"
  p_gtValue :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (()))

-- | p_leValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THDoubleTensor_leValue"
  p_leValue :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (()))

-- | p_geValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THDoubleTensor_geValue"
  p_geValue :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (()))

-- | p_eqValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THDoubleTensor_eqValue"
  p_eqValue :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (()))

-- | p_neValue : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THDoubleTensor_neValue"
  p_neValue :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (()))

-- | p_ltValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THDoubleTensor_ltValueT"
  p_ltValueT :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (()))

-- | p_gtValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THDoubleTensor_gtValueT"
  p_gtValueT :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (()))

-- | p_leValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THDoubleTensor_leValueT"
  p_leValueT :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (()))

-- | p_geValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THDoubleTensor_geValueT"
  p_geValueT :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (()))

-- | p_eqValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THDoubleTensor_eqValueT"
  p_eqValueT :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (()))

-- | p_neValueT : Pointer to function : state self_ src value -> void
foreign import ccall "THCTensorMathCompare.h &THDoubleTensor_neValueT"
  p_neValueT :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (()))