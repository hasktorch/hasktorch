{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Half.TensorMathPointwise where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_sign :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaHalfTensor_sign"
  c_sign :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ()

-- | c_clamp :  state self src min_value max_value -> void
foreign import ccall "THCTensorMathPointwise.h THCudaHalfTensor_clamp"
  c_clamp :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> CTHHalf -> IO ()

-- | c_cross :  state self src1 src2 dimension -> void
foreign import ccall "THCTensorMathPointwise.h THCudaHalfTensor_cross"
  c_cross :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CInt -> IO ()

-- | c_cadd :  state self src1 value src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaHalfTensor_cadd"
  c_cadd :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> Ptr C'THCudaHalfTensor -> IO ()

-- | c_csub :  state self src1 value src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaHalfTensor_csub"
  c_csub :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> Ptr C'THCudaHalfTensor -> IO ()

-- | c_cmul :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaHalfTensor_cmul"
  c_cmul :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ()

-- | c_cpow :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaHalfTensor_cpow"
  c_cpow :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ()

-- | c_cdiv :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaHalfTensor_cdiv"
  c_cdiv :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ()

-- | c_clshift :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaHalfTensor_clshift"
  c_clshift :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ()

-- | c_crshift :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaHalfTensor_crshift"
  c_crshift :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ()

-- | c_cmax :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaHalfTensor_cmax"
  c_cmax :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ()

-- | c_cmin :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaHalfTensor_cmin"
  c_cmin :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ()

-- | c_cfmod :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaHalfTensor_cfmod"
  c_cfmod :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ()

-- | c_cremainder :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaHalfTensor_cremainder"
  c_cremainder :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ()

-- | c_cmaxValue :  state self src value -> void
foreign import ccall "THCTensorMathPointwise.h THCudaHalfTensor_cmaxValue"
  c_cmaxValue :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> IO ()

-- | c_cminValue :  state self src value -> void
foreign import ccall "THCTensorMathPointwise.h THCudaHalfTensor_cminValue"
  c_cminValue :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> IO ()

-- | c_cbitand :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaHalfTensor_cbitand"
  c_cbitand :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ()

-- | c_cbitor :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaHalfTensor_cbitor"
  c_cbitor :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ()

-- | c_cbitxor :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaHalfTensor_cbitxor"
  c_cbitxor :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ()

-- | c_addcmul :  state self t value src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaHalfTensor_addcmul"
  c_addcmul :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ()

-- | c_addcdiv :  state self t value src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaHalfTensor_addcdiv"
  c_addcdiv :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ()

-- | p_sign : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaHalfTensor_sign"
  p_sign :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ())

-- | p_clamp : Pointer to function : state self src min_value max_value -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaHalfTensor_clamp"
  p_clamp :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> CTHHalf -> IO ())

-- | p_cross : Pointer to function : state self src1 src2 dimension -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaHalfTensor_cross"
  p_cross :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CInt -> IO ())

-- | p_cadd : Pointer to function : state self src1 value src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaHalfTensor_cadd"
  p_cadd :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> Ptr C'THCudaHalfTensor -> IO ())

-- | p_csub : Pointer to function : state self src1 value src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaHalfTensor_csub"
  p_csub :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> Ptr C'THCudaHalfTensor -> IO ())

-- | p_cmul : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaHalfTensor_cmul"
  p_cmul :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ())

-- | p_cpow : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaHalfTensor_cpow"
  p_cpow :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ())

-- | p_cdiv : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaHalfTensor_cdiv"
  p_cdiv :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ())

-- | p_clshift : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaHalfTensor_clshift"
  p_clshift :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ())

-- | p_crshift : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaHalfTensor_crshift"
  p_crshift :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ())

-- | p_cmax : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaHalfTensor_cmax"
  p_cmax :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ())

-- | p_cmin : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaHalfTensor_cmin"
  p_cmin :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ())

-- | p_cfmod : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaHalfTensor_cfmod"
  p_cfmod :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ())

-- | p_cremainder : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaHalfTensor_cremainder"
  p_cremainder :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ())

-- | p_cmaxValue : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaHalfTensor_cmaxValue"
  p_cmaxValue :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> IO ())

-- | p_cminValue : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaHalfTensor_cminValue"
  p_cminValue :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> IO ())

-- | p_cbitand : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaHalfTensor_cbitand"
  p_cbitand :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ())

-- | p_cbitor : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaHalfTensor_cbitor"
  p_cbitor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ())

-- | p_cbitxor : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaHalfTensor_cbitxor"
  p_cbitxor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ())

-- | p_addcmul : Pointer to function : state self t value src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaHalfTensor_addcmul"
  p_addcmul :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ())

-- | p_addcdiv : Pointer to function : state self t value src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaHalfTensor_addcdiv"
  p_addcdiv :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO ())