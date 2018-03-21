{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Int.TensorMathPointwise where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_cpow :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaIntTensor_cpow"
  c_cpow :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_neg :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaIntTensor_neg"
  c_neg :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_abs :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaIntTensor_abs"
  c_abs :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_sign :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaIntTensor_sign"
  c_sign :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_clamp :  state self src min_value max_value -> void
foreign import ccall "THCTensorMathPointwise.h THCudaIntTensor_clamp"
  c_clamp :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> CInt -> IO ()

-- | c_cross :  state self src1 src2 dimension -> void
foreign import ccall "THCTensorMathPointwise.h THCudaIntTensor_cross"
  c_cross :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> IO ()

-- | c_cadd :  state self src1 value src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaIntTensor_cadd"
  c_cadd :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> IO ()

-- | c_csub :  state self src1 value src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaIntTensor_csub"
  c_csub :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> IO ()

-- | c_cmul :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaIntTensor_cmul"
  c_cmul :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_cdiv :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaIntTensor_cdiv"
  c_cdiv :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_clshift :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaIntTensor_clshift"
  c_clshift :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_crshift :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaIntTensor_crshift"
  c_crshift :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_cmax :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaIntTensor_cmax"
  c_cmax :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_cmin :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaIntTensor_cmin"
  c_cmin :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_cfmod :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaIntTensor_cfmod"
  c_cfmod :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_cremainder :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaIntTensor_cremainder"
  c_cremainder :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_cmaxValue :  state self src value -> void
foreign import ccall "THCTensorMathPointwise.h THCudaIntTensor_cmaxValue"
  c_cmaxValue :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> IO ()

-- | c_cminValue :  state self src value -> void
foreign import ccall "THCTensorMathPointwise.h THCudaIntTensor_cminValue"
  c_cminValue :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> IO ()

-- | c_cbitand :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaIntTensor_cbitand"
  c_cbitand :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_cbitor :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaIntTensor_cbitor"
  c_cbitor :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_cbitxor :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaIntTensor_cbitxor"
  c_cbitxor :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_addcmul :  state self t value src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaIntTensor_addcmul"
  c_addcmul :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_addcdiv :  state self t value src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaIntTensor_addcdiv"
  c_addcdiv :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | p_cpow : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaIntTensor_cpow"
  p_cpow :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_neg : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaIntTensor_neg"
  p_neg :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_abs : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaIntTensor_abs"
  p_abs :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_sign : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaIntTensor_sign"
  p_sign :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_clamp : Pointer to function : state self src min_value max_value -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaIntTensor_clamp"
  p_clamp :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> CInt -> IO ())

-- | p_cross : Pointer to function : state self src1 src2 dimension -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaIntTensor_cross"
  p_cross :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> IO ())

-- | p_cadd : Pointer to function : state self src1 value src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaIntTensor_cadd"
  p_cadd :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> IO ())

-- | p_csub : Pointer to function : state self src1 value src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaIntTensor_csub"
  p_csub :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> IO ())

-- | p_cmul : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaIntTensor_cmul"
  p_cmul :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_cdiv : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaIntTensor_cdiv"
  p_cdiv :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_clshift : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaIntTensor_clshift"
  p_clshift :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_crshift : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaIntTensor_crshift"
  p_crshift :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_cmax : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaIntTensor_cmax"
  p_cmax :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_cmin : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaIntTensor_cmin"
  p_cmin :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_cfmod : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaIntTensor_cfmod"
  p_cfmod :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_cremainder : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaIntTensor_cremainder"
  p_cremainder :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_cmaxValue : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaIntTensor_cmaxValue"
  p_cmaxValue :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> IO ())

-- | p_cminValue : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaIntTensor_cminValue"
  p_cminValue :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> IO ())

-- | p_cbitand : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaIntTensor_cbitand"
  p_cbitand :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_cbitor : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaIntTensor_cbitor"
  p_cbitor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_cbitxor : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaIntTensor_cbitxor"
  p_cbitxor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_addcmul : Pointer to function : state self t value src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaIntTensor_addcmul"
  p_addcmul :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_addcdiv : Pointer to function : state self t value src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaIntTensor_addcdiv"
  p_addcdiv :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())