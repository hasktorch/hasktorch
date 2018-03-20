{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Byte.TensorMathPointwise where

import Foreign
import Foreign.C.Types
import Torch.Types.THC
import Data.Word
import Data.Int

-- | c_cpow :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCByteTensor_cpow"
  c_cpow :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ()

-- | c_sign :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCByteTensor_sign"
  c_sign :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ()

-- | c_clamp :  state self src min_value max_value -> void
foreign import ccall "THCTensorMathPointwise.h THCByteTensor_clamp"
  c_clamp :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> CUChar -> IO ()

-- | c_cross :  state self src1 src2 dimension -> void
foreign import ccall "THCTensorMathPointwise.h THCByteTensor_cross"
  c_cross :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CInt -> IO ()

-- | c_cadd :  state self src1 value src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCByteTensor_cadd"
  c_cadd :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> Ptr C'THCudaByteTensor -> IO ()

-- | c_csub :  state self src1 value src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCByteTensor_csub"
  c_csub :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> Ptr C'THCudaByteTensor -> IO ()

-- | c_cmul :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCByteTensor_cmul"
  c_cmul :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ()

-- | c_cdiv :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCByteTensor_cdiv"
  c_cdiv :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ()

-- | c_clshift :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCByteTensor_clshift"
  c_clshift :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ()

-- | c_crshift :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCByteTensor_crshift"
  c_crshift :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ()

-- | c_cmax :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCByteTensor_cmax"
  c_cmax :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ()

-- | c_cmin :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCByteTensor_cmin"
  c_cmin :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ()

-- | c_cfmod :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCByteTensor_cfmod"
  c_cfmod :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ()

-- | c_cremainder :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCByteTensor_cremainder"
  c_cremainder :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ()

-- | c_cmaxValue :  state self src value -> void
foreign import ccall "THCTensorMathPointwise.h THCByteTensor_cmaxValue"
  c_cmaxValue :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ()

-- | c_cminValue :  state self src value -> void
foreign import ccall "THCTensorMathPointwise.h THCByteTensor_cminValue"
  c_cminValue :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ()

-- | c_cbitand :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCByteTensor_cbitand"
  c_cbitand :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ()

-- | c_cbitor :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCByteTensor_cbitor"
  c_cbitor :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ()

-- | c_cbitxor :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCByteTensor_cbitxor"
  c_cbitxor :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ()

-- | c_addcmul :  state self t value src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCByteTensor_addcmul"
  c_addcmul :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ()

-- | c_addcdiv :  state self t value src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCByteTensor_addcdiv"
  c_addcdiv :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ()

-- | p_cpow : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCByteTensor_cpow"
  p_cpow :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ())

-- | p_sign : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCByteTensor_sign"
  p_sign :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ())

-- | p_clamp : Pointer to function : state self src min_value max_value -> void
foreign import ccall "THCTensorMathPointwise.h &THCByteTensor_clamp"
  p_clamp :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> CUChar -> IO ())

-- | p_cross : Pointer to function : state self src1 src2 dimension -> void
foreign import ccall "THCTensorMathPointwise.h &THCByteTensor_cross"
  p_cross :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CInt -> IO ())

-- | p_cadd : Pointer to function : state self src1 value src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCByteTensor_cadd"
  p_cadd :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> Ptr C'THCudaByteTensor -> IO ())

-- | p_csub : Pointer to function : state self src1 value src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCByteTensor_csub"
  p_csub :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> Ptr C'THCudaByteTensor -> IO ())

-- | p_cmul : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCByteTensor_cmul"
  p_cmul :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ())

-- | p_cdiv : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCByteTensor_cdiv"
  p_cdiv :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ())

-- | p_clshift : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCByteTensor_clshift"
  p_clshift :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ())

-- | p_crshift : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCByteTensor_crshift"
  p_crshift :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ())

-- | p_cmax : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCByteTensor_cmax"
  p_cmax :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ())

-- | p_cmin : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCByteTensor_cmin"
  p_cmin :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ())

-- | p_cfmod : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCByteTensor_cfmod"
  p_cfmod :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ())

-- | p_cremainder : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCByteTensor_cremainder"
  p_cremainder :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ())

-- | p_cmaxValue : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPointwise.h &THCByteTensor_cmaxValue"
  p_cmaxValue :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ())

-- | p_cminValue : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPointwise.h &THCByteTensor_cminValue"
  p_cminValue :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> IO ())

-- | p_cbitand : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCByteTensor_cbitand"
  p_cbitand :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ())

-- | p_cbitor : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCByteTensor_cbitor"
  p_cbitor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ())

-- | p_cbitxor : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCByteTensor_cbitxor"
  p_cbitxor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ())

-- | p_addcmul : Pointer to function : state self t value src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCByteTensor_addcmul"
  p_addcmul :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ())

-- | p_addcdiv : Pointer to function : state self t value src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCByteTensor_addcdiv"
  p_addcdiv :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> CUChar -> Ptr C'THCudaByteTensor -> Ptr C'THCudaByteTensor -> IO ())