{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Byte.TensorMath where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_fill :  r_ value -> void
foreign import ccall "THTensorMath.h THByteTensor_fill"
  c_fill_ :: Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_fill_ with unused argument (for CTHState) to unify backpack signatures.
c_fill :: Ptr C'THState -> Ptr C'THByteTensor -> CUChar -> IO ()
c_fill = const c_fill_

-- | c_zero :  r_ -> void
foreign import ccall "THTensorMath.h THByteTensor_zero"
  c_zero_ :: Ptr C'THByteTensor -> IO ()

-- | alias of c_zero_ with unused argument (for CTHState) to unify backpack signatures.
c_zero :: Ptr C'THState -> Ptr C'THByteTensor -> IO ()
c_zero = const c_zero_

-- | c_maskedFill :  tensor mask value -> void
foreign import ccall "THTensorMath.h THByteTensor_maskedFill"
  c_maskedFill_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_maskedFill_ with unused argument (for CTHState) to unify backpack signatures.
c_maskedFill :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_maskedFill = const c_maskedFill_

-- | c_maskedCopy :  tensor mask src -> void
foreign import ccall "THTensorMath.h THByteTensor_maskedCopy"
  c_maskedCopy_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_maskedCopy_ with unused argument (for CTHState) to unify backpack signatures.
c_maskedCopy :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_maskedCopy = const c_maskedCopy_

-- | c_maskedSelect :  tensor src mask -> void
foreign import ccall "THTensorMath.h THByteTensor_maskedSelect"
  c_maskedSelect_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_maskedSelect_ with unused argument (for CTHState) to unify backpack signatures.
c_maskedSelect :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_maskedSelect = const c_maskedSelect_

-- | c_nonzero :  subscript tensor -> void
foreign import ccall "THTensorMath.h THByteTensor_nonzero"
  c_nonzero_ :: Ptr C'THLongTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_nonzero_ with unused argument (for CTHState) to unify backpack signatures.
c_nonzero :: Ptr C'THState -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> IO ()
c_nonzero = const c_nonzero_

-- | c_indexSelect :  tensor src dim index -> void
foreign import ccall "THTensorMath.h THByteTensor_indexSelect"
  c_indexSelect_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> Ptr C'THLongTensor -> IO ()

-- | alias of c_indexSelect_ with unused argument (for CTHState) to unify backpack signatures.
c_indexSelect :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> Ptr C'THLongTensor -> IO ()
c_indexSelect = const c_indexSelect_

-- | c_indexCopy :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THByteTensor_indexCopy"
  c_indexCopy_ :: Ptr C'THByteTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_indexCopy_ with unused argument (for CTHState) to unify backpack signatures.
c_indexCopy :: Ptr C'THState -> Ptr C'THByteTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> IO ()
c_indexCopy = const c_indexCopy_

-- | c_indexAdd :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THByteTensor_indexAdd"
  c_indexAdd_ :: Ptr C'THByteTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_indexAdd_ with unused argument (for CTHState) to unify backpack signatures.
c_indexAdd :: Ptr C'THState -> Ptr C'THByteTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> IO ()
c_indexAdd = const c_indexAdd_

-- | c_indexFill :  tensor dim index val -> void
foreign import ccall "THTensorMath.h THByteTensor_indexFill"
  c_indexFill_ :: Ptr C'THByteTensor -> CInt -> Ptr C'THLongTensor -> CUChar -> IO ()

-- | alias of c_indexFill_ with unused argument (for CTHState) to unify backpack signatures.
c_indexFill :: Ptr C'THState -> Ptr C'THByteTensor -> CInt -> Ptr C'THLongTensor -> CUChar -> IO ()
c_indexFill = const c_indexFill_

-- | c_take :  tensor src index -> void
foreign import ccall "THTensorMath.h THByteTensor_take"
  c_take_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_take_ with unused argument (for CTHState) to unify backpack signatures.
c_take :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THLongTensor -> IO ()
c_take = const c_take_

-- | c_put :  tensor index src accumulate -> void
foreign import ccall "THTensorMath.h THByteTensor_put"
  c_put_ :: Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> CInt -> IO ()

-- | alias of c_put_ with unused argument (for CTHState) to unify backpack signatures.
c_put :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> CInt -> IO ()
c_put = const c_put_

-- | c_gather :  tensor src dim index -> void
foreign import ccall "THTensorMath.h THByteTensor_gather"
  c_gather_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> Ptr C'THLongTensor -> IO ()

-- | alias of c_gather_ with unused argument (for CTHState) to unify backpack signatures.
c_gather :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> Ptr C'THLongTensor -> IO ()
c_gather = const c_gather_

-- | c_scatter :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THByteTensor_scatter"
  c_scatter_ :: Ptr C'THByteTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_scatter_ with unused argument (for CTHState) to unify backpack signatures.
c_scatter :: Ptr C'THState -> Ptr C'THByteTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> IO ()
c_scatter = const c_scatter_

-- | c_scatterAdd :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THByteTensor_scatterAdd"
  c_scatterAdd_ :: Ptr C'THByteTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_scatterAdd_ with unused argument (for CTHState) to unify backpack signatures.
c_scatterAdd :: Ptr C'THState -> Ptr C'THByteTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> IO ()
c_scatterAdd = const c_scatterAdd_

-- | c_scatterFill :  tensor dim index val -> void
foreign import ccall "THTensorMath.h THByteTensor_scatterFill"
  c_scatterFill_ :: Ptr C'THByteTensor -> CInt -> Ptr C'THLongTensor -> CUChar -> IO ()

-- | alias of c_scatterFill_ with unused argument (for CTHState) to unify backpack signatures.
c_scatterFill :: Ptr C'THState -> Ptr C'THByteTensor -> CInt -> Ptr C'THLongTensor -> CUChar -> IO ()
c_scatterFill = const c_scatterFill_

-- | c_dot :  t src -> accreal
foreign import ccall "THTensorMath.h THByteTensor_dot"
  c_dot_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO CLong

-- | alias of c_dot_ with unused argument (for CTHState) to unify backpack signatures.
c_dot :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO CLong
c_dot = const c_dot_

-- | c_minall :  t -> real
foreign import ccall "THTensorMath.h THByteTensor_minall"
  c_minall_ :: Ptr C'THByteTensor -> IO CUChar

-- | alias of c_minall_ with unused argument (for CTHState) to unify backpack signatures.
c_minall :: Ptr C'THState -> Ptr C'THByteTensor -> IO CUChar
c_minall = const c_minall_

-- | c_maxall :  t -> real
foreign import ccall "THTensorMath.h THByteTensor_maxall"
  c_maxall_ :: Ptr C'THByteTensor -> IO CUChar

-- | alias of c_maxall_ with unused argument (for CTHState) to unify backpack signatures.
c_maxall :: Ptr C'THState -> Ptr C'THByteTensor -> IO CUChar
c_maxall = const c_maxall_

-- | c_medianall :  t -> real
foreign import ccall "THTensorMath.h THByteTensor_medianall"
  c_medianall_ :: Ptr C'THByteTensor -> IO CUChar

-- | alias of c_medianall_ with unused argument (for CTHState) to unify backpack signatures.
c_medianall :: Ptr C'THState -> Ptr C'THByteTensor -> IO CUChar
c_medianall = const c_medianall_

-- | c_sumall :  t -> accreal
foreign import ccall "THTensorMath.h THByteTensor_sumall"
  c_sumall_ :: Ptr C'THByteTensor -> IO CLong

-- | alias of c_sumall_ with unused argument (for CTHState) to unify backpack signatures.
c_sumall :: Ptr C'THState -> Ptr C'THByteTensor -> IO CLong
c_sumall = const c_sumall_

-- | c_prodall :  t -> accreal
foreign import ccall "THTensorMath.h THByteTensor_prodall"
  c_prodall_ :: Ptr C'THByteTensor -> IO CLong

-- | alias of c_prodall_ with unused argument (for CTHState) to unify backpack signatures.
c_prodall :: Ptr C'THState -> Ptr C'THByteTensor -> IO CLong
c_prodall = const c_prodall_

-- | c_add :  r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_add"
  c_add_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_add_ with unused argument (for CTHState) to unify backpack signatures.
c_add :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_add = const c_add_

-- | c_sub :  r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_sub"
  c_sub_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_sub_ with unused argument (for CTHState) to unify backpack signatures.
c_sub :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_sub = const c_sub_

-- | c_add_scaled :  r_ t value alpha -> void
foreign import ccall "THTensorMath.h THByteTensor_add_scaled"
  c_add_scaled_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> CUChar -> IO ()

-- | alias of c_add_scaled_ with unused argument (for CTHState) to unify backpack signatures.
c_add_scaled :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> CUChar -> IO ()
c_add_scaled = const c_add_scaled_

-- | c_sub_scaled :  r_ t value alpha -> void
foreign import ccall "THTensorMath.h THByteTensor_sub_scaled"
  c_sub_scaled_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> CUChar -> IO ()

-- | alias of c_sub_scaled_ with unused argument (for CTHState) to unify backpack signatures.
c_sub_scaled :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> CUChar -> IO ()
c_sub_scaled = const c_sub_scaled_

-- | c_mul :  r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_mul"
  c_mul_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_mul_ with unused argument (for CTHState) to unify backpack signatures.
c_mul :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_mul = const c_mul_

-- | c_div :  r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_div"
  c_div_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_div_ with unused argument (for CTHState) to unify backpack signatures.
c_div :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_div = const c_div_

-- | c_lshift :  r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_lshift"
  c_lshift_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_lshift_ with unused argument (for CTHState) to unify backpack signatures.
c_lshift :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_lshift = const c_lshift_

-- | c_rshift :  r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_rshift"
  c_rshift_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_rshift_ with unused argument (for CTHState) to unify backpack signatures.
c_rshift :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_rshift = const c_rshift_

-- | c_fmod :  r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_fmod"
  c_fmod_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_fmod_ with unused argument (for CTHState) to unify backpack signatures.
c_fmod :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_fmod = const c_fmod_

-- | c_remainder :  r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_remainder"
  c_remainder_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_remainder_ with unused argument (for CTHState) to unify backpack signatures.
c_remainder :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_remainder = const c_remainder_

-- | c_clamp :  r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h THByteTensor_clamp"
  c_clamp_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> CUChar -> IO ()

-- | alias of c_clamp_ with unused argument (for CTHState) to unify backpack signatures.
c_clamp :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> CUChar -> IO ()
c_clamp = const c_clamp_

-- | c_bitand :  r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_bitand"
  c_bitand_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_bitand_ with unused argument (for CTHState) to unify backpack signatures.
c_bitand :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_bitand = const c_bitand_

-- | c_bitor :  r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_bitor"
  c_bitor_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_bitor_ with unused argument (for CTHState) to unify backpack signatures.
c_bitor :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_bitor = const c_bitor_

-- | c_bitxor :  r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_bitxor"
  c_bitxor_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_bitxor_ with unused argument (for CTHState) to unify backpack signatures.
c_bitxor :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_bitxor = const c_bitxor_

-- | c_cadd :  r_ t value src -> void
foreign import ccall "THTensorMath.h THByteTensor_cadd"
  c_cadd_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> IO ()

-- | alias of c_cadd_ with unused argument (for CTHState) to unify backpack signatures.
c_cadd :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> IO ()
c_cadd = const c_cadd_

-- | c_csub :  self src1 value src2 -> void
foreign import ccall "THTensorMath.h THByteTensor_csub"
  c_csub_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> IO ()

-- | alias of c_csub_ with unused argument (for CTHState) to unify backpack signatures.
c_csub :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> IO ()
c_csub = const c_csub_

-- | c_cmul :  r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensor_cmul"
  c_cmul_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_cmul_ with unused argument (for CTHState) to unify backpack signatures.
c_cmul :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_cmul = const c_cmul_

-- | c_cpow :  r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensor_cpow"
  c_cpow_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_cpow_ with unused argument (for CTHState) to unify backpack signatures.
c_cpow :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_cpow = const c_cpow_

-- | c_cdiv :  r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensor_cdiv"
  c_cdiv_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_cdiv_ with unused argument (for CTHState) to unify backpack signatures.
c_cdiv :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_cdiv = const c_cdiv_

-- | c_clshift :  r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensor_clshift"
  c_clshift_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_clshift_ with unused argument (for CTHState) to unify backpack signatures.
c_clshift :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_clshift = const c_clshift_

-- | c_crshift :  r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensor_crshift"
  c_crshift_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_crshift_ with unused argument (for CTHState) to unify backpack signatures.
c_crshift :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_crshift = const c_crshift_

-- | c_cfmod :  r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensor_cfmod"
  c_cfmod_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_cfmod_ with unused argument (for CTHState) to unify backpack signatures.
c_cfmod :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_cfmod = const c_cfmod_

-- | c_cremainder :  r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensor_cremainder"
  c_cremainder_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_cremainder_ with unused argument (for CTHState) to unify backpack signatures.
c_cremainder :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_cremainder = const c_cremainder_

-- | c_cbitand :  r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensor_cbitand"
  c_cbitand_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_cbitand_ with unused argument (for CTHState) to unify backpack signatures.
c_cbitand :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_cbitand = const c_cbitand_

-- | c_cbitor :  r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensor_cbitor"
  c_cbitor_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_cbitor_ with unused argument (for CTHState) to unify backpack signatures.
c_cbitor :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_cbitor = const c_cbitor_

-- | c_cbitxor :  r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensor_cbitxor"
  c_cbitxor_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_cbitxor_ with unused argument (for CTHState) to unify backpack signatures.
c_cbitxor :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_cbitxor = const c_cbitxor_

-- | c_addcmul :  r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THByteTensor_addcmul"
  c_addcmul_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_addcmul_ with unused argument (for CTHState) to unify backpack signatures.
c_addcmul :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_addcmul = const c_addcmul_

-- | c_addcdiv :  r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THByteTensor_addcdiv"
  c_addcdiv_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_addcdiv_ with unused argument (for CTHState) to unify backpack signatures.
c_addcdiv :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_addcdiv = const c_addcdiv_

-- | c_addmv :  r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h THByteTensor_addmv"
  c_addmv_ :: Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_addmv_ with unused argument (for CTHState) to unify backpack signatures.
c_addmv :: Ptr C'THState -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_addmv = const c_addmv_

-- | c_addmm :  r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h THByteTensor_addmm"
  c_addmm_ :: Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_addmm_ with unused argument (for CTHState) to unify backpack signatures.
c_addmm :: Ptr C'THState -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_addmm = const c_addmm_

-- | c_addr :  r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h THByteTensor_addr"
  c_addr_ :: Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_addr_ with unused argument (for CTHState) to unify backpack signatures.
c_addr :: Ptr C'THState -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_addr = const c_addr_

-- | c_addbmm :  r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THByteTensor_addbmm"
  c_addbmm_ :: Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_addbmm_ with unused argument (for CTHState) to unify backpack signatures.
c_addbmm :: Ptr C'THState -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_addbmm = const c_addbmm_

-- | c_baddbmm :  r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THByteTensor_baddbmm"
  c_baddbmm_ :: Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_baddbmm_ with unused argument (for CTHState) to unify backpack signatures.
c_baddbmm :: Ptr C'THState -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_baddbmm = const c_baddbmm_

-- | c_match :  r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h THByteTensor_match"
  c_match_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_match_ with unused argument (for CTHState) to unify backpack signatures.
c_match :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_match = const c_match_

-- | c_numel :  t -> ptrdiff_t
foreign import ccall "THTensorMath.h THByteTensor_numel"
  c_numel_ :: Ptr C'THByteTensor -> IO CPtrdiff

-- | alias of c_numel_ with unused argument (for CTHState) to unify backpack signatures.
c_numel :: Ptr C'THState -> Ptr C'THByteTensor -> IO CPtrdiff
c_numel = const c_numel_

-- | c_preserveReduceDimSemantics :  r_ in_dims reduce_dimension keepdim -> void
foreign import ccall "THTensorMath.h THByteTensor_preserveReduceDimSemantics"
  c_preserveReduceDimSemantics_ :: Ptr C'THByteTensor -> CInt -> CInt -> CInt -> IO ()

-- | alias of c_preserveReduceDimSemantics_ with unused argument (for CTHState) to unify backpack signatures.
c_preserveReduceDimSemantics :: Ptr C'THState -> Ptr C'THByteTensor -> CInt -> CInt -> CInt -> IO ()
c_preserveReduceDimSemantics = const c_preserveReduceDimSemantics_

-- | c_max :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THByteTensor_max"
  c_max_ :: Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> CInt -> CInt -> IO ()

-- | alias of c_max_ with unused argument (for CTHState) to unify backpack signatures.
c_max :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> CInt -> CInt -> IO ()
c_max = const c_max_

-- | c_min :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THByteTensor_min"
  c_min_ :: Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> CInt -> CInt -> IO ()

-- | alias of c_min_ with unused argument (for CTHState) to unify backpack signatures.
c_min :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> CInt -> CInt -> IO ()
c_min = const c_min_

-- | c_kthvalue :  values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h THByteTensor_kthvalue"
  c_kthvalue_ :: Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> CLLong -> CInt -> CInt -> IO ()

-- | alias of c_kthvalue_ with unused argument (for CTHState) to unify backpack signatures.
c_kthvalue :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> CLLong -> CInt -> CInt -> IO ()
c_kthvalue = const c_kthvalue_

-- | c_mode :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THByteTensor_mode"
  c_mode_ :: Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> CInt -> CInt -> IO ()

-- | alias of c_mode_ with unused argument (for CTHState) to unify backpack signatures.
c_mode :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> CInt -> CInt -> IO ()
c_mode = const c_mode_

-- | c_median :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THByteTensor_median"
  c_median_ :: Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> CInt -> CInt -> IO ()

-- | alias of c_median_ with unused argument (for CTHState) to unify backpack signatures.
c_median :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> CInt -> CInt -> IO ()
c_median = const c_median_

-- | c_sum :  r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THByteTensor_sum"
  c_sum_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> CInt -> IO ()

-- | alias of c_sum_ with unused argument (for CTHState) to unify backpack signatures.
c_sum :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> CInt -> IO ()
c_sum = const c_sum_

-- | c_prod :  r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THByteTensor_prod"
  c_prod_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> CInt -> IO ()

-- | alias of c_prod_ with unused argument (for CTHState) to unify backpack signatures.
c_prod :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> CInt -> IO ()
c_prod = const c_prod_

-- | c_cumsum :  r_ t dimension -> void
foreign import ccall "THTensorMath.h THByteTensor_cumsum"
  c_cumsum_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> IO ()

-- | alias of c_cumsum_ with unused argument (for CTHState) to unify backpack signatures.
c_cumsum :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> IO ()
c_cumsum = const c_cumsum_

-- | c_cumprod :  r_ t dimension -> void
foreign import ccall "THTensorMath.h THByteTensor_cumprod"
  c_cumprod_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> IO ()

-- | alias of c_cumprod_ with unused argument (for CTHState) to unify backpack signatures.
c_cumprod :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> IO ()
c_cumprod = const c_cumprod_

-- | c_sign :  r_ t -> void
foreign import ccall "THTensorMath.h THByteTensor_sign"
  c_sign_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_sign_ with unused argument (for CTHState) to unify backpack signatures.
c_sign :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_sign = const c_sign_

-- | c_trace :  t -> accreal
foreign import ccall "THTensorMath.h THByteTensor_trace"
  c_trace_ :: Ptr C'THByteTensor -> IO CLong

-- | alias of c_trace_ with unused argument (for CTHState) to unify backpack signatures.
c_trace :: Ptr C'THState -> Ptr C'THByteTensor -> IO CLong
c_trace = const c_trace_

-- | c_cross :  r_ a b dimension -> void
foreign import ccall "THTensorMath.h THByteTensor_cross"
  c_cross_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> IO ()

-- | alias of c_cross_ with unused argument (for CTHState) to unify backpack signatures.
c_cross :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> IO ()
c_cross = const c_cross_

-- | c_cmax :  r t src -> void
foreign import ccall "THTensorMath.h THByteTensor_cmax"
  c_cmax_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_cmax_ with unused argument (for CTHState) to unify backpack signatures.
c_cmax :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_cmax = const c_cmax_

-- | c_cmin :  r t src -> void
foreign import ccall "THTensorMath.h THByteTensor_cmin"
  c_cmin_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_cmin_ with unused argument (for CTHState) to unify backpack signatures.
c_cmin :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_cmin = const c_cmin_

-- | c_cmaxValue :  r t value -> void
foreign import ccall "THTensorMath.h THByteTensor_cmaxValue"
  c_cmaxValue_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_cmaxValue_ with unused argument (for CTHState) to unify backpack signatures.
c_cmaxValue :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_cmaxValue = const c_cmaxValue_

-- | c_cminValue :  r t value -> void
foreign import ccall "THTensorMath.h THByteTensor_cminValue"
  c_cminValue_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_cminValue_ with unused argument (for CTHState) to unify backpack signatures.
c_cminValue :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_cminValue = const c_cminValue_

-- | c_zeros :  r_ size -> void
foreign import ccall "THTensorMath.h THByteTensor_zeros"
  c_zeros_ :: Ptr C'THByteTensor -> Ptr C'THLongStorage -> IO ()

-- | alias of c_zeros_ with unused argument (for CTHState) to unify backpack signatures.
c_zeros :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THLongStorage -> IO ()
c_zeros = const c_zeros_

-- | c_zerosLike :  r_ input -> void
foreign import ccall "THTensorMath.h THByteTensor_zerosLike"
  c_zerosLike_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_zerosLike_ with unused argument (for CTHState) to unify backpack signatures.
c_zerosLike :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_zerosLike = const c_zerosLike_

-- | c_ones :  r_ size -> void
foreign import ccall "THTensorMath.h THByteTensor_ones"
  c_ones_ :: Ptr C'THByteTensor -> Ptr C'THLongStorage -> IO ()

-- | alias of c_ones_ with unused argument (for CTHState) to unify backpack signatures.
c_ones :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THLongStorage -> IO ()
c_ones = const c_ones_

-- | c_onesLike :  r_ input -> void
foreign import ccall "THTensorMath.h THByteTensor_onesLike"
  c_onesLike_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_onesLike_ with unused argument (for CTHState) to unify backpack signatures.
c_onesLike :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_onesLike = const c_onesLike_

-- | c_diag :  r_ t k -> void
foreign import ccall "THTensorMath.h THByteTensor_diag"
  c_diag_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> IO ()

-- | alias of c_diag_ with unused argument (for CTHState) to unify backpack signatures.
c_diag :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> IO ()
c_diag = const c_diag_

-- | c_eye :  r_ n m -> void
foreign import ccall "THTensorMath.h THByteTensor_eye"
  c_eye_ :: Ptr C'THByteTensor -> CLLong -> CLLong -> IO ()

-- | alias of c_eye_ with unused argument (for CTHState) to unify backpack signatures.
c_eye :: Ptr C'THState -> Ptr C'THByteTensor -> CLLong -> CLLong -> IO ()
c_eye = const c_eye_

-- | c_arange :  r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THByteTensor_arange"
  c_arange_ :: Ptr C'THByteTensor -> CLong -> CLong -> CLong -> IO ()

-- | alias of c_arange_ with unused argument (for CTHState) to unify backpack signatures.
c_arange :: Ptr C'THState -> Ptr C'THByteTensor -> CLong -> CLong -> CLong -> IO ()
c_arange = const c_arange_

-- | c_range :  r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THByteTensor_range"
  c_range_ :: Ptr C'THByteTensor -> CLong -> CLong -> CLong -> IO ()

-- | alias of c_range_ with unused argument (for CTHState) to unify backpack signatures.
c_range :: Ptr C'THState -> Ptr C'THByteTensor -> CLong -> CLong -> CLong -> IO ()
c_range = const c_range_

-- | c_randperm :  r_ _generator n -> void
foreign import ccall "THTensorMath.h THByteTensor_randperm"
  c_randperm_ :: Ptr C'THByteTensor -> Ptr C'THGenerator -> CLLong -> IO ()

-- | alias of c_randperm_ with unused argument (for CTHState) to unify backpack signatures.
c_randperm :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THGenerator -> CLLong -> IO ()
c_randperm = const c_randperm_

-- | c_reshape :  r_ t size -> void
foreign import ccall "THTensorMath.h THByteTensor_reshape"
  c_reshape_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THLongStorage -> IO ()

-- | alias of c_reshape_ with unused argument (for CTHState) to unify backpack signatures.
c_reshape :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THLongStorage -> IO ()
c_reshape = const c_reshape_

-- | c_sort :  rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h THByteTensor_sort"
  c_sort_ :: Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> CInt -> CInt -> IO ()

-- | alias of c_sort_ with unused argument (for CTHState) to unify backpack signatures.
c_sort :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> CInt -> CInt -> IO ()
c_sort = const c_sort_

-- | c_topk :  rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h THByteTensor_topk"
  c_topk_ :: Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> CLLong -> CInt -> CInt -> CInt -> IO ()

-- | alias of c_topk_ with unused argument (for CTHState) to unify backpack signatures.
c_topk :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> CLLong -> CInt -> CInt -> CInt -> IO ()
c_topk = const c_topk_

-- | c_tril :  r_ t k -> void
foreign import ccall "THTensorMath.h THByteTensor_tril"
  c_tril_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CLLong -> IO ()

-- | alias of c_tril_ with unused argument (for CTHState) to unify backpack signatures.
c_tril :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CLLong -> IO ()
c_tril = const c_tril_

-- | c_triu :  r_ t k -> void
foreign import ccall "THTensorMath.h THByteTensor_triu"
  c_triu_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CLLong -> IO ()

-- | alias of c_triu_ with unused argument (for CTHState) to unify backpack signatures.
c_triu :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CLLong -> IO ()
c_triu = const c_triu_

-- | c_cat :  r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h THByteTensor_cat"
  c_cat_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> IO ()

-- | alias of c_cat_ with unused argument (for CTHState) to unify backpack signatures.
c_cat :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> IO ()
c_cat = const c_cat_

-- | c_catArray :  result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h THByteTensor_catArray"
  c_catArray_ :: Ptr C'THByteTensor -> Ptr (Ptr C'THByteTensor) -> CInt -> CInt -> IO ()

-- | alias of c_catArray_ with unused argument (for CTHState) to unify backpack signatures.
c_catArray :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr (Ptr C'THByteTensor) -> CInt -> CInt -> IO ()
c_catArray = const c_catArray_

-- | c_equal :  ta tb -> int
foreign import ccall "THTensorMath.h THByteTensor_equal"
  c_equal_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO CInt

-- | alias of c_equal_ with unused argument (for CTHState) to unify backpack signatures.
c_equal :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO CInt
c_equal = const c_equal_

-- | c_ltValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_ltValue"
  c_ltValue_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_ltValue_ with unused argument (for CTHState) to unify backpack signatures.
c_ltValue :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_ltValue = const c_ltValue_

-- | c_leValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_leValue"
  c_leValue_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_leValue_ with unused argument (for CTHState) to unify backpack signatures.
c_leValue :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_leValue = const c_leValue_

-- | c_gtValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_gtValue"
  c_gtValue_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_gtValue_ with unused argument (for CTHState) to unify backpack signatures.
c_gtValue :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_gtValue = const c_gtValue_

-- | c_geValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_geValue"
  c_geValue_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_geValue_ with unused argument (for CTHState) to unify backpack signatures.
c_geValue :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_geValue = const c_geValue_

-- | c_neValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_neValue"
  c_neValue_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_neValue_ with unused argument (for CTHState) to unify backpack signatures.
c_neValue :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_neValue = const c_neValue_

-- | c_eqValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_eqValue"
  c_eqValue_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_eqValue_ with unused argument (for CTHState) to unify backpack signatures.
c_eqValue :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_eqValue = const c_eqValue_

-- | c_ltValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_ltValueT"
  c_ltValueT_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_ltValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_ltValueT :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_ltValueT = const c_ltValueT_

-- | c_leValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_leValueT"
  c_leValueT_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_leValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_leValueT :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_leValueT = const c_leValueT_

-- | c_gtValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_gtValueT"
  c_gtValueT_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_gtValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_gtValueT :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_gtValueT = const c_gtValueT_

-- | c_geValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_geValueT"
  c_geValueT_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_geValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_geValueT :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_geValueT = const c_geValueT_

-- | c_neValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_neValueT"
  c_neValueT_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_neValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_neValueT :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_neValueT = const c_neValueT_

-- | c_eqValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_eqValueT"
  c_eqValueT_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()

-- | alias of c_eqValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_eqValueT :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ()
c_eqValueT = const c_eqValueT_

-- | c_ltTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensor_ltTensor"
  c_ltTensor_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_ltTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_ltTensor :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_ltTensor = const c_ltTensor_

-- | c_leTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensor_leTensor"
  c_leTensor_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_leTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_leTensor :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_leTensor = const c_leTensor_

-- | c_gtTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensor_gtTensor"
  c_gtTensor_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_gtTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_gtTensor :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_gtTensor = const c_gtTensor_

-- | c_geTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensor_geTensor"
  c_geTensor_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_geTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_geTensor :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_geTensor = const c_geTensor_

-- | c_neTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensor_neTensor"
  c_neTensor_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_neTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_neTensor :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_neTensor = const c_neTensor_

-- | c_eqTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensor_eqTensor"
  c_eqTensor_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_eqTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_eqTensor :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_eqTensor = const c_eqTensor_

-- | c_ltTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensor_ltTensorT"
  c_ltTensorT_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_ltTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_ltTensorT :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_ltTensorT = const c_ltTensorT_

-- | c_leTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensor_leTensorT"
  c_leTensorT_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_leTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_leTensorT :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_leTensorT = const c_leTensorT_

-- | c_gtTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensor_gtTensorT"
  c_gtTensorT_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_gtTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_gtTensorT :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_gtTensorT = const c_gtTensorT_

-- | c_geTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensor_geTensorT"
  c_geTensorT_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_geTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_geTensorT :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_geTensorT = const c_geTensorT_

-- | c_neTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensor_neTensorT"
  c_neTensorT_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_neTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_neTensorT :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_neTensorT = const c_neTensorT_

-- | c_eqTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensor_eqTensorT"
  c_eqTensorT_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_eqTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_eqTensorT :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ()
c_eqTensorT = const c_eqTensorT_

-- | c_logicalall :  self -> int
foreign import ccall "THTensorMath.h THByteTensor_logicalall"
  c_logicalall_ :: Ptr C'THByteTensor -> IO CInt

-- | alias of c_logicalall_ with unused argument (for CTHState) to unify backpack signatures.
c_logicalall :: Ptr C'THState -> Ptr C'THByteTensor -> IO CInt
c_logicalall = const c_logicalall_

-- | c_logicalany :  self -> int
foreign import ccall "THTensorMath.h THByteTensor_logicalany"
  c_logicalany_ :: Ptr C'THByteTensor -> IO CInt

-- | alias of c_logicalany_ with unused argument (for CTHState) to unify backpack signatures.
c_logicalany :: Ptr C'THState -> Ptr C'THByteTensor -> IO CInt
c_logicalany = const c_logicalany_

-- | p_fill : Pointer to function : r_ value -> void
foreign import ccall "THTensorMath.h &THByteTensor_fill"
  p_fill :: FunPtr (Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_zero : Pointer to function : r_ -> void
foreign import ccall "THTensorMath.h &THByteTensor_zero"
  p_zero :: FunPtr (Ptr C'THByteTensor -> IO ())

-- | p_maskedFill : Pointer to function : tensor mask value -> void
foreign import ccall "THTensorMath.h &THByteTensor_maskedFill"
  p_maskedFill :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_maskedCopy : Pointer to function : tensor mask src -> void
foreign import ccall "THTensorMath.h &THByteTensor_maskedCopy"
  p_maskedCopy :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_maskedSelect : Pointer to function : tensor src mask -> void
foreign import ccall "THTensorMath.h &THByteTensor_maskedSelect"
  p_maskedSelect :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_nonzero : Pointer to function : subscript tensor -> void
foreign import ccall "THTensorMath.h &THByteTensor_nonzero"
  p_nonzero :: FunPtr (Ptr C'THLongTensor -> Ptr C'THByteTensor -> IO ())

-- | p_indexSelect : Pointer to function : tensor src dim index -> void
foreign import ccall "THTensorMath.h &THByteTensor_indexSelect"
  p_indexSelect :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> Ptr C'THLongTensor -> IO ())

-- | p_indexCopy : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THByteTensor_indexCopy"
  p_indexCopy :: FunPtr (Ptr C'THByteTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> IO ())

-- | p_indexAdd : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THByteTensor_indexAdd"
  p_indexAdd :: FunPtr (Ptr C'THByteTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> IO ())

-- | p_indexFill : Pointer to function : tensor dim index val -> void
foreign import ccall "THTensorMath.h &THByteTensor_indexFill"
  p_indexFill :: FunPtr (Ptr C'THByteTensor -> CInt -> Ptr C'THLongTensor -> CUChar -> IO ())

-- | p_take : Pointer to function : tensor src index -> void
foreign import ccall "THTensorMath.h &THByteTensor_take"
  p_take :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THLongTensor -> IO ())

-- | p_put : Pointer to function : tensor index src accumulate -> void
foreign import ccall "THTensorMath.h &THByteTensor_put"
  p_put :: FunPtr (Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> CInt -> IO ())

-- | p_gather : Pointer to function : tensor src dim index -> void
foreign import ccall "THTensorMath.h &THByteTensor_gather"
  p_gather :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> Ptr C'THLongTensor -> IO ())

-- | p_scatter : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THByteTensor_scatter"
  p_scatter :: FunPtr (Ptr C'THByteTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> IO ())

-- | p_scatterAdd : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THByteTensor_scatterAdd"
  p_scatterAdd :: FunPtr (Ptr C'THByteTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> IO ())

-- | p_scatterFill : Pointer to function : tensor dim index val -> void
foreign import ccall "THTensorMath.h &THByteTensor_scatterFill"
  p_scatterFill :: FunPtr (Ptr C'THByteTensor -> CInt -> Ptr C'THLongTensor -> CUChar -> IO ())

-- | p_dot : Pointer to function : t src -> accreal
foreign import ccall "THTensorMath.h &THByteTensor_dot"
  p_dot :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO CLong)

-- | p_minall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THByteTensor_minall"
  p_minall :: FunPtr (Ptr C'THByteTensor -> IO CUChar)

-- | p_maxall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THByteTensor_maxall"
  p_maxall :: FunPtr (Ptr C'THByteTensor -> IO CUChar)

-- | p_medianall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THByteTensor_medianall"
  p_medianall :: FunPtr (Ptr C'THByteTensor -> IO CUChar)

-- | p_sumall : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THByteTensor_sumall"
  p_sumall :: FunPtr (Ptr C'THByteTensor -> IO CLong)

-- | p_prodall : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THByteTensor_prodall"
  p_prodall :: FunPtr (Ptr C'THByteTensor -> IO CLong)

-- | p_add : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THByteTensor_add"
  p_add :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_sub : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THByteTensor_sub"
  p_sub :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_add_scaled : Pointer to function : r_ t value alpha -> void
foreign import ccall "THTensorMath.h &THByteTensor_add_scaled"
  p_add_scaled :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> CUChar -> IO ())

-- | p_sub_scaled : Pointer to function : r_ t value alpha -> void
foreign import ccall "THTensorMath.h &THByteTensor_sub_scaled"
  p_sub_scaled :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> CUChar -> IO ())

-- | p_mul : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THByteTensor_mul"
  p_mul :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_div : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THByteTensor_div"
  p_div :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_lshift : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THByteTensor_lshift"
  p_lshift :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_rshift : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THByteTensor_rshift"
  p_rshift :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_fmod : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THByteTensor_fmod"
  p_fmod :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_remainder : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THByteTensor_remainder"
  p_remainder :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_clamp : Pointer to function : r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h &THByteTensor_clamp"
  p_clamp :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> CUChar -> IO ())

-- | p_bitand : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THByteTensor_bitand"
  p_bitand :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_bitor : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THByteTensor_bitor"
  p_bitor :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_bitxor : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THByteTensor_bitxor"
  p_bitxor :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_cadd : Pointer to function : r_ t value src -> void
foreign import ccall "THTensorMath.h &THByteTensor_cadd"
  p_cadd :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> IO ())

-- | p_csub : Pointer to function : self src1 value src2 -> void
foreign import ccall "THTensorMath.h &THByteTensor_csub"
  p_csub :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> IO ())

-- | p_cmul : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THByteTensor_cmul"
  p_cmul :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_cpow : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THByteTensor_cpow"
  p_cpow :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_cdiv : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THByteTensor_cdiv"
  p_cdiv :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_clshift : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THByteTensor_clshift"
  p_clshift :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_crshift : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THByteTensor_crshift"
  p_crshift :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_cfmod : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THByteTensor_cfmod"
  p_cfmod :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_cremainder : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THByteTensor_cremainder"
  p_cremainder :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_cbitand : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THByteTensor_cbitand"
  p_cbitand :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_cbitor : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THByteTensor_cbitor"
  p_cbitor :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_cbitxor : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THByteTensor_cbitxor"
  p_cbitxor :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_addcmul : Pointer to function : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h &THByteTensor_addcmul"
  p_addcmul :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_addcdiv : Pointer to function : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h &THByteTensor_addcdiv"
  p_addcdiv :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_addmv : Pointer to function : r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h &THByteTensor_addmv"
  p_addmv :: FunPtr (Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_addmm : Pointer to function : r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h &THByteTensor_addmm"
  p_addmm :: FunPtr (Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_addr : Pointer to function : r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h &THByteTensor_addr"
  p_addr :: FunPtr (Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_addbmm : Pointer to function : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h &THByteTensor_addbmm"
  p_addbmm :: FunPtr (Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_baddbmm : Pointer to function : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h &THByteTensor_baddbmm"
  p_baddbmm :: FunPtr (Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> CUChar -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_match : Pointer to function : r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h &THByteTensor_match"
  p_match :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_numel : Pointer to function : t -> ptrdiff_t
foreign import ccall "THTensorMath.h &THByteTensor_numel"
  p_numel :: FunPtr (Ptr C'THByteTensor -> IO CPtrdiff)

-- | p_preserveReduceDimSemantics : Pointer to function : r_ in_dims reduce_dimension keepdim -> void
foreign import ccall "THTensorMath.h &THByteTensor_preserveReduceDimSemantics"
  p_preserveReduceDimSemantics :: FunPtr (Ptr C'THByteTensor -> CInt -> CInt -> CInt -> IO ())

-- | p_max : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THByteTensor_max"
  p_max :: FunPtr (Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> CInt -> CInt -> IO ())

-- | p_min : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THByteTensor_min"
  p_min :: FunPtr (Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> CInt -> CInt -> IO ())

-- | p_kthvalue : Pointer to function : values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h &THByteTensor_kthvalue"
  p_kthvalue :: FunPtr (Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> CLLong -> CInt -> CInt -> IO ())

-- | p_mode : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THByteTensor_mode"
  p_mode :: FunPtr (Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> CInt -> CInt -> IO ())

-- | p_median : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THByteTensor_median"
  p_median :: FunPtr (Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> CInt -> CInt -> IO ())

-- | p_sum : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THByteTensor_sum"
  p_sum :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> CInt -> IO ())

-- | p_prod : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THByteTensor_prod"
  p_prod :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> CInt -> IO ())

-- | p_cumsum : Pointer to function : r_ t dimension -> void
foreign import ccall "THTensorMath.h &THByteTensor_cumsum"
  p_cumsum :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> IO ())

-- | p_cumprod : Pointer to function : r_ t dimension -> void
foreign import ccall "THTensorMath.h &THByteTensor_cumprod"
  p_cumprod :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> IO ())

-- | p_sign : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THByteTensor_sign"
  p_sign :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_trace : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THByteTensor_trace"
  p_trace :: FunPtr (Ptr C'THByteTensor -> IO CLong)

-- | p_cross : Pointer to function : r_ a b dimension -> void
foreign import ccall "THTensorMath.h &THByteTensor_cross"
  p_cross :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> IO ())

-- | p_cmax : Pointer to function : r t src -> void
foreign import ccall "THTensorMath.h &THByteTensor_cmax"
  p_cmax :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_cmin : Pointer to function : r t src -> void
foreign import ccall "THTensorMath.h &THByteTensor_cmin"
  p_cmin :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_cmaxValue : Pointer to function : r t value -> void
foreign import ccall "THTensorMath.h &THByteTensor_cmaxValue"
  p_cmaxValue :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_cminValue : Pointer to function : r t value -> void
foreign import ccall "THTensorMath.h &THByteTensor_cminValue"
  p_cminValue :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_zeros : Pointer to function : r_ size -> void
foreign import ccall "THTensorMath.h &THByteTensor_zeros"
  p_zeros :: FunPtr (Ptr C'THByteTensor -> Ptr C'THLongStorage -> IO ())

-- | p_zerosLike : Pointer to function : r_ input -> void
foreign import ccall "THTensorMath.h &THByteTensor_zerosLike"
  p_zerosLike :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_ones : Pointer to function : r_ size -> void
foreign import ccall "THTensorMath.h &THByteTensor_ones"
  p_ones :: FunPtr (Ptr C'THByteTensor -> Ptr C'THLongStorage -> IO ())

-- | p_onesLike : Pointer to function : r_ input -> void
foreign import ccall "THTensorMath.h &THByteTensor_onesLike"
  p_onesLike :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_diag : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THByteTensor_diag"
  p_diag :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> IO ())

-- | p_eye : Pointer to function : r_ n m -> void
foreign import ccall "THTensorMath.h &THByteTensor_eye"
  p_eye :: FunPtr (Ptr C'THByteTensor -> CLLong -> CLLong -> IO ())

-- | p_arange : Pointer to function : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h &THByteTensor_arange"
  p_arange :: FunPtr (Ptr C'THByteTensor -> CLong -> CLong -> CLong -> IO ())

-- | p_range : Pointer to function : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h &THByteTensor_range"
  p_range :: FunPtr (Ptr C'THByteTensor -> CLong -> CLong -> CLong -> IO ())

-- | p_randperm : Pointer to function : r_ _generator n -> void
foreign import ccall "THTensorMath.h &THByteTensor_randperm"
  p_randperm :: FunPtr (Ptr C'THByteTensor -> Ptr C'THGenerator -> CLLong -> IO ())

-- | p_reshape : Pointer to function : r_ t size -> void
foreign import ccall "THTensorMath.h &THByteTensor_reshape"
  p_reshape :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THLongStorage -> IO ())

-- | p_sort : Pointer to function : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h &THByteTensor_sort"
  p_sort :: FunPtr (Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> CInt -> CInt -> IO ())

-- | p_topk : Pointer to function : rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h &THByteTensor_topk"
  p_topk :: FunPtr (Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> CLLong -> CInt -> CInt -> CInt -> IO ())

-- | p_tril : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THByteTensor_tril"
  p_tril :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CLLong -> IO ())

-- | p_triu : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THByteTensor_triu"
  p_triu :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CLLong -> IO ())

-- | p_cat : Pointer to function : r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h &THByteTensor_cat"
  p_cat :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> CInt -> IO ())

-- | p_catArray : Pointer to function : result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h &THByteTensor_catArray"
  p_catArray :: FunPtr (Ptr C'THByteTensor -> Ptr (Ptr C'THByteTensor) -> CInt -> CInt -> IO ())

-- | p_equal : Pointer to function : ta tb -> int
foreign import ccall "THTensorMath.h &THByteTensor_equal"
  p_equal :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO CInt)

-- | p_ltValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THByteTensor_ltValue"
  p_ltValue :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_leValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THByteTensor_leValue"
  p_leValue :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_gtValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THByteTensor_gtValue"
  p_gtValue :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_geValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THByteTensor_geValue"
  p_geValue :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_neValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THByteTensor_neValue"
  p_neValue :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_eqValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THByteTensor_eqValue"
  p_eqValue :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_ltValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THByteTensor_ltValueT"
  p_ltValueT :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_leValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THByteTensor_leValueT"
  p_leValueT :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_gtValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THByteTensor_gtValueT"
  p_gtValueT :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_geValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THByteTensor_geValueT"
  p_geValueT :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_neValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THByteTensor_neValueT"
  p_neValueT :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_eqValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THByteTensor_eqValueT"
  p_eqValueT :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> CUChar -> IO ())

-- | p_ltTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THByteTensor_ltTensor"
  p_ltTensor :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_leTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THByteTensor_leTensor"
  p_leTensor :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_gtTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THByteTensor_gtTensor"
  p_gtTensor :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_geTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THByteTensor_geTensor"
  p_geTensor :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_neTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THByteTensor_neTensor"
  p_neTensor :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_eqTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THByteTensor_eqTensor"
  p_eqTensor :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_ltTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THByteTensor_ltTensorT"
  p_ltTensorT :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_leTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THByteTensor_leTensorT"
  p_leTensorT :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_gtTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THByteTensor_gtTensorT"
  p_gtTensorT :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_geTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THByteTensor_geTensorT"
  p_geTensorT :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_neTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THByteTensor_neTensorT"
  p_neTensorT :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_eqTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THByteTensor_eqTensorT"
  p_eqTensorT :: FunPtr (Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> IO ())

-- | p_logicalall : Pointer to function : self -> int
foreign import ccall "THTensorMath.h &THByteTensor_logicalall"
  p_logicalall :: FunPtr (Ptr C'THByteTensor -> IO CInt)

-- | p_logicalany : Pointer to function : self -> int
foreign import ccall "THTensorMath.h &THByteTensor_logicalany"
  p_logicalany :: FunPtr (Ptr C'THByteTensor -> IO CInt)