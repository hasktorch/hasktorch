{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Int.TensorMath where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_fill :  r_ value -> void
foreign import ccall "THTensorMath.h THIntTensor_fill"
  c_fill_ :: Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_fill_ with unused argument (for CTHState) to unify backpack signatures.
c_fill :: Ptr C'THState -> Ptr C'THIntTensor -> CInt -> IO ()
c_fill = const c_fill_

-- | c_zero :  r_ -> void
foreign import ccall "THTensorMath.h THIntTensor_zero"
  c_zero_ :: Ptr C'THIntTensor -> IO ()

-- | alias of c_zero_ with unused argument (for CTHState) to unify backpack signatures.
c_zero :: Ptr C'THState -> Ptr C'THIntTensor -> IO ()
c_zero = const c_zero_

-- | c_maskedFill :  tensor mask value -> void
foreign import ccall "THTensorMath.h THIntTensor_maskedFill"
  c_maskedFill_ :: Ptr C'THIntTensor -> Ptr C'THByteTensor -> CInt -> IO ()

-- | alias of c_maskedFill_ with unused argument (for CTHState) to unify backpack signatures.
c_maskedFill :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THByteTensor -> CInt -> IO ()
c_maskedFill = const c_maskedFill_

-- | c_maskedCopy :  tensor mask src -> void
foreign import ccall "THTensorMath.h THIntTensor_maskedCopy"
  c_maskedCopy_ :: Ptr C'THIntTensor -> Ptr C'THByteTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_maskedCopy_ with unused argument (for CTHState) to unify backpack signatures.
c_maskedCopy :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THByteTensor -> Ptr C'THIntTensor -> IO ()
c_maskedCopy = const c_maskedCopy_

-- | c_maskedSelect :  tensor src mask -> void
foreign import ccall "THTensorMath.h THIntTensor_maskedSelect"
  c_maskedSelect_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_maskedSelect_ with unused argument (for CTHState) to unify backpack signatures.
c_maskedSelect :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THByteTensor -> IO ()
c_maskedSelect = const c_maskedSelect_

-- | c_nonzero :  subscript tensor -> void
foreign import ccall "THTensorMath.h THIntTensor_nonzero"
  c_nonzero_ :: Ptr C'THLongTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_nonzero_ with unused argument (for CTHState) to unify backpack signatures.
c_nonzero :: Ptr C'THState -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> IO ()
c_nonzero = const c_nonzero_

-- | c_indexSelect :  tensor src dim index -> void
foreign import ccall "THTensorMath.h THIntTensor_indexSelect"
  c_indexSelect_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> Ptr C'THLongTensor -> IO ()

-- | alias of c_indexSelect_ with unused argument (for CTHState) to unify backpack signatures.
c_indexSelect :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> Ptr C'THLongTensor -> IO ()
c_indexSelect = const c_indexSelect_

-- | c_indexCopy :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THIntTensor_indexCopy"
  c_indexCopy_ :: Ptr C'THIntTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_indexCopy_ with unused argument (for CTHState) to unify backpack signatures.
c_indexCopy :: Ptr C'THState -> Ptr C'THIntTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> IO ()
c_indexCopy = const c_indexCopy_

-- | c_indexAdd :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THIntTensor_indexAdd"
  c_indexAdd_ :: Ptr C'THIntTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_indexAdd_ with unused argument (for CTHState) to unify backpack signatures.
c_indexAdd :: Ptr C'THState -> Ptr C'THIntTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> IO ()
c_indexAdd = const c_indexAdd_

-- | c_indexFill :  tensor dim index val -> void
foreign import ccall "THTensorMath.h THIntTensor_indexFill"
  c_indexFill_ :: Ptr C'THIntTensor -> CInt -> Ptr C'THLongTensor -> CInt -> IO ()

-- | alias of c_indexFill_ with unused argument (for CTHState) to unify backpack signatures.
c_indexFill :: Ptr C'THState -> Ptr C'THIntTensor -> CInt -> Ptr C'THLongTensor -> CInt -> IO ()
c_indexFill = const c_indexFill_

-- | c_take :  tensor src index -> void
foreign import ccall "THTensorMath.h THIntTensor_take"
  c_take_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_take_ with unused argument (for CTHState) to unify backpack signatures.
c_take :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THLongTensor -> IO ()
c_take = const c_take_

-- | c_put :  tensor index src accumulate -> void
foreign import ccall "THTensorMath.h THIntTensor_put"
  c_put_ :: Ptr C'THIntTensor -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_put_ with unused argument (for CTHState) to unify backpack signatures.
c_put :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_put = const c_put_

-- | c_gather :  tensor src dim index -> void
foreign import ccall "THTensorMath.h THIntTensor_gather"
  c_gather_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> Ptr C'THLongTensor -> IO ()

-- | alias of c_gather_ with unused argument (for CTHState) to unify backpack signatures.
c_gather :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> Ptr C'THLongTensor -> IO ()
c_gather = const c_gather_

-- | c_scatter :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THIntTensor_scatter"
  c_scatter_ :: Ptr C'THIntTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_scatter_ with unused argument (for CTHState) to unify backpack signatures.
c_scatter :: Ptr C'THState -> Ptr C'THIntTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> IO ()
c_scatter = const c_scatter_

-- | c_scatterAdd :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THIntTensor_scatterAdd"
  c_scatterAdd_ :: Ptr C'THIntTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_scatterAdd_ with unused argument (for CTHState) to unify backpack signatures.
c_scatterAdd :: Ptr C'THState -> Ptr C'THIntTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> IO ()
c_scatterAdd = const c_scatterAdd_

-- | c_scatterFill :  tensor dim index val -> void
foreign import ccall "THTensorMath.h THIntTensor_scatterFill"
  c_scatterFill_ :: Ptr C'THIntTensor -> CInt -> Ptr C'THLongTensor -> CInt -> IO ()

-- | alias of c_scatterFill_ with unused argument (for CTHState) to unify backpack signatures.
c_scatterFill :: Ptr C'THState -> Ptr C'THIntTensor -> CInt -> Ptr C'THLongTensor -> CInt -> IO ()
c_scatterFill = const c_scatterFill_

-- | c_dot :  t src -> accreal
foreign import ccall "THTensorMath.h THIntTensor_dot"
  c_dot_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO CLong

-- | alias of c_dot_ with unused argument (for CTHState) to unify backpack signatures.
c_dot :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO CLong
c_dot = const c_dot_

-- | c_minall :  t -> real
foreign import ccall "THTensorMath.h THIntTensor_minall"
  c_minall_ :: Ptr C'THIntTensor -> IO CInt

-- | alias of c_minall_ with unused argument (for CTHState) to unify backpack signatures.
c_minall :: Ptr C'THState -> Ptr C'THIntTensor -> IO CInt
c_minall = const c_minall_

-- | c_maxall :  t -> real
foreign import ccall "THTensorMath.h THIntTensor_maxall"
  c_maxall_ :: Ptr C'THIntTensor -> IO CInt

-- | alias of c_maxall_ with unused argument (for CTHState) to unify backpack signatures.
c_maxall :: Ptr C'THState -> Ptr C'THIntTensor -> IO CInt
c_maxall = const c_maxall_

-- | c_medianall :  t -> real
foreign import ccall "THTensorMath.h THIntTensor_medianall"
  c_medianall_ :: Ptr C'THIntTensor -> IO CInt

-- | alias of c_medianall_ with unused argument (for CTHState) to unify backpack signatures.
c_medianall :: Ptr C'THState -> Ptr C'THIntTensor -> IO CInt
c_medianall = const c_medianall_

-- | c_sumall :  t -> accreal
foreign import ccall "THTensorMath.h THIntTensor_sumall"
  c_sumall_ :: Ptr C'THIntTensor -> IO CLong

-- | alias of c_sumall_ with unused argument (for CTHState) to unify backpack signatures.
c_sumall :: Ptr C'THState -> Ptr C'THIntTensor -> IO CLong
c_sumall = const c_sumall_

-- | c_prodall :  t -> accreal
foreign import ccall "THTensorMath.h THIntTensor_prodall"
  c_prodall_ :: Ptr C'THIntTensor -> IO CLong

-- | alias of c_prodall_ with unused argument (for CTHState) to unify backpack signatures.
c_prodall :: Ptr C'THState -> Ptr C'THIntTensor -> IO CLong
c_prodall = const c_prodall_

-- | c_neg :  self src -> void
foreign import ccall "THTensorMath.h THIntTensor_neg"
  c_neg_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_neg_ with unused argument (for CTHState) to unify backpack signatures.
c_neg :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_neg = const c_neg_

-- | c_add :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_add"
  c_add_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_add_ with unused argument (for CTHState) to unify backpack signatures.
c_add :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_add = const c_add_

-- | c_sub :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_sub"
  c_sub_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_sub_ with unused argument (for CTHState) to unify backpack signatures.
c_sub :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_sub = const c_sub_

-- | c_add_scaled :  r_ t value alpha -> void
foreign import ccall "THTensorMath.h THIntTensor_add_scaled"
  c_add_scaled_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ()

-- | alias of c_add_scaled_ with unused argument (for CTHState) to unify backpack signatures.
c_add_scaled :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ()
c_add_scaled = const c_add_scaled_

-- | c_sub_scaled :  r_ t value alpha -> void
foreign import ccall "THTensorMath.h THIntTensor_sub_scaled"
  c_sub_scaled_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ()

-- | alias of c_sub_scaled_ with unused argument (for CTHState) to unify backpack signatures.
c_sub_scaled :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ()
c_sub_scaled = const c_sub_scaled_

-- | c_mul :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_mul"
  c_mul_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_mul_ with unused argument (for CTHState) to unify backpack signatures.
c_mul :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_mul = const c_mul_

-- | c_div :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_div"
  c_div_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_div_ with unused argument (for CTHState) to unify backpack signatures.
c_div :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_div = const c_div_

-- | c_lshift :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_lshift"
  c_lshift_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_lshift_ with unused argument (for CTHState) to unify backpack signatures.
c_lshift :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_lshift = const c_lshift_

-- | c_rshift :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_rshift"
  c_rshift_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_rshift_ with unused argument (for CTHState) to unify backpack signatures.
c_rshift :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_rshift = const c_rshift_

-- | c_fmod :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_fmod"
  c_fmod_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_fmod_ with unused argument (for CTHState) to unify backpack signatures.
c_fmod :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_fmod = const c_fmod_

-- | c_remainder :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_remainder"
  c_remainder_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_remainder_ with unused argument (for CTHState) to unify backpack signatures.
c_remainder :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_remainder = const c_remainder_

-- | c_clamp :  r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h THIntTensor_clamp"
  c_clamp_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ()

-- | alias of c_clamp_ with unused argument (for CTHState) to unify backpack signatures.
c_clamp :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ()
c_clamp = const c_clamp_

-- | c_bitand :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_bitand"
  c_bitand_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_bitand_ with unused argument (for CTHState) to unify backpack signatures.
c_bitand :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_bitand = const c_bitand_

-- | c_bitor :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_bitor"
  c_bitor_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_bitor_ with unused argument (for CTHState) to unify backpack signatures.
c_bitor :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_bitor = const c_bitor_

-- | c_bitxor :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_bitxor"
  c_bitxor_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_bitxor_ with unused argument (for CTHState) to unify backpack signatures.
c_bitxor :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_bitxor = const c_bitxor_

-- | c_cadd :  r_ t value src -> void
foreign import ccall "THTensorMath.h THIntTensor_cadd"
  c_cadd_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> IO ()

-- | alias of c_cadd_ with unused argument (for CTHState) to unify backpack signatures.
c_cadd :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> IO ()
c_cadd = const c_cadd_

-- | c_csub :  self src1 value src2 -> void
foreign import ccall "THTensorMath.h THIntTensor_csub"
  c_csub_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> IO ()

-- | alias of c_csub_ with unused argument (for CTHState) to unify backpack signatures.
c_csub :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> IO ()
c_csub = const c_csub_

-- | c_cmul :  r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cmul"
  c_cmul_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_cmul_ with unused argument (for CTHState) to unify backpack signatures.
c_cmul :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_cmul = const c_cmul_

-- | c_cpow :  r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cpow"
  c_cpow_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_cpow_ with unused argument (for CTHState) to unify backpack signatures.
c_cpow :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_cpow = const c_cpow_

-- | c_cdiv :  r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cdiv"
  c_cdiv_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_cdiv_ with unused argument (for CTHState) to unify backpack signatures.
c_cdiv :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_cdiv = const c_cdiv_

-- | c_clshift :  r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_clshift"
  c_clshift_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_clshift_ with unused argument (for CTHState) to unify backpack signatures.
c_clshift :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_clshift = const c_clshift_

-- | c_crshift :  r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_crshift"
  c_crshift_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_crshift_ with unused argument (for CTHState) to unify backpack signatures.
c_crshift :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_crshift = const c_crshift_

-- | c_cfmod :  r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cfmod"
  c_cfmod_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_cfmod_ with unused argument (for CTHState) to unify backpack signatures.
c_cfmod :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_cfmod = const c_cfmod_

-- | c_cremainder :  r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cremainder"
  c_cremainder_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_cremainder_ with unused argument (for CTHState) to unify backpack signatures.
c_cremainder :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_cremainder = const c_cremainder_

-- | c_cbitand :  r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cbitand"
  c_cbitand_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_cbitand_ with unused argument (for CTHState) to unify backpack signatures.
c_cbitand :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_cbitand = const c_cbitand_

-- | c_cbitor :  r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cbitor"
  c_cbitor_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_cbitor_ with unused argument (for CTHState) to unify backpack signatures.
c_cbitor :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_cbitor = const c_cbitor_

-- | c_cbitxor :  r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cbitxor"
  c_cbitxor_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_cbitxor_ with unused argument (for CTHState) to unify backpack signatures.
c_cbitxor :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_cbitxor = const c_cbitxor_

-- | c_addcmul :  r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THIntTensor_addcmul"
  c_addcmul_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_addcmul_ with unused argument (for CTHState) to unify backpack signatures.
c_addcmul :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_addcmul = const c_addcmul_

-- | c_addcdiv :  r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THIntTensor_addcdiv"
  c_addcdiv_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_addcdiv_ with unused argument (for CTHState) to unify backpack signatures.
c_addcdiv :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_addcdiv = const c_addcdiv_

-- | c_addmv :  r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h THIntTensor_addmv"
  c_addmv_ :: Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_addmv_ with unused argument (for CTHState) to unify backpack signatures.
c_addmv :: Ptr C'THState -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_addmv = const c_addmv_

-- | c_addmm :  r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h THIntTensor_addmm"
  c_addmm_ :: Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_addmm_ with unused argument (for CTHState) to unify backpack signatures.
c_addmm :: Ptr C'THState -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_addmm = const c_addmm_

-- | c_addr :  r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h THIntTensor_addr"
  c_addr_ :: Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_addr_ with unused argument (for CTHState) to unify backpack signatures.
c_addr :: Ptr C'THState -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_addr = const c_addr_

-- | c_addbmm :  r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THIntTensor_addbmm"
  c_addbmm_ :: Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_addbmm_ with unused argument (for CTHState) to unify backpack signatures.
c_addbmm :: Ptr C'THState -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_addbmm = const c_addbmm_

-- | c_baddbmm :  r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THIntTensor_baddbmm"
  c_baddbmm_ :: Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_baddbmm_ with unused argument (for CTHState) to unify backpack signatures.
c_baddbmm :: Ptr C'THState -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_baddbmm = const c_baddbmm_

-- | c_match :  r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h THIntTensor_match"
  c_match_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_match_ with unused argument (for CTHState) to unify backpack signatures.
c_match :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_match = const c_match_

-- | c_numel :  t -> ptrdiff_t
foreign import ccall "THTensorMath.h THIntTensor_numel"
  c_numel_ :: Ptr C'THIntTensor -> IO CPtrdiff

-- | alias of c_numel_ with unused argument (for CTHState) to unify backpack signatures.
c_numel :: Ptr C'THState -> Ptr C'THIntTensor -> IO CPtrdiff
c_numel = const c_numel_

-- | c_preserveReduceDimSemantics :  r_ in_dims reduce_dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensor_preserveReduceDimSemantics"
  c_preserveReduceDimSemantics_ :: Ptr C'THIntTensor -> CInt -> CInt -> CInt -> IO ()

-- | alias of c_preserveReduceDimSemantics_ with unused argument (for CTHState) to unify backpack signatures.
c_preserveReduceDimSemantics :: Ptr C'THState -> Ptr C'THIntTensor -> CInt -> CInt -> CInt -> IO ()
c_preserveReduceDimSemantics = const c_preserveReduceDimSemantics_

-- | c_max :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensor_max"
  c_max_ :: Ptr C'THIntTensor -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ()

-- | alias of c_max_ with unused argument (for CTHState) to unify backpack signatures.
c_max :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ()
c_max = const c_max_

-- | c_min :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensor_min"
  c_min_ :: Ptr C'THIntTensor -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ()

-- | alias of c_min_ with unused argument (for CTHState) to unify backpack signatures.
c_min :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ()
c_min = const c_min_

-- | c_kthvalue :  values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensor_kthvalue"
  c_kthvalue_ :: Ptr C'THIntTensor -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> CLLong -> CInt -> CInt -> IO ()

-- | alias of c_kthvalue_ with unused argument (for CTHState) to unify backpack signatures.
c_kthvalue :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> CLLong -> CInt -> CInt -> IO ()
c_kthvalue = const c_kthvalue_

-- | c_mode :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensor_mode"
  c_mode_ :: Ptr C'THIntTensor -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ()

-- | alias of c_mode_ with unused argument (for CTHState) to unify backpack signatures.
c_mode :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ()
c_mode = const c_mode_

-- | c_median :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensor_median"
  c_median_ :: Ptr C'THIntTensor -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ()

-- | alias of c_median_ with unused argument (for CTHState) to unify backpack signatures.
c_median :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ()
c_median = const c_median_

-- | c_sum :  r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensor_sum"
  c_sum_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ()

-- | alias of c_sum_ with unused argument (for CTHState) to unify backpack signatures.
c_sum :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ()
c_sum = const c_sum_

-- | c_prod :  r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensor_prod"
  c_prod_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ()

-- | alias of c_prod_ with unused argument (for CTHState) to unify backpack signatures.
c_prod :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ()
c_prod = const c_prod_

-- | c_cumsum :  r_ t dimension -> void
foreign import ccall "THTensorMath.h THIntTensor_cumsum"
  c_cumsum_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_cumsum_ with unused argument (for CTHState) to unify backpack signatures.
c_cumsum :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_cumsum = const c_cumsum_

-- | c_cumprod :  r_ t dimension -> void
foreign import ccall "THTensorMath.h THIntTensor_cumprod"
  c_cumprod_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_cumprod_ with unused argument (for CTHState) to unify backpack signatures.
c_cumprod :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_cumprod = const c_cumprod_

-- | c_sign :  r_ t -> void
foreign import ccall "THTensorMath.h THIntTensor_sign"
  c_sign_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_sign_ with unused argument (for CTHState) to unify backpack signatures.
c_sign :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_sign = const c_sign_

-- | c_trace :  t -> accreal
foreign import ccall "THTensorMath.h THIntTensor_trace"
  c_trace_ :: Ptr C'THIntTensor -> IO CLong

-- | alias of c_trace_ with unused argument (for CTHState) to unify backpack signatures.
c_trace :: Ptr C'THState -> Ptr C'THIntTensor -> IO CLong
c_trace = const c_trace_

-- | c_cross :  r_ a b dimension -> void
foreign import ccall "THTensorMath.h THIntTensor_cross"
  c_cross_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_cross_ with unused argument (for CTHState) to unify backpack signatures.
c_cross :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_cross = const c_cross_

-- | c_cmax :  r t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cmax"
  c_cmax_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_cmax_ with unused argument (for CTHState) to unify backpack signatures.
c_cmax :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_cmax = const c_cmax_

-- | c_cmin :  r t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cmin"
  c_cmin_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_cmin_ with unused argument (for CTHState) to unify backpack signatures.
c_cmin :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_cmin = const c_cmin_

-- | c_cmaxValue :  r t value -> void
foreign import ccall "THTensorMath.h THIntTensor_cmaxValue"
  c_cmaxValue_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_cmaxValue_ with unused argument (for CTHState) to unify backpack signatures.
c_cmaxValue :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_cmaxValue = const c_cmaxValue_

-- | c_cminValue :  r t value -> void
foreign import ccall "THTensorMath.h THIntTensor_cminValue"
  c_cminValue_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_cminValue_ with unused argument (for CTHState) to unify backpack signatures.
c_cminValue :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_cminValue = const c_cminValue_

-- | c_zeros :  r_ size -> void
foreign import ccall "THTensorMath.h THIntTensor_zeros"
  c_zeros_ :: Ptr C'THIntTensor -> Ptr C'THLongStorage -> IO ()

-- | alias of c_zeros_ with unused argument (for CTHState) to unify backpack signatures.
c_zeros :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THLongStorage -> IO ()
c_zeros = const c_zeros_

-- | c_zerosLike :  r_ input -> void
foreign import ccall "THTensorMath.h THIntTensor_zerosLike"
  c_zerosLike_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_zerosLike_ with unused argument (for CTHState) to unify backpack signatures.
c_zerosLike :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_zerosLike = const c_zerosLike_

-- | c_ones :  r_ size -> void
foreign import ccall "THTensorMath.h THIntTensor_ones"
  c_ones_ :: Ptr C'THIntTensor -> Ptr C'THLongStorage -> IO ()

-- | alias of c_ones_ with unused argument (for CTHState) to unify backpack signatures.
c_ones :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THLongStorage -> IO ()
c_ones = const c_ones_

-- | c_onesLike :  r_ input -> void
foreign import ccall "THTensorMath.h THIntTensor_onesLike"
  c_onesLike_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_onesLike_ with unused argument (for CTHState) to unify backpack signatures.
c_onesLike :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_onesLike = const c_onesLike_

-- | c_diag :  r_ t k -> void
foreign import ccall "THTensorMath.h THIntTensor_diag"
  c_diag_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_diag_ with unused argument (for CTHState) to unify backpack signatures.
c_diag :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_diag = const c_diag_

-- | c_eye :  r_ n m -> void
foreign import ccall "THTensorMath.h THIntTensor_eye"
  c_eye_ :: Ptr C'THIntTensor -> CLLong -> CLLong -> IO ()

-- | alias of c_eye_ with unused argument (for CTHState) to unify backpack signatures.
c_eye :: Ptr C'THState -> Ptr C'THIntTensor -> CLLong -> CLLong -> IO ()
c_eye = const c_eye_

-- | c_arange :  r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THIntTensor_arange"
  c_arange_ :: Ptr C'THIntTensor -> CLong -> CLong -> CLong -> IO ()

-- | alias of c_arange_ with unused argument (for CTHState) to unify backpack signatures.
c_arange :: Ptr C'THState -> Ptr C'THIntTensor -> CLong -> CLong -> CLong -> IO ()
c_arange = const c_arange_

-- | c_range :  r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THIntTensor_range"
  c_range_ :: Ptr C'THIntTensor -> CLong -> CLong -> CLong -> IO ()

-- | alias of c_range_ with unused argument (for CTHState) to unify backpack signatures.
c_range :: Ptr C'THState -> Ptr C'THIntTensor -> CLong -> CLong -> CLong -> IO ()
c_range = const c_range_

-- | c_randperm :  r_ _generator n -> void
foreign import ccall "THTensorMath.h THIntTensor_randperm"
  c_randperm_ :: Ptr C'THIntTensor -> Ptr C'THGenerator -> CLLong -> IO ()

-- | alias of c_randperm_ with unused argument (for CTHState) to unify backpack signatures.
c_randperm :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THGenerator -> CLLong -> IO ()
c_randperm = const c_randperm_

-- | c_reshape :  r_ t size -> void
foreign import ccall "THTensorMath.h THIntTensor_reshape"
  c_reshape_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THLongStorage -> IO ()

-- | alias of c_reshape_ with unused argument (for CTHState) to unify backpack signatures.
c_reshape :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THLongStorage -> IO ()
c_reshape = const c_reshape_

-- | c_sort :  rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h THIntTensor_sort"
  c_sort_ :: Ptr C'THIntTensor -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ()

-- | alias of c_sort_ with unused argument (for CTHState) to unify backpack signatures.
c_sort :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ()
c_sort = const c_sort_

-- | c_topk :  rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h THIntTensor_topk"
  c_topk_ :: Ptr C'THIntTensor -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> CLLong -> CInt -> CInt -> CInt -> IO ()

-- | alias of c_topk_ with unused argument (for CTHState) to unify backpack signatures.
c_topk :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> CLLong -> CInt -> CInt -> CInt -> IO ()
c_topk = const c_topk_

-- | c_tril :  r_ t k -> void
foreign import ccall "THTensorMath.h THIntTensor_tril"
  c_tril_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CLLong -> IO ()

-- | alias of c_tril_ with unused argument (for CTHState) to unify backpack signatures.
c_tril :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CLLong -> IO ()
c_tril = const c_tril_

-- | c_triu :  r_ t k -> void
foreign import ccall "THTensorMath.h THIntTensor_triu"
  c_triu_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CLLong -> IO ()

-- | alias of c_triu_ with unused argument (for CTHState) to unify backpack signatures.
c_triu :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CLLong -> IO ()
c_triu = const c_triu_

-- | c_cat :  r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h THIntTensor_cat"
  c_cat_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_cat_ with unused argument (for CTHState) to unify backpack signatures.
c_cat :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_cat = const c_cat_

-- | c_catArray :  result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h THIntTensor_catArray"
  c_catArray_ :: Ptr C'THIntTensor -> Ptr (Ptr C'THIntTensor) -> CInt -> CInt -> IO ()

-- | alias of c_catArray_ with unused argument (for CTHState) to unify backpack signatures.
c_catArray :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr (Ptr C'THIntTensor) -> CInt -> CInt -> IO ()
c_catArray = const c_catArray_

-- | c_equal :  ta tb -> int
foreign import ccall "THTensorMath.h THIntTensor_equal"
  c_equal_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO CInt

-- | alias of c_equal_ with unused argument (for CTHState) to unify backpack signatures.
c_equal :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO CInt
c_equal = const c_equal_

-- | c_ltValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_ltValue"
  c_ltValue_ :: Ptr C'THByteTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_ltValue_ with unused argument (for CTHState) to unify backpack signatures.
c_ltValue :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_ltValue = const c_ltValue_

-- | c_leValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_leValue"
  c_leValue_ :: Ptr C'THByteTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_leValue_ with unused argument (for CTHState) to unify backpack signatures.
c_leValue :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_leValue = const c_leValue_

-- | c_gtValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_gtValue"
  c_gtValue_ :: Ptr C'THByteTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_gtValue_ with unused argument (for CTHState) to unify backpack signatures.
c_gtValue :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_gtValue = const c_gtValue_

-- | c_geValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_geValue"
  c_geValue_ :: Ptr C'THByteTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_geValue_ with unused argument (for CTHState) to unify backpack signatures.
c_geValue :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_geValue = const c_geValue_

-- | c_neValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_neValue"
  c_neValue_ :: Ptr C'THByteTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_neValue_ with unused argument (for CTHState) to unify backpack signatures.
c_neValue :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_neValue = const c_neValue_

-- | c_eqValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_eqValue"
  c_eqValue_ :: Ptr C'THByteTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_eqValue_ with unused argument (for CTHState) to unify backpack signatures.
c_eqValue :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_eqValue = const c_eqValue_

-- | c_ltValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_ltValueT"
  c_ltValueT_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_ltValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_ltValueT :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_ltValueT = const c_ltValueT_

-- | c_leValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_leValueT"
  c_leValueT_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_leValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_leValueT :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_leValueT = const c_leValueT_

-- | c_gtValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_gtValueT"
  c_gtValueT_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_gtValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_gtValueT :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_gtValueT = const c_gtValueT_

-- | c_geValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_geValueT"
  c_geValueT_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_geValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_geValueT :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_geValueT = const c_geValueT_

-- | c_neValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_neValueT"
  c_neValueT_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_neValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_neValueT :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_neValueT = const c_neValueT_

-- | c_eqValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_eqValueT"
  c_eqValueT_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()

-- | alias of c_eqValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_eqValueT :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ()
c_eqValueT = const c_eqValueT_

-- | c_ltTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_ltTensor"
  c_ltTensor_ :: Ptr C'THByteTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_ltTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_ltTensor :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_ltTensor = const c_ltTensor_

-- | c_leTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_leTensor"
  c_leTensor_ :: Ptr C'THByteTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_leTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_leTensor :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_leTensor = const c_leTensor_

-- | c_gtTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_gtTensor"
  c_gtTensor_ :: Ptr C'THByteTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_gtTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_gtTensor :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_gtTensor = const c_gtTensor_

-- | c_geTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_geTensor"
  c_geTensor_ :: Ptr C'THByteTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_geTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_geTensor :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_geTensor = const c_geTensor_

-- | c_neTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_neTensor"
  c_neTensor_ :: Ptr C'THByteTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_neTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_neTensor :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_neTensor = const c_neTensor_

-- | c_eqTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_eqTensor"
  c_eqTensor_ :: Ptr C'THByteTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_eqTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_eqTensor :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_eqTensor = const c_eqTensor_

-- | c_ltTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_ltTensorT"
  c_ltTensorT_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_ltTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_ltTensorT :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_ltTensorT = const c_ltTensorT_

-- | c_leTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_leTensorT"
  c_leTensorT_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_leTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_leTensorT :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_leTensorT = const c_leTensorT_

-- | c_gtTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_gtTensorT"
  c_gtTensorT_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_gtTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_gtTensorT :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_gtTensorT = const c_gtTensorT_

-- | c_geTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_geTensorT"
  c_geTensorT_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_geTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_geTensorT :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_geTensorT = const c_geTensorT_

-- | c_neTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_neTensorT"
  c_neTensorT_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_neTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_neTensorT :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_neTensorT = const c_neTensorT_

-- | c_eqTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_eqTensorT"
  c_eqTensorT_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_eqTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_eqTensorT :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_eqTensorT = const c_eqTensorT_

-- | c_abs :  r_ t -> void
foreign import ccall "THTensorMath.h THIntTensor_abs"
  c_abs_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_abs_ with unused argument (for CTHState) to unify backpack signatures.
c_abs :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ()
c_abs = const c_abs_

-- | p_fill : Pointer to function : r_ value -> void
foreign import ccall "THTensorMath.h &THIntTensor_fill"
  p_fill :: FunPtr (Ptr C'THIntTensor -> CInt -> IO ())

-- | p_zero : Pointer to function : r_ -> void
foreign import ccall "THTensorMath.h &THIntTensor_zero"
  p_zero :: FunPtr (Ptr C'THIntTensor -> IO ())

-- | p_maskedFill : Pointer to function : tensor mask value -> void
foreign import ccall "THTensorMath.h &THIntTensor_maskedFill"
  p_maskedFill :: FunPtr (Ptr C'THIntTensor -> Ptr C'THByteTensor -> CInt -> IO ())

-- | p_maskedCopy : Pointer to function : tensor mask src -> void
foreign import ccall "THTensorMath.h &THIntTensor_maskedCopy"
  p_maskedCopy :: FunPtr (Ptr C'THIntTensor -> Ptr C'THByteTensor -> Ptr C'THIntTensor -> IO ())

-- | p_maskedSelect : Pointer to function : tensor src mask -> void
foreign import ccall "THTensorMath.h &THIntTensor_maskedSelect"
  p_maskedSelect :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THByteTensor -> IO ())

-- | p_nonzero : Pointer to function : subscript tensor -> void
foreign import ccall "THTensorMath.h &THIntTensor_nonzero"
  p_nonzero :: FunPtr (Ptr C'THLongTensor -> Ptr C'THIntTensor -> IO ())

-- | p_indexSelect : Pointer to function : tensor src dim index -> void
foreign import ccall "THTensorMath.h &THIntTensor_indexSelect"
  p_indexSelect :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> Ptr C'THLongTensor -> IO ())

-- | p_indexCopy : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THIntTensor_indexCopy"
  p_indexCopy :: FunPtr (Ptr C'THIntTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> IO ())

-- | p_indexAdd : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THIntTensor_indexAdd"
  p_indexAdd :: FunPtr (Ptr C'THIntTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> IO ())

-- | p_indexFill : Pointer to function : tensor dim index val -> void
foreign import ccall "THTensorMath.h &THIntTensor_indexFill"
  p_indexFill :: FunPtr (Ptr C'THIntTensor -> CInt -> Ptr C'THLongTensor -> CInt -> IO ())

-- | p_take : Pointer to function : tensor src index -> void
foreign import ccall "THTensorMath.h &THIntTensor_take"
  p_take :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THLongTensor -> IO ())

-- | p_put : Pointer to function : tensor index src accumulate -> void
foreign import ccall "THTensorMath.h &THIntTensor_put"
  p_put :: FunPtr (Ptr C'THIntTensor -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_gather : Pointer to function : tensor src dim index -> void
foreign import ccall "THTensorMath.h &THIntTensor_gather"
  p_gather :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> Ptr C'THLongTensor -> IO ())

-- | p_scatter : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THIntTensor_scatter"
  p_scatter :: FunPtr (Ptr C'THIntTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> IO ())

-- | p_scatterAdd : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THIntTensor_scatterAdd"
  p_scatterAdd :: FunPtr (Ptr C'THIntTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> IO ())

-- | p_scatterFill : Pointer to function : tensor dim index val -> void
foreign import ccall "THTensorMath.h &THIntTensor_scatterFill"
  p_scatterFill :: FunPtr (Ptr C'THIntTensor -> CInt -> Ptr C'THLongTensor -> CInt -> IO ())

-- | p_dot : Pointer to function : t src -> accreal
foreign import ccall "THTensorMath.h &THIntTensor_dot"
  p_dot :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO CLong)

-- | p_minall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THIntTensor_minall"
  p_minall :: FunPtr (Ptr C'THIntTensor -> IO CInt)

-- | p_maxall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THIntTensor_maxall"
  p_maxall :: FunPtr (Ptr C'THIntTensor -> IO CInt)

-- | p_medianall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THIntTensor_medianall"
  p_medianall :: FunPtr (Ptr C'THIntTensor -> IO CInt)

-- | p_sumall : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THIntTensor_sumall"
  p_sumall :: FunPtr (Ptr C'THIntTensor -> IO CLong)

-- | p_prodall : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THIntTensor_prodall"
  p_prodall :: FunPtr (Ptr C'THIntTensor -> IO CLong)

-- | p_neg : Pointer to function : self src -> void
foreign import ccall "THTensorMath.h &THIntTensor_neg"
  p_neg :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_add : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_add"
  p_add :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_sub : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_sub"
  p_sub :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_add_scaled : Pointer to function : r_ t value alpha -> void
foreign import ccall "THTensorMath.h &THIntTensor_add_scaled"
  p_add_scaled :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ())

-- | p_sub_scaled : Pointer to function : r_ t value alpha -> void
foreign import ccall "THTensorMath.h &THIntTensor_sub_scaled"
  p_sub_scaled :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ())

-- | p_mul : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_mul"
  p_mul :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_div : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_div"
  p_div :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_lshift : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_lshift"
  p_lshift :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_rshift : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_rshift"
  p_rshift :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_fmod : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_fmod"
  p_fmod :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_remainder : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_remainder"
  p_remainder :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_clamp : Pointer to function : r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h &THIntTensor_clamp"
  p_clamp :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ())

-- | p_bitand : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_bitand"
  p_bitand :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_bitor : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_bitor"
  p_bitor :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_bitxor : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_bitxor"
  p_bitxor :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_cadd : Pointer to function : r_ t value src -> void
foreign import ccall "THTensorMath.h &THIntTensor_cadd"
  p_cadd :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> IO ())

-- | p_csub : Pointer to function : self src1 value src2 -> void
foreign import ccall "THTensorMath.h &THIntTensor_csub"
  p_csub :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> IO ())

-- | p_cmul : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THIntTensor_cmul"
  p_cmul :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_cpow : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THIntTensor_cpow"
  p_cpow :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_cdiv : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THIntTensor_cdiv"
  p_cdiv :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_clshift : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THIntTensor_clshift"
  p_clshift :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_crshift : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THIntTensor_crshift"
  p_crshift :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_cfmod : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THIntTensor_cfmod"
  p_cfmod :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_cremainder : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THIntTensor_cremainder"
  p_cremainder :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_cbitand : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THIntTensor_cbitand"
  p_cbitand :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_cbitor : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THIntTensor_cbitor"
  p_cbitor :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_cbitxor : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THIntTensor_cbitxor"
  p_cbitxor :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_addcmul : Pointer to function : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h &THIntTensor_addcmul"
  p_addcmul :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_addcdiv : Pointer to function : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h &THIntTensor_addcdiv"
  p_addcdiv :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_addmv : Pointer to function : r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h &THIntTensor_addmv"
  p_addmv :: FunPtr (Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_addmm : Pointer to function : r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h &THIntTensor_addmm"
  p_addmm :: FunPtr (Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_addr : Pointer to function : r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h &THIntTensor_addr"
  p_addr :: FunPtr (Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_addbmm : Pointer to function : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h &THIntTensor_addbmm"
  p_addbmm :: FunPtr (Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_baddbmm : Pointer to function : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h &THIntTensor_baddbmm"
  p_baddbmm :: FunPtr (Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> CInt -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_match : Pointer to function : r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h &THIntTensor_match"
  p_match :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_numel : Pointer to function : t -> ptrdiff_t
foreign import ccall "THTensorMath.h &THIntTensor_numel"
  p_numel :: FunPtr (Ptr C'THIntTensor -> IO CPtrdiff)

-- | p_preserveReduceDimSemantics : Pointer to function : r_ in_dims reduce_dimension keepdim -> void
foreign import ccall "THTensorMath.h &THIntTensor_preserveReduceDimSemantics"
  p_preserveReduceDimSemantics :: FunPtr (Ptr C'THIntTensor -> CInt -> CInt -> CInt -> IO ())

-- | p_max : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THIntTensor_max"
  p_max :: FunPtr (Ptr C'THIntTensor -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ())

-- | p_min : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THIntTensor_min"
  p_min :: FunPtr (Ptr C'THIntTensor -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ())

-- | p_kthvalue : Pointer to function : values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h &THIntTensor_kthvalue"
  p_kthvalue :: FunPtr (Ptr C'THIntTensor -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> CLLong -> CInt -> CInt -> IO ())

-- | p_mode : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THIntTensor_mode"
  p_mode :: FunPtr (Ptr C'THIntTensor -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ())

-- | p_median : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THIntTensor_median"
  p_median :: FunPtr (Ptr C'THIntTensor -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ())

-- | p_sum : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THIntTensor_sum"
  p_sum :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ())

-- | p_prod : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THIntTensor_prod"
  p_prod :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ())

-- | p_cumsum : Pointer to function : r_ t dimension -> void
foreign import ccall "THTensorMath.h &THIntTensor_cumsum"
  p_cumsum :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_cumprod : Pointer to function : r_ t dimension -> void
foreign import ccall "THTensorMath.h &THIntTensor_cumprod"
  p_cumprod :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_sign : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THIntTensor_sign"
  p_sign :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_trace : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THIntTensor_trace"
  p_trace :: FunPtr (Ptr C'THIntTensor -> IO CLong)

-- | p_cross : Pointer to function : r_ a b dimension -> void
foreign import ccall "THTensorMath.h &THIntTensor_cross"
  p_cross :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_cmax : Pointer to function : r t src -> void
foreign import ccall "THTensorMath.h &THIntTensor_cmax"
  p_cmax :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_cmin : Pointer to function : r t src -> void
foreign import ccall "THTensorMath.h &THIntTensor_cmin"
  p_cmin :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_cmaxValue : Pointer to function : r t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_cmaxValue"
  p_cmaxValue :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_cminValue : Pointer to function : r t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_cminValue"
  p_cminValue :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_zeros : Pointer to function : r_ size -> void
foreign import ccall "THTensorMath.h &THIntTensor_zeros"
  p_zeros :: FunPtr (Ptr C'THIntTensor -> Ptr C'THLongStorage -> IO ())

-- | p_zerosLike : Pointer to function : r_ input -> void
foreign import ccall "THTensorMath.h &THIntTensor_zerosLike"
  p_zerosLike :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_ones : Pointer to function : r_ size -> void
foreign import ccall "THTensorMath.h &THIntTensor_ones"
  p_ones :: FunPtr (Ptr C'THIntTensor -> Ptr C'THLongStorage -> IO ())

-- | p_onesLike : Pointer to function : r_ input -> void
foreign import ccall "THTensorMath.h &THIntTensor_onesLike"
  p_onesLike :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_diag : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THIntTensor_diag"
  p_diag :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_eye : Pointer to function : r_ n m -> void
foreign import ccall "THTensorMath.h &THIntTensor_eye"
  p_eye :: FunPtr (Ptr C'THIntTensor -> CLLong -> CLLong -> IO ())

-- | p_arange : Pointer to function : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h &THIntTensor_arange"
  p_arange :: FunPtr (Ptr C'THIntTensor -> CLong -> CLong -> CLong -> IO ())

-- | p_range : Pointer to function : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h &THIntTensor_range"
  p_range :: FunPtr (Ptr C'THIntTensor -> CLong -> CLong -> CLong -> IO ())

-- | p_randperm : Pointer to function : r_ _generator n -> void
foreign import ccall "THTensorMath.h &THIntTensor_randperm"
  p_randperm :: FunPtr (Ptr C'THIntTensor -> Ptr C'THGenerator -> CLLong -> IO ())

-- | p_reshape : Pointer to function : r_ t size -> void
foreign import ccall "THTensorMath.h &THIntTensor_reshape"
  p_reshape :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THLongStorage -> IO ())

-- | p_sort : Pointer to function : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h &THIntTensor_sort"
  p_sort :: FunPtr (Ptr C'THIntTensor -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> CInt -> CInt -> IO ())

-- | p_topk : Pointer to function : rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h &THIntTensor_topk"
  p_topk :: FunPtr (Ptr C'THIntTensor -> Ptr C'THLongTensor -> Ptr C'THIntTensor -> CLLong -> CInt -> CInt -> CInt -> IO ())

-- | p_tril : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THIntTensor_tril"
  p_tril :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CLLong -> IO ())

-- | p_triu : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THIntTensor_triu"
  p_triu :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CLLong -> IO ())

-- | p_cat : Pointer to function : r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h &THIntTensor_cat"
  p_cat :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_catArray : Pointer to function : result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h &THIntTensor_catArray"
  p_catArray :: FunPtr (Ptr C'THIntTensor -> Ptr (Ptr C'THIntTensor) -> CInt -> CInt -> IO ())

-- | p_equal : Pointer to function : ta tb -> int
foreign import ccall "THTensorMath.h &THIntTensor_equal"
  p_equal :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO CInt)

-- | p_ltValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_ltValue"
  p_ltValue :: FunPtr (Ptr C'THByteTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_leValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_leValue"
  p_leValue :: FunPtr (Ptr C'THByteTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_gtValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_gtValue"
  p_gtValue :: FunPtr (Ptr C'THByteTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_geValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_geValue"
  p_geValue :: FunPtr (Ptr C'THByteTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_neValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_neValue"
  p_neValue :: FunPtr (Ptr C'THByteTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_eqValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_eqValue"
  p_eqValue :: FunPtr (Ptr C'THByteTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_ltValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_ltValueT"
  p_ltValueT :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_leValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_leValueT"
  p_leValueT :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_gtValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_gtValueT"
  p_gtValueT :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_geValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_geValueT"
  p_geValueT :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_neValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_neValueT"
  p_neValueT :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_eqValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THIntTensor_eqValueT"
  p_eqValueT :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> IO ())

-- | p_ltTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THIntTensor_ltTensor"
  p_ltTensor :: FunPtr (Ptr C'THByteTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_leTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THIntTensor_leTensor"
  p_leTensor :: FunPtr (Ptr C'THByteTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_gtTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THIntTensor_gtTensor"
  p_gtTensor :: FunPtr (Ptr C'THByteTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_geTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THIntTensor_geTensor"
  p_geTensor :: FunPtr (Ptr C'THByteTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_neTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THIntTensor_neTensor"
  p_neTensor :: FunPtr (Ptr C'THByteTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_eqTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THIntTensor_eqTensor"
  p_eqTensor :: FunPtr (Ptr C'THByteTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_ltTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THIntTensor_ltTensorT"
  p_ltTensorT :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_leTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THIntTensor_leTensorT"
  p_leTensorT :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_gtTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THIntTensor_gtTensorT"
  p_gtTensorT :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_geTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THIntTensor_geTensorT"
  p_geTensorT :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_neTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THIntTensor_neTensorT"
  p_neTensorT :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_eqTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THIntTensor_eqTensorT"
  p_eqTensorT :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_abs : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THIntTensor_abs"
  p_abs :: FunPtr (Ptr C'THIntTensor -> Ptr C'THIntTensor -> IO ())