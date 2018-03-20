{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Long.TensorMath where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_fill :  r_ value -> void
foreign import ccall "THTensorMath.h THLongTensor_fill"
  c_fill_ :: Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_fill_ with unused argument (for CTHState) to unify backpack signatures.
c_fill = const c_fill_

-- | c_zero :  r_ -> void
foreign import ccall "THTensorMath.h THLongTensor_zero"
  c_zero_ :: Ptr C'THLongTensor -> IO ()

-- | alias of c_zero_ with unused argument (for CTHState) to unify backpack signatures.
c_zero = const c_zero_

-- | c_maskedFill :  tensor mask value -> void
foreign import ccall "THTensorMath.h THLongTensor_maskedFill"
  c_maskedFill_ :: Ptr C'THLongTensor -> Ptr C'THByteTensor -> CLong -> IO ()

-- | alias of c_maskedFill_ with unused argument (for CTHState) to unify backpack signatures.
c_maskedFill = const c_maskedFill_

-- | c_maskedCopy :  tensor mask src -> void
foreign import ccall "THTensorMath.h THLongTensor_maskedCopy"
  c_maskedCopy_ :: Ptr C'THLongTensor -> Ptr C'THByteTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_maskedCopy_ with unused argument (for CTHState) to unify backpack signatures.
c_maskedCopy = const c_maskedCopy_

-- | c_maskedSelect :  tensor src mask -> void
foreign import ccall "THTensorMath.h THLongTensor_maskedSelect"
  c_maskedSelect_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_maskedSelect_ with unused argument (for CTHState) to unify backpack signatures.
c_maskedSelect = const c_maskedSelect_

-- | c_nonzero :  subscript tensor -> void
foreign import ccall "THTensorMath.h THLongTensor_nonzero"
  c_nonzero_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_nonzero_ with unused argument (for CTHState) to unify backpack signatures.
c_nonzero = const c_nonzero_

-- | c_indexSelect :  tensor src dim index -> void
foreign import ccall "THTensorMath.h THLongTensor_indexSelect"
  c_indexSelect_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> Ptr C'THLongTensor -> IO ()

-- | alias of c_indexSelect_ with unused argument (for CTHState) to unify backpack signatures.
c_indexSelect = const c_indexSelect_

-- | c_indexCopy :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THLongTensor_indexCopy"
  c_indexCopy_ :: Ptr C'THLongTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_indexCopy_ with unused argument (for CTHState) to unify backpack signatures.
c_indexCopy = const c_indexCopy_

-- | c_indexAdd :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THLongTensor_indexAdd"
  c_indexAdd_ :: Ptr C'THLongTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_indexAdd_ with unused argument (for CTHState) to unify backpack signatures.
c_indexAdd = const c_indexAdd_

-- | c_indexFill :  tensor dim index val -> void
foreign import ccall "THTensorMath.h THLongTensor_indexFill"
  c_indexFill_ :: Ptr C'THLongTensor -> CInt -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_indexFill_ with unused argument (for CTHState) to unify backpack signatures.
c_indexFill = const c_indexFill_

-- | c_take :  tensor src index -> void
foreign import ccall "THTensorMath.h THLongTensor_take"
  c_take_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_take_ with unused argument (for CTHState) to unify backpack signatures.
c_take = const c_take_

-- | c_put :  tensor index src accumulate -> void
foreign import ccall "THTensorMath.h THLongTensor_put"
  c_put_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> IO ()

-- | alias of c_put_ with unused argument (for CTHState) to unify backpack signatures.
c_put = const c_put_

-- | c_gather :  tensor src dim index -> void
foreign import ccall "THTensorMath.h THLongTensor_gather"
  c_gather_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> Ptr C'THLongTensor -> IO ()

-- | alias of c_gather_ with unused argument (for CTHState) to unify backpack signatures.
c_gather = const c_gather_

-- | c_scatter :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THLongTensor_scatter"
  c_scatter_ :: Ptr C'THLongTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_scatter_ with unused argument (for CTHState) to unify backpack signatures.
c_scatter = const c_scatter_

-- | c_scatterAdd :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THLongTensor_scatterAdd"
  c_scatterAdd_ :: Ptr C'THLongTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_scatterAdd_ with unused argument (for CTHState) to unify backpack signatures.
c_scatterAdd = const c_scatterAdd_

-- | c_scatterFill :  tensor dim index val -> void
foreign import ccall "THTensorMath.h THLongTensor_scatterFill"
  c_scatterFill_ :: Ptr C'THLongTensor -> CInt -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_scatterFill_ with unused argument (for CTHState) to unify backpack signatures.
c_scatterFill = const c_scatterFill_

-- | c_dot :  t src -> accreal
foreign import ccall "THTensorMath.h THLongTensor_dot"
  c_dot_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO CLong

-- | alias of c_dot_ with unused argument (for CTHState) to unify backpack signatures.
c_dot = const c_dot_

-- | c_minall :  t -> real
foreign import ccall "THTensorMath.h THLongTensor_minall"
  c_minall_ :: Ptr C'THLongTensor -> IO CLong

-- | alias of c_minall_ with unused argument (for CTHState) to unify backpack signatures.
c_minall = const c_minall_

-- | c_maxall :  t -> real
foreign import ccall "THTensorMath.h THLongTensor_maxall"
  c_maxall_ :: Ptr C'THLongTensor -> IO CLong

-- | alias of c_maxall_ with unused argument (for CTHState) to unify backpack signatures.
c_maxall = const c_maxall_

-- | c_medianall :  t -> real
foreign import ccall "THTensorMath.h THLongTensor_medianall"
  c_medianall_ :: Ptr C'THLongTensor -> IO CLong

-- | alias of c_medianall_ with unused argument (for CTHState) to unify backpack signatures.
c_medianall = const c_medianall_

-- | c_sumall :  t -> accreal
foreign import ccall "THTensorMath.h THLongTensor_sumall"
  c_sumall_ :: Ptr C'THLongTensor -> IO CLong

-- | alias of c_sumall_ with unused argument (for CTHState) to unify backpack signatures.
c_sumall = const c_sumall_

-- | c_prodall :  t -> accreal
foreign import ccall "THTensorMath.h THLongTensor_prodall"
  c_prodall_ :: Ptr C'THLongTensor -> IO CLong

-- | alias of c_prodall_ with unused argument (for CTHState) to unify backpack signatures.
c_prodall = const c_prodall_

-- | c_neg :  self src -> void
foreign import ccall "THTensorMath.h THLongTensor_neg"
  c_neg_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_neg_ with unused argument (for CTHState) to unify backpack signatures.
c_neg = const c_neg_

-- | c_add :  r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_add"
  c_add_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_add_ with unused argument (for CTHState) to unify backpack signatures.
c_add = const c_add_

-- | c_sub :  r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_sub"
  c_sub_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_sub_ with unused argument (for CTHState) to unify backpack signatures.
c_sub = const c_sub_

-- | c_add_scaled :  r_ t value alpha -> void
foreign import ccall "THTensorMath.h THLongTensor_add_scaled"
  c_add_scaled_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> CLong -> IO ()

-- | alias of c_add_scaled_ with unused argument (for CTHState) to unify backpack signatures.
c_add_scaled = const c_add_scaled_

-- | c_sub_scaled :  r_ t value alpha -> void
foreign import ccall "THTensorMath.h THLongTensor_sub_scaled"
  c_sub_scaled_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> CLong -> IO ()

-- | alias of c_sub_scaled_ with unused argument (for CTHState) to unify backpack signatures.
c_sub_scaled = const c_sub_scaled_

-- | c_mul :  r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_mul"
  c_mul_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_mul_ with unused argument (for CTHState) to unify backpack signatures.
c_mul = const c_mul_

-- | c_div :  r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_div"
  c_div_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_div_ with unused argument (for CTHState) to unify backpack signatures.
c_div = const c_div_

-- | c_lshift :  r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_lshift"
  c_lshift_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_lshift_ with unused argument (for CTHState) to unify backpack signatures.
c_lshift = const c_lshift_

-- | c_rshift :  r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_rshift"
  c_rshift_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_rshift_ with unused argument (for CTHState) to unify backpack signatures.
c_rshift = const c_rshift_

-- | c_fmod :  r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_fmod"
  c_fmod_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_fmod_ with unused argument (for CTHState) to unify backpack signatures.
c_fmod = const c_fmod_

-- | c_remainder :  r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_remainder"
  c_remainder_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_remainder_ with unused argument (for CTHState) to unify backpack signatures.
c_remainder = const c_remainder_

-- | c_clamp :  r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h THLongTensor_clamp"
  c_clamp_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> CLong -> IO ()

-- | alias of c_clamp_ with unused argument (for CTHState) to unify backpack signatures.
c_clamp = const c_clamp_

-- | c_bitand :  r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_bitand"
  c_bitand_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_bitand_ with unused argument (for CTHState) to unify backpack signatures.
c_bitand = const c_bitand_

-- | c_bitor :  r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_bitor"
  c_bitor_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_bitor_ with unused argument (for CTHState) to unify backpack signatures.
c_bitor = const c_bitor_

-- | c_bitxor :  r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_bitxor"
  c_bitxor_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_bitxor_ with unused argument (for CTHState) to unify backpack signatures.
c_bitxor = const c_bitxor_

-- | c_cadd :  r_ t value src -> void
foreign import ccall "THTensorMath.h THLongTensor_cadd"
  c_cadd_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> IO ()

-- | alias of c_cadd_ with unused argument (for CTHState) to unify backpack signatures.
c_cadd = const c_cadd_

-- | c_csub :  self src1 value src2 -> void
foreign import ccall "THTensorMath.h THLongTensor_csub"
  c_csub_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> IO ()

-- | alias of c_csub_ with unused argument (for CTHState) to unify backpack signatures.
c_csub = const c_csub_

-- | c_cmul :  r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensor_cmul"
  c_cmul_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_cmul_ with unused argument (for CTHState) to unify backpack signatures.
c_cmul = const c_cmul_

-- | c_cpow :  r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensor_cpow"
  c_cpow_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_cpow_ with unused argument (for CTHState) to unify backpack signatures.
c_cpow = const c_cpow_

-- | c_cdiv :  r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensor_cdiv"
  c_cdiv_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_cdiv_ with unused argument (for CTHState) to unify backpack signatures.
c_cdiv = const c_cdiv_

-- | c_clshift :  r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensor_clshift"
  c_clshift_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_clshift_ with unused argument (for CTHState) to unify backpack signatures.
c_clshift = const c_clshift_

-- | c_crshift :  r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensor_crshift"
  c_crshift_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_crshift_ with unused argument (for CTHState) to unify backpack signatures.
c_crshift = const c_crshift_

-- | c_cfmod :  r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensor_cfmod"
  c_cfmod_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_cfmod_ with unused argument (for CTHState) to unify backpack signatures.
c_cfmod = const c_cfmod_

-- | c_cremainder :  r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensor_cremainder"
  c_cremainder_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_cremainder_ with unused argument (for CTHState) to unify backpack signatures.
c_cremainder = const c_cremainder_

-- | c_cbitand :  r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensor_cbitand"
  c_cbitand_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_cbitand_ with unused argument (for CTHState) to unify backpack signatures.
c_cbitand = const c_cbitand_

-- | c_cbitor :  r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensor_cbitor"
  c_cbitor_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_cbitor_ with unused argument (for CTHState) to unify backpack signatures.
c_cbitor = const c_cbitor_

-- | c_cbitxor :  r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensor_cbitxor"
  c_cbitxor_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_cbitxor_ with unused argument (for CTHState) to unify backpack signatures.
c_cbitxor = const c_cbitxor_

-- | c_addcmul :  r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THLongTensor_addcmul"
  c_addcmul_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_addcmul_ with unused argument (for CTHState) to unify backpack signatures.
c_addcmul = const c_addcmul_

-- | c_addcdiv :  r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THLongTensor_addcdiv"
  c_addcdiv_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_addcdiv_ with unused argument (for CTHState) to unify backpack signatures.
c_addcdiv = const c_addcdiv_

-- | c_addmv :  r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h THLongTensor_addmv"
  c_addmv_ :: Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_addmv_ with unused argument (for CTHState) to unify backpack signatures.
c_addmv = const c_addmv_

-- | c_addmm :  r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h THLongTensor_addmm"
  c_addmm_ :: Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_addmm_ with unused argument (for CTHState) to unify backpack signatures.
c_addmm = const c_addmm_

-- | c_addr :  r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h THLongTensor_addr"
  c_addr_ :: Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_addr_ with unused argument (for CTHState) to unify backpack signatures.
c_addr = const c_addr_

-- | c_addbmm :  r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THLongTensor_addbmm"
  c_addbmm_ :: Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_addbmm_ with unused argument (for CTHState) to unify backpack signatures.
c_addbmm = const c_addbmm_

-- | c_baddbmm :  r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THLongTensor_baddbmm"
  c_baddbmm_ :: Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_baddbmm_ with unused argument (for CTHState) to unify backpack signatures.
c_baddbmm = const c_baddbmm_

-- | c_match :  r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h THLongTensor_match"
  c_match_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_match_ with unused argument (for CTHState) to unify backpack signatures.
c_match = const c_match_

-- | c_numel :  t -> ptrdiff_t
foreign import ccall "THTensorMath.h THLongTensor_numel"
  c_numel_ :: Ptr C'THLongTensor -> IO CPtrdiff

-- | alias of c_numel_ with unused argument (for CTHState) to unify backpack signatures.
c_numel = const c_numel_

-- | c_preserveReduceDimSemantics :  r_ in_dims reduce_dimension keepdim -> void
foreign import ccall "THTensorMath.h THLongTensor_preserveReduceDimSemantics"
  c_preserveReduceDimSemantics_ :: Ptr C'THLongTensor -> CInt -> CInt -> CInt -> IO ()

-- | alias of c_preserveReduceDimSemantics_ with unused argument (for CTHState) to unify backpack signatures.
c_preserveReduceDimSemantics = const c_preserveReduceDimSemantics_

-- | c_max :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THLongTensor_max"
  c_max_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> CInt -> IO ()

-- | alias of c_max_ with unused argument (for CTHState) to unify backpack signatures.
c_max = const c_max_

-- | c_min :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THLongTensor_min"
  c_min_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> CInt -> IO ()

-- | alias of c_min_ with unused argument (for CTHState) to unify backpack signatures.
c_min = const c_min_

-- | c_kthvalue :  values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h THLongTensor_kthvalue"
  c_kthvalue_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLLong -> CInt -> CInt -> IO ()

-- | alias of c_kthvalue_ with unused argument (for CTHState) to unify backpack signatures.
c_kthvalue = const c_kthvalue_

-- | c_mode :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THLongTensor_mode"
  c_mode_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> CInt -> IO ()

-- | alias of c_mode_ with unused argument (for CTHState) to unify backpack signatures.
c_mode = const c_mode_

-- | c_median :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THLongTensor_median"
  c_median_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> CInt -> IO ()

-- | alias of c_median_ with unused argument (for CTHState) to unify backpack signatures.
c_median = const c_median_

-- | c_sum :  r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THLongTensor_sum"
  c_sum_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> CInt -> IO ()

-- | alias of c_sum_ with unused argument (for CTHState) to unify backpack signatures.
c_sum = const c_sum_

-- | c_prod :  r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THLongTensor_prod"
  c_prod_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> CInt -> IO ()

-- | alias of c_prod_ with unused argument (for CTHState) to unify backpack signatures.
c_prod = const c_prod_

-- | c_cumsum :  r_ t dimension -> void
foreign import ccall "THTensorMath.h THLongTensor_cumsum"
  c_cumsum_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> IO ()

-- | alias of c_cumsum_ with unused argument (for CTHState) to unify backpack signatures.
c_cumsum = const c_cumsum_

-- | c_cumprod :  r_ t dimension -> void
foreign import ccall "THTensorMath.h THLongTensor_cumprod"
  c_cumprod_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> IO ()

-- | alias of c_cumprod_ with unused argument (for CTHState) to unify backpack signatures.
c_cumprod = const c_cumprod_

-- | c_sign :  r_ t -> void
foreign import ccall "THTensorMath.h THLongTensor_sign"
  c_sign_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_sign_ with unused argument (for CTHState) to unify backpack signatures.
c_sign = const c_sign_

-- | c_trace :  t -> accreal
foreign import ccall "THTensorMath.h THLongTensor_trace"
  c_trace_ :: Ptr C'THLongTensor -> IO CLong

-- | alias of c_trace_ with unused argument (for CTHState) to unify backpack signatures.
c_trace = const c_trace_

-- | c_cross :  r_ a b dimension -> void
foreign import ccall "THTensorMath.h THLongTensor_cross"
  c_cross_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> IO ()

-- | alias of c_cross_ with unused argument (for CTHState) to unify backpack signatures.
c_cross = const c_cross_

-- | c_cmax :  r t src -> void
foreign import ccall "THTensorMath.h THLongTensor_cmax"
  c_cmax_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_cmax_ with unused argument (for CTHState) to unify backpack signatures.
c_cmax = const c_cmax_

-- | c_cmin :  r t src -> void
foreign import ccall "THTensorMath.h THLongTensor_cmin"
  c_cmin_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_cmin_ with unused argument (for CTHState) to unify backpack signatures.
c_cmin = const c_cmin_

-- | c_cmaxValue :  r t value -> void
foreign import ccall "THTensorMath.h THLongTensor_cmaxValue"
  c_cmaxValue_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_cmaxValue_ with unused argument (for CTHState) to unify backpack signatures.
c_cmaxValue = const c_cmaxValue_

-- | c_cminValue :  r t value -> void
foreign import ccall "THTensorMath.h THLongTensor_cminValue"
  c_cminValue_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_cminValue_ with unused argument (for CTHState) to unify backpack signatures.
c_cminValue = const c_cminValue_

-- | c_zeros :  r_ size -> void
foreign import ccall "THTensorMath.h THLongTensor_zeros"
  c_zeros_ :: Ptr C'THLongTensor -> Ptr C'THLongStorage -> IO ()

-- | alias of c_zeros_ with unused argument (for CTHState) to unify backpack signatures.
c_zeros = const c_zeros_

-- | c_zerosLike :  r_ input -> void
foreign import ccall "THTensorMath.h THLongTensor_zerosLike"
  c_zerosLike_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_zerosLike_ with unused argument (for CTHState) to unify backpack signatures.
c_zerosLike = const c_zerosLike_

-- | c_ones :  r_ size -> void
foreign import ccall "THTensorMath.h THLongTensor_ones"
  c_ones_ :: Ptr C'THLongTensor -> Ptr C'THLongStorage -> IO ()

-- | alias of c_ones_ with unused argument (for CTHState) to unify backpack signatures.
c_ones = const c_ones_

-- | c_onesLike :  r_ input -> void
foreign import ccall "THTensorMath.h THLongTensor_onesLike"
  c_onesLike_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_onesLike_ with unused argument (for CTHState) to unify backpack signatures.
c_onesLike = const c_onesLike_

-- | c_diag :  r_ t k -> void
foreign import ccall "THTensorMath.h THLongTensor_diag"
  c_diag_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> IO ()

-- | alias of c_diag_ with unused argument (for CTHState) to unify backpack signatures.
c_diag = const c_diag_

-- | c_eye :  r_ n m -> void
foreign import ccall "THTensorMath.h THLongTensor_eye"
  c_eye_ :: Ptr C'THLongTensor -> CLLong -> CLLong -> IO ()

-- | alias of c_eye_ with unused argument (for CTHState) to unify backpack signatures.
c_eye = const c_eye_

-- | c_arange :  r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THLongTensor_arange"
  c_arange_ :: Ptr C'THLongTensor -> CLong -> CLong -> CLong -> IO ()

-- | alias of c_arange_ with unused argument (for CTHState) to unify backpack signatures.
c_arange = const c_arange_

-- | c_range :  r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THLongTensor_range"
  c_range_ :: Ptr C'THLongTensor -> CLong -> CLong -> CLong -> IO ()

-- | alias of c_range_ with unused argument (for CTHState) to unify backpack signatures.
c_range = const c_range_

-- | c_randperm :  r_ _generator n -> void
foreign import ccall "THTensorMath.h THLongTensor_randperm"
  c_randperm_ :: Ptr C'THLongTensor -> Ptr C'THGenerator -> CLLong -> IO ()

-- | alias of c_randperm_ with unused argument (for CTHState) to unify backpack signatures.
c_randperm = const c_randperm_

-- | c_reshape :  r_ t size -> void
foreign import ccall "THTensorMath.h THLongTensor_reshape"
  c_reshape_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongStorage -> IO ()

-- | alias of c_reshape_ with unused argument (for CTHState) to unify backpack signatures.
c_reshape = const c_reshape_

-- | c_sort :  rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h THLongTensor_sort"
  c_sort_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> CInt -> IO ()

-- | alias of c_sort_ with unused argument (for CTHState) to unify backpack signatures.
c_sort = const c_sort_

-- | c_topk :  rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h THLongTensor_topk"
  c_topk_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLLong -> CInt -> CInt -> CInt -> IO ()

-- | alias of c_topk_ with unused argument (for CTHState) to unify backpack signatures.
c_topk = const c_topk_

-- | c_tril :  r_ t k -> void
foreign import ccall "THTensorMath.h THLongTensor_tril"
  c_tril_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLLong -> IO ()

-- | alias of c_tril_ with unused argument (for CTHState) to unify backpack signatures.
c_tril = const c_tril_

-- | c_triu :  r_ t k -> void
foreign import ccall "THTensorMath.h THLongTensor_triu"
  c_triu_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLLong -> IO ()

-- | alias of c_triu_ with unused argument (for CTHState) to unify backpack signatures.
c_triu = const c_triu_

-- | c_cat :  r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h THLongTensor_cat"
  c_cat_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> IO ()

-- | alias of c_cat_ with unused argument (for CTHState) to unify backpack signatures.
c_cat = const c_cat_

-- | c_catArray :  result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h THLongTensor_catArray"
  c_catArray_ :: Ptr C'THLongTensor -> Ptr (Ptr C'THLongTensor) -> CInt -> CInt -> IO ()

-- | alias of c_catArray_ with unused argument (for CTHState) to unify backpack signatures.
c_catArray = const c_catArray_

-- | c_equal :  ta tb -> int
foreign import ccall "THTensorMath.h THLongTensor_equal"
  c_equal_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO CInt

-- | alias of c_equal_ with unused argument (for CTHState) to unify backpack signatures.
c_equal = const c_equal_

-- | c_ltValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_ltValue"
  c_ltValue_ :: Ptr C'THByteTensor -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_ltValue_ with unused argument (for CTHState) to unify backpack signatures.
c_ltValue = const c_ltValue_

-- | c_leValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_leValue"
  c_leValue_ :: Ptr C'THByteTensor -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_leValue_ with unused argument (for CTHState) to unify backpack signatures.
c_leValue = const c_leValue_

-- | c_gtValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_gtValue"
  c_gtValue_ :: Ptr C'THByteTensor -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_gtValue_ with unused argument (for CTHState) to unify backpack signatures.
c_gtValue = const c_gtValue_

-- | c_geValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_geValue"
  c_geValue_ :: Ptr C'THByteTensor -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_geValue_ with unused argument (for CTHState) to unify backpack signatures.
c_geValue = const c_geValue_

-- | c_neValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_neValue"
  c_neValue_ :: Ptr C'THByteTensor -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_neValue_ with unused argument (for CTHState) to unify backpack signatures.
c_neValue = const c_neValue_

-- | c_eqValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_eqValue"
  c_eqValue_ :: Ptr C'THByteTensor -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_eqValue_ with unused argument (for CTHState) to unify backpack signatures.
c_eqValue = const c_eqValue_

-- | c_ltValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_ltValueT"
  c_ltValueT_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_ltValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_ltValueT = const c_ltValueT_

-- | c_leValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_leValueT"
  c_leValueT_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_leValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_leValueT = const c_leValueT_

-- | c_gtValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_gtValueT"
  c_gtValueT_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_gtValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_gtValueT = const c_gtValueT_

-- | c_geValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_geValueT"
  c_geValueT_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_geValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_geValueT = const c_geValueT_

-- | c_neValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_neValueT"
  c_neValueT_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_neValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_neValueT = const c_neValueT_

-- | c_eqValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_eqValueT"
  c_eqValueT_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ()

-- | alias of c_eqValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_eqValueT = const c_eqValueT_

-- | c_ltTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensor_ltTensor"
  c_ltTensor_ :: Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_ltTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_ltTensor = const c_ltTensor_

-- | c_leTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensor_leTensor"
  c_leTensor_ :: Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_leTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_leTensor = const c_leTensor_

-- | c_gtTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensor_gtTensor"
  c_gtTensor_ :: Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_gtTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_gtTensor = const c_gtTensor_

-- | c_geTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensor_geTensor"
  c_geTensor_ :: Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_geTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_geTensor = const c_geTensor_

-- | c_neTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensor_neTensor"
  c_neTensor_ :: Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_neTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_neTensor = const c_neTensor_

-- | c_eqTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensor_eqTensor"
  c_eqTensor_ :: Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_eqTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_eqTensor = const c_eqTensor_

-- | c_ltTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensor_ltTensorT"
  c_ltTensorT_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_ltTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_ltTensorT = const c_ltTensorT_

-- | c_leTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensor_leTensorT"
  c_leTensorT_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_leTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_leTensorT = const c_leTensorT_

-- | c_gtTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensor_gtTensorT"
  c_gtTensorT_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_gtTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_gtTensorT = const c_gtTensorT_

-- | c_geTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensor_geTensorT"
  c_geTensorT_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_geTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_geTensorT = const c_geTensorT_

-- | c_neTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensor_neTensorT"
  c_neTensorT_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_neTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_neTensorT = const c_neTensorT_

-- | c_eqTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensor_eqTensorT"
  c_eqTensorT_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_eqTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_eqTensorT = const c_eqTensorT_

-- | c_abs :  r_ t -> void
foreign import ccall "THTensorMath.h THLongTensor_abs"
  c_abs_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_abs_ with unused argument (for CTHState) to unify backpack signatures.
c_abs = const c_abs_

-- | p_fill : Pointer to function : r_ value -> void
foreign import ccall "THTensorMath.h &THLongTensor_fill"
  p_fill_ :: FunPtr (Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_fill_ with unused argument (for CTHState) to unify backpack signatures.
p_fill = const p_fill_

-- | p_zero : Pointer to function : r_ -> void
foreign import ccall "THTensorMath.h &THLongTensor_zero"
  p_zero_ :: FunPtr (Ptr C'THLongTensor -> IO ())

-- | alias of p_zero_ with unused argument (for CTHState) to unify backpack signatures.
p_zero = const p_zero_

-- | p_maskedFill : Pointer to function : tensor mask value -> void
foreign import ccall "THTensorMath.h &THLongTensor_maskedFill"
  p_maskedFill_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THByteTensor -> CLong -> IO ())

-- | alias of p_maskedFill_ with unused argument (for CTHState) to unify backpack signatures.
p_maskedFill = const p_maskedFill_

-- | p_maskedCopy : Pointer to function : tensor mask src -> void
foreign import ccall "THTensorMath.h &THLongTensor_maskedCopy"
  p_maskedCopy_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THByteTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_maskedCopy_ with unused argument (for CTHState) to unify backpack signatures.
p_maskedCopy = const p_maskedCopy_

-- | p_maskedSelect : Pointer to function : tensor src mask -> void
foreign import ccall "THTensorMath.h &THLongTensor_maskedSelect"
  p_maskedSelect_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THByteTensor -> IO ())

-- | alias of p_maskedSelect_ with unused argument (for CTHState) to unify backpack signatures.
p_maskedSelect = const p_maskedSelect_

-- | p_nonzero : Pointer to function : subscript tensor -> void
foreign import ccall "THTensorMath.h &THLongTensor_nonzero"
  p_nonzero_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_nonzero_ with unused argument (for CTHState) to unify backpack signatures.
p_nonzero = const p_nonzero_

-- | p_indexSelect : Pointer to function : tensor src dim index -> void
foreign import ccall "THTensorMath.h &THLongTensor_indexSelect"
  p_indexSelect_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> Ptr C'THLongTensor -> IO ())

-- | alias of p_indexSelect_ with unused argument (for CTHState) to unify backpack signatures.
p_indexSelect = const p_indexSelect_

-- | p_indexCopy : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THLongTensor_indexCopy"
  p_indexCopy_ :: FunPtr (Ptr C'THLongTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_indexCopy_ with unused argument (for CTHState) to unify backpack signatures.
p_indexCopy = const p_indexCopy_

-- | p_indexAdd : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THLongTensor_indexAdd"
  p_indexAdd_ :: FunPtr (Ptr C'THLongTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_indexAdd_ with unused argument (for CTHState) to unify backpack signatures.
p_indexAdd = const p_indexAdd_

-- | p_indexFill : Pointer to function : tensor dim index val -> void
foreign import ccall "THTensorMath.h &THLongTensor_indexFill"
  p_indexFill_ :: FunPtr (Ptr C'THLongTensor -> CInt -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_indexFill_ with unused argument (for CTHState) to unify backpack signatures.
p_indexFill = const p_indexFill_

-- | p_take : Pointer to function : tensor src index -> void
foreign import ccall "THTensorMath.h &THLongTensor_take"
  p_take_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_take_ with unused argument (for CTHState) to unify backpack signatures.
p_take = const p_take_

-- | p_put : Pointer to function : tensor index src accumulate -> void
foreign import ccall "THTensorMath.h &THLongTensor_put"
  p_put_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> IO ())

-- | alias of p_put_ with unused argument (for CTHState) to unify backpack signatures.
p_put = const p_put_

-- | p_gather : Pointer to function : tensor src dim index -> void
foreign import ccall "THTensorMath.h &THLongTensor_gather"
  p_gather_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> Ptr C'THLongTensor -> IO ())

-- | alias of p_gather_ with unused argument (for CTHState) to unify backpack signatures.
p_gather = const p_gather_

-- | p_scatter : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THLongTensor_scatter"
  p_scatter_ :: FunPtr (Ptr C'THLongTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_scatter_ with unused argument (for CTHState) to unify backpack signatures.
p_scatter = const p_scatter_

-- | p_scatterAdd : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THLongTensor_scatterAdd"
  p_scatterAdd_ :: FunPtr (Ptr C'THLongTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_scatterAdd_ with unused argument (for CTHState) to unify backpack signatures.
p_scatterAdd = const p_scatterAdd_

-- | p_scatterFill : Pointer to function : tensor dim index val -> void
foreign import ccall "THTensorMath.h &THLongTensor_scatterFill"
  p_scatterFill_ :: FunPtr (Ptr C'THLongTensor -> CInt -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_scatterFill_ with unused argument (for CTHState) to unify backpack signatures.
p_scatterFill = const p_scatterFill_

-- | p_dot : Pointer to function : t src -> accreal
foreign import ccall "THTensorMath.h &THLongTensor_dot"
  p_dot_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO CLong)

-- | alias of p_dot_ with unused argument (for CTHState) to unify backpack signatures.
p_dot = const p_dot_

-- | p_minall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THLongTensor_minall"
  p_minall_ :: FunPtr (Ptr C'THLongTensor -> IO CLong)

-- | alias of p_minall_ with unused argument (for CTHState) to unify backpack signatures.
p_minall = const p_minall_

-- | p_maxall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THLongTensor_maxall"
  p_maxall_ :: FunPtr (Ptr C'THLongTensor -> IO CLong)

-- | alias of p_maxall_ with unused argument (for CTHState) to unify backpack signatures.
p_maxall = const p_maxall_

-- | p_medianall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THLongTensor_medianall"
  p_medianall_ :: FunPtr (Ptr C'THLongTensor -> IO CLong)

-- | alias of p_medianall_ with unused argument (for CTHState) to unify backpack signatures.
p_medianall = const p_medianall_

-- | p_sumall : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THLongTensor_sumall"
  p_sumall_ :: FunPtr (Ptr C'THLongTensor -> IO CLong)

-- | alias of p_sumall_ with unused argument (for CTHState) to unify backpack signatures.
p_sumall = const p_sumall_

-- | p_prodall : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THLongTensor_prodall"
  p_prodall_ :: FunPtr (Ptr C'THLongTensor -> IO CLong)

-- | alias of p_prodall_ with unused argument (for CTHState) to unify backpack signatures.
p_prodall = const p_prodall_

-- | p_neg : Pointer to function : self src -> void
foreign import ccall "THTensorMath.h &THLongTensor_neg"
  p_neg_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_neg_ with unused argument (for CTHState) to unify backpack signatures.
p_neg = const p_neg_

-- | p_add : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THLongTensor_add"
  p_add_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_add_ with unused argument (for CTHState) to unify backpack signatures.
p_add = const p_add_

-- | p_sub : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THLongTensor_sub"
  p_sub_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_sub_ with unused argument (for CTHState) to unify backpack signatures.
p_sub = const p_sub_

-- | p_add_scaled : Pointer to function : r_ t value alpha -> void
foreign import ccall "THTensorMath.h &THLongTensor_add_scaled"
  p_add_scaled_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> CLong -> IO ())

-- | alias of p_add_scaled_ with unused argument (for CTHState) to unify backpack signatures.
p_add_scaled = const p_add_scaled_

-- | p_sub_scaled : Pointer to function : r_ t value alpha -> void
foreign import ccall "THTensorMath.h &THLongTensor_sub_scaled"
  p_sub_scaled_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> CLong -> IO ())

-- | alias of p_sub_scaled_ with unused argument (for CTHState) to unify backpack signatures.
p_sub_scaled = const p_sub_scaled_

-- | p_mul : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THLongTensor_mul"
  p_mul_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_mul_ with unused argument (for CTHState) to unify backpack signatures.
p_mul = const p_mul_

-- | p_div : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THLongTensor_div"
  p_div_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_div_ with unused argument (for CTHState) to unify backpack signatures.
p_div = const p_div_

-- | p_lshift : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THLongTensor_lshift"
  p_lshift_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_lshift_ with unused argument (for CTHState) to unify backpack signatures.
p_lshift = const p_lshift_

-- | p_rshift : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THLongTensor_rshift"
  p_rshift_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_rshift_ with unused argument (for CTHState) to unify backpack signatures.
p_rshift = const p_rshift_

-- | p_fmod : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THLongTensor_fmod"
  p_fmod_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_fmod_ with unused argument (for CTHState) to unify backpack signatures.
p_fmod = const p_fmod_

-- | p_remainder : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THLongTensor_remainder"
  p_remainder_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_remainder_ with unused argument (for CTHState) to unify backpack signatures.
p_remainder = const p_remainder_

-- | p_clamp : Pointer to function : r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h &THLongTensor_clamp"
  p_clamp_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> CLong -> IO ())

-- | alias of p_clamp_ with unused argument (for CTHState) to unify backpack signatures.
p_clamp = const p_clamp_

-- | p_bitand : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THLongTensor_bitand"
  p_bitand_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_bitand_ with unused argument (for CTHState) to unify backpack signatures.
p_bitand = const p_bitand_

-- | p_bitor : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THLongTensor_bitor"
  p_bitor_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_bitor_ with unused argument (for CTHState) to unify backpack signatures.
p_bitor = const p_bitor_

-- | p_bitxor : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THLongTensor_bitxor"
  p_bitxor_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_bitxor_ with unused argument (for CTHState) to unify backpack signatures.
p_bitxor = const p_bitxor_

-- | p_cadd : Pointer to function : r_ t value src -> void
foreign import ccall "THTensorMath.h &THLongTensor_cadd"
  p_cadd_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> IO ())

-- | alias of p_cadd_ with unused argument (for CTHState) to unify backpack signatures.
p_cadd = const p_cadd_

-- | p_csub : Pointer to function : self src1 value src2 -> void
foreign import ccall "THTensorMath.h &THLongTensor_csub"
  p_csub_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> IO ())

-- | alias of p_csub_ with unused argument (for CTHState) to unify backpack signatures.
p_csub = const p_csub_

-- | p_cmul : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THLongTensor_cmul"
  p_cmul_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_cmul_ with unused argument (for CTHState) to unify backpack signatures.
p_cmul = const p_cmul_

-- | p_cpow : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THLongTensor_cpow"
  p_cpow_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_cpow_ with unused argument (for CTHState) to unify backpack signatures.
p_cpow = const p_cpow_

-- | p_cdiv : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THLongTensor_cdiv"
  p_cdiv_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_cdiv_ with unused argument (for CTHState) to unify backpack signatures.
p_cdiv = const p_cdiv_

-- | p_clshift : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THLongTensor_clshift"
  p_clshift_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_clshift_ with unused argument (for CTHState) to unify backpack signatures.
p_clshift = const p_clshift_

-- | p_crshift : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THLongTensor_crshift"
  p_crshift_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_crshift_ with unused argument (for CTHState) to unify backpack signatures.
p_crshift = const p_crshift_

-- | p_cfmod : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THLongTensor_cfmod"
  p_cfmod_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_cfmod_ with unused argument (for CTHState) to unify backpack signatures.
p_cfmod = const p_cfmod_

-- | p_cremainder : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THLongTensor_cremainder"
  p_cremainder_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_cremainder_ with unused argument (for CTHState) to unify backpack signatures.
p_cremainder = const p_cremainder_

-- | p_cbitand : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THLongTensor_cbitand"
  p_cbitand_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_cbitand_ with unused argument (for CTHState) to unify backpack signatures.
p_cbitand = const p_cbitand_

-- | p_cbitor : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THLongTensor_cbitor"
  p_cbitor_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_cbitor_ with unused argument (for CTHState) to unify backpack signatures.
p_cbitor = const p_cbitor_

-- | p_cbitxor : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THLongTensor_cbitxor"
  p_cbitxor_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_cbitxor_ with unused argument (for CTHState) to unify backpack signatures.
p_cbitxor = const p_cbitxor_

-- | p_addcmul : Pointer to function : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h &THLongTensor_addcmul"
  p_addcmul_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_addcmul_ with unused argument (for CTHState) to unify backpack signatures.
p_addcmul = const p_addcmul_

-- | p_addcdiv : Pointer to function : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h &THLongTensor_addcdiv"
  p_addcdiv_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_addcdiv_ with unused argument (for CTHState) to unify backpack signatures.
p_addcdiv = const p_addcdiv_

-- | p_addmv : Pointer to function : r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h &THLongTensor_addmv"
  p_addmv_ :: FunPtr (Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_addmv_ with unused argument (for CTHState) to unify backpack signatures.
p_addmv = const p_addmv_

-- | p_addmm : Pointer to function : r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h &THLongTensor_addmm"
  p_addmm_ :: FunPtr (Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_addmm_ with unused argument (for CTHState) to unify backpack signatures.
p_addmm = const p_addmm_

-- | p_addr : Pointer to function : r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h &THLongTensor_addr"
  p_addr_ :: FunPtr (Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_addr_ with unused argument (for CTHState) to unify backpack signatures.
p_addr = const p_addr_

-- | p_addbmm : Pointer to function : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h &THLongTensor_addbmm"
  p_addbmm_ :: FunPtr (Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_addbmm_ with unused argument (for CTHState) to unify backpack signatures.
p_addbmm = const p_addbmm_

-- | p_baddbmm : Pointer to function : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h &THLongTensor_baddbmm"
  p_baddbmm_ :: FunPtr (Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> CLong -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_baddbmm_ with unused argument (for CTHState) to unify backpack signatures.
p_baddbmm = const p_baddbmm_

-- | p_match : Pointer to function : r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h &THLongTensor_match"
  p_match_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_match_ with unused argument (for CTHState) to unify backpack signatures.
p_match = const p_match_

-- | p_numel : Pointer to function : t -> ptrdiff_t
foreign import ccall "THTensorMath.h &THLongTensor_numel"
  p_numel_ :: FunPtr (Ptr C'THLongTensor -> IO CPtrdiff)

-- | alias of p_numel_ with unused argument (for CTHState) to unify backpack signatures.
p_numel = const p_numel_

-- | p_preserveReduceDimSemantics : Pointer to function : r_ in_dims reduce_dimension keepdim -> void
foreign import ccall "THTensorMath.h &THLongTensor_preserveReduceDimSemantics"
  p_preserveReduceDimSemantics_ :: FunPtr (Ptr C'THLongTensor -> CInt -> CInt -> CInt -> IO ())

-- | alias of p_preserveReduceDimSemantics_ with unused argument (for CTHState) to unify backpack signatures.
p_preserveReduceDimSemantics = const p_preserveReduceDimSemantics_

-- | p_max : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THLongTensor_max"
  p_max_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> CInt -> IO ())

-- | alias of p_max_ with unused argument (for CTHState) to unify backpack signatures.
p_max = const p_max_

-- | p_min : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THLongTensor_min"
  p_min_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> CInt -> IO ())

-- | alias of p_min_ with unused argument (for CTHState) to unify backpack signatures.
p_min = const p_min_

-- | p_kthvalue : Pointer to function : values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h &THLongTensor_kthvalue"
  p_kthvalue_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLLong -> CInt -> CInt -> IO ())

-- | alias of p_kthvalue_ with unused argument (for CTHState) to unify backpack signatures.
p_kthvalue = const p_kthvalue_

-- | p_mode : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THLongTensor_mode"
  p_mode_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> CInt -> IO ())

-- | alias of p_mode_ with unused argument (for CTHState) to unify backpack signatures.
p_mode = const p_mode_

-- | p_median : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THLongTensor_median"
  p_median_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> CInt -> IO ())

-- | alias of p_median_ with unused argument (for CTHState) to unify backpack signatures.
p_median = const p_median_

-- | p_sum : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THLongTensor_sum"
  p_sum_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> CInt -> IO ())

-- | alias of p_sum_ with unused argument (for CTHState) to unify backpack signatures.
p_sum = const p_sum_

-- | p_prod : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THLongTensor_prod"
  p_prod_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> CInt -> IO ())

-- | alias of p_prod_ with unused argument (for CTHState) to unify backpack signatures.
p_prod = const p_prod_

-- | p_cumsum : Pointer to function : r_ t dimension -> void
foreign import ccall "THTensorMath.h &THLongTensor_cumsum"
  p_cumsum_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> IO ())

-- | alias of p_cumsum_ with unused argument (for CTHState) to unify backpack signatures.
p_cumsum = const p_cumsum_

-- | p_cumprod : Pointer to function : r_ t dimension -> void
foreign import ccall "THTensorMath.h &THLongTensor_cumprod"
  p_cumprod_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> IO ())

-- | alias of p_cumprod_ with unused argument (for CTHState) to unify backpack signatures.
p_cumprod = const p_cumprod_

-- | p_sign : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THLongTensor_sign"
  p_sign_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_sign_ with unused argument (for CTHState) to unify backpack signatures.
p_sign = const p_sign_

-- | p_trace : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THLongTensor_trace"
  p_trace_ :: FunPtr (Ptr C'THLongTensor -> IO CLong)

-- | alias of p_trace_ with unused argument (for CTHState) to unify backpack signatures.
p_trace = const p_trace_

-- | p_cross : Pointer to function : r_ a b dimension -> void
foreign import ccall "THTensorMath.h &THLongTensor_cross"
  p_cross_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> IO ())

-- | alias of p_cross_ with unused argument (for CTHState) to unify backpack signatures.
p_cross = const p_cross_

-- | p_cmax : Pointer to function : r t src -> void
foreign import ccall "THTensorMath.h &THLongTensor_cmax"
  p_cmax_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_cmax_ with unused argument (for CTHState) to unify backpack signatures.
p_cmax = const p_cmax_

-- | p_cmin : Pointer to function : r t src -> void
foreign import ccall "THTensorMath.h &THLongTensor_cmin"
  p_cmin_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_cmin_ with unused argument (for CTHState) to unify backpack signatures.
p_cmin = const p_cmin_

-- | p_cmaxValue : Pointer to function : r t value -> void
foreign import ccall "THTensorMath.h &THLongTensor_cmaxValue"
  p_cmaxValue_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_cmaxValue_ with unused argument (for CTHState) to unify backpack signatures.
p_cmaxValue = const p_cmaxValue_

-- | p_cminValue : Pointer to function : r t value -> void
foreign import ccall "THTensorMath.h &THLongTensor_cminValue"
  p_cminValue_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_cminValue_ with unused argument (for CTHState) to unify backpack signatures.
p_cminValue = const p_cminValue_

-- | p_zeros : Pointer to function : r_ size -> void
foreign import ccall "THTensorMath.h &THLongTensor_zeros"
  p_zeros_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongStorage -> IO ())

-- | alias of p_zeros_ with unused argument (for CTHState) to unify backpack signatures.
p_zeros = const p_zeros_

-- | p_zerosLike : Pointer to function : r_ input -> void
foreign import ccall "THTensorMath.h &THLongTensor_zerosLike"
  p_zerosLike_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_zerosLike_ with unused argument (for CTHState) to unify backpack signatures.
p_zerosLike = const p_zerosLike_

-- | p_ones : Pointer to function : r_ size -> void
foreign import ccall "THTensorMath.h &THLongTensor_ones"
  p_ones_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongStorage -> IO ())

-- | alias of p_ones_ with unused argument (for CTHState) to unify backpack signatures.
p_ones = const p_ones_

-- | p_onesLike : Pointer to function : r_ input -> void
foreign import ccall "THTensorMath.h &THLongTensor_onesLike"
  p_onesLike_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_onesLike_ with unused argument (for CTHState) to unify backpack signatures.
p_onesLike = const p_onesLike_

-- | p_diag : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THLongTensor_diag"
  p_diag_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> IO ())

-- | alias of p_diag_ with unused argument (for CTHState) to unify backpack signatures.
p_diag = const p_diag_

-- | p_eye : Pointer to function : r_ n m -> void
foreign import ccall "THTensorMath.h &THLongTensor_eye"
  p_eye_ :: FunPtr (Ptr C'THLongTensor -> CLLong -> CLLong -> IO ())

-- | alias of p_eye_ with unused argument (for CTHState) to unify backpack signatures.
p_eye = const p_eye_

-- | p_arange : Pointer to function : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h &THLongTensor_arange"
  p_arange_ :: FunPtr (Ptr C'THLongTensor -> CLong -> CLong -> CLong -> IO ())

-- | alias of p_arange_ with unused argument (for CTHState) to unify backpack signatures.
p_arange = const p_arange_

-- | p_range : Pointer to function : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h &THLongTensor_range"
  p_range_ :: FunPtr (Ptr C'THLongTensor -> CLong -> CLong -> CLong -> IO ())

-- | alias of p_range_ with unused argument (for CTHState) to unify backpack signatures.
p_range = const p_range_

-- | p_randperm : Pointer to function : r_ _generator n -> void
foreign import ccall "THTensorMath.h &THLongTensor_randperm"
  p_randperm_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THGenerator -> CLLong -> IO ())

-- | alias of p_randperm_ with unused argument (for CTHState) to unify backpack signatures.
p_randperm = const p_randperm_

-- | p_reshape : Pointer to function : r_ t size -> void
foreign import ccall "THTensorMath.h &THLongTensor_reshape"
  p_reshape_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongStorage -> IO ())

-- | alias of p_reshape_ with unused argument (for CTHState) to unify backpack signatures.
p_reshape = const p_reshape_

-- | p_sort : Pointer to function : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h &THLongTensor_sort"
  p_sort_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> CInt -> IO ())

-- | alias of p_sort_ with unused argument (for CTHState) to unify backpack signatures.
p_sort = const p_sort_

-- | p_topk : Pointer to function : rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h &THLongTensor_topk"
  p_topk_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLLong -> CInt -> CInt -> CInt -> IO ())

-- | alias of p_topk_ with unused argument (for CTHState) to unify backpack signatures.
p_topk = const p_topk_

-- | p_tril : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THLongTensor_tril"
  p_tril_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLLong -> IO ())

-- | alias of p_tril_ with unused argument (for CTHState) to unify backpack signatures.
p_tril = const p_tril_

-- | p_triu : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THLongTensor_triu"
  p_triu_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLLong -> IO ())

-- | alias of p_triu_ with unused argument (for CTHState) to unify backpack signatures.
p_triu = const p_triu_

-- | p_cat : Pointer to function : r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h &THLongTensor_cat"
  p_cat_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> CInt -> IO ())

-- | alias of p_cat_ with unused argument (for CTHState) to unify backpack signatures.
p_cat = const p_cat_

-- | p_catArray : Pointer to function : result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h &THLongTensor_catArray"
  p_catArray_ :: FunPtr (Ptr C'THLongTensor -> Ptr (Ptr C'THLongTensor) -> CInt -> CInt -> IO ())

-- | alias of p_catArray_ with unused argument (for CTHState) to unify backpack signatures.
p_catArray = const p_catArray_

-- | p_equal : Pointer to function : ta tb -> int
foreign import ccall "THTensorMath.h &THLongTensor_equal"
  p_equal_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO CInt)

-- | alias of p_equal_ with unused argument (for CTHState) to unify backpack signatures.
p_equal = const p_equal_

-- | p_ltValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THLongTensor_ltValue"
  p_ltValue_ :: FunPtr (Ptr C'THByteTensor -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_ltValue_ with unused argument (for CTHState) to unify backpack signatures.
p_ltValue = const p_ltValue_

-- | p_leValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THLongTensor_leValue"
  p_leValue_ :: FunPtr (Ptr C'THByteTensor -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_leValue_ with unused argument (for CTHState) to unify backpack signatures.
p_leValue = const p_leValue_

-- | p_gtValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THLongTensor_gtValue"
  p_gtValue_ :: FunPtr (Ptr C'THByteTensor -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_gtValue_ with unused argument (for CTHState) to unify backpack signatures.
p_gtValue = const p_gtValue_

-- | p_geValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THLongTensor_geValue"
  p_geValue_ :: FunPtr (Ptr C'THByteTensor -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_geValue_ with unused argument (for CTHState) to unify backpack signatures.
p_geValue = const p_geValue_

-- | p_neValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THLongTensor_neValue"
  p_neValue_ :: FunPtr (Ptr C'THByteTensor -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_neValue_ with unused argument (for CTHState) to unify backpack signatures.
p_neValue = const p_neValue_

-- | p_eqValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THLongTensor_eqValue"
  p_eqValue_ :: FunPtr (Ptr C'THByteTensor -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_eqValue_ with unused argument (for CTHState) to unify backpack signatures.
p_eqValue = const p_eqValue_

-- | p_ltValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THLongTensor_ltValueT"
  p_ltValueT_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_ltValueT_ with unused argument (for CTHState) to unify backpack signatures.
p_ltValueT = const p_ltValueT_

-- | p_leValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THLongTensor_leValueT"
  p_leValueT_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_leValueT_ with unused argument (for CTHState) to unify backpack signatures.
p_leValueT = const p_leValueT_

-- | p_gtValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THLongTensor_gtValueT"
  p_gtValueT_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_gtValueT_ with unused argument (for CTHState) to unify backpack signatures.
p_gtValueT = const p_gtValueT_

-- | p_geValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THLongTensor_geValueT"
  p_geValueT_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_geValueT_ with unused argument (for CTHState) to unify backpack signatures.
p_geValueT = const p_geValueT_

-- | p_neValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THLongTensor_neValueT"
  p_neValueT_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_neValueT_ with unused argument (for CTHState) to unify backpack signatures.
p_neValueT = const p_neValueT_

-- | p_eqValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THLongTensor_eqValueT"
  p_eqValueT_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> CLong -> IO ())

-- | alias of p_eqValueT_ with unused argument (for CTHState) to unify backpack signatures.
p_eqValueT = const p_eqValueT_

-- | p_ltTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THLongTensor_ltTensor"
  p_ltTensor_ :: FunPtr (Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_ltTensor_ with unused argument (for CTHState) to unify backpack signatures.
p_ltTensor = const p_ltTensor_

-- | p_leTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THLongTensor_leTensor"
  p_leTensor_ :: FunPtr (Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_leTensor_ with unused argument (for CTHState) to unify backpack signatures.
p_leTensor = const p_leTensor_

-- | p_gtTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THLongTensor_gtTensor"
  p_gtTensor_ :: FunPtr (Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_gtTensor_ with unused argument (for CTHState) to unify backpack signatures.
p_gtTensor = const p_gtTensor_

-- | p_geTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THLongTensor_geTensor"
  p_geTensor_ :: FunPtr (Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_geTensor_ with unused argument (for CTHState) to unify backpack signatures.
p_geTensor = const p_geTensor_

-- | p_neTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THLongTensor_neTensor"
  p_neTensor_ :: FunPtr (Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_neTensor_ with unused argument (for CTHState) to unify backpack signatures.
p_neTensor = const p_neTensor_

-- | p_eqTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THLongTensor_eqTensor"
  p_eqTensor_ :: FunPtr (Ptr C'THByteTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_eqTensor_ with unused argument (for CTHState) to unify backpack signatures.
p_eqTensor = const p_eqTensor_

-- | p_ltTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THLongTensor_ltTensorT"
  p_ltTensorT_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_ltTensorT_ with unused argument (for CTHState) to unify backpack signatures.
p_ltTensorT = const p_ltTensorT_

-- | p_leTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THLongTensor_leTensorT"
  p_leTensorT_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_leTensorT_ with unused argument (for CTHState) to unify backpack signatures.
p_leTensorT = const p_leTensorT_

-- | p_gtTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THLongTensor_gtTensorT"
  p_gtTensorT_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_gtTensorT_ with unused argument (for CTHState) to unify backpack signatures.
p_gtTensorT = const p_gtTensorT_

-- | p_geTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THLongTensor_geTensorT"
  p_geTensorT_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_geTensorT_ with unused argument (for CTHState) to unify backpack signatures.
p_geTensorT = const p_geTensorT_

-- | p_neTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THLongTensor_neTensorT"
  p_neTensorT_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_neTensorT_ with unused argument (for CTHState) to unify backpack signatures.
p_neTensorT = const p_neTensorT_

-- | p_eqTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THLongTensor_eqTensorT"
  p_eqTensorT_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_eqTensorT_ with unused argument (for CTHState) to unify backpack signatures.
p_eqTensorT = const p_eqTensorT_

-- | p_abs : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THLongTensor_abs"
  p_abs_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_abs_ with unused argument (for CTHState) to unify backpack signatures.
p_abs = const p_abs_