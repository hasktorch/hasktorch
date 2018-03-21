{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Float.TensorMath where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_fill :  r_ value -> void
foreign import ccall "THTensorMath.h THFloatTensor_fill"
  c_fill_ :: Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_fill_ with unused argument (for CTHState) to unify backpack signatures.
c_fill = const c_fill_

-- | c_zero :  r_ -> void
foreign import ccall "THTensorMath.h THFloatTensor_zero"
  c_zero_ :: Ptr C'THFloatTensor -> IO ()

-- | alias of c_zero_ with unused argument (for CTHState) to unify backpack signatures.
c_zero = const c_zero_

-- | c_maskedFill :  tensor mask value -> void
foreign import ccall "THTensorMath.h THFloatTensor_maskedFill"
  c_maskedFill_ :: Ptr C'THFloatTensor -> Ptr C'THByteTensor -> CFloat -> IO ()

-- | alias of c_maskedFill_ with unused argument (for CTHState) to unify backpack signatures.
c_maskedFill = const c_maskedFill_

-- | c_maskedCopy :  tensor mask src -> void
foreign import ccall "THTensorMath.h THFloatTensor_maskedCopy"
  c_maskedCopy_ :: Ptr C'THFloatTensor -> Ptr C'THByteTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_maskedCopy_ with unused argument (for CTHState) to unify backpack signatures.
c_maskedCopy = const c_maskedCopy_

-- | c_maskedSelect :  tensor src mask -> void
foreign import ccall "THTensorMath.h THFloatTensor_maskedSelect"
  c_maskedSelect_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_maskedSelect_ with unused argument (for CTHState) to unify backpack signatures.
c_maskedSelect = const c_maskedSelect_

-- | c_nonzero :  subscript tensor -> void
foreign import ccall "THTensorMath.h THFloatTensor_nonzero"
  c_nonzero_ :: Ptr C'THLongTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_nonzero_ with unused argument (for CTHState) to unify backpack signatures.
c_nonzero = const c_nonzero_

-- | c_indexSelect :  tensor src dim index -> void
foreign import ccall "THTensorMath.h THFloatTensor_indexSelect"
  c_indexSelect_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> Ptr C'THLongTensor -> IO ()

-- | alias of c_indexSelect_ with unused argument (for CTHState) to unify backpack signatures.
c_indexSelect = const c_indexSelect_

-- | c_indexCopy :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THFloatTensor_indexCopy"
  c_indexCopy_ :: Ptr C'THFloatTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_indexCopy_ with unused argument (for CTHState) to unify backpack signatures.
c_indexCopy = const c_indexCopy_

-- | c_indexAdd :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THFloatTensor_indexAdd"
  c_indexAdd_ :: Ptr C'THFloatTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_indexAdd_ with unused argument (for CTHState) to unify backpack signatures.
c_indexAdd = const c_indexAdd_

-- | c_indexFill :  tensor dim index val -> void
foreign import ccall "THTensorMath.h THFloatTensor_indexFill"
  c_indexFill_ :: Ptr C'THFloatTensor -> CInt -> Ptr C'THLongTensor -> CFloat -> IO ()

-- | alias of c_indexFill_ with unused argument (for CTHState) to unify backpack signatures.
c_indexFill = const c_indexFill_

-- | c_take :  tensor src index -> void
foreign import ccall "THTensorMath.h THFloatTensor_take"
  c_take_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_take_ with unused argument (for CTHState) to unify backpack signatures.
c_take = const c_take_

-- | c_put :  tensor index src accumulate -> void
foreign import ccall "THTensorMath.h THFloatTensor_put"
  c_put_ :: Ptr C'THFloatTensor -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> CInt -> IO ()

-- | alias of c_put_ with unused argument (for CTHState) to unify backpack signatures.
c_put = const c_put_

-- | c_gather :  tensor src dim index -> void
foreign import ccall "THTensorMath.h THFloatTensor_gather"
  c_gather_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> Ptr C'THLongTensor -> IO ()

-- | alias of c_gather_ with unused argument (for CTHState) to unify backpack signatures.
c_gather = const c_gather_

-- | c_scatter :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THFloatTensor_scatter"
  c_scatter_ :: Ptr C'THFloatTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_scatter_ with unused argument (for CTHState) to unify backpack signatures.
c_scatter = const c_scatter_

-- | c_scatterAdd :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THFloatTensor_scatterAdd"
  c_scatterAdd_ :: Ptr C'THFloatTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_scatterAdd_ with unused argument (for CTHState) to unify backpack signatures.
c_scatterAdd = const c_scatterAdd_

-- | c_scatterFill :  tensor dim index val -> void
foreign import ccall "THTensorMath.h THFloatTensor_scatterFill"
  c_scatterFill_ :: Ptr C'THFloatTensor -> CInt -> Ptr C'THLongTensor -> CFloat -> IO ()

-- | alias of c_scatterFill_ with unused argument (for CTHState) to unify backpack signatures.
c_scatterFill = const c_scatterFill_

-- | c_dot :  t src -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_dot"
  c_dot_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO CDouble

-- | alias of c_dot_ with unused argument (for CTHState) to unify backpack signatures.
c_dot = const c_dot_

-- | c_minall :  t -> real
foreign import ccall "THTensorMath.h THFloatTensor_minall"
  c_minall_ :: Ptr C'THFloatTensor -> IO CFloat

-- | alias of c_minall_ with unused argument (for CTHState) to unify backpack signatures.
c_minall = const c_minall_

-- | c_maxall :  t -> real
foreign import ccall "THTensorMath.h THFloatTensor_maxall"
  c_maxall_ :: Ptr C'THFloatTensor -> IO CFloat

-- | alias of c_maxall_ with unused argument (for CTHState) to unify backpack signatures.
c_maxall = const c_maxall_

-- | c_medianall :  t -> real
foreign import ccall "THTensorMath.h THFloatTensor_medianall"
  c_medianall_ :: Ptr C'THFloatTensor -> IO CFloat

-- | alias of c_medianall_ with unused argument (for CTHState) to unify backpack signatures.
c_medianall = const c_medianall_

-- | c_sumall :  t -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_sumall"
  c_sumall_ :: Ptr C'THFloatTensor -> IO CDouble

-- | alias of c_sumall_ with unused argument (for CTHState) to unify backpack signatures.
c_sumall = const c_sumall_

-- | c_prodall :  t -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_prodall"
  c_prodall_ :: Ptr C'THFloatTensor -> IO CDouble

-- | alias of c_prodall_ with unused argument (for CTHState) to unify backpack signatures.
c_prodall = const c_prodall_

-- | c_neg :  self src -> void
foreign import ccall "THTensorMath.h THFloatTensor_neg"
  c_neg_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_neg_ with unused argument (for CTHState) to unify backpack signatures.
c_neg = const c_neg_

-- | c_cinv :  self src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cinv"
  c_cinv_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_cinv_ with unused argument (for CTHState) to unify backpack signatures.
c_cinv = const c_cinv_

-- | c_add :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_add"
  c_add_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_add_ with unused argument (for CTHState) to unify backpack signatures.
c_add = const c_add_

-- | c_sub :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_sub"
  c_sub_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_sub_ with unused argument (for CTHState) to unify backpack signatures.
c_sub = const c_sub_

-- | c_add_scaled :  r_ t value alpha -> void
foreign import ccall "THTensorMath.h THFloatTensor_add_scaled"
  c_add_scaled_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> CFloat -> IO ()

-- | alias of c_add_scaled_ with unused argument (for CTHState) to unify backpack signatures.
c_add_scaled = const c_add_scaled_

-- | c_sub_scaled :  r_ t value alpha -> void
foreign import ccall "THTensorMath.h THFloatTensor_sub_scaled"
  c_sub_scaled_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> CFloat -> IO ()

-- | alias of c_sub_scaled_ with unused argument (for CTHState) to unify backpack signatures.
c_sub_scaled = const c_sub_scaled_

-- | c_mul :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_mul"
  c_mul_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_mul_ with unused argument (for CTHState) to unify backpack signatures.
c_mul = const c_mul_

-- | c_div :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_div"
  c_div_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_div_ with unused argument (for CTHState) to unify backpack signatures.
c_div = const c_div_

-- | c_lshift :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_lshift"
  c_lshift_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_lshift_ with unused argument (for CTHState) to unify backpack signatures.
c_lshift = const c_lshift_

-- | c_rshift :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_rshift"
  c_rshift_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_rshift_ with unused argument (for CTHState) to unify backpack signatures.
c_rshift = const c_rshift_

-- | c_fmod :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_fmod"
  c_fmod_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_fmod_ with unused argument (for CTHState) to unify backpack signatures.
c_fmod = const c_fmod_

-- | c_remainder :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_remainder"
  c_remainder_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_remainder_ with unused argument (for CTHState) to unify backpack signatures.
c_remainder = const c_remainder_

-- | c_clamp :  r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h THFloatTensor_clamp"
  c_clamp_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> CFloat -> IO ()

-- | alias of c_clamp_ with unused argument (for CTHState) to unify backpack signatures.
c_clamp = const c_clamp_

-- | c_bitand :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_bitand"
  c_bitand_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_bitand_ with unused argument (for CTHState) to unify backpack signatures.
c_bitand = const c_bitand_

-- | c_bitor :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_bitor"
  c_bitor_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_bitor_ with unused argument (for CTHState) to unify backpack signatures.
c_bitor = const c_bitor_

-- | c_bitxor :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_bitxor"
  c_bitxor_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_bitxor_ with unused argument (for CTHState) to unify backpack signatures.
c_bitxor = const c_bitxor_

-- | c_cadd :  r_ t value src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cadd"
  c_cadd_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_cadd_ with unused argument (for CTHState) to unify backpack signatures.
c_cadd = const c_cadd_

-- | c_csub :  self src1 value src2 -> void
foreign import ccall "THTensorMath.h THFloatTensor_csub"
  c_csub_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_csub_ with unused argument (for CTHState) to unify backpack signatures.
c_csub = const c_csub_

-- | c_cmul :  r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cmul"
  c_cmul_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_cmul_ with unused argument (for CTHState) to unify backpack signatures.
c_cmul = const c_cmul_

-- | c_cpow :  r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cpow"
  c_cpow_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_cpow_ with unused argument (for CTHState) to unify backpack signatures.
c_cpow = const c_cpow_

-- | c_cdiv :  r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cdiv"
  c_cdiv_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_cdiv_ with unused argument (for CTHState) to unify backpack signatures.
c_cdiv = const c_cdiv_

-- | c_clshift :  r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_clshift"
  c_clshift_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_clshift_ with unused argument (for CTHState) to unify backpack signatures.
c_clshift = const c_clshift_

-- | c_crshift :  r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_crshift"
  c_crshift_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_crshift_ with unused argument (for CTHState) to unify backpack signatures.
c_crshift = const c_crshift_

-- | c_cfmod :  r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cfmod"
  c_cfmod_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_cfmod_ with unused argument (for CTHState) to unify backpack signatures.
c_cfmod = const c_cfmod_

-- | c_cremainder :  r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cremainder"
  c_cremainder_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_cremainder_ with unused argument (for CTHState) to unify backpack signatures.
c_cremainder = const c_cremainder_

-- | c_cbitand :  r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cbitand"
  c_cbitand_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_cbitand_ with unused argument (for CTHState) to unify backpack signatures.
c_cbitand = const c_cbitand_

-- | c_cbitor :  r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cbitor"
  c_cbitor_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_cbitor_ with unused argument (for CTHState) to unify backpack signatures.
c_cbitor = const c_cbitor_

-- | c_cbitxor :  r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cbitxor"
  c_cbitxor_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_cbitxor_ with unused argument (for CTHState) to unify backpack signatures.
c_cbitxor = const c_cbitxor_

-- | c_addcmul :  r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THFloatTensor_addcmul"
  c_addcmul_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_addcmul_ with unused argument (for CTHState) to unify backpack signatures.
c_addcmul = const c_addcmul_

-- | c_addcdiv :  r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THFloatTensor_addcdiv"
  c_addcdiv_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_addcdiv_ with unused argument (for CTHState) to unify backpack signatures.
c_addcdiv = const c_addcdiv_

-- | c_addmv :  r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h THFloatTensor_addmv"
  c_addmv_ :: Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_addmv_ with unused argument (for CTHState) to unify backpack signatures.
c_addmv = const c_addmv_

-- | c_addmm :  r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h THFloatTensor_addmm"
  c_addmm_ :: Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_addmm_ with unused argument (for CTHState) to unify backpack signatures.
c_addmm = const c_addmm_

-- | c_addr :  r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h THFloatTensor_addr"
  c_addr_ :: Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_addr_ with unused argument (for CTHState) to unify backpack signatures.
c_addr = const c_addr_

-- | c_addbmm :  r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THFloatTensor_addbmm"
  c_addbmm_ :: Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_addbmm_ with unused argument (for CTHState) to unify backpack signatures.
c_addbmm = const c_addbmm_

-- | c_baddbmm :  r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THFloatTensor_baddbmm"
  c_baddbmm_ :: Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_baddbmm_ with unused argument (for CTHState) to unify backpack signatures.
c_baddbmm = const c_baddbmm_

-- | c_match :  r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h THFloatTensor_match"
  c_match_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_match_ with unused argument (for CTHState) to unify backpack signatures.
c_match = const c_match_

-- | c_numel :  t -> ptrdiff_t
foreign import ccall "THTensorMath.h THFloatTensor_numel"
  c_numel_ :: Ptr C'THFloatTensor -> IO CPtrdiff

-- | alias of c_numel_ with unused argument (for CTHState) to unify backpack signatures.
c_numel = const c_numel_

-- | c_preserveReduceDimSemantics :  r_ in_dims reduce_dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_preserveReduceDimSemantics"
  c_preserveReduceDimSemantics_ :: Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> IO ()

-- | alias of c_preserveReduceDimSemantics_ with unused argument (for CTHState) to unify backpack signatures.
c_preserveReduceDimSemantics = const c_preserveReduceDimSemantics_

-- | c_max :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_max"
  c_max_ :: Ptr C'THFloatTensor -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ()

-- | alias of c_max_ with unused argument (for CTHState) to unify backpack signatures.
c_max = const c_max_

-- | c_min :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_min"
  c_min_ :: Ptr C'THFloatTensor -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ()

-- | alias of c_min_ with unused argument (for CTHState) to unify backpack signatures.
c_min = const c_min_

-- | c_kthvalue :  values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_kthvalue"
  c_kthvalue_ :: Ptr C'THFloatTensor -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> CLLong -> CInt -> CInt -> IO ()

-- | alias of c_kthvalue_ with unused argument (for CTHState) to unify backpack signatures.
c_kthvalue = const c_kthvalue_

-- | c_mode :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_mode"
  c_mode_ :: Ptr C'THFloatTensor -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ()

-- | alias of c_mode_ with unused argument (for CTHState) to unify backpack signatures.
c_mode = const c_mode_

-- | c_median :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_median"
  c_median_ :: Ptr C'THFloatTensor -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ()

-- | alias of c_median_ with unused argument (for CTHState) to unify backpack signatures.
c_median = const c_median_

-- | c_sum :  r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_sum"
  c_sum_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ()

-- | alias of c_sum_ with unused argument (for CTHState) to unify backpack signatures.
c_sum = const c_sum_

-- | c_prod :  r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_prod"
  c_prod_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ()

-- | alias of c_prod_ with unused argument (for CTHState) to unify backpack signatures.
c_prod = const c_prod_

-- | c_cumsum :  r_ t dimension -> void
foreign import ccall "THTensorMath.h THFloatTensor_cumsum"
  c_cumsum_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ()

-- | alias of c_cumsum_ with unused argument (for CTHState) to unify backpack signatures.
c_cumsum = const c_cumsum_

-- | c_cumprod :  r_ t dimension -> void
foreign import ccall "THTensorMath.h THFloatTensor_cumprod"
  c_cumprod_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ()

-- | alias of c_cumprod_ with unused argument (for CTHState) to unify backpack signatures.
c_cumprod = const c_cumprod_

-- | c_sign :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_sign"
  c_sign_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_sign_ with unused argument (for CTHState) to unify backpack signatures.
c_sign = const c_sign_

-- | c_trace :  t -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_trace"
  c_trace_ :: Ptr C'THFloatTensor -> IO CDouble

-- | alias of c_trace_ with unused argument (for CTHState) to unify backpack signatures.
c_trace = const c_trace_

-- | c_cross :  r_ a b dimension -> void
foreign import ccall "THTensorMath.h THFloatTensor_cross"
  c_cross_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ()

-- | alias of c_cross_ with unused argument (for CTHState) to unify backpack signatures.
c_cross = const c_cross_

-- | c_cmax :  r t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cmax"
  c_cmax_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_cmax_ with unused argument (for CTHState) to unify backpack signatures.
c_cmax = const c_cmax_

-- | c_cmin :  r t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cmin"
  c_cmin_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_cmin_ with unused argument (for CTHState) to unify backpack signatures.
c_cmin = const c_cmin_

-- | c_cmaxValue :  r t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_cmaxValue"
  c_cmaxValue_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_cmaxValue_ with unused argument (for CTHState) to unify backpack signatures.
c_cmaxValue = const c_cmaxValue_

-- | c_cminValue :  r t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_cminValue"
  c_cminValue_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_cminValue_ with unused argument (for CTHState) to unify backpack signatures.
c_cminValue = const c_cminValue_

-- | c_zeros :  r_ size -> void
foreign import ccall "THTensorMath.h THFloatTensor_zeros"
  c_zeros_ :: Ptr C'THFloatTensor -> Ptr C'THLongStorage -> IO ()

-- | alias of c_zeros_ with unused argument (for CTHState) to unify backpack signatures.
c_zeros = const c_zeros_

-- | c_zerosLike :  r_ input -> void
foreign import ccall "THTensorMath.h THFloatTensor_zerosLike"
  c_zerosLike_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_zerosLike_ with unused argument (for CTHState) to unify backpack signatures.
c_zerosLike = const c_zerosLike_

-- | c_ones :  r_ size -> void
foreign import ccall "THTensorMath.h THFloatTensor_ones"
  c_ones_ :: Ptr C'THFloatTensor -> Ptr C'THLongStorage -> IO ()

-- | alias of c_ones_ with unused argument (for CTHState) to unify backpack signatures.
c_ones = const c_ones_

-- | c_onesLike :  r_ input -> void
foreign import ccall "THTensorMath.h THFloatTensor_onesLike"
  c_onesLike_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_onesLike_ with unused argument (for CTHState) to unify backpack signatures.
c_onesLike = const c_onesLike_

-- | c_diag :  r_ t k -> void
foreign import ccall "THTensorMath.h THFloatTensor_diag"
  c_diag_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ()

-- | alias of c_diag_ with unused argument (for CTHState) to unify backpack signatures.
c_diag = const c_diag_

-- | c_eye :  r_ n m -> void
foreign import ccall "THTensorMath.h THFloatTensor_eye"
  c_eye_ :: Ptr C'THFloatTensor -> CLLong -> CLLong -> IO ()

-- | alias of c_eye_ with unused argument (for CTHState) to unify backpack signatures.
c_eye = const c_eye_

-- | c_arange :  r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THFloatTensor_arange"
  c_arange_ :: Ptr C'THFloatTensor -> CDouble -> CDouble -> CDouble -> IO ()

-- | alias of c_arange_ with unused argument (for CTHState) to unify backpack signatures.
c_arange = const c_arange_

-- | c_range :  r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THFloatTensor_range"
  c_range_ :: Ptr C'THFloatTensor -> CDouble -> CDouble -> CDouble -> IO ()

-- | alias of c_range_ with unused argument (for CTHState) to unify backpack signatures.
c_range = const c_range_

-- | c_randperm :  r_ _generator n -> void
foreign import ccall "THTensorMath.h THFloatTensor_randperm"
  c_randperm_ :: Ptr C'THFloatTensor -> Ptr C'THGenerator -> CLLong -> IO ()

-- | alias of c_randperm_ with unused argument (for CTHState) to unify backpack signatures.
c_randperm = const c_randperm_

-- | c_reshape :  r_ t size -> void
foreign import ccall "THTensorMath.h THFloatTensor_reshape"
  c_reshape_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THLongStorage -> IO ()

-- | alias of c_reshape_ with unused argument (for CTHState) to unify backpack signatures.
c_reshape = const c_reshape_

-- | c_sort :  rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h THFloatTensor_sort"
  c_sort_ :: Ptr C'THFloatTensor -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ()

-- | alias of c_sort_ with unused argument (for CTHState) to unify backpack signatures.
c_sort = const c_sort_

-- | c_topk :  rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h THFloatTensor_topk"
  c_topk_ :: Ptr C'THFloatTensor -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> CLLong -> CInt -> CInt -> CInt -> IO ()

-- | alias of c_topk_ with unused argument (for CTHState) to unify backpack signatures.
c_topk = const c_topk_

-- | c_tril :  r_ t k -> void
foreign import ccall "THTensorMath.h THFloatTensor_tril"
  c_tril_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CLLong -> IO ()

-- | alias of c_tril_ with unused argument (for CTHState) to unify backpack signatures.
c_tril = const c_tril_

-- | c_triu :  r_ t k -> void
foreign import ccall "THTensorMath.h THFloatTensor_triu"
  c_triu_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CLLong -> IO ()

-- | alias of c_triu_ with unused argument (for CTHState) to unify backpack signatures.
c_triu = const c_triu_

-- | c_cat :  r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h THFloatTensor_cat"
  c_cat_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ()

-- | alias of c_cat_ with unused argument (for CTHState) to unify backpack signatures.
c_cat = const c_cat_

-- | c_catArray :  result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h THFloatTensor_catArray"
  c_catArray_ :: Ptr C'THFloatTensor -> Ptr (Ptr C'THFloatTensor) -> CInt -> CInt -> IO ()

-- | alias of c_catArray_ with unused argument (for CTHState) to unify backpack signatures.
c_catArray = const c_catArray_

-- | c_equal :  ta tb -> int
foreign import ccall "THTensorMath.h THFloatTensor_equal"
  c_equal_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO CInt

-- | alias of c_equal_ with unused argument (for CTHState) to unify backpack signatures.
c_equal = const c_equal_

-- | c_ltValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_ltValue"
  c_ltValue_ :: Ptr C'THByteTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_ltValue_ with unused argument (for CTHState) to unify backpack signatures.
c_ltValue = const c_ltValue_

-- | c_leValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_leValue"
  c_leValue_ :: Ptr C'THByteTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_leValue_ with unused argument (for CTHState) to unify backpack signatures.
c_leValue = const c_leValue_

-- | c_gtValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_gtValue"
  c_gtValue_ :: Ptr C'THByteTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_gtValue_ with unused argument (for CTHState) to unify backpack signatures.
c_gtValue = const c_gtValue_

-- | c_geValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_geValue"
  c_geValue_ :: Ptr C'THByteTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_geValue_ with unused argument (for CTHState) to unify backpack signatures.
c_geValue = const c_geValue_

-- | c_neValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_neValue"
  c_neValue_ :: Ptr C'THByteTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_neValue_ with unused argument (for CTHState) to unify backpack signatures.
c_neValue = const c_neValue_

-- | c_eqValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_eqValue"
  c_eqValue_ :: Ptr C'THByteTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_eqValue_ with unused argument (for CTHState) to unify backpack signatures.
c_eqValue = const c_eqValue_

-- | c_ltValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_ltValueT"
  c_ltValueT_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_ltValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_ltValueT = const c_ltValueT_

-- | c_leValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_leValueT"
  c_leValueT_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_leValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_leValueT = const c_leValueT_

-- | c_gtValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_gtValueT"
  c_gtValueT_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_gtValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_gtValueT = const c_gtValueT_

-- | c_geValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_geValueT"
  c_geValueT_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_geValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_geValueT = const c_geValueT_

-- | c_neValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_neValueT"
  c_neValueT_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_neValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_neValueT = const c_neValueT_

-- | c_eqValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_eqValueT"
  c_eqValueT_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_eqValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_eqValueT = const c_eqValueT_

-- | c_ltTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_ltTensor"
  c_ltTensor_ :: Ptr C'THByteTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_ltTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_ltTensor = const c_ltTensor_

-- | c_leTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_leTensor"
  c_leTensor_ :: Ptr C'THByteTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_leTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_leTensor = const c_leTensor_

-- | c_gtTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_gtTensor"
  c_gtTensor_ :: Ptr C'THByteTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_gtTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_gtTensor = const c_gtTensor_

-- | c_geTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_geTensor"
  c_geTensor_ :: Ptr C'THByteTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_geTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_geTensor = const c_geTensor_

-- | c_neTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_neTensor"
  c_neTensor_ :: Ptr C'THByteTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_neTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_neTensor = const c_neTensor_

-- | c_eqTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_eqTensor"
  c_eqTensor_ :: Ptr C'THByteTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_eqTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_eqTensor = const c_eqTensor_

-- | c_ltTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_ltTensorT"
  c_ltTensorT_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_ltTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_ltTensorT = const c_ltTensorT_

-- | c_leTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_leTensorT"
  c_leTensorT_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_leTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_leTensorT = const c_leTensorT_

-- | c_gtTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_gtTensorT"
  c_gtTensorT_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_gtTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_gtTensorT = const c_gtTensorT_

-- | c_geTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_geTensorT"
  c_geTensorT_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_geTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_geTensorT = const c_geTensorT_

-- | c_neTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_neTensorT"
  c_neTensorT_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_neTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_neTensorT = const c_neTensorT_

-- | c_eqTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_eqTensorT"
  c_eqTensorT_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_eqTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_eqTensorT = const c_eqTensorT_

-- | c_pow :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_pow"
  c_pow_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_pow_ with unused argument (for CTHState) to unify backpack signatures.
c_pow = const c_pow_

-- | c_tpow :  r_ value t -> void
foreign import ccall "THTensorMath.h THFloatTensor_tpow"
  c_tpow_ :: Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_tpow_ with unused argument (for CTHState) to unify backpack signatures.
c_tpow = const c_tpow_

-- | c_abs :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_abs"
  c_abs_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_abs_ with unused argument (for CTHState) to unify backpack signatures.
c_abs = const c_abs_

-- | c_sigmoid :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_sigmoid"
  c_sigmoid_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_sigmoid_ with unused argument (for CTHState) to unify backpack signatures.
c_sigmoid = const c_sigmoid_

-- | c_log :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_log"
  c_log_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_log_ with unused argument (for CTHState) to unify backpack signatures.
c_log = const c_log_

-- | c_lgamma :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_lgamma"
  c_lgamma_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_lgamma_ with unused argument (for CTHState) to unify backpack signatures.
c_lgamma = const c_lgamma_

-- | c_digamma :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_digamma"
  c_digamma_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_digamma_ with unused argument (for CTHState) to unify backpack signatures.
c_digamma = const c_digamma_

-- | c_trigamma :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_trigamma"
  c_trigamma_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_trigamma_ with unused argument (for CTHState) to unify backpack signatures.
c_trigamma = const c_trigamma_

-- | c_polygamma :  r_ n t -> void
foreign import ccall "THTensorMath.h THFloatTensor_polygamma"
  c_polygamma_ :: Ptr C'THFloatTensor -> CLLong -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_polygamma_ with unused argument (for CTHState) to unify backpack signatures.
c_polygamma = const c_polygamma_

-- | c_log1p :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_log1p"
  c_log1p_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_log1p_ with unused argument (for CTHState) to unify backpack signatures.
c_log1p = const c_log1p_

-- | c_exp :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_exp"
  c_exp_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_exp_ with unused argument (for CTHState) to unify backpack signatures.
c_exp = const c_exp_

-- | c_expm1 :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_expm1"
  c_expm1_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_expm1_ with unused argument (for CTHState) to unify backpack signatures.
c_expm1 = const c_expm1_

-- | c_cos :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_cos"
  c_cos_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_cos_ with unused argument (for CTHState) to unify backpack signatures.
c_cos = const c_cos_

-- | c_acos :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_acos"
  c_acos_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_acos_ with unused argument (for CTHState) to unify backpack signatures.
c_acos = const c_acos_

-- | c_cosh :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_cosh"
  c_cosh_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_cosh_ with unused argument (for CTHState) to unify backpack signatures.
c_cosh = const c_cosh_

-- | c_sin :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_sin"
  c_sin_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_sin_ with unused argument (for CTHState) to unify backpack signatures.
c_sin = const c_sin_

-- | c_asin :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_asin"
  c_asin_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_asin_ with unused argument (for CTHState) to unify backpack signatures.
c_asin = const c_asin_

-- | c_sinh :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_sinh"
  c_sinh_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_sinh_ with unused argument (for CTHState) to unify backpack signatures.
c_sinh = const c_sinh_

-- | c_tan :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_tan"
  c_tan_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_tan_ with unused argument (for CTHState) to unify backpack signatures.
c_tan = const c_tan_

-- | c_atan :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_atan"
  c_atan_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_atan_ with unused argument (for CTHState) to unify backpack signatures.
c_atan = const c_atan_

-- | c_atan2 :  r_ tx ty -> void
foreign import ccall "THTensorMath.h THFloatTensor_atan2"
  c_atan2_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_atan2_ with unused argument (for CTHState) to unify backpack signatures.
c_atan2 = const c_atan2_

-- | c_tanh :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_tanh"
  c_tanh_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_tanh_ with unused argument (for CTHState) to unify backpack signatures.
c_tanh = const c_tanh_

-- | c_erf :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_erf"
  c_erf_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_erf_ with unused argument (for CTHState) to unify backpack signatures.
c_erf = const c_erf_

-- | c_erfinv :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_erfinv"
  c_erfinv_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_erfinv_ with unused argument (for CTHState) to unify backpack signatures.
c_erfinv = const c_erfinv_

-- | c_sqrt :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_sqrt"
  c_sqrt_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_sqrt_ with unused argument (for CTHState) to unify backpack signatures.
c_sqrt = const c_sqrt_

-- | c_rsqrt :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_rsqrt"
  c_rsqrt_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_rsqrt_ with unused argument (for CTHState) to unify backpack signatures.
c_rsqrt = const c_rsqrt_

-- | c_ceil :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_ceil"
  c_ceil_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_ceil_ with unused argument (for CTHState) to unify backpack signatures.
c_ceil = const c_ceil_

-- | c_floor :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_floor"
  c_floor_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_floor_ with unused argument (for CTHState) to unify backpack signatures.
c_floor = const c_floor_

-- | c_round :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_round"
  c_round_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_round_ with unused argument (for CTHState) to unify backpack signatures.
c_round = const c_round_

-- | c_trunc :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_trunc"
  c_trunc_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_trunc_ with unused argument (for CTHState) to unify backpack signatures.
c_trunc = const c_trunc_

-- | c_frac :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_frac"
  c_frac_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_frac_ with unused argument (for CTHState) to unify backpack signatures.
c_frac = const c_frac_

-- | c_lerp :  r_ a b weight -> void
foreign import ccall "THTensorMath.h THFloatTensor_lerp"
  c_lerp_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ()

-- | alias of c_lerp_ with unused argument (for CTHState) to unify backpack signatures.
c_lerp = const c_lerp_

-- | c_mean :  r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_mean"
  c_mean_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ()

-- | alias of c_mean_ with unused argument (for CTHState) to unify backpack signatures.
c_mean = const c_mean_

-- | c_std :  r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_std"
  c_std_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> IO ()

-- | alias of c_std_ with unused argument (for CTHState) to unify backpack signatures.
c_std = const c_std_

-- | c_var :  r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_var"
  c_var_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> IO ()

-- | alias of c_var_ with unused argument (for CTHState) to unify backpack signatures.
c_var = const c_var_

-- | c_norm :  r_ t value dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_norm"
  c_norm_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> CInt -> CInt -> IO ()

-- | alias of c_norm_ with unused argument (for CTHState) to unify backpack signatures.
c_norm = const c_norm_

-- | c_renorm :  r_ t value dimension maxnorm -> void
foreign import ccall "THTensorMath.h THFloatTensor_renorm"
  c_renorm_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> CInt -> CFloat -> IO ()

-- | alias of c_renorm_ with unused argument (for CTHState) to unify backpack signatures.
c_renorm = const c_renorm_

-- | c_dist :  a b value -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_dist"
  c_dist_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO CDouble

-- | alias of c_dist_ with unused argument (for CTHState) to unify backpack signatures.
c_dist = const c_dist_

-- | c_histc :  hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THFloatTensor_histc"
  c_histc_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CLLong -> CFloat -> CFloat -> IO ()

-- | alias of c_histc_ with unused argument (for CTHState) to unify backpack signatures.
c_histc = const c_histc_

-- | c_bhistc :  hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THFloatTensor_bhistc"
  c_bhistc_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CLLong -> CFloat -> CFloat -> IO ()

-- | alias of c_bhistc_ with unused argument (for CTHState) to unify backpack signatures.
c_bhistc = const c_bhistc_

-- | c_meanall :  self -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_meanall"
  c_meanall_ :: Ptr C'THFloatTensor -> IO CDouble

-- | alias of c_meanall_ with unused argument (for CTHState) to unify backpack signatures.
c_meanall = const c_meanall_

-- | c_varall :  self biased -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_varall"
  c_varall_ :: Ptr C'THFloatTensor -> CInt -> IO CDouble

-- | alias of c_varall_ with unused argument (for CTHState) to unify backpack signatures.
c_varall = const c_varall_

-- | c_stdall :  self biased -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_stdall"
  c_stdall_ :: Ptr C'THFloatTensor -> CInt -> IO CDouble

-- | alias of c_stdall_ with unused argument (for CTHState) to unify backpack signatures.
c_stdall = const c_stdall_

-- | c_normall :  t value -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_normall"
  c_normall_ :: Ptr C'THFloatTensor -> CFloat -> IO CDouble

-- | alias of c_normall_ with unused argument (for CTHState) to unify backpack signatures.
c_normall = const c_normall_

-- | c_linspace :  r_ a b n -> void
foreign import ccall "THTensorMath.h THFloatTensor_linspace"
  c_linspace_ :: Ptr C'THFloatTensor -> CFloat -> CFloat -> CLLong -> IO ()

-- | alias of c_linspace_ with unused argument (for CTHState) to unify backpack signatures.
c_linspace = const c_linspace_

-- | c_logspace :  r_ a b n -> void
foreign import ccall "THTensorMath.h THFloatTensor_logspace"
  c_logspace_ :: Ptr C'THFloatTensor -> CFloat -> CFloat -> CLLong -> IO ()

-- | alias of c_logspace_ with unused argument (for CTHState) to unify backpack signatures.
c_logspace = const c_logspace_

-- | c_rand :  r_ _generator size -> void
foreign import ccall "THTensorMath.h THFloatTensor_rand"
  c_rand_ :: Ptr C'THFloatTensor -> Ptr C'THGenerator -> Ptr C'THLongStorage -> IO ()

-- | alias of c_rand_ with unused argument (for CTHState) to unify backpack signatures.
c_rand = const c_rand_

-- | c_randn :  r_ _generator size -> void
foreign import ccall "THTensorMath.h THFloatTensor_randn"
  c_randn_ :: Ptr C'THFloatTensor -> Ptr C'THGenerator -> Ptr C'THLongStorage -> IO ()

-- | alias of c_randn_ with unused argument (for CTHState) to unify backpack signatures.
c_randn = const c_randn_

-- | c_dirichlet_grad :  self x alpha total -> void
foreign import ccall "THTensorMath.h THFloatTensor_dirichlet_grad"
  c_dirichlet_grad_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_dirichlet_grad_ with unused argument (for CTHState) to unify backpack signatures.
c_dirichlet_grad = const c_dirichlet_grad_

-- | p_fill : Pointer to function : r_ value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_fill"
  p_fill_ :: FunPtr (Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_fill_ with unused argument (for CTHState) to unify backpack signatures.
p_fill = const p_fill_

-- | p_zero : Pointer to function : r_ -> void
foreign import ccall "THTensorMath.h &THFloatTensor_zero"
  p_zero_ :: FunPtr (Ptr C'THFloatTensor -> IO ())

-- | alias of p_zero_ with unused argument (for CTHState) to unify backpack signatures.
p_zero = const p_zero_

-- | p_maskedFill : Pointer to function : tensor mask value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_maskedFill"
  p_maskedFill_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THByteTensor -> CFloat -> IO ())

-- | alias of p_maskedFill_ with unused argument (for CTHState) to unify backpack signatures.
p_maskedFill = const p_maskedFill_

-- | p_maskedCopy : Pointer to function : tensor mask src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_maskedCopy"
  p_maskedCopy_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THByteTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_maskedCopy_ with unused argument (for CTHState) to unify backpack signatures.
p_maskedCopy = const p_maskedCopy_

-- | p_maskedSelect : Pointer to function : tensor src mask -> void
foreign import ccall "THTensorMath.h &THFloatTensor_maskedSelect"
  p_maskedSelect_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THByteTensor -> IO ())

-- | alias of p_maskedSelect_ with unused argument (for CTHState) to unify backpack signatures.
p_maskedSelect = const p_maskedSelect_

-- | p_nonzero : Pointer to function : subscript tensor -> void
foreign import ccall "THTensorMath.h &THFloatTensor_nonzero"
  p_nonzero_ :: FunPtr (Ptr C'THLongTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_nonzero_ with unused argument (for CTHState) to unify backpack signatures.
p_nonzero = const p_nonzero_

-- | p_indexSelect : Pointer to function : tensor src dim index -> void
foreign import ccall "THTensorMath.h &THFloatTensor_indexSelect"
  p_indexSelect_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> Ptr C'THLongTensor -> IO ())

-- | alias of p_indexSelect_ with unused argument (for CTHState) to unify backpack signatures.
p_indexSelect = const p_indexSelect_

-- | p_indexCopy : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_indexCopy"
  p_indexCopy_ :: FunPtr (Ptr C'THFloatTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_indexCopy_ with unused argument (for CTHState) to unify backpack signatures.
p_indexCopy = const p_indexCopy_

-- | p_indexAdd : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_indexAdd"
  p_indexAdd_ :: FunPtr (Ptr C'THFloatTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_indexAdd_ with unused argument (for CTHState) to unify backpack signatures.
p_indexAdd = const p_indexAdd_

-- | p_indexFill : Pointer to function : tensor dim index val -> void
foreign import ccall "THTensorMath.h &THFloatTensor_indexFill"
  p_indexFill_ :: FunPtr (Ptr C'THFloatTensor -> CInt -> Ptr C'THLongTensor -> CFloat -> IO ())

-- | alias of p_indexFill_ with unused argument (for CTHState) to unify backpack signatures.
p_indexFill = const p_indexFill_

-- | p_take : Pointer to function : tensor src index -> void
foreign import ccall "THTensorMath.h &THFloatTensor_take"
  p_take_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THLongTensor -> IO ())

-- | alias of p_take_ with unused argument (for CTHState) to unify backpack signatures.
p_take = const p_take_

-- | p_put : Pointer to function : tensor index src accumulate -> void
foreign import ccall "THTensorMath.h &THFloatTensor_put"
  p_put_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> CInt -> IO ())

-- | alias of p_put_ with unused argument (for CTHState) to unify backpack signatures.
p_put = const p_put_

-- | p_gather : Pointer to function : tensor src dim index -> void
foreign import ccall "THTensorMath.h &THFloatTensor_gather"
  p_gather_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> Ptr C'THLongTensor -> IO ())

-- | alias of p_gather_ with unused argument (for CTHState) to unify backpack signatures.
p_gather = const p_gather_

-- | p_scatter : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_scatter"
  p_scatter_ :: FunPtr (Ptr C'THFloatTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_scatter_ with unused argument (for CTHState) to unify backpack signatures.
p_scatter = const p_scatter_

-- | p_scatterAdd : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_scatterAdd"
  p_scatterAdd_ :: FunPtr (Ptr C'THFloatTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_scatterAdd_ with unused argument (for CTHState) to unify backpack signatures.
p_scatterAdd = const p_scatterAdd_

-- | p_scatterFill : Pointer to function : tensor dim index val -> void
foreign import ccall "THTensorMath.h &THFloatTensor_scatterFill"
  p_scatterFill_ :: FunPtr (Ptr C'THFloatTensor -> CInt -> Ptr C'THLongTensor -> CFloat -> IO ())

-- | alias of p_scatterFill_ with unused argument (for CTHState) to unify backpack signatures.
p_scatterFill = const p_scatterFill_

-- | p_dot : Pointer to function : t src -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_dot"
  p_dot_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO CDouble)

-- | alias of p_dot_ with unused argument (for CTHState) to unify backpack signatures.
p_dot = const p_dot_

-- | p_minall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THFloatTensor_minall"
  p_minall_ :: FunPtr (Ptr C'THFloatTensor -> IO CFloat)

-- | alias of p_minall_ with unused argument (for CTHState) to unify backpack signatures.
p_minall = const p_minall_

-- | p_maxall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THFloatTensor_maxall"
  p_maxall_ :: FunPtr (Ptr C'THFloatTensor -> IO CFloat)

-- | alias of p_maxall_ with unused argument (for CTHState) to unify backpack signatures.
p_maxall = const p_maxall_

-- | p_medianall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THFloatTensor_medianall"
  p_medianall_ :: FunPtr (Ptr C'THFloatTensor -> IO CFloat)

-- | alias of p_medianall_ with unused argument (for CTHState) to unify backpack signatures.
p_medianall = const p_medianall_

-- | p_sumall : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_sumall"
  p_sumall_ :: FunPtr (Ptr C'THFloatTensor -> IO CDouble)

-- | alias of p_sumall_ with unused argument (for CTHState) to unify backpack signatures.
p_sumall = const p_sumall_

-- | p_prodall : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_prodall"
  p_prodall_ :: FunPtr (Ptr C'THFloatTensor -> IO CDouble)

-- | alias of p_prodall_ with unused argument (for CTHState) to unify backpack signatures.
p_prodall = const p_prodall_

-- | p_neg : Pointer to function : self src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_neg"
  p_neg_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_neg_ with unused argument (for CTHState) to unify backpack signatures.
p_neg = const p_neg_

-- | p_cinv : Pointer to function : self src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cinv"
  p_cinv_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_cinv_ with unused argument (for CTHState) to unify backpack signatures.
p_cinv = const p_cinv_

-- | p_add : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_add"
  p_add_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_add_ with unused argument (for CTHState) to unify backpack signatures.
p_add = const p_add_

-- | p_sub : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sub"
  p_sub_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_sub_ with unused argument (for CTHState) to unify backpack signatures.
p_sub = const p_sub_

-- | p_add_scaled : Pointer to function : r_ t value alpha -> void
foreign import ccall "THTensorMath.h &THFloatTensor_add_scaled"
  p_add_scaled_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> CFloat -> IO ())

-- | alias of p_add_scaled_ with unused argument (for CTHState) to unify backpack signatures.
p_add_scaled = const p_add_scaled_

-- | p_sub_scaled : Pointer to function : r_ t value alpha -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sub_scaled"
  p_sub_scaled_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> CFloat -> IO ())

-- | alias of p_sub_scaled_ with unused argument (for CTHState) to unify backpack signatures.
p_sub_scaled = const p_sub_scaled_

-- | p_mul : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_mul"
  p_mul_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_mul_ with unused argument (for CTHState) to unify backpack signatures.
p_mul = const p_mul_

-- | p_div : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_div"
  p_div_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_div_ with unused argument (for CTHState) to unify backpack signatures.
p_div = const p_div_

-- | p_lshift : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_lshift"
  p_lshift_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_lshift_ with unused argument (for CTHState) to unify backpack signatures.
p_lshift = const p_lshift_

-- | p_rshift : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_rshift"
  p_rshift_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_rshift_ with unused argument (for CTHState) to unify backpack signatures.
p_rshift = const p_rshift_

-- | p_fmod : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_fmod"
  p_fmod_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_fmod_ with unused argument (for CTHState) to unify backpack signatures.
p_fmod = const p_fmod_

-- | p_remainder : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_remainder"
  p_remainder_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_remainder_ with unused argument (for CTHState) to unify backpack signatures.
p_remainder = const p_remainder_

-- | p_clamp : Pointer to function : r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_clamp"
  p_clamp_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> CFloat -> IO ())

-- | alias of p_clamp_ with unused argument (for CTHState) to unify backpack signatures.
p_clamp = const p_clamp_

-- | p_bitand : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_bitand"
  p_bitand_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_bitand_ with unused argument (for CTHState) to unify backpack signatures.
p_bitand = const p_bitand_

-- | p_bitor : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_bitor"
  p_bitor_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_bitor_ with unused argument (for CTHState) to unify backpack signatures.
p_bitor = const p_bitor_

-- | p_bitxor : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_bitxor"
  p_bitxor_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_bitxor_ with unused argument (for CTHState) to unify backpack signatures.
p_bitxor = const p_bitxor_

-- | p_cadd : Pointer to function : r_ t value src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cadd"
  p_cadd_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_cadd_ with unused argument (for CTHState) to unify backpack signatures.
p_cadd = const p_cadd_

-- | p_csub : Pointer to function : self src1 value src2 -> void
foreign import ccall "THTensorMath.h &THFloatTensor_csub"
  p_csub_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_csub_ with unused argument (for CTHState) to unify backpack signatures.
p_csub = const p_csub_

-- | p_cmul : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cmul"
  p_cmul_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_cmul_ with unused argument (for CTHState) to unify backpack signatures.
p_cmul = const p_cmul_

-- | p_cpow : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cpow"
  p_cpow_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_cpow_ with unused argument (for CTHState) to unify backpack signatures.
p_cpow = const p_cpow_

-- | p_cdiv : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cdiv"
  p_cdiv_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_cdiv_ with unused argument (for CTHState) to unify backpack signatures.
p_cdiv = const p_cdiv_

-- | p_clshift : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_clshift"
  p_clshift_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_clshift_ with unused argument (for CTHState) to unify backpack signatures.
p_clshift = const p_clshift_

-- | p_crshift : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_crshift"
  p_crshift_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_crshift_ with unused argument (for CTHState) to unify backpack signatures.
p_crshift = const p_crshift_

-- | p_cfmod : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cfmod"
  p_cfmod_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_cfmod_ with unused argument (for CTHState) to unify backpack signatures.
p_cfmod = const p_cfmod_

-- | p_cremainder : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cremainder"
  p_cremainder_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_cremainder_ with unused argument (for CTHState) to unify backpack signatures.
p_cremainder = const p_cremainder_

-- | p_cbitand : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cbitand"
  p_cbitand_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_cbitand_ with unused argument (for CTHState) to unify backpack signatures.
p_cbitand = const p_cbitand_

-- | p_cbitor : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cbitor"
  p_cbitor_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_cbitor_ with unused argument (for CTHState) to unify backpack signatures.
p_cbitor = const p_cbitor_

-- | p_cbitxor : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cbitxor"
  p_cbitxor_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_cbitxor_ with unused argument (for CTHState) to unify backpack signatures.
p_cbitxor = const p_cbitxor_

-- | p_addcmul : Pointer to function : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h &THFloatTensor_addcmul"
  p_addcmul_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_addcmul_ with unused argument (for CTHState) to unify backpack signatures.
p_addcmul = const p_addcmul_

-- | p_addcdiv : Pointer to function : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h &THFloatTensor_addcdiv"
  p_addcdiv_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_addcdiv_ with unused argument (for CTHState) to unify backpack signatures.
p_addcdiv = const p_addcdiv_

-- | p_addmv : Pointer to function : r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h &THFloatTensor_addmv"
  p_addmv_ :: FunPtr (Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_addmv_ with unused argument (for CTHState) to unify backpack signatures.
p_addmv = const p_addmv_

-- | p_addmm : Pointer to function : r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h &THFloatTensor_addmm"
  p_addmm_ :: FunPtr (Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_addmm_ with unused argument (for CTHState) to unify backpack signatures.
p_addmm = const p_addmm_

-- | p_addr : Pointer to function : r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h &THFloatTensor_addr"
  p_addr_ :: FunPtr (Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_addr_ with unused argument (for CTHState) to unify backpack signatures.
p_addr = const p_addr_

-- | p_addbmm : Pointer to function : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h &THFloatTensor_addbmm"
  p_addbmm_ :: FunPtr (Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_addbmm_ with unused argument (for CTHState) to unify backpack signatures.
p_addbmm = const p_addbmm_

-- | p_baddbmm : Pointer to function : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h &THFloatTensor_baddbmm"
  p_baddbmm_ :: FunPtr (Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_baddbmm_ with unused argument (for CTHState) to unify backpack signatures.
p_baddbmm = const p_baddbmm_

-- | p_match : Pointer to function : r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h &THFloatTensor_match"
  p_match_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_match_ with unused argument (for CTHState) to unify backpack signatures.
p_match = const p_match_

-- | p_numel : Pointer to function : t -> ptrdiff_t
foreign import ccall "THTensorMath.h &THFloatTensor_numel"
  p_numel_ :: FunPtr (Ptr C'THFloatTensor -> IO CPtrdiff)

-- | alias of p_numel_ with unused argument (for CTHState) to unify backpack signatures.
p_numel = const p_numel_

-- | p_preserveReduceDimSemantics : Pointer to function : r_ in_dims reduce_dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_preserveReduceDimSemantics"
  p_preserveReduceDimSemantics_ :: FunPtr (Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> IO ())

-- | alias of p_preserveReduceDimSemantics_ with unused argument (for CTHState) to unify backpack signatures.
p_preserveReduceDimSemantics = const p_preserveReduceDimSemantics_

-- | p_max : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_max"
  p_max_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ())

-- | alias of p_max_ with unused argument (for CTHState) to unify backpack signatures.
p_max = const p_max_

-- | p_min : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_min"
  p_min_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ())

-- | alias of p_min_ with unused argument (for CTHState) to unify backpack signatures.
p_min = const p_min_

-- | p_kthvalue : Pointer to function : values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_kthvalue"
  p_kthvalue_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> CLLong -> CInt -> CInt -> IO ())

-- | alias of p_kthvalue_ with unused argument (for CTHState) to unify backpack signatures.
p_kthvalue = const p_kthvalue_

-- | p_mode : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_mode"
  p_mode_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ())

-- | alias of p_mode_ with unused argument (for CTHState) to unify backpack signatures.
p_mode = const p_mode_

-- | p_median : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_median"
  p_median_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ())

-- | alias of p_median_ with unused argument (for CTHState) to unify backpack signatures.
p_median = const p_median_

-- | p_sum : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sum"
  p_sum_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ())

-- | alias of p_sum_ with unused argument (for CTHState) to unify backpack signatures.
p_sum = const p_sum_

-- | p_prod : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_prod"
  p_prod_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ())

-- | alias of p_prod_ with unused argument (for CTHState) to unify backpack signatures.
p_prod = const p_prod_

-- | p_cumsum : Pointer to function : r_ t dimension -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cumsum"
  p_cumsum_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ())

-- | alias of p_cumsum_ with unused argument (for CTHState) to unify backpack signatures.
p_cumsum = const p_cumsum_

-- | p_cumprod : Pointer to function : r_ t dimension -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cumprod"
  p_cumprod_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ())

-- | alias of p_cumprod_ with unused argument (for CTHState) to unify backpack signatures.
p_cumprod = const p_cumprod_

-- | p_sign : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sign"
  p_sign_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_sign_ with unused argument (for CTHState) to unify backpack signatures.
p_sign = const p_sign_

-- | p_trace : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_trace"
  p_trace_ :: FunPtr (Ptr C'THFloatTensor -> IO CDouble)

-- | alias of p_trace_ with unused argument (for CTHState) to unify backpack signatures.
p_trace = const p_trace_

-- | p_cross : Pointer to function : r_ a b dimension -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cross"
  p_cross_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ())

-- | alias of p_cross_ with unused argument (for CTHState) to unify backpack signatures.
p_cross = const p_cross_

-- | p_cmax : Pointer to function : r t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cmax"
  p_cmax_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_cmax_ with unused argument (for CTHState) to unify backpack signatures.
p_cmax = const p_cmax_

-- | p_cmin : Pointer to function : r t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cmin"
  p_cmin_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_cmin_ with unused argument (for CTHState) to unify backpack signatures.
p_cmin = const p_cmin_

-- | p_cmaxValue : Pointer to function : r t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cmaxValue"
  p_cmaxValue_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_cmaxValue_ with unused argument (for CTHState) to unify backpack signatures.
p_cmaxValue = const p_cmaxValue_

-- | p_cminValue : Pointer to function : r t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cminValue"
  p_cminValue_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_cminValue_ with unused argument (for CTHState) to unify backpack signatures.
p_cminValue = const p_cminValue_

-- | p_zeros : Pointer to function : r_ size -> void
foreign import ccall "THTensorMath.h &THFloatTensor_zeros"
  p_zeros_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THLongStorage -> IO ())

-- | alias of p_zeros_ with unused argument (for CTHState) to unify backpack signatures.
p_zeros = const p_zeros_

-- | p_zerosLike : Pointer to function : r_ input -> void
foreign import ccall "THTensorMath.h &THFloatTensor_zerosLike"
  p_zerosLike_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_zerosLike_ with unused argument (for CTHState) to unify backpack signatures.
p_zerosLike = const p_zerosLike_

-- | p_ones : Pointer to function : r_ size -> void
foreign import ccall "THTensorMath.h &THFloatTensor_ones"
  p_ones_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THLongStorage -> IO ())

-- | alias of p_ones_ with unused argument (for CTHState) to unify backpack signatures.
p_ones = const p_ones_

-- | p_onesLike : Pointer to function : r_ input -> void
foreign import ccall "THTensorMath.h &THFloatTensor_onesLike"
  p_onesLike_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_onesLike_ with unused argument (for CTHState) to unify backpack signatures.
p_onesLike = const p_onesLike_

-- | p_diag : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THFloatTensor_diag"
  p_diag_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ())

-- | alias of p_diag_ with unused argument (for CTHState) to unify backpack signatures.
p_diag = const p_diag_

-- | p_eye : Pointer to function : r_ n m -> void
foreign import ccall "THTensorMath.h &THFloatTensor_eye"
  p_eye_ :: FunPtr (Ptr C'THFloatTensor -> CLLong -> CLLong -> IO ())

-- | alias of p_eye_ with unused argument (for CTHState) to unify backpack signatures.
p_eye = const p_eye_

-- | p_arange : Pointer to function : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h &THFloatTensor_arange"
  p_arange_ :: FunPtr (Ptr C'THFloatTensor -> CDouble -> CDouble -> CDouble -> IO ())

-- | alias of p_arange_ with unused argument (for CTHState) to unify backpack signatures.
p_arange = const p_arange_

-- | p_range : Pointer to function : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h &THFloatTensor_range"
  p_range_ :: FunPtr (Ptr C'THFloatTensor -> CDouble -> CDouble -> CDouble -> IO ())

-- | alias of p_range_ with unused argument (for CTHState) to unify backpack signatures.
p_range = const p_range_

-- | p_randperm : Pointer to function : r_ _generator n -> void
foreign import ccall "THTensorMath.h &THFloatTensor_randperm"
  p_randperm_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THGenerator -> CLLong -> IO ())

-- | alias of p_randperm_ with unused argument (for CTHState) to unify backpack signatures.
p_randperm = const p_randperm_

-- | p_reshape : Pointer to function : r_ t size -> void
foreign import ccall "THTensorMath.h &THFloatTensor_reshape"
  p_reshape_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THLongStorage -> IO ())

-- | alias of p_reshape_ with unused argument (for CTHState) to unify backpack signatures.
p_reshape = const p_reshape_

-- | p_sort : Pointer to function : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sort"
  p_sort_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ())

-- | alias of p_sort_ with unused argument (for CTHState) to unify backpack signatures.
p_sort = const p_sort_

-- | p_topk : Pointer to function : rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h &THFloatTensor_topk"
  p_topk_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> CLLong -> CInt -> CInt -> CInt -> IO ())

-- | alias of p_topk_ with unused argument (for CTHState) to unify backpack signatures.
p_topk = const p_topk_

-- | p_tril : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THFloatTensor_tril"
  p_tril_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CLLong -> IO ())

-- | alias of p_tril_ with unused argument (for CTHState) to unify backpack signatures.
p_tril = const p_tril_

-- | p_triu : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THFloatTensor_triu"
  p_triu_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CLLong -> IO ())

-- | alias of p_triu_ with unused argument (for CTHState) to unify backpack signatures.
p_triu = const p_triu_

-- | p_cat : Pointer to function : r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cat"
  p_cat_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> IO ())

-- | alias of p_cat_ with unused argument (for CTHState) to unify backpack signatures.
p_cat = const p_cat_

-- | p_catArray : Pointer to function : result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h &THFloatTensor_catArray"
  p_catArray_ :: FunPtr (Ptr C'THFloatTensor -> Ptr (Ptr C'THFloatTensor) -> CInt -> CInt -> IO ())

-- | alias of p_catArray_ with unused argument (for CTHState) to unify backpack signatures.
p_catArray = const p_catArray_

-- | p_equal : Pointer to function : ta tb -> int
foreign import ccall "THTensorMath.h &THFloatTensor_equal"
  p_equal_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO CInt)

-- | alias of p_equal_ with unused argument (for CTHState) to unify backpack signatures.
p_equal = const p_equal_

-- | p_ltValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_ltValue"
  p_ltValue_ :: FunPtr (Ptr C'THByteTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_ltValue_ with unused argument (for CTHState) to unify backpack signatures.
p_ltValue = const p_ltValue_

-- | p_leValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_leValue"
  p_leValue_ :: FunPtr (Ptr C'THByteTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_leValue_ with unused argument (for CTHState) to unify backpack signatures.
p_leValue = const p_leValue_

-- | p_gtValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_gtValue"
  p_gtValue_ :: FunPtr (Ptr C'THByteTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_gtValue_ with unused argument (for CTHState) to unify backpack signatures.
p_gtValue = const p_gtValue_

-- | p_geValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_geValue"
  p_geValue_ :: FunPtr (Ptr C'THByteTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_geValue_ with unused argument (for CTHState) to unify backpack signatures.
p_geValue = const p_geValue_

-- | p_neValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_neValue"
  p_neValue_ :: FunPtr (Ptr C'THByteTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_neValue_ with unused argument (for CTHState) to unify backpack signatures.
p_neValue = const p_neValue_

-- | p_eqValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_eqValue"
  p_eqValue_ :: FunPtr (Ptr C'THByteTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_eqValue_ with unused argument (for CTHState) to unify backpack signatures.
p_eqValue = const p_eqValue_

-- | p_ltValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_ltValueT"
  p_ltValueT_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_ltValueT_ with unused argument (for CTHState) to unify backpack signatures.
p_ltValueT = const p_ltValueT_

-- | p_leValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_leValueT"
  p_leValueT_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_leValueT_ with unused argument (for CTHState) to unify backpack signatures.
p_leValueT = const p_leValueT_

-- | p_gtValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_gtValueT"
  p_gtValueT_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_gtValueT_ with unused argument (for CTHState) to unify backpack signatures.
p_gtValueT = const p_gtValueT_

-- | p_geValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_geValueT"
  p_geValueT_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_geValueT_ with unused argument (for CTHState) to unify backpack signatures.
p_geValueT = const p_geValueT_

-- | p_neValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_neValueT"
  p_neValueT_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_neValueT_ with unused argument (for CTHState) to unify backpack signatures.
p_neValueT = const p_neValueT_

-- | p_eqValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_eqValueT"
  p_eqValueT_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_eqValueT_ with unused argument (for CTHState) to unify backpack signatures.
p_eqValueT = const p_eqValueT_

-- | p_ltTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_ltTensor"
  p_ltTensor_ :: FunPtr (Ptr C'THByteTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_ltTensor_ with unused argument (for CTHState) to unify backpack signatures.
p_ltTensor = const p_ltTensor_

-- | p_leTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_leTensor"
  p_leTensor_ :: FunPtr (Ptr C'THByteTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_leTensor_ with unused argument (for CTHState) to unify backpack signatures.
p_leTensor = const p_leTensor_

-- | p_gtTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_gtTensor"
  p_gtTensor_ :: FunPtr (Ptr C'THByteTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_gtTensor_ with unused argument (for CTHState) to unify backpack signatures.
p_gtTensor = const p_gtTensor_

-- | p_geTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_geTensor"
  p_geTensor_ :: FunPtr (Ptr C'THByteTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_geTensor_ with unused argument (for CTHState) to unify backpack signatures.
p_geTensor = const p_geTensor_

-- | p_neTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_neTensor"
  p_neTensor_ :: FunPtr (Ptr C'THByteTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_neTensor_ with unused argument (for CTHState) to unify backpack signatures.
p_neTensor = const p_neTensor_

-- | p_eqTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_eqTensor"
  p_eqTensor_ :: FunPtr (Ptr C'THByteTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_eqTensor_ with unused argument (for CTHState) to unify backpack signatures.
p_eqTensor = const p_eqTensor_

-- | p_ltTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_ltTensorT"
  p_ltTensorT_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_ltTensorT_ with unused argument (for CTHState) to unify backpack signatures.
p_ltTensorT = const p_ltTensorT_

-- | p_leTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_leTensorT"
  p_leTensorT_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_leTensorT_ with unused argument (for CTHState) to unify backpack signatures.
p_leTensorT = const p_leTensorT_

-- | p_gtTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_gtTensorT"
  p_gtTensorT_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_gtTensorT_ with unused argument (for CTHState) to unify backpack signatures.
p_gtTensorT = const p_gtTensorT_

-- | p_geTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_geTensorT"
  p_geTensorT_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_geTensorT_ with unused argument (for CTHState) to unify backpack signatures.
p_geTensorT = const p_geTensorT_

-- | p_neTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_neTensorT"
  p_neTensorT_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_neTensorT_ with unused argument (for CTHState) to unify backpack signatures.
p_neTensorT = const p_neTensorT_

-- | p_eqTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_eqTensorT"
  p_eqTensorT_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_eqTensorT_ with unused argument (for CTHState) to unify backpack signatures.
p_eqTensorT = const p_eqTensorT_

-- | p_pow : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_pow"
  p_pow_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_pow_ with unused argument (for CTHState) to unify backpack signatures.
p_pow = const p_pow_

-- | p_tpow : Pointer to function : r_ value t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_tpow"
  p_tpow_ :: FunPtr (Ptr C'THFloatTensor -> CFloat -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_tpow_ with unused argument (for CTHState) to unify backpack signatures.
p_tpow = const p_tpow_

-- | p_abs : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_abs"
  p_abs_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_abs_ with unused argument (for CTHState) to unify backpack signatures.
p_abs = const p_abs_

-- | p_sigmoid : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sigmoid"
  p_sigmoid_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_sigmoid_ with unused argument (for CTHState) to unify backpack signatures.
p_sigmoid = const p_sigmoid_

-- | p_log : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_log"
  p_log_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_log_ with unused argument (for CTHState) to unify backpack signatures.
p_log = const p_log_

-- | p_lgamma : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_lgamma"
  p_lgamma_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_lgamma_ with unused argument (for CTHState) to unify backpack signatures.
p_lgamma = const p_lgamma_

-- | p_digamma : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_digamma"
  p_digamma_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_digamma_ with unused argument (for CTHState) to unify backpack signatures.
p_digamma = const p_digamma_

-- | p_trigamma : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_trigamma"
  p_trigamma_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_trigamma_ with unused argument (for CTHState) to unify backpack signatures.
p_trigamma = const p_trigamma_

-- | p_polygamma : Pointer to function : r_ n t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_polygamma"
  p_polygamma_ :: FunPtr (Ptr C'THFloatTensor -> CLLong -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_polygamma_ with unused argument (for CTHState) to unify backpack signatures.
p_polygamma = const p_polygamma_

-- | p_log1p : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_log1p"
  p_log1p_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_log1p_ with unused argument (for CTHState) to unify backpack signatures.
p_log1p = const p_log1p_

-- | p_exp : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_exp"
  p_exp_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_exp_ with unused argument (for CTHState) to unify backpack signatures.
p_exp = const p_exp_

-- | p_expm1 : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_expm1"
  p_expm1_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_expm1_ with unused argument (for CTHState) to unify backpack signatures.
p_expm1 = const p_expm1_

-- | p_cos : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cos"
  p_cos_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_cos_ with unused argument (for CTHState) to unify backpack signatures.
p_cos = const p_cos_

-- | p_acos : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_acos"
  p_acos_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_acos_ with unused argument (for CTHState) to unify backpack signatures.
p_acos = const p_acos_

-- | p_cosh : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cosh"
  p_cosh_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_cosh_ with unused argument (for CTHState) to unify backpack signatures.
p_cosh = const p_cosh_

-- | p_sin : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sin"
  p_sin_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_sin_ with unused argument (for CTHState) to unify backpack signatures.
p_sin = const p_sin_

-- | p_asin : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_asin"
  p_asin_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_asin_ with unused argument (for CTHState) to unify backpack signatures.
p_asin = const p_asin_

-- | p_sinh : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sinh"
  p_sinh_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_sinh_ with unused argument (for CTHState) to unify backpack signatures.
p_sinh = const p_sinh_

-- | p_tan : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_tan"
  p_tan_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_tan_ with unused argument (for CTHState) to unify backpack signatures.
p_tan = const p_tan_

-- | p_atan : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_atan"
  p_atan_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_atan_ with unused argument (for CTHState) to unify backpack signatures.
p_atan = const p_atan_

-- | p_atan2 : Pointer to function : r_ tx ty -> void
foreign import ccall "THTensorMath.h &THFloatTensor_atan2"
  p_atan2_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_atan2_ with unused argument (for CTHState) to unify backpack signatures.
p_atan2 = const p_atan2_

-- | p_tanh : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_tanh"
  p_tanh_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_tanh_ with unused argument (for CTHState) to unify backpack signatures.
p_tanh = const p_tanh_

-- | p_erf : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_erf"
  p_erf_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_erf_ with unused argument (for CTHState) to unify backpack signatures.
p_erf = const p_erf_

-- | p_erfinv : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_erfinv"
  p_erfinv_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_erfinv_ with unused argument (for CTHState) to unify backpack signatures.
p_erfinv = const p_erfinv_

-- | p_sqrt : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sqrt"
  p_sqrt_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_sqrt_ with unused argument (for CTHState) to unify backpack signatures.
p_sqrt = const p_sqrt_

-- | p_rsqrt : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_rsqrt"
  p_rsqrt_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_rsqrt_ with unused argument (for CTHState) to unify backpack signatures.
p_rsqrt = const p_rsqrt_

-- | p_ceil : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_ceil"
  p_ceil_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_ceil_ with unused argument (for CTHState) to unify backpack signatures.
p_ceil = const p_ceil_

-- | p_floor : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_floor"
  p_floor_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_floor_ with unused argument (for CTHState) to unify backpack signatures.
p_floor = const p_floor_

-- | p_round : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_round"
  p_round_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_round_ with unused argument (for CTHState) to unify backpack signatures.
p_round = const p_round_

-- | p_trunc : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_trunc"
  p_trunc_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_trunc_ with unused argument (for CTHState) to unify backpack signatures.
p_trunc = const p_trunc_

-- | p_frac : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_frac"
  p_frac_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_frac_ with unused argument (for CTHState) to unify backpack signatures.
p_frac = const p_frac_

-- | p_lerp : Pointer to function : r_ a b weight -> void
foreign import ccall "THTensorMath.h &THFloatTensor_lerp"
  p_lerp_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO ())

-- | alias of p_lerp_ with unused argument (for CTHState) to unify backpack signatures.
p_lerp = const p_lerp_

-- | p_mean : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_mean"
  p_mean_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ())

-- | alias of p_mean_ with unused argument (for CTHState) to unify backpack signatures.
p_mean = const p_mean_

-- | p_std : Pointer to function : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_std"
  p_std_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> IO ())

-- | alias of p_std_ with unused argument (for CTHState) to unify backpack signatures.
p_std = const p_std_

-- | p_var : Pointer to function : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_var"
  p_var_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CInt -> CInt -> CInt -> IO ())

-- | alias of p_var_ with unused argument (for CTHState) to unify backpack signatures.
p_var = const p_var_

-- | p_norm : Pointer to function : r_ t value dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_norm"
  p_norm_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> CInt -> CInt -> IO ())

-- | alias of p_norm_ with unused argument (for CTHState) to unify backpack signatures.
p_norm = const p_norm_

-- | p_renorm : Pointer to function : r_ t value dimension maxnorm -> void
foreign import ccall "THTensorMath.h &THFloatTensor_renorm"
  p_renorm_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> CInt -> CFloat -> IO ())

-- | alias of p_renorm_ with unused argument (for CTHState) to unify backpack signatures.
p_renorm = const p_renorm_

-- | p_dist : Pointer to function : a b value -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_dist"
  p_dist_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CFloat -> IO CDouble)

-- | alias of p_dist_ with unused argument (for CTHState) to unify backpack signatures.
p_dist = const p_dist_

-- | p_histc : Pointer to function : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h &THFloatTensor_histc"
  p_histc_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CLLong -> CFloat -> CFloat -> IO ())

-- | alias of p_histc_ with unused argument (for CTHState) to unify backpack signatures.
p_histc = const p_histc_

-- | p_bhistc : Pointer to function : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h &THFloatTensor_bhistc"
  p_bhistc_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> CLLong -> CFloat -> CFloat -> IO ())

-- | alias of p_bhistc_ with unused argument (for CTHState) to unify backpack signatures.
p_bhistc = const p_bhistc_

-- | p_meanall : Pointer to function : self -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_meanall"
  p_meanall_ :: FunPtr (Ptr C'THFloatTensor -> IO CDouble)

-- | alias of p_meanall_ with unused argument (for CTHState) to unify backpack signatures.
p_meanall = const p_meanall_

-- | p_varall : Pointer to function : self biased -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_varall"
  p_varall_ :: FunPtr (Ptr C'THFloatTensor -> CInt -> IO CDouble)

-- | alias of p_varall_ with unused argument (for CTHState) to unify backpack signatures.
p_varall = const p_varall_

-- | p_stdall : Pointer to function : self biased -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_stdall"
  p_stdall_ :: FunPtr (Ptr C'THFloatTensor -> CInt -> IO CDouble)

-- | alias of p_stdall_ with unused argument (for CTHState) to unify backpack signatures.
p_stdall = const p_stdall_

-- | p_normall : Pointer to function : t value -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_normall"
  p_normall_ :: FunPtr (Ptr C'THFloatTensor -> CFloat -> IO CDouble)

-- | alias of p_normall_ with unused argument (for CTHState) to unify backpack signatures.
p_normall = const p_normall_

-- | p_linspace : Pointer to function : r_ a b n -> void
foreign import ccall "THTensorMath.h &THFloatTensor_linspace"
  p_linspace_ :: FunPtr (Ptr C'THFloatTensor -> CFloat -> CFloat -> CLLong -> IO ())

-- | alias of p_linspace_ with unused argument (for CTHState) to unify backpack signatures.
p_linspace = const p_linspace_

-- | p_logspace : Pointer to function : r_ a b n -> void
foreign import ccall "THTensorMath.h &THFloatTensor_logspace"
  p_logspace_ :: FunPtr (Ptr C'THFloatTensor -> CFloat -> CFloat -> CLLong -> IO ())

-- | alias of p_logspace_ with unused argument (for CTHState) to unify backpack signatures.
p_logspace = const p_logspace_

-- | p_rand : Pointer to function : r_ _generator size -> void
foreign import ccall "THTensorMath.h &THFloatTensor_rand"
  p_rand_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THGenerator -> Ptr C'THLongStorage -> IO ())

-- | alias of p_rand_ with unused argument (for CTHState) to unify backpack signatures.
p_rand = const p_rand_

-- | p_randn : Pointer to function : r_ _generator size -> void
foreign import ccall "THTensorMath.h &THFloatTensor_randn"
  p_randn_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THGenerator -> Ptr C'THLongStorage -> IO ())

-- | alias of p_randn_ with unused argument (for CTHState) to unify backpack signatures.
p_randn = const p_randn_

-- | p_dirichlet_grad : Pointer to function : self x alpha total -> void
foreign import ccall "THTensorMath.h &THFloatTensor_dirichlet_grad"
  p_dirichlet_grad_ :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | alias of p_dirichlet_grad_ with unused argument (for CTHState) to unify backpack signatures.
p_dirichlet_grad = const p_dirichlet_grad_