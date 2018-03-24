{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Double.TensorMath where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_fill :  r_ value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_fill"
  c_fill_ :: Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_fill_ with unused argument (for CTHState) to unify backpack signatures.
c_fill :: Ptr C'THState -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_fill = const c_fill_

-- | c_zero :  r_ -> void
foreign import ccall "THTensorMath.h THDoubleTensor_zero"
  c_zero_ :: Ptr C'THDoubleTensor -> IO ()

-- | alias of c_zero_ with unused argument (for CTHState) to unify backpack signatures.
c_zero :: Ptr C'THState -> Ptr C'THDoubleTensor -> IO ()
c_zero = const c_zero_

-- | c_maskedFill :  tensor mask value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_maskedFill"
  c_maskedFill_ :: Ptr C'THDoubleTensor -> Ptr C'THByteTensor -> CDouble -> IO ()

-- | alias of c_maskedFill_ with unused argument (for CTHState) to unify backpack signatures.
c_maskedFill :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THByteTensor -> CDouble -> IO ()
c_maskedFill = const c_maskedFill_

-- | c_maskedCopy :  tensor mask src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_maskedCopy"
  c_maskedCopy_ :: Ptr C'THDoubleTensor -> Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_maskedCopy_ with unused argument (for CTHState) to unify backpack signatures.
c_maskedCopy :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> IO ()
c_maskedCopy = const c_maskedCopy_

-- | c_maskedSelect :  tensor src mask -> void
foreign import ccall "THTensorMath.h THDoubleTensor_maskedSelect"
  c_maskedSelect_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_maskedSelect_ with unused argument (for CTHState) to unify backpack signatures.
c_maskedSelect :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THByteTensor -> IO ()
c_maskedSelect = const c_maskedSelect_

-- | c_nonzero :  subscript tensor -> void
foreign import ccall "THTensorMath.h THDoubleTensor_nonzero"
  c_nonzero_ :: Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_nonzero_ with unused argument (for CTHState) to unify backpack signatures.
c_nonzero :: Ptr C'THState -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> IO ()
c_nonzero = const c_nonzero_

-- | c_indexSelect :  tensor src dim index -> void
foreign import ccall "THTensorMath.h THDoubleTensor_indexSelect"
  c_indexSelect_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> Ptr C'THLongTensor -> IO ()

-- | alias of c_indexSelect_ with unused argument (for CTHState) to unify backpack signatures.
c_indexSelect :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> Ptr C'THLongTensor -> IO ()
c_indexSelect = const c_indexSelect_

-- | c_indexCopy :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_indexCopy"
  c_indexCopy_ :: Ptr C'THDoubleTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_indexCopy_ with unused argument (for CTHState) to unify backpack signatures.
c_indexCopy :: Ptr C'THState -> Ptr C'THDoubleTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> IO ()
c_indexCopy = const c_indexCopy_

-- | c_indexAdd :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_indexAdd"
  c_indexAdd_ :: Ptr C'THDoubleTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_indexAdd_ with unused argument (for CTHState) to unify backpack signatures.
c_indexAdd :: Ptr C'THState -> Ptr C'THDoubleTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> IO ()
c_indexAdd = const c_indexAdd_

-- | c_indexFill :  tensor dim index val -> void
foreign import ccall "THTensorMath.h THDoubleTensor_indexFill"
  c_indexFill_ :: Ptr C'THDoubleTensor -> CInt -> Ptr C'THLongTensor -> CDouble -> IO ()

-- | alias of c_indexFill_ with unused argument (for CTHState) to unify backpack signatures.
c_indexFill :: Ptr C'THState -> Ptr C'THDoubleTensor -> CInt -> Ptr C'THLongTensor -> CDouble -> IO ()
c_indexFill = const c_indexFill_

-- | c_take :  tensor src index -> void
foreign import ccall "THTensorMath.h THDoubleTensor_take"
  c_take_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_take_ with unused argument (for CTHState) to unify backpack signatures.
c_take :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> IO ()
c_take = const c_take_

-- | c_put :  tensor index src accumulate -> void
foreign import ccall "THTensorMath.h THDoubleTensor_put"
  c_put_ :: Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> CInt -> IO ()

-- | alias of c_put_ with unused argument (for CTHState) to unify backpack signatures.
c_put :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> CInt -> IO ()
c_put = const c_put_

-- | c_gather :  tensor src dim index -> void
foreign import ccall "THTensorMath.h THDoubleTensor_gather"
  c_gather_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> Ptr C'THLongTensor -> IO ()

-- | alias of c_gather_ with unused argument (for CTHState) to unify backpack signatures.
c_gather :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> Ptr C'THLongTensor -> IO ()
c_gather = const c_gather_

-- | c_scatter :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_scatter"
  c_scatter_ :: Ptr C'THDoubleTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_scatter_ with unused argument (for CTHState) to unify backpack signatures.
c_scatter :: Ptr C'THState -> Ptr C'THDoubleTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> IO ()
c_scatter = const c_scatter_

-- | c_scatterAdd :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_scatterAdd"
  c_scatterAdd_ :: Ptr C'THDoubleTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_scatterAdd_ with unused argument (for CTHState) to unify backpack signatures.
c_scatterAdd :: Ptr C'THState -> Ptr C'THDoubleTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> IO ()
c_scatterAdd = const c_scatterAdd_

-- | c_scatterFill :  tensor dim index val -> void
foreign import ccall "THTensorMath.h THDoubleTensor_scatterFill"
  c_scatterFill_ :: Ptr C'THDoubleTensor -> CInt -> Ptr C'THLongTensor -> CDouble -> IO ()

-- | alias of c_scatterFill_ with unused argument (for CTHState) to unify backpack signatures.
c_scatterFill :: Ptr C'THState -> Ptr C'THDoubleTensor -> CInt -> Ptr C'THLongTensor -> CDouble -> IO ()
c_scatterFill = const c_scatterFill_

-- | c_dot :  t src -> accreal
foreign import ccall "THTensorMath.h THDoubleTensor_dot"
  c_dot_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO CDouble

-- | alias of c_dot_ with unused argument (for CTHState) to unify backpack signatures.
c_dot :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO CDouble
c_dot = const c_dot_

-- | c_minall :  t -> real
foreign import ccall "THTensorMath.h THDoubleTensor_minall"
  c_minall_ :: Ptr C'THDoubleTensor -> IO CDouble

-- | alias of c_minall_ with unused argument (for CTHState) to unify backpack signatures.
c_minall :: Ptr C'THState -> Ptr C'THDoubleTensor -> IO CDouble
c_minall = const c_minall_

-- | c_maxall :  t -> real
foreign import ccall "THTensorMath.h THDoubleTensor_maxall"
  c_maxall_ :: Ptr C'THDoubleTensor -> IO CDouble

-- | alias of c_maxall_ with unused argument (for CTHState) to unify backpack signatures.
c_maxall :: Ptr C'THState -> Ptr C'THDoubleTensor -> IO CDouble
c_maxall = const c_maxall_

-- | c_medianall :  t -> real
foreign import ccall "THTensorMath.h THDoubleTensor_medianall"
  c_medianall_ :: Ptr C'THDoubleTensor -> IO CDouble

-- | alias of c_medianall_ with unused argument (for CTHState) to unify backpack signatures.
c_medianall :: Ptr C'THState -> Ptr C'THDoubleTensor -> IO CDouble
c_medianall = const c_medianall_

-- | c_sumall :  t -> accreal
foreign import ccall "THTensorMath.h THDoubleTensor_sumall"
  c_sumall_ :: Ptr C'THDoubleTensor -> IO CDouble

-- | alias of c_sumall_ with unused argument (for CTHState) to unify backpack signatures.
c_sumall :: Ptr C'THState -> Ptr C'THDoubleTensor -> IO CDouble
c_sumall = const c_sumall_

-- | c_prodall :  t -> accreal
foreign import ccall "THTensorMath.h THDoubleTensor_prodall"
  c_prodall_ :: Ptr C'THDoubleTensor -> IO CDouble

-- | alias of c_prodall_ with unused argument (for CTHState) to unify backpack signatures.
c_prodall :: Ptr C'THState -> Ptr C'THDoubleTensor -> IO CDouble
c_prodall = const c_prodall_

-- | c_neg :  self src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_neg"
  c_neg_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_neg_ with unused argument (for CTHState) to unify backpack signatures.
c_neg :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_neg = const c_neg_

-- | c_cinv :  self src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cinv"
  c_cinv_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_cinv_ with unused argument (for CTHState) to unify backpack signatures.
c_cinv :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_cinv = const c_cinv_

-- | c_add :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_add"
  c_add_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_add_ with unused argument (for CTHState) to unify backpack signatures.
c_add :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_add = const c_add_

-- | c_sub :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_sub"
  c_sub_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_sub_ with unused argument (for CTHState) to unify backpack signatures.
c_sub :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_sub = const c_sub_

-- | c_add_scaled :  r_ t value alpha -> void
foreign import ccall "THTensorMath.h THDoubleTensor_add_scaled"
  c_add_scaled_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> IO ()

-- | alias of c_add_scaled_ with unused argument (for CTHState) to unify backpack signatures.
c_add_scaled :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> IO ()
c_add_scaled = const c_add_scaled_

-- | c_sub_scaled :  r_ t value alpha -> void
foreign import ccall "THTensorMath.h THDoubleTensor_sub_scaled"
  c_sub_scaled_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> IO ()

-- | alias of c_sub_scaled_ with unused argument (for CTHState) to unify backpack signatures.
c_sub_scaled :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> IO ()
c_sub_scaled = const c_sub_scaled_

-- | c_mul :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_mul"
  c_mul_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_mul_ with unused argument (for CTHState) to unify backpack signatures.
c_mul :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_mul = const c_mul_

-- | c_div :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_div"
  c_div_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_div_ with unused argument (for CTHState) to unify backpack signatures.
c_div :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_div = const c_div_

-- | c_lshift :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_lshift"
  c_lshift_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_lshift_ with unused argument (for CTHState) to unify backpack signatures.
c_lshift :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_lshift = const c_lshift_

-- | c_rshift :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_rshift"
  c_rshift_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_rshift_ with unused argument (for CTHState) to unify backpack signatures.
c_rshift :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_rshift = const c_rshift_

-- | c_fmod :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_fmod"
  c_fmod_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_fmod_ with unused argument (for CTHState) to unify backpack signatures.
c_fmod :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_fmod = const c_fmod_

-- | c_remainder :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_remainder"
  c_remainder_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_remainder_ with unused argument (for CTHState) to unify backpack signatures.
c_remainder :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_remainder = const c_remainder_

-- | c_clamp :  r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_clamp"
  c_clamp_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> IO ()

-- | alias of c_clamp_ with unused argument (for CTHState) to unify backpack signatures.
c_clamp :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> IO ()
c_clamp = const c_clamp_

-- | c_bitand :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_bitand"
  c_bitand_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_bitand_ with unused argument (for CTHState) to unify backpack signatures.
c_bitand :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_bitand = const c_bitand_

-- | c_bitor :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_bitor"
  c_bitor_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_bitor_ with unused argument (for CTHState) to unify backpack signatures.
c_bitor :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_bitor = const c_bitor_

-- | c_bitxor :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_bitxor"
  c_bitxor_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_bitxor_ with unused argument (for CTHState) to unify backpack signatures.
c_bitxor :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_bitxor = const c_bitxor_

-- | c_cadd :  r_ t value src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cadd"
  c_cadd_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_cadd_ with unused argument (for CTHState) to unify backpack signatures.
c_cadd :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> IO ()
c_cadd = const c_cadd_

-- | c_csub :  self src1 value src2 -> void
foreign import ccall "THTensorMath.h THDoubleTensor_csub"
  c_csub_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_csub_ with unused argument (for CTHState) to unify backpack signatures.
c_csub :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> IO ()
c_csub = const c_csub_

-- | c_cmul :  r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cmul"
  c_cmul_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_cmul_ with unused argument (for CTHState) to unify backpack signatures.
c_cmul :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_cmul = const c_cmul_

-- | c_cpow :  r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cpow"
  c_cpow_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_cpow_ with unused argument (for CTHState) to unify backpack signatures.
c_cpow :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_cpow = const c_cpow_

-- | c_cdiv :  r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cdiv"
  c_cdiv_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_cdiv_ with unused argument (for CTHState) to unify backpack signatures.
c_cdiv :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_cdiv = const c_cdiv_

-- | c_clshift :  r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_clshift"
  c_clshift_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_clshift_ with unused argument (for CTHState) to unify backpack signatures.
c_clshift :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_clshift = const c_clshift_

-- | c_crshift :  r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_crshift"
  c_crshift_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_crshift_ with unused argument (for CTHState) to unify backpack signatures.
c_crshift :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_crshift = const c_crshift_

-- | c_cfmod :  r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cfmod"
  c_cfmod_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_cfmod_ with unused argument (for CTHState) to unify backpack signatures.
c_cfmod :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_cfmod = const c_cfmod_

-- | c_cremainder :  r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cremainder"
  c_cremainder_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_cremainder_ with unused argument (for CTHState) to unify backpack signatures.
c_cremainder :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_cremainder = const c_cremainder_

-- | c_cbitand :  r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cbitand"
  c_cbitand_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_cbitand_ with unused argument (for CTHState) to unify backpack signatures.
c_cbitand :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_cbitand = const c_cbitand_

-- | c_cbitor :  r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cbitor"
  c_cbitor_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_cbitor_ with unused argument (for CTHState) to unify backpack signatures.
c_cbitor :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_cbitor = const c_cbitor_

-- | c_cbitxor :  r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cbitxor"
  c_cbitxor_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_cbitxor_ with unused argument (for CTHState) to unify backpack signatures.
c_cbitxor :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_cbitxor = const c_cbitxor_

-- | c_addcmul :  r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THDoubleTensor_addcmul"
  c_addcmul_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_addcmul_ with unused argument (for CTHState) to unify backpack signatures.
c_addcmul :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_addcmul = const c_addcmul_

-- | c_addcdiv :  r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THDoubleTensor_addcdiv"
  c_addcdiv_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_addcdiv_ with unused argument (for CTHState) to unify backpack signatures.
c_addcdiv :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_addcdiv = const c_addcdiv_

-- | c_addmv :  r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h THDoubleTensor_addmv"
  c_addmv_ :: Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_addmv_ with unused argument (for CTHState) to unify backpack signatures.
c_addmv :: Ptr C'THState -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_addmv = const c_addmv_

-- | c_addmm :  r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h THDoubleTensor_addmm"
  c_addmm_ :: Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_addmm_ with unused argument (for CTHState) to unify backpack signatures.
c_addmm :: Ptr C'THState -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_addmm = const c_addmm_

-- | c_addr :  r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h THDoubleTensor_addr"
  c_addr_ :: Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_addr_ with unused argument (for CTHState) to unify backpack signatures.
c_addr :: Ptr C'THState -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_addr = const c_addr_

-- | c_addbmm :  r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THDoubleTensor_addbmm"
  c_addbmm_ :: Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_addbmm_ with unused argument (for CTHState) to unify backpack signatures.
c_addbmm :: Ptr C'THState -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_addbmm = const c_addbmm_

-- | c_baddbmm :  r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THDoubleTensor_baddbmm"
  c_baddbmm_ :: Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_baddbmm_ with unused argument (for CTHState) to unify backpack signatures.
c_baddbmm :: Ptr C'THState -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_baddbmm = const c_baddbmm_

-- | c_match :  r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h THDoubleTensor_match"
  c_match_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_match_ with unused argument (for CTHState) to unify backpack signatures.
c_match :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_match = const c_match_

-- | c_numel :  t -> ptrdiff_t
foreign import ccall "THTensorMath.h THDoubleTensor_numel"
  c_numel_ :: Ptr C'THDoubleTensor -> IO CPtrdiff

-- | alias of c_numel_ with unused argument (for CTHState) to unify backpack signatures.
c_numel :: Ptr C'THState -> Ptr C'THDoubleTensor -> IO CPtrdiff
c_numel = const c_numel_

-- | c_preserveReduceDimSemantics :  r_ in_dims reduce_dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensor_preserveReduceDimSemantics"
  c_preserveReduceDimSemantics_ :: Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> IO ()

-- | alias of c_preserveReduceDimSemantics_ with unused argument (for CTHState) to unify backpack signatures.
c_preserveReduceDimSemantics :: Ptr C'THState -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> IO ()
c_preserveReduceDimSemantics = const c_preserveReduceDimSemantics_

-- | c_max :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensor_max"
  c_max_ :: Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ()

-- | alias of c_max_ with unused argument (for CTHState) to unify backpack signatures.
c_max :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ()
c_max = const c_max_

-- | c_min :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensor_min"
  c_min_ :: Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ()

-- | alias of c_min_ with unused argument (for CTHState) to unify backpack signatures.
c_min :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ()
c_min = const c_min_

-- | c_kthvalue :  values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensor_kthvalue"
  c_kthvalue_ :: Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> CLLong -> CInt -> CInt -> IO ()

-- | alias of c_kthvalue_ with unused argument (for CTHState) to unify backpack signatures.
c_kthvalue :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> CLLong -> CInt -> CInt -> IO ()
c_kthvalue = const c_kthvalue_

-- | c_mode :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensor_mode"
  c_mode_ :: Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ()

-- | alias of c_mode_ with unused argument (for CTHState) to unify backpack signatures.
c_mode :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ()
c_mode = const c_mode_

-- | c_median :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensor_median"
  c_median_ :: Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ()

-- | alias of c_median_ with unused argument (for CTHState) to unify backpack signatures.
c_median :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ()
c_median = const c_median_

-- | c_sum :  r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensor_sum"
  c_sum_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ()

-- | alias of c_sum_ with unused argument (for CTHState) to unify backpack signatures.
c_sum :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ()
c_sum = const c_sum_

-- | c_prod :  r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensor_prod"
  c_prod_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ()

-- | alias of c_prod_ with unused argument (for CTHState) to unify backpack signatures.
c_prod :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ()
c_prod = const c_prod_

-- | c_cumsum :  r_ t dimension -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cumsum"
  c_cumsum_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ()

-- | alias of c_cumsum_ with unused argument (for CTHState) to unify backpack signatures.
c_cumsum :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ()
c_cumsum = const c_cumsum_

-- | c_cumprod :  r_ t dimension -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cumprod"
  c_cumprod_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ()

-- | alias of c_cumprod_ with unused argument (for CTHState) to unify backpack signatures.
c_cumprod :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ()
c_cumprod = const c_cumprod_

-- | c_sign :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_sign"
  c_sign_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_sign_ with unused argument (for CTHState) to unify backpack signatures.
c_sign :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_sign = const c_sign_

-- | c_trace :  t -> accreal
foreign import ccall "THTensorMath.h THDoubleTensor_trace"
  c_trace_ :: Ptr C'THDoubleTensor -> IO CDouble

-- | alias of c_trace_ with unused argument (for CTHState) to unify backpack signatures.
c_trace :: Ptr C'THState -> Ptr C'THDoubleTensor -> IO CDouble
c_trace = const c_trace_

-- | c_cross :  r_ a b dimension -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cross"
  c_cross_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ()

-- | alias of c_cross_ with unused argument (for CTHState) to unify backpack signatures.
c_cross :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ()
c_cross = const c_cross_

-- | c_cmax :  r t src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cmax"
  c_cmax_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_cmax_ with unused argument (for CTHState) to unify backpack signatures.
c_cmax :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_cmax = const c_cmax_

-- | c_cmin :  r t src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cmin"
  c_cmin_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_cmin_ with unused argument (for CTHState) to unify backpack signatures.
c_cmin :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_cmin = const c_cmin_

-- | c_cmaxValue :  r t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cmaxValue"
  c_cmaxValue_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_cmaxValue_ with unused argument (for CTHState) to unify backpack signatures.
c_cmaxValue :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_cmaxValue = const c_cmaxValue_

-- | c_cminValue :  r t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cminValue"
  c_cminValue_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_cminValue_ with unused argument (for CTHState) to unify backpack signatures.
c_cminValue :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_cminValue = const c_cminValue_

-- | c_zeros :  r_ size -> void
foreign import ccall "THTensorMath.h THDoubleTensor_zeros"
  c_zeros_ :: Ptr C'THDoubleTensor -> Ptr C'THLongStorage -> IO ()

-- | alias of c_zeros_ with unused argument (for CTHState) to unify backpack signatures.
c_zeros :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THLongStorage -> IO ()
c_zeros = const c_zeros_

-- | c_zerosLike :  r_ input -> void
foreign import ccall "THTensorMath.h THDoubleTensor_zerosLike"
  c_zerosLike_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_zerosLike_ with unused argument (for CTHState) to unify backpack signatures.
c_zerosLike :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_zerosLike = const c_zerosLike_

-- | c_ones :  r_ size -> void
foreign import ccall "THTensorMath.h THDoubleTensor_ones"
  c_ones_ :: Ptr C'THDoubleTensor -> Ptr C'THLongStorage -> IO ()

-- | alias of c_ones_ with unused argument (for CTHState) to unify backpack signatures.
c_ones :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THLongStorage -> IO ()
c_ones = const c_ones_

-- | c_onesLike :  r_ input -> void
foreign import ccall "THTensorMath.h THDoubleTensor_onesLike"
  c_onesLike_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_onesLike_ with unused argument (for CTHState) to unify backpack signatures.
c_onesLike :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_onesLike = const c_onesLike_

-- | c_diag :  r_ t k -> void
foreign import ccall "THTensorMath.h THDoubleTensor_diag"
  c_diag_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ()

-- | alias of c_diag_ with unused argument (for CTHState) to unify backpack signatures.
c_diag :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ()
c_diag = const c_diag_

-- | c_eye :  r_ n m -> void
foreign import ccall "THTensorMath.h THDoubleTensor_eye"
  c_eye_ :: Ptr C'THDoubleTensor -> CLLong -> CLLong -> IO ()

-- | alias of c_eye_ with unused argument (for CTHState) to unify backpack signatures.
c_eye :: Ptr C'THState -> Ptr C'THDoubleTensor -> CLLong -> CLLong -> IO ()
c_eye = const c_eye_

-- | c_arange :  r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THDoubleTensor_arange"
  c_arange_ :: Ptr C'THDoubleTensor -> CDouble -> CDouble -> CDouble -> IO ()

-- | alias of c_arange_ with unused argument (for CTHState) to unify backpack signatures.
c_arange :: Ptr C'THState -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> CDouble -> IO ()
c_arange = const c_arange_

-- | c_range :  r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THDoubleTensor_range"
  c_range_ :: Ptr C'THDoubleTensor -> CDouble -> CDouble -> CDouble -> IO ()

-- | alias of c_range_ with unused argument (for CTHState) to unify backpack signatures.
c_range :: Ptr C'THState -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> CDouble -> IO ()
c_range = const c_range_

-- | c_randperm :  r_ _generator n -> void
foreign import ccall "THTensorMath.h THDoubleTensor_randperm"
  c_randperm_ :: Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CLLong -> IO ()

-- | alias of c_randperm_ with unused argument (for CTHState) to unify backpack signatures.
c_randperm :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CLLong -> IO ()
c_randperm = const c_randperm_

-- | c_reshape :  r_ t size -> void
foreign import ccall "THTensorMath.h THDoubleTensor_reshape"
  c_reshape_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THLongStorage -> IO ()

-- | alias of c_reshape_ with unused argument (for CTHState) to unify backpack signatures.
c_reshape :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THLongStorage -> IO ()
c_reshape = const c_reshape_

-- | c_sort :  rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h THDoubleTensor_sort"
  c_sort_ :: Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ()

-- | alias of c_sort_ with unused argument (for CTHState) to unify backpack signatures.
c_sort :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ()
c_sort = const c_sort_

-- | c_topk :  rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h THDoubleTensor_topk"
  c_topk_ :: Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> CLLong -> CInt -> CInt -> CInt -> IO ()

-- | alias of c_topk_ with unused argument (for CTHState) to unify backpack signatures.
c_topk :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> CLLong -> CInt -> CInt -> CInt -> IO ()
c_topk = const c_topk_

-- | c_tril :  r_ t k -> void
foreign import ccall "THTensorMath.h THDoubleTensor_tril"
  c_tril_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CLLong -> IO ()

-- | alias of c_tril_ with unused argument (for CTHState) to unify backpack signatures.
c_tril :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CLLong -> IO ()
c_tril = const c_tril_

-- | c_triu :  r_ t k -> void
foreign import ccall "THTensorMath.h THDoubleTensor_triu"
  c_triu_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CLLong -> IO ()

-- | alias of c_triu_ with unused argument (for CTHState) to unify backpack signatures.
c_triu :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CLLong -> IO ()
c_triu = const c_triu_

-- | c_cat :  r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cat"
  c_cat_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ()

-- | alias of c_cat_ with unused argument (for CTHState) to unify backpack signatures.
c_cat :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ()
c_cat = const c_cat_

-- | c_catArray :  result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h THDoubleTensor_catArray"
  c_catArray_ :: Ptr C'THDoubleTensor -> Ptr (Ptr C'THDoubleTensor) -> CInt -> CInt -> IO ()

-- | alias of c_catArray_ with unused argument (for CTHState) to unify backpack signatures.
c_catArray :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr (Ptr C'THDoubleTensor) -> CInt -> CInt -> IO ()
c_catArray = const c_catArray_

-- | c_equal :  ta tb -> int
foreign import ccall "THTensorMath.h THDoubleTensor_equal"
  c_equal_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO CInt

-- | alias of c_equal_ with unused argument (for CTHState) to unify backpack signatures.
c_equal :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO CInt
c_equal = const c_equal_

-- | c_ltValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_ltValue"
  c_ltValue_ :: Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_ltValue_ with unused argument (for CTHState) to unify backpack signatures.
c_ltValue :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_ltValue = const c_ltValue_

-- | c_leValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_leValue"
  c_leValue_ :: Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_leValue_ with unused argument (for CTHState) to unify backpack signatures.
c_leValue :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_leValue = const c_leValue_

-- | c_gtValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_gtValue"
  c_gtValue_ :: Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_gtValue_ with unused argument (for CTHState) to unify backpack signatures.
c_gtValue :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_gtValue = const c_gtValue_

-- | c_geValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_geValue"
  c_geValue_ :: Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_geValue_ with unused argument (for CTHState) to unify backpack signatures.
c_geValue :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_geValue = const c_geValue_

-- | c_neValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_neValue"
  c_neValue_ :: Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_neValue_ with unused argument (for CTHState) to unify backpack signatures.
c_neValue :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_neValue = const c_neValue_

-- | c_eqValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_eqValue"
  c_eqValue_ :: Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_eqValue_ with unused argument (for CTHState) to unify backpack signatures.
c_eqValue :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_eqValue = const c_eqValue_

-- | c_ltValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_ltValueT"
  c_ltValueT_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_ltValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_ltValueT :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_ltValueT = const c_ltValueT_

-- | c_leValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_leValueT"
  c_leValueT_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_leValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_leValueT :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_leValueT = const c_leValueT_

-- | c_gtValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_gtValueT"
  c_gtValueT_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_gtValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_gtValueT :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_gtValueT = const c_gtValueT_

-- | c_geValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_geValueT"
  c_geValueT_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_geValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_geValueT :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_geValueT = const c_geValueT_

-- | c_neValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_neValueT"
  c_neValueT_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_neValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_neValueT :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_neValueT = const c_neValueT_

-- | c_eqValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_eqValueT"
  c_eqValueT_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_eqValueT_ with unused argument (for CTHState) to unify backpack signatures.
c_eqValueT :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_eqValueT = const c_eqValueT_

-- | c_ltTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensor_ltTensor"
  c_ltTensor_ :: Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_ltTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_ltTensor :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_ltTensor = const c_ltTensor_

-- | c_leTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensor_leTensor"
  c_leTensor_ :: Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_leTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_leTensor :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_leTensor = const c_leTensor_

-- | c_gtTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensor_gtTensor"
  c_gtTensor_ :: Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_gtTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_gtTensor :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_gtTensor = const c_gtTensor_

-- | c_geTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensor_geTensor"
  c_geTensor_ :: Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_geTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_geTensor :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_geTensor = const c_geTensor_

-- | c_neTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensor_neTensor"
  c_neTensor_ :: Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_neTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_neTensor :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_neTensor = const c_neTensor_

-- | c_eqTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensor_eqTensor"
  c_eqTensor_ :: Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_eqTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_eqTensor :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_eqTensor = const c_eqTensor_

-- | c_ltTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensor_ltTensorT"
  c_ltTensorT_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_ltTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_ltTensorT :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_ltTensorT = const c_ltTensorT_

-- | c_leTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensor_leTensorT"
  c_leTensorT_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_leTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_leTensorT :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_leTensorT = const c_leTensorT_

-- | c_gtTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensor_gtTensorT"
  c_gtTensorT_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_gtTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_gtTensorT :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_gtTensorT = const c_gtTensorT_

-- | c_geTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensor_geTensorT"
  c_geTensorT_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_geTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_geTensorT :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_geTensorT = const c_geTensorT_

-- | c_neTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensor_neTensorT"
  c_neTensorT_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_neTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_neTensorT :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_neTensorT = const c_neTensorT_

-- | c_eqTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensor_eqTensorT"
  c_eqTensorT_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_eqTensorT_ with unused argument (for CTHState) to unify backpack signatures.
c_eqTensorT :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_eqTensorT = const c_eqTensorT_

-- | c_pow :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_pow"
  c_pow_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_pow_ with unused argument (for CTHState) to unify backpack signatures.
c_pow :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_pow = const c_pow_

-- | c_tpow :  r_ value t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_tpow"
  c_tpow_ :: Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_tpow_ with unused argument (for CTHState) to unify backpack signatures.
c_tpow :: Ptr C'THState -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> IO ()
c_tpow = const c_tpow_

-- | c_abs :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_abs"
  c_abs_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_abs_ with unused argument (for CTHState) to unify backpack signatures.
c_abs :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_abs = const c_abs_

-- | c_sigmoid :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_sigmoid"
  c_sigmoid_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_sigmoid_ with unused argument (for CTHState) to unify backpack signatures.
c_sigmoid :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_sigmoid = const c_sigmoid_

-- | c_log :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_log"
  c_log_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_log_ with unused argument (for CTHState) to unify backpack signatures.
c_log :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_log = const c_log_

-- | c_lgamma :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_lgamma"
  c_lgamma_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_lgamma_ with unused argument (for CTHState) to unify backpack signatures.
c_lgamma :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_lgamma = const c_lgamma_

-- | c_digamma :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_digamma"
  c_digamma_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_digamma_ with unused argument (for CTHState) to unify backpack signatures.
c_digamma :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_digamma = const c_digamma_

-- | c_trigamma :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_trigamma"
  c_trigamma_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_trigamma_ with unused argument (for CTHState) to unify backpack signatures.
c_trigamma :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_trigamma = const c_trigamma_

-- | c_polygamma :  r_ n t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_polygamma"
  c_polygamma_ :: Ptr C'THDoubleTensor -> CLLong -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_polygamma_ with unused argument (for CTHState) to unify backpack signatures.
c_polygamma :: Ptr C'THState -> Ptr C'THDoubleTensor -> CLLong -> Ptr C'THDoubleTensor -> IO ()
c_polygamma = const c_polygamma_

-- | c_log1p :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_log1p"
  c_log1p_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_log1p_ with unused argument (for CTHState) to unify backpack signatures.
c_log1p :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_log1p = const c_log1p_

-- | c_exp :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_exp"
  c_exp_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_exp_ with unused argument (for CTHState) to unify backpack signatures.
c_exp :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_exp = const c_exp_

-- | c_expm1 :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_expm1"
  c_expm1_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_expm1_ with unused argument (for CTHState) to unify backpack signatures.
c_expm1 :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_expm1 = const c_expm1_

-- | c_cos :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cos"
  c_cos_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_cos_ with unused argument (for CTHState) to unify backpack signatures.
c_cos :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_cos = const c_cos_

-- | c_acos :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_acos"
  c_acos_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_acos_ with unused argument (for CTHState) to unify backpack signatures.
c_acos :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_acos = const c_acos_

-- | c_cosh :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cosh"
  c_cosh_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_cosh_ with unused argument (for CTHState) to unify backpack signatures.
c_cosh :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_cosh = const c_cosh_

-- | c_sin :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_sin"
  c_sin_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_sin_ with unused argument (for CTHState) to unify backpack signatures.
c_sin :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_sin = const c_sin_

-- | c_asin :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_asin"
  c_asin_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_asin_ with unused argument (for CTHState) to unify backpack signatures.
c_asin :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_asin = const c_asin_

-- | c_sinh :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_sinh"
  c_sinh_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_sinh_ with unused argument (for CTHState) to unify backpack signatures.
c_sinh :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_sinh = const c_sinh_

-- | c_tan :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_tan"
  c_tan_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_tan_ with unused argument (for CTHState) to unify backpack signatures.
c_tan :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_tan = const c_tan_

-- | c_atan :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_atan"
  c_atan_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_atan_ with unused argument (for CTHState) to unify backpack signatures.
c_atan :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_atan = const c_atan_

-- | c_atan2 :  r_ tx ty -> void
foreign import ccall "THTensorMath.h THDoubleTensor_atan2"
  c_atan2_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_atan2_ with unused argument (for CTHState) to unify backpack signatures.
c_atan2 :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_atan2 = const c_atan2_

-- | c_tanh :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_tanh"
  c_tanh_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_tanh_ with unused argument (for CTHState) to unify backpack signatures.
c_tanh :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_tanh = const c_tanh_

-- | c_erf :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_erf"
  c_erf_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_erf_ with unused argument (for CTHState) to unify backpack signatures.
c_erf :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_erf = const c_erf_

-- | c_erfinv :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_erfinv"
  c_erfinv_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_erfinv_ with unused argument (for CTHState) to unify backpack signatures.
c_erfinv :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_erfinv = const c_erfinv_

-- | c_sqrt :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_sqrt"
  c_sqrt_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_sqrt_ with unused argument (for CTHState) to unify backpack signatures.
c_sqrt :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_sqrt = const c_sqrt_

-- | c_rsqrt :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_rsqrt"
  c_rsqrt_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_rsqrt_ with unused argument (for CTHState) to unify backpack signatures.
c_rsqrt :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_rsqrt = const c_rsqrt_

-- | c_ceil :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_ceil"
  c_ceil_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_ceil_ with unused argument (for CTHState) to unify backpack signatures.
c_ceil :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_ceil = const c_ceil_

-- | c_floor :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_floor"
  c_floor_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_floor_ with unused argument (for CTHState) to unify backpack signatures.
c_floor :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_floor = const c_floor_

-- | c_round :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_round"
  c_round_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_round_ with unused argument (for CTHState) to unify backpack signatures.
c_round :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_round = const c_round_

-- | c_trunc :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_trunc"
  c_trunc_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_trunc_ with unused argument (for CTHState) to unify backpack signatures.
c_trunc :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_trunc = const c_trunc_

-- | c_frac :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_frac"
  c_frac_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_frac_ with unused argument (for CTHState) to unify backpack signatures.
c_frac :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_frac = const c_frac_

-- | c_lerp :  r_ a b weight -> void
foreign import ccall "THTensorMath.h THDoubleTensor_lerp"
  c_lerp_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_lerp_ with unused argument (for CTHState) to unify backpack signatures.
c_lerp :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_lerp = const c_lerp_

-- | c_mean :  r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensor_mean"
  c_mean_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ()

-- | alias of c_mean_ with unused argument (for CTHState) to unify backpack signatures.
c_mean :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ()
c_mean = const c_mean_

-- | c_std :  r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensor_std"
  c_std_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> IO ()

-- | alias of c_std_ with unused argument (for CTHState) to unify backpack signatures.
c_std :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> IO ()
c_std = const c_std_

-- | c_var :  r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensor_var"
  c_var_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> IO ()

-- | alias of c_var_ with unused argument (for CTHState) to unify backpack signatures.
c_var :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> IO ()
c_var = const c_var_

-- | c_norm :  r_ t value dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensor_norm"
  c_norm_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CInt -> CInt -> IO ()

-- | alias of c_norm_ with unused argument (for CTHState) to unify backpack signatures.
c_norm :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CInt -> CInt -> IO ()
c_norm = const c_norm_

-- | c_renorm :  r_ t value dimension maxnorm -> void
foreign import ccall "THTensorMath.h THDoubleTensor_renorm"
  c_renorm_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CInt -> CDouble -> IO ()

-- | alias of c_renorm_ with unused argument (for CTHState) to unify backpack signatures.
c_renorm :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CInt -> CDouble -> IO ()
c_renorm = const c_renorm_

-- | c_dist :  a b value -> accreal
foreign import ccall "THTensorMath.h THDoubleTensor_dist"
  c_dist_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO CDouble

-- | alias of c_dist_ with unused argument (for CTHState) to unify backpack signatures.
c_dist :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO CDouble
c_dist = const c_dist_

-- | c_histc :  hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THDoubleTensor_histc"
  c_histc_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CLLong -> CDouble -> CDouble -> IO ()

-- | alias of c_histc_ with unused argument (for CTHState) to unify backpack signatures.
c_histc :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CLLong -> CDouble -> CDouble -> IO ()
c_histc = const c_histc_

-- | c_bhistc :  hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THDoubleTensor_bhistc"
  c_bhistc_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CLLong -> CDouble -> CDouble -> IO ()

-- | alias of c_bhistc_ with unused argument (for CTHState) to unify backpack signatures.
c_bhistc :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CLLong -> CDouble -> CDouble -> IO ()
c_bhistc = const c_bhistc_

-- | c_meanall :  self -> accreal
foreign import ccall "THTensorMath.h THDoubleTensor_meanall"
  c_meanall_ :: Ptr C'THDoubleTensor -> IO CDouble

-- | alias of c_meanall_ with unused argument (for CTHState) to unify backpack signatures.
c_meanall :: Ptr C'THState -> Ptr C'THDoubleTensor -> IO CDouble
c_meanall = const c_meanall_

-- | c_varall :  self biased -> accreal
foreign import ccall "THTensorMath.h THDoubleTensor_varall"
  c_varall_ :: Ptr C'THDoubleTensor -> CInt -> IO CDouble

-- | alias of c_varall_ with unused argument (for CTHState) to unify backpack signatures.
c_varall :: Ptr C'THState -> Ptr C'THDoubleTensor -> CInt -> IO CDouble
c_varall = const c_varall_

-- | c_stdall :  self biased -> accreal
foreign import ccall "THTensorMath.h THDoubleTensor_stdall"
  c_stdall_ :: Ptr C'THDoubleTensor -> CInt -> IO CDouble

-- | alias of c_stdall_ with unused argument (for CTHState) to unify backpack signatures.
c_stdall :: Ptr C'THState -> Ptr C'THDoubleTensor -> CInt -> IO CDouble
c_stdall = const c_stdall_

-- | c_normall :  t value -> accreal
foreign import ccall "THTensorMath.h THDoubleTensor_normall"
  c_normall_ :: Ptr C'THDoubleTensor -> CDouble -> IO CDouble

-- | alias of c_normall_ with unused argument (for CTHState) to unify backpack signatures.
c_normall :: Ptr C'THState -> Ptr C'THDoubleTensor -> CDouble -> IO CDouble
c_normall = const c_normall_

-- | c_linspace :  r_ a b n -> void
foreign import ccall "THTensorMath.h THDoubleTensor_linspace"
  c_linspace_ :: Ptr C'THDoubleTensor -> CDouble -> CDouble -> CLLong -> IO ()

-- | alias of c_linspace_ with unused argument (for CTHState) to unify backpack signatures.
c_linspace :: Ptr C'THState -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> CLLong -> IO ()
c_linspace = const c_linspace_

-- | c_logspace :  r_ a b n -> void
foreign import ccall "THTensorMath.h THDoubleTensor_logspace"
  c_logspace_ :: Ptr C'THDoubleTensor -> CDouble -> CDouble -> CLLong -> IO ()

-- | alias of c_logspace_ with unused argument (for CTHState) to unify backpack signatures.
c_logspace :: Ptr C'THState -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> CLLong -> IO ()
c_logspace = const c_logspace_

-- | c_rand :  r_ _generator size -> void
foreign import ccall "THTensorMath.h THDoubleTensor_rand"
  c_rand_ :: Ptr C'THDoubleTensor -> Ptr C'THGenerator -> Ptr C'THLongStorage -> IO ()

-- | alias of c_rand_ with unused argument (for CTHState) to unify backpack signatures.
c_rand :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THGenerator -> Ptr C'THLongStorage -> IO ()
c_rand = const c_rand_

-- | c_randn :  r_ _generator size -> void
foreign import ccall "THTensorMath.h THDoubleTensor_randn"
  c_randn_ :: Ptr C'THDoubleTensor -> Ptr C'THGenerator -> Ptr C'THLongStorage -> IO ()

-- | alias of c_randn_ with unused argument (for CTHState) to unify backpack signatures.
c_randn :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THGenerator -> Ptr C'THLongStorage -> IO ()
c_randn = const c_randn_

-- | c_dirichlet_grad :  self x alpha total -> void
foreign import ccall "THTensorMath.h THDoubleTensor_dirichlet_grad"
  c_dirichlet_grad_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_dirichlet_grad_ with unused argument (for CTHState) to unify backpack signatures.
c_dirichlet_grad :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_dirichlet_grad = const c_dirichlet_grad_

-- | p_fill : Pointer to function : r_ value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_fill"
  p_fill :: FunPtr (Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_zero : Pointer to function : r_ -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_zero"
  p_zero :: FunPtr (Ptr C'THDoubleTensor -> IO ())

-- | p_maskedFill : Pointer to function : tensor mask value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_maskedFill"
  p_maskedFill :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THByteTensor -> CDouble -> IO ())

-- | p_maskedCopy : Pointer to function : tensor mask src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_maskedCopy"
  p_maskedCopy :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_maskedSelect : Pointer to function : tensor src mask -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_maskedSelect"
  p_maskedSelect :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THByteTensor -> IO ())

-- | p_nonzero : Pointer to function : subscript tensor -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_nonzero"
  p_nonzero :: FunPtr (Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_indexSelect : Pointer to function : tensor src dim index -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_indexSelect"
  p_indexSelect :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> Ptr C'THLongTensor -> IO ())

-- | p_indexCopy : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_indexCopy"
  p_indexCopy :: FunPtr (Ptr C'THDoubleTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_indexAdd : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_indexAdd"
  p_indexAdd :: FunPtr (Ptr C'THDoubleTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_indexFill : Pointer to function : tensor dim index val -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_indexFill"
  p_indexFill :: FunPtr (Ptr C'THDoubleTensor -> CInt -> Ptr C'THLongTensor -> CDouble -> IO ())

-- | p_take : Pointer to function : tensor src index -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_take"
  p_take :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> IO ())

-- | p_put : Pointer to function : tensor index src accumulate -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_put"
  p_put :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> CInt -> IO ())

-- | p_gather : Pointer to function : tensor src dim index -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_gather"
  p_gather :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> Ptr C'THLongTensor -> IO ())

-- | p_scatter : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_scatter"
  p_scatter :: FunPtr (Ptr C'THDoubleTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_scatterAdd : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_scatterAdd"
  p_scatterAdd :: FunPtr (Ptr C'THDoubleTensor -> CInt -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_scatterFill : Pointer to function : tensor dim index val -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_scatterFill"
  p_scatterFill :: FunPtr (Ptr C'THDoubleTensor -> CInt -> Ptr C'THLongTensor -> CDouble -> IO ())

-- | p_dot : Pointer to function : t src -> accreal
foreign import ccall "THTensorMath.h &THDoubleTensor_dot"
  p_dot :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO CDouble)

-- | p_minall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THDoubleTensor_minall"
  p_minall :: FunPtr (Ptr C'THDoubleTensor -> IO CDouble)

-- | p_maxall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THDoubleTensor_maxall"
  p_maxall :: FunPtr (Ptr C'THDoubleTensor -> IO CDouble)

-- | p_medianall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THDoubleTensor_medianall"
  p_medianall :: FunPtr (Ptr C'THDoubleTensor -> IO CDouble)

-- | p_sumall : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THDoubleTensor_sumall"
  p_sumall :: FunPtr (Ptr C'THDoubleTensor -> IO CDouble)

-- | p_prodall : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THDoubleTensor_prodall"
  p_prodall :: FunPtr (Ptr C'THDoubleTensor -> IO CDouble)

-- | p_neg : Pointer to function : self src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_neg"
  p_neg :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_cinv : Pointer to function : self src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cinv"
  p_cinv :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_add : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_add"
  p_add :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_sub : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_sub"
  p_sub :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_add_scaled : Pointer to function : r_ t value alpha -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_add_scaled"
  p_add_scaled :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> IO ())

-- | p_sub_scaled : Pointer to function : r_ t value alpha -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_sub_scaled"
  p_sub_scaled :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> IO ())

-- | p_mul : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_mul"
  p_mul :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_div : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_div"
  p_div :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_lshift : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_lshift"
  p_lshift :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_rshift : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_rshift"
  p_rshift :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_fmod : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_fmod"
  p_fmod :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_remainder : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_remainder"
  p_remainder :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_clamp : Pointer to function : r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_clamp"
  p_clamp :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CDouble -> IO ())

-- | p_bitand : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_bitand"
  p_bitand :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_bitor : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_bitor"
  p_bitor :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_bitxor : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_bitxor"
  p_bitxor :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_cadd : Pointer to function : r_ t value src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cadd"
  p_cadd :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> IO ())

-- | p_csub : Pointer to function : self src1 value src2 -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_csub"
  p_csub :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> IO ())

-- | p_cmul : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cmul"
  p_cmul :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_cpow : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cpow"
  p_cpow :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_cdiv : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cdiv"
  p_cdiv :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_clshift : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_clshift"
  p_clshift :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_crshift : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_crshift"
  p_crshift :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_cfmod : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cfmod"
  p_cfmod :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_cremainder : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cremainder"
  p_cremainder :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_cbitand : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cbitand"
  p_cbitand :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_cbitor : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cbitor"
  p_cbitor :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_cbitxor : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cbitxor"
  p_cbitxor :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_addcmul : Pointer to function : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_addcmul"
  p_addcmul :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_addcdiv : Pointer to function : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_addcdiv"
  p_addcdiv :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_addmv : Pointer to function : r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_addmv"
  p_addmv :: FunPtr (Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_addmm : Pointer to function : r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_addmm"
  p_addmm :: FunPtr (Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_addr : Pointer to function : r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_addr"
  p_addr :: FunPtr (Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_addbmm : Pointer to function : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_addbmm"
  p_addbmm :: FunPtr (Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_baddbmm : Pointer to function : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_baddbmm"
  p_baddbmm :: FunPtr (Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_match : Pointer to function : r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_match"
  p_match :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_numel : Pointer to function : t -> ptrdiff_t
foreign import ccall "THTensorMath.h &THDoubleTensor_numel"
  p_numel :: FunPtr (Ptr C'THDoubleTensor -> IO CPtrdiff)

-- | p_preserveReduceDimSemantics : Pointer to function : r_ in_dims reduce_dimension keepdim -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_preserveReduceDimSemantics"
  p_preserveReduceDimSemantics :: FunPtr (Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> IO ())

-- | p_max : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_max"
  p_max :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ())

-- | p_min : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_min"
  p_min :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ())

-- | p_kthvalue : Pointer to function : values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_kthvalue"
  p_kthvalue :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> CLLong -> CInt -> CInt -> IO ())

-- | p_mode : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_mode"
  p_mode :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ())

-- | p_median : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_median"
  p_median :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ())

-- | p_sum : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_sum"
  p_sum :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ())

-- | p_prod : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_prod"
  p_prod :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ())

-- | p_cumsum : Pointer to function : r_ t dimension -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cumsum"
  p_cumsum :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ())

-- | p_cumprod : Pointer to function : r_ t dimension -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cumprod"
  p_cumprod :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ())

-- | p_sign : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_sign"
  p_sign :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_trace : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THDoubleTensor_trace"
  p_trace :: FunPtr (Ptr C'THDoubleTensor -> IO CDouble)

-- | p_cross : Pointer to function : r_ a b dimension -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cross"
  p_cross :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ())

-- | p_cmax : Pointer to function : r t src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cmax"
  p_cmax :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_cmin : Pointer to function : r t src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cmin"
  p_cmin :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_cmaxValue : Pointer to function : r t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cmaxValue"
  p_cmaxValue :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_cminValue : Pointer to function : r t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cminValue"
  p_cminValue :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_zeros : Pointer to function : r_ size -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_zeros"
  p_zeros :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THLongStorage -> IO ())

-- | p_zerosLike : Pointer to function : r_ input -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_zerosLike"
  p_zerosLike :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_ones : Pointer to function : r_ size -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_ones"
  p_ones :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THLongStorage -> IO ())

-- | p_onesLike : Pointer to function : r_ input -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_onesLike"
  p_onesLike :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_diag : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_diag"
  p_diag :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ())

-- | p_eye : Pointer to function : r_ n m -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_eye"
  p_eye :: FunPtr (Ptr C'THDoubleTensor -> CLLong -> CLLong -> IO ())

-- | p_arange : Pointer to function : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_arange"
  p_arange :: FunPtr (Ptr C'THDoubleTensor -> CDouble -> CDouble -> CDouble -> IO ())

-- | p_range : Pointer to function : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_range"
  p_range :: FunPtr (Ptr C'THDoubleTensor -> CDouble -> CDouble -> CDouble -> IO ())

-- | p_randperm : Pointer to function : r_ _generator n -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_randperm"
  p_randperm :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CLLong -> IO ())

-- | p_reshape : Pointer to function : r_ t size -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_reshape"
  p_reshape :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THLongStorage -> IO ())

-- | p_sort : Pointer to function : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_sort"
  p_sort :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ())

-- | p_topk : Pointer to function : rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_topk"
  p_topk :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> CLLong -> CInt -> CInt -> CInt -> IO ())

-- | p_tril : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_tril"
  p_tril :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CLLong -> IO ())

-- | p_triu : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_triu"
  p_triu :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CLLong -> IO ())

-- | p_cat : Pointer to function : r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cat"
  p_cat :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> IO ())

-- | p_catArray : Pointer to function : result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_catArray"
  p_catArray :: FunPtr (Ptr C'THDoubleTensor -> Ptr (Ptr C'THDoubleTensor) -> CInt -> CInt -> IO ())

-- | p_equal : Pointer to function : ta tb -> int
foreign import ccall "THTensorMath.h &THDoubleTensor_equal"
  p_equal :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO CInt)

-- | p_ltValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_ltValue"
  p_ltValue :: FunPtr (Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_leValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_leValue"
  p_leValue :: FunPtr (Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_gtValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_gtValue"
  p_gtValue :: FunPtr (Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_geValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_geValue"
  p_geValue :: FunPtr (Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_neValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_neValue"
  p_neValue :: FunPtr (Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_eqValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_eqValue"
  p_eqValue :: FunPtr (Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_ltValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_ltValueT"
  p_ltValueT :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_leValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_leValueT"
  p_leValueT :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_gtValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_gtValueT"
  p_gtValueT :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_geValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_geValueT"
  p_geValueT :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_neValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_neValueT"
  p_neValueT :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_eqValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_eqValueT"
  p_eqValueT :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_ltTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_ltTensor"
  p_ltTensor :: FunPtr (Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_leTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_leTensor"
  p_leTensor :: FunPtr (Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_gtTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_gtTensor"
  p_gtTensor :: FunPtr (Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_geTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_geTensor"
  p_geTensor :: FunPtr (Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_neTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_neTensor"
  p_neTensor :: FunPtr (Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_eqTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_eqTensor"
  p_eqTensor :: FunPtr (Ptr C'THByteTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_ltTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_ltTensorT"
  p_ltTensorT :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_leTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_leTensorT"
  p_leTensorT :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_gtTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_gtTensorT"
  p_gtTensorT :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_geTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_geTensorT"
  p_geTensorT :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_neTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_neTensorT"
  p_neTensorT :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_eqTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_eqTensorT"
  p_eqTensorT :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_pow : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_pow"
  p_pow :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_tpow : Pointer to function : r_ value t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_tpow"
  p_tpow :: FunPtr (Ptr C'THDoubleTensor -> CDouble -> Ptr C'THDoubleTensor -> IO ())

-- | p_abs : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_abs"
  p_abs :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_sigmoid : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_sigmoid"
  p_sigmoid :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_log : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_log"
  p_log :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_lgamma : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_lgamma"
  p_lgamma :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_digamma : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_digamma"
  p_digamma :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_trigamma : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_trigamma"
  p_trigamma :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_polygamma : Pointer to function : r_ n t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_polygamma"
  p_polygamma :: FunPtr (Ptr C'THDoubleTensor -> CLLong -> Ptr C'THDoubleTensor -> IO ())

-- | p_log1p : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_log1p"
  p_log1p :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_exp : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_exp"
  p_exp :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_expm1 : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_expm1"
  p_expm1 :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_cos : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cos"
  p_cos :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_acos : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_acos"
  p_acos :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_cosh : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cosh"
  p_cosh :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_sin : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_sin"
  p_sin :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_asin : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_asin"
  p_asin :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_sinh : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_sinh"
  p_sinh :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_tan : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_tan"
  p_tan :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_atan : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_atan"
  p_atan :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_atan2 : Pointer to function : r_ tx ty -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_atan2"
  p_atan2 :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_tanh : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_tanh"
  p_tanh :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_erf : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_erf"
  p_erf :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_erfinv : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_erfinv"
  p_erfinv :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_sqrt : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_sqrt"
  p_sqrt :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_rsqrt : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_rsqrt"
  p_rsqrt :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_ceil : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_ceil"
  p_ceil :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_floor : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_floor"
  p_floor :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_round : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_round"
  p_round :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_trunc : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_trunc"
  p_trunc :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_frac : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_frac"
  p_frac :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_lerp : Pointer to function : r_ a b weight -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_lerp"
  p_lerp :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_mean : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_mean"
  p_mean :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ())

-- | p_std : Pointer to function : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_std"
  p_std :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> IO ())

-- | p_var : Pointer to function : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_var"
  p_var :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CInt -> CInt -> CInt -> IO ())

-- | p_norm : Pointer to function : r_ t value dimension keepdim -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_norm"
  p_norm :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CInt -> CInt -> IO ())

-- | p_renorm : Pointer to function : r_ t value dimension maxnorm -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_renorm"
  p_renorm :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> CInt -> CDouble -> IO ())

-- | p_dist : Pointer to function : a b value -> accreal
foreign import ccall "THTensorMath.h &THDoubleTensor_dist"
  p_dist :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CDouble -> IO CDouble)

-- | p_histc : Pointer to function : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_histc"
  p_histc :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CLLong -> CDouble -> CDouble -> IO ())

-- | p_bhistc : Pointer to function : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_bhistc"
  p_bhistc :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> CLLong -> CDouble -> CDouble -> IO ())

-- | p_meanall : Pointer to function : self -> accreal
foreign import ccall "THTensorMath.h &THDoubleTensor_meanall"
  p_meanall :: FunPtr (Ptr C'THDoubleTensor -> IO CDouble)

-- | p_varall : Pointer to function : self biased -> accreal
foreign import ccall "THTensorMath.h &THDoubleTensor_varall"
  p_varall :: FunPtr (Ptr C'THDoubleTensor -> CInt -> IO CDouble)

-- | p_stdall : Pointer to function : self biased -> accreal
foreign import ccall "THTensorMath.h &THDoubleTensor_stdall"
  p_stdall :: FunPtr (Ptr C'THDoubleTensor -> CInt -> IO CDouble)

-- | p_normall : Pointer to function : t value -> accreal
foreign import ccall "THTensorMath.h &THDoubleTensor_normall"
  p_normall :: FunPtr (Ptr C'THDoubleTensor -> CDouble -> IO CDouble)

-- | p_linspace : Pointer to function : r_ a b n -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_linspace"
  p_linspace :: FunPtr (Ptr C'THDoubleTensor -> CDouble -> CDouble -> CLLong -> IO ())

-- | p_logspace : Pointer to function : r_ a b n -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_logspace"
  p_logspace :: FunPtr (Ptr C'THDoubleTensor -> CDouble -> CDouble -> CLLong -> IO ())

-- | p_rand : Pointer to function : r_ _generator size -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_rand"
  p_rand :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THGenerator -> Ptr C'THLongStorage -> IO ())

-- | p_randn : Pointer to function : r_ _generator size -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_randn"
  p_randn :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THGenerator -> Ptr C'THLongStorage -> IO ())

-- | p_dirichlet_grad : Pointer to function : self x alpha total -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_dirichlet_grad"
  p_dirichlet_grad :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())