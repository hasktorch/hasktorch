{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Long.TensorMath
  ( c_fill
  , c_zero
  , c_maskedFill
  , c_maskedCopy
  , c_maskedSelect
  , c_nonzero
  , c_indexSelect
  , c_indexCopy
  , c_indexAdd
  , c_indexFill
  , c_take
  , c_put
  , c_gather
  , c_scatter
  , c_scatterAdd
  , c_scatterFill
  , c_dot
  , c_minall
  , c_maxall
  , c_medianall
  , c_sumall
  , c_prodall
  , c_neg
  , c_add
  , c_sub
  , c_add_scaled
  , c_sub_scaled
  , c_mul
  , c_div
  , c_lshift
  , c_rshift
  , c_fmod
  , c_remainder
  , c_clamp
  , c_bitand
  , c_bitor
  , c_bitxor
  , c_cadd
  , c_csub
  , c_cmul
  , c_cpow
  , c_cdiv
  , c_clshift
  , c_crshift
  , c_cfmod
  , c_cremainder
  , c_cbitand
  , c_cbitor
  , c_cbitxor
  , c_addcmul
  , c_addcdiv
  , c_addmv
  , c_addmm
  , c_addr
  , c_addbmm
  , c_baddbmm
  , c_match
  , c_numel
  , c_max
  , c_min
  , c_kthvalue
  , c_mode
  , c_median
  , c_sum
  , c_prod
  , c_cumsum
  , c_cumprod
  , c_sign
  , c_trace
  , c_cross
  , c_cmax
  , c_cmin
  , c_cmaxValue
  , c_cminValue
  , c_zeros
  , c_zerosLike
  , c_ones
  , c_onesLike
  , c_diag
  , c_eye
  , c_arange
  , c_range
  , c_randperm
  , c_reshape
  , c_sort
  , c_topk
  , c_tril
  , c_triu
  , c_cat
  , c_catArray
  , c_equal
  , c_ltValue
  , c_leValue
  , c_gtValue
  , c_geValue
  , c_neValue
  , c_eqValue
  , c_ltValueT
  , c_leValueT
  , c_gtValueT
  , c_geValueT
  , c_neValueT
  , c_eqValueT
  , c_ltTensor
  , c_leTensor
  , c_gtTensor
  , c_geTensor
  , c_neTensor
  , c_eqTensor
  , c_ltTensorT
  , c_leTensorT
  , c_gtTensorT
  , c_geTensorT
  , c_neTensorT
  , c_eqTensorT
  , c_abs
  , p_fill
  , p_zero
  , p_maskedFill
  , p_maskedCopy
  , p_maskedSelect
  , p_nonzero
  , p_indexSelect
  , p_indexCopy
  , p_indexAdd
  , p_indexFill
  , p_take
  , p_put
  , p_gather
  , p_scatter
  , p_scatterAdd
  , p_scatterFill
  , p_dot
  , p_minall
  , p_maxall
  , p_medianall
  , p_sumall
  , p_prodall
  , p_neg
  , p_add
  , p_sub
  , p_add_scaled
  , p_sub_scaled
  , p_mul
  , p_div
  , p_lshift
  , p_rshift
  , p_fmod
  , p_remainder
  , p_clamp
  , p_bitand
  , p_bitor
  , p_bitxor
  , p_cadd
  , p_csub
  , p_cmul
  , p_cpow
  , p_cdiv
  , p_clshift
  , p_crshift
  , p_cfmod
  , p_cremainder
  , p_cbitand
  , p_cbitor
  , p_cbitxor
  , p_addcmul
  , p_addcdiv
  , p_addmv
  , p_addmm
  , p_addr
  , p_addbmm
  , p_baddbmm
  , p_match
  , p_numel
  , p_max
  , p_min
  , p_kthvalue
  , p_mode
  , p_median
  , p_sum
  , p_prod
  , p_cumsum
  , p_cumprod
  , p_sign
  , p_trace
  , p_cross
  , p_cmax
  , p_cmin
  , p_cmaxValue
  , p_cminValue
  , p_zeros
  , p_zerosLike
  , p_ones
  , p_onesLike
  , p_diag
  , p_eye
  , p_arange
  , p_range
  , p_randperm
  , p_reshape
  , p_sort
  , p_topk
  , p_tril
  , p_triu
  , p_cat
  , p_catArray
  , p_equal
  , p_ltValue
  , p_leValue
  , p_gtValue
  , p_geValue
  , p_neValue
  , p_eqValue
  , p_ltValueT
  , p_leValueT
  , p_gtValueT
  , p_geValueT
  , p_neValueT
  , p_eqValueT
  , p_ltTensor
  , p_leTensor
  , p_gtTensor
  , p_geTensor
  , p_neTensor
  , p_eqTensor
  , p_ltTensorT
  , p_leTensorT
  , p_gtTensorT
  , p_geTensorT
  , p_neTensorT
  , p_eqTensorT
  , p_abs
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_fill :  r_ value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_fill"
  c_fill :: Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_zero :  r_ -> void
foreign import ccall "THTensorMath.h c_THTensorLong_zero"
  c_zero :: Ptr (CTHLongTensor) -> IO (())

-- | c_maskedFill :  tensor mask value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_maskedFill"
  c_maskedFill :: Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> CLong -> IO (())

-- | c_maskedCopy :  tensor mask src -> void
foreign import ccall "THTensorMath.h c_THTensorLong_maskedCopy"
  c_maskedCopy :: Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_maskedSelect :  tensor src mask -> void
foreign import ccall "THTensorMath.h c_THTensorLong_maskedSelect"
  c_maskedSelect :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_nonzero :  subscript tensor -> void
foreign import ccall "THTensorMath.h c_THTensorLong_nonzero"
  c_nonzero :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_indexSelect :  tensor src dim index -> void
foreign import ccall "THTensorMath.h c_THTensorLong_indexSelect"
  c_indexSelect :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (())

-- | c_indexCopy :  tensor dim index src -> void
foreign import ccall "THTensorMath.h c_THTensorLong_indexCopy"
  c_indexCopy :: Ptr (CTHLongTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_indexAdd :  tensor dim index src -> void
foreign import ccall "THTensorMath.h c_THTensorLong_indexAdd"
  c_indexAdd :: Ptr (CTHLongTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_indexFill :  tensor dim index val -> void
foreign import ccall "THTensorMath.h c_THTensorLong_indexFill"
  c_indexFill :: Ptr (CTHLongTensor) -> CInt -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_take :  tensor src index -> void
foreign import ccall "THTensorMath.h c_THTensorLong_take"
  c_take :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_put :  tensor index src accumulate -> void
foreign import ccall "THTensorMath.h c_THTensorLong_put"
  c_put :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> IO (())

-- | c_gather :  tensor src dim index -> void
foreign import ccall "THTensorMath.h c_THTensorLong_gather"
  c_gather :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (())

-- | c_scatter :  tensor dim index src -> void
foreign import ccall "THTensorMath.h c_THTensorLong_scatter"
  c_scatter :: Ptr (CTHLongTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_scatterAdd :  tensor dim index src -> void
foreign import ccall "THTensorMath.h c_THTensorLong_scatterAdd"
  c_scatterAdd :: Ptr (CTHLongTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_scatterFill :  tensor dim index val -> void
foreign import ccall "THTensorMath.h c_THTensorLong_scatterFill"
  c_scatterFill :: Ptr (CTHLongTensor) -> CInt -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_dot :  t src -> accreal
foreign import ccall "THTensorMath.h c_THTensorLong_dot"
  c_dot :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (CLong)

-- | c_minall :  t -> real
foreign import ccall "THTensorMath.h c_THTensorLong_minall"
  c_minall :: Ptr (CTHLongTensor) -> IO (CLong)

-- | c_maxall :  t -> real
foreign import ccall "THTensorMath.h c_THTensorLong_maxall"
  c_maxall :: Ptr (CTHLongTensor) -> IO (CLong)

-- | c_medianall :  t -> real
foreign import ccall "THTensorMath.h c_THTensorLong_medianall"
  c_medianall :: Ptr (CTHLongTensor) -> IO (CLong)

-- | c_sumall :  t -> accreal
foreign import ccall "THTensorMath.h c_THTensorLong_sumall"
  c_sumall :: Ptr (CTHLongTensor) -> IO (CLong)

-- | c_prodall :  t -> accreal
foreign import ccall "THTensorMath.h c_THTensorLong_prodall"
  c_prodall :: Ptr (CTHLongTensor) -> IO (CLong)

-- | c_neg :  self src -> void
foreign import ccall "THTensorMath.h c_THTensorLong_neg"
  c_neg :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_add :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_add"
  c_add :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_sub :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_sub"
  c_sub :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_add_scaled :  r_ t value alpha -> void
foreign import ccall "THTensorMath.h c_THTensorLong_add_scaled"
  c_add_scaled :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> CLong -> IO (())

-- | c_sub_scaled :  r_ t value alpha -> void
foreign import ccall "THTensorMath.h c_THTensorLong_sub_scaled"
  c_sub_scaled :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> CLong -> IO (())

-- | c_mul :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_mul"
  c_mul :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_div :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_div"
  c_div :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_lshift :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_lshift"
  c_lshift :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_rshift :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_rshift"
  c_rshift :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_fmod :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_fmod"
  c_fmod :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_remainder :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_remainder"
  c_remainder :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_clamp :  r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_clamp"
  c_clamp :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> CLong -> IO (())

-- | c_bitand :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_bitand"
  c_bitand :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_bitor :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_bitor"
  c_bitor :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_bitxor :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_bitxor"
  c_bitxor :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_cadd :  r_ t value src -> void
foreign import ccall "THTensorMath.h c_THTensorLong_cadd"
  c_cadd :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> IO (())

-- | c_csub :  self src1 value src2 -> void
foreign import ccall "THTensorMath.h c_THTensorLong_csub"
  c_csub :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> IO (())

-- | c_cmul :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorLong_cmul"
  c_cmul :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_cpow :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorLong_cpow"
  c_cpow :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_cdiv :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorLong_cdiv"
  c_cdiv :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_clshift :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorLong_clshift"
  c_clshift :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_crshift :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorLong_crshift"
  c_crshift :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_cfmod :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorLong_cfmod"
  c_cfmod :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_cremainder :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorLong_cremainder"
  c_cremainder :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_cbitand :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorLong_cbitand"
  c_cbitand :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_cbitor :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorLong_cbitor"
  c_cbitor :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_cbitxor :  r_ t src -> void
foreign import ccall "THTensorMath.h c_THTensorLong_cbitxor"
  c_cbitxor :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_addcmul :  r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h c_THTensorLong_addcmul"
  c_addcmul :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_addcdiv :  r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h c_THTensorLong_addcdiv"
  c_addcdiv :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_addmv :  r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h c_THTensorLong_addmv"
  c_addmv :: Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_addmm :  r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h c_THTensorLong_addmm"
  c_addmm :: Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_addr :  r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h c_THTensorLong_addr"
  c_addr :: Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_addbmm :  r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h c_THTensorLong_addbmm"
  c_addbmm :: Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_baddbmm :  r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h c_THTensorLong_baddbmm"
  c_baddbmm :: Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_match :  r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h c_THTensorLong_match"
  c_match :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_numel :  t -> ptrdiff_t
foreign import ccall "THTensorMath.h c_THTensorLong_numel"
  c_numel :: Ptr (CTHLongTensor) -> IO (CPtrdiff)

-- | c_max :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h c_THTensorLong_max"
  c_max :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> CInt -> IO (())

-- | c_min :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h c_THTensorLong_min"
  c_min :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> CInt -> IO (())

-- | c_kthvalue :  values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h c_THTensorLong_kthvalue"
  c_kthvalue :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLLong -> CInt -> CInt -> IO (())

-- | c_mode :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h c_THTensorLong_mode"
  c_mode :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> CInt -> IO (())

-- | c_median :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h c_THTensorLong_median"
  c_median :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> CInt -> IO (())

-- | c_sum :  r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h c_THTensorLong_sum"
  c_sum :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> CInt -> IO (())

-- | c_prod :  r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h c_THTensorLong_prod"
  c_prod :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> CInt -> IO (())

-- | c_cumsum :  r_ t dimension -> void
foreign import ccall "THTensorMath.h c_THTensorLong_cumsum"
  c_cumsum :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> IO (())

-- | c_cumprod :  r_ t dimension -> void
foreign import ccall "THTensorMath.h c_THTensorLong_cumprod"
  c_cumprod :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> IO (())

-- | c_sign :  r_ t -> void
foreign import ccall "THTensorMath.h c_THTensorLong_sign"
  c_sign :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_trace :  t -> accreal
foreign import ccall "THTensorMath.h c_THTensorLong_trace"
  c_trace :: Ptr (CTHLongTensor) -> IO (CLong)

-- | c_cross :  r_ a b dimension -> void
foreign import ccall "THTensorMath.h c_THTensorLong_cross"
  c_cross :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> IO (())

-- | c_cmax :  r t src -> void
foreign import ccall "THTensorMath.h c_THTensorLong_cmax"
  c_cmax :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_cmin :  r t src -> void
foreign import ccall "THTensorMath.h c_THTensorLong_cmin"
  c_cmin :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_cmaxValue :  r t value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_cmaxValue"
  c_cmaxValue :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_cminValue :  r t value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_cminValue"
  c_cminValue :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_zeros :  r_ size -> void
foreign import ccall "THTensorMath.h c_THTensorLong_zeros"
  c_zeros :: Ptr (CTHLongTensor) -> Ptr (CTHLongStorage) -> IO (())

-- | c_zerosLike :  r_ input -> void
foreign import ccall "THTensorMath.h c_THTensorLong_zerosLike"
  c_zerosLike :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_ones :  r_ size -> void
foreign import ccall "THTensorMath.h c_THTensorLong_ones"
  c_ones :: Ptr (CTHLongTensor) -> Ptr (CTHLongStorage) -> IO (())

-- | c_onesLike :  r_ input -> void
foreign import ccall "THTensorMath.h c_THTensorLong_onesLike"
  c_onesLike :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_diag :  r_ t k -> void
foreign import ccall "THTensorMath.h c_THTensorLong_diag"
  c_diag :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> IO (())

-- | c_eye :  r_ n m -> void
foreign import ccall "THTensorMath.h c_THTensorLong_eye"
  c_eye :: Ptr (CTHLongTensor) -> CLLong -> CLLong -> IO (())

-- | c_arange :  r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h c_THTensorLong_arange"
  c_arange :: Ptr (CTHLongTensor) -> CLong -> CLong -> CLong -> IO (())

-- | c_range :  r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h c_THTensorLong_range"
  c_range :: Ptr (CTHLongTensor) -> CLong -> CLong -> CLong -> IO (())

-- | c_randperm :  r_ _generator n -> void
foreign import ccall "THTensorMath.h c_THTensorLong_randperm"
  c_randperm :: Ptr (CTHLongTensor) -> Ptr (CTHGenerator) -> CLLong -> IO (())

-- | c_reshape :  r_ t size -> void
foreign import ccall "THTensorMath.h c_THTensorLong_reshape"
  c_reshape :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongStorage) -> IO (())

-- | c_sort :  rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h c_THTensorLong_sort"
  c_sort :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> CInt -> IO (())

-- | c_topk :  rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h c_THTensorLong_topk"
  c_topk :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLLong -> CInt -> CInt -> CInt -> IO (())

-- | c_tril :  r_ t k -> void
foreign import ccall "THTensorMath.h c_THTensorLong_tril"
  c_tril :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLLong -> IO (())

-- | c_triu :  r_ t k -> void
foreign import ccall "THTensorMath.h c_THTensorLong_triu"
  c_triu :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLLong -> IO (())

-- | c_cat :  r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h c_THTensorLong_cat"
  c_cat :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> IO (())

-- | c_catArray :  result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h c_THTensorLong_catArray"
  c_catArray :: Ptr (CTHLongTensor) -> Ptr (Ptr (CTHLongTensor)) -> CInt -> CInt -> IO (())

-- | c_equal :  ta tb -> int
foreign import ccall "THTensorMath.h c_THTensorLong_equal"
  c_equal :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (CInt)

-- | c_ltValue :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_ltValue"
  c_ltValue :: Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_leValue :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_leValue"
  c_leValue :: Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_gtValue :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_gtValue"
  c_gtValue :: Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_geValue :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_geValue"
  c_geValue :: Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_neValue :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_neValue"
  c_neValue :: Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_eqValue :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_eqValue"
  c_eqValue :: Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_ltValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_ltValueT"
  c_ltValueT :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_leValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_leValueT"
  c_leValueT :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_gtValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_gtValueT"
  c_gtValueT :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_geValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_geValueT"
  c_geValueT :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_neValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_neValueT"
  c_neValueT :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_eqValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h c_THTensorLong_eqValueT"
  c_eqValueT :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (())

-- | c_ltTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorLong_ltTensor"
  c_ltTensor :: Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_leTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorLong_leTensor"
  c_leTensor :: Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_gtTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorLong_gtTensor"
  c_gtTensor :: Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_geTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorLong_geTensor"
  c_geTensor :: Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_neTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorLong_neTensor"
  c_neTensor :: Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_eqTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorLong_eqTensor"
  c_eqTensor :: Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_ltTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorLong_ltTensorT"
  c_ltTensorT :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_leTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorLong_leTensorT"
  c_leTensorT :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_gtTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorLong_gtTensorT"
  c_gtTensorT :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_geTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorLong_geTensorT"
  c_geTensorT :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_neTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorLong_neTensorT"
  c_neTensorT :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_eqTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h c_THTensorLong_eqTensorT"
  c_eqTensorT :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_abs :  r_ t -> void
foreign import ccall "THTensorMath.h c_THTensorLong_abs"
  c_abs :: Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | p_fill : Pointer to function : r_ value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_fill"
  p_fill :: FunPtr (Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_zero : Pointer to function : r_ -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_zero"
  p_zero :: FunPtr (Ptr (CTHLongTensor) -> IO (()))

-- | p_maskedFill : Pointer to function : tensor mask value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_maskedFill"
  p_maskedFill :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> CLong -> IO (()))

-- | p_maskedCopy : Pointer to function : tensor mask src -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_maskedCopy"
  p_maskedCopy :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_maskedSelect : Pointer to function : tensor src mask -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_maskedSelect"
  p_maskedSelect :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_nonzero : Pointer to function : subscript tensor -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_nonzero"
  p_nonzero :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_indexSelect : Pointer to function : tensor src dim index -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_indexSelect"
  p_indexSelect :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (()))

-- | p_indexCopy : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_indexCopy"
  p_indexCopy :: FunPtr (Ptr (CTHLongTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_indexAdd : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_indexAdd"
  p_indexAdd :: FunPtr (Ptr (CTHLongTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_indexFill : Pointer to function : tensor dim index val -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_indexFill"
  p_indexFill :: FunPtr (Ptr (CTHLongTensor) -> CInt -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_take : Pointer to function : tensor src index -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_take"
  p_take :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_put : Pointer to function : tensor index src accumulate -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_put"
  p_put :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> IO (()))

-- | p_gather : Pointer to function : tensor src dim index -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_gather"
  p_gather :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (()))

-- | p_scatter : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_scatter"
  p_scatter :: FunPtr (Ptr (CTHLongTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_scatterAdd : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_scatterAdd"
  p_scatterAdd :: FunPtr (Ptr (CTHLongTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_scatterFill : Pointer to function : tensor dim index val -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_scatterFill"
  p_scatterFill :: FunPtr (Ptr (CTHLongTensor) -> CInt -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_dot : Pointer to function : t src -> accreal
foreign import ccall "THTensorMath.h &p_THTensorLong_dot"
  p_dot :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (CLong))

-- | p_minall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &p_THTensorLong_minall"
  p_minall :: FunPtr (Ptr (CTHLongTensor) -> IO (CLong))

-- | p_maxall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &p_THTensorLong_maxall"
  p_maxall :: FunPtr (Ptr (CTHLongTensor) -> IO (CLong))

-- | p_medianall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &p_THTensorLong_medianall"
  p_medianall :: FunPtr (Ptr (CTHLongTensor) -> IO (CLong))

-- | p_sumall : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &p_THTensorLong_sumall"
  p_sumall :: FunPtr (Ptr (CTHLongTensor) -> IO (CLong))

-- | p_prodall : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &p_THTensorLong_prodall"
  p_prodall :: FunPtr (Ptr (CTHLongTensor) -> IO (CLong))

-- | p_neg : Pointer to function : self src -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_neg"
  p_neg :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_add : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_add"
  p_add :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_sub : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_sub"
  p_sub :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_add_scaled : Pointer to function : r_ t value alpha -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_add_scaled"
  p_add_scaled :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> CLong -> IO (()))

-- | p_sub_scaled : Pointer to function : r_ t value alpha -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_sub_scaled"
  p_sub_scaled :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> CLong -> IO (()))

-- | p_mul : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_mul"
  p_mul :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_div : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_div"
  p_div :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_lshift : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_lshift"
  p_lshift :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_rshift : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_rshift"
  p_rshift :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_fmod : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_fmod"
  p_fmod :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_remainder : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_remainder"
  p_remainder :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_clamp : Pointer to function : r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_clamp"
  p_clamp :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> CLong -> IO (()))

-- | p_bitand : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_bitand"
  p_bitand :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_bitor : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_bitor"
  p_bitor :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_bitxor : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_bitxor"
  p_bitxor :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_cadd : Pointer to function : r_ t value src -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_cadd"
  p_cadd :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> IO (()))

-- | p_csub : Pointer to function : self src1 value src2 -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_csub"
  p_csub :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> IO (()))

-- | p_cmul : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_cmul"
  p_cmul :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_cpow : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_cpow"
  p_cpow :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_cdiv : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_cdiv"
  p_cdiv :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_clshift : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_clshift"
  p_clshift :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_crshift : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_crshift"
  p_crshift :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_cfmod : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_cfmod"
  p_cfmod :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_cremainder : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_cremainder"
  p_cremainder :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_cbitand : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_cbitand"
  p_cbitand :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_cbitor : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_cbitor"
  p_cbitor :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_cbitxor : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_cbitxor"
  p_cbitxor :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_addcmul : Pointer to function : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_addcmul"
  p_addcmul :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_addcdiv : Pointer to function : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_addcdiv"
  p_addcdiv :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_addmv : Pointer to function : r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_addmv"
  p_addmv :: FunPtr (Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_addmm : Pointer to function : r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_addmm"
  p_addmm :: FunPtr (Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_addr : Pointer to function : r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_addr"
  p_addr :: FunPtr (Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_addbmm : Pointer to function : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_addbmm"
  p_addbmm :: FunPtr (Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_baddbmm : Pointer to function : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_baddbmm"
  p_baddbmm :: FunPtr (Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> CLong -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_match : Pointer to function : r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_match"
  p_match :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_numel : Pointer to function : t -> ptrdiff_t
foreign import ccall "THTensorMath.h &p_THTensorLong_numel"
  p_numel :: FunPtr (Ptr (CTHLongTensor) -> IO (CPtrdiff))

-- | p_max : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_max"
  p_max :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> CInt -> IO (()))

-- | p_min : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_min"
  p_min :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> CInt -> IO (()))

-- | p_kthvalue : Pointer to function : values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_kthvalue"
  p_kthvalue :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLLong -> CInt -> CInt -> IO (()))

-- | p_mode : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_mode"
  p_mode :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> CInt -> IO (()))

-- | p_median : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_median"
  p_median :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> CInt -> IO (()))

-- | p_sum : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_sum"
  p_sum :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> CInt -> IO (()))

-- | p_prod : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_prod"
  p_prod :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> CInt -> IO (()))

-- | p_cumsum : Pointer to function : r_ t dimension -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_cumsum"
  p_cumsum :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> IO (()))

-- | p_cumprod : Pointer to function : r_ t dimension -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_cumprod"
  p_cumprod :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> IO (()))

-- | p_sign : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_sign"
  p_sign :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_trace : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &p_THTensorLong_trace"
  p_trace :: FunPtr (Ptr (CTHLongTensor) -> IO (CLong))

-- | p_cross : Pointer to function : r_ a b dimension -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_cross"
  p_cross :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> IO (()))

-- | p_cmax : Pointer to function : r t src -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_cmax"
  p_cmax :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_cmin : Pointer to function : r t src -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_cmin"
  p_cmin :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_cmaxValue : Pointer to function : r t value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_cmaxValue"
  p_cmaxValue :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_cminValue : Pointer to function : r t value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_cminValue"
  p_cminValue :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_zeros : Pointer to function : r_ size -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_zeros"
  p_zeros :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_zerosLike : Pointer to function : r_ input -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_zerosLike"
  p_zerosLike :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_ones : Pointer to function : r_ size -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_ones"
  p_ones :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_onesLike : Pointer to function : r_ input -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_onesLike"
  p_onesLike :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_diag : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_diag"
  p_diag :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> IO (()))

-- | p_eye : Pointer to function : r_ n m -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_eye"
  p_eye :: FunPtr (Ptr (CTHLongTensor) -> CLLong -> CLLong -> IO (()))

-- | p_arange : Pointer to function : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_arange"
  p_arange :: FunPtr (Ptr (CTHLongTensor) -> CLong -> CLong -> CLong -> IO (()))

-- | p_range : Pointer to function : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_range"
  p_range :: FunPtr (Ptr (CTHLongTensor) -> CLong -> CLong -> CLong -> IO (()))

-- | p_randperm : Pointer to function : r_ _generator n -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_randperm"
  p_randperm :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHGenerator) -> CLLong -> IO (()))

-- | p_reshape : Pointer to function : r_ t size -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_reshape"
  p_reshape :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_sort : Pointer to function : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_sort"
  p_sort :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> CInt -> IO (()))

-- | p_topk : Pointer to function : rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_topk"
  p_topk :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLLong -> CInt -> CInt -> CInt -> IO (()))

-- | p_tril : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_tril"
  p_tril :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLLong -> IO (()))

-- | p_triu : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_triu"
  p_triu :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLLong -> IO (()))

-- | p_cat : Pointer to function : r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_cat"
  p_cat :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> IO (()))

-- | p_catArray : Pointer to function : result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_catArray"
  p_catArray :: FunPtr (Ptr (CTHLongTensor) -> Ptr (Ptr (CTHLongTensor)) -> CInt -> CInt -> IO (()))

-- | p_equal : Pointer to function : ta tb -> int
foreign import ccall "THTensorMath.h &p_THTensorLong_equal"
  p_equal :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (CInt))

-- | p_ltValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_ltValue"
  p_ltValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_leValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_leValue"
  p_leValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_gtValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_gtValue"
  p_gtValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_geValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_geValue"
  p_geValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_neValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_neValue"
  p_neValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_eqValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_eqValue"
  p_eqValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_ltValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_ltValueT"
  p_ltValueT :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_leValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_leValueT"
  p_leValueT :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_gtValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_gtValueT"
  p_gtValueT :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_geValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_geValueT"
  p_geValueT :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_neValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_neValueT"
  p_neValueT :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_eqValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_eqValueT"
  p_eqValueT :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CLong -> IO (()))

-- | p_ltTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_ltTensor"
  p_ltTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_leTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_leTensor"
  p_leTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_gtTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_gtTensor"
  p_gtTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_geTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_geTensor"
  p_geTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_neTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_neTensor"
  p_neTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_eqTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_eqTensor"
  p_eqTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_ltTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_ltTensorT"
  p_ltTensorT :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_leTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_leTensorT"
  p_leTensorT :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_gtTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_gtTensorT"
  p_gtTensorT :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_geTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_geTensorT"
  p_geTensorT :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_neTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_neTensorT"
  p_neTensorT :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_eqTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_eqTensorT"
  p_eqTensorT :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_abs : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &p_THTensorLong_abs"
  p_abs :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> IO (()))