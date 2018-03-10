{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Double.TensorMath
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
  , c_cinv
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
  , c_sigmoid
  , c_log
  , c_lgamma
  , c_digamma
  , c_trigamma
  , c_polygamma
  , c_log1p
  , c_exp
  , c_expm1
  , c_cos
  , c_acos
  , c_cosh
  , c_sin
  , c_asin
  , c_sinh
  , c_tan
  , c_atan
  , c_atan2
  , c_tanh
  , c_erf
  , c_erfinv
  , c_pow
  , c_tpow
  , c_sqrt
  , c_rsqrt
  , c_ceil
  , c_floor
  , c_round
  , c_trunc
  , c_frac
  , c_lerp
  , c_mean
  , c_std
  , c_var
  , c_norm
  , c_renorm
  , c_dist
  , c_histc
  , c_bhistc
  , c_meanall
  , c_varall
  , c_stdall
  , c_normall
  , c_linspace
  , c_logspace
  , c_rand
  , c_randn
  , c_dirichlet_grad
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
  , p_cinv
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
  , p_sigmoid
  , p_log
  , p_lgamma
  , p_digamma
  , p_trigamma
  , p_polygamma
  , p_log1p
  , p_exp
  , p_expm1
  , p_cos
  , p_acos
  , p_cosh
  , p_sin
  , p_asin
  , p_sinh
  , p_tan
  , p_atan
  , p_atan2
  , p_tanh
  , p_erf
  , p_erfinv
  , p_pow
  , p_tpow
  , p_sqrt
  , p_rsqrt
  , p_ceil
  , p_floor
  , p_round
  , p_trunc
  , p_frac
  , p_lerp
  , p_mean
  , p_std
  , p_var
  , p_norm
  , p_renorm
  , p_dist
  , p_histc
  , p_bhistc
  , p_meanall
  , p_varall
  , p_stdall
  , p_normall
  , p_linspace
  , p_logspace
  , p_rand
  , p_randn
  , p_dirichlet_grad
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_fill :  r_ value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_fill"
  c_fill :: Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_zero :  r_ -> void
foreign import ccall "THTensorMath.h THDoubleTensor_zero"
  c_zero :: Ptr CTHDoubleTensor -> IO ()

-- | c_maskedFill :  tensor mask value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_maskedFill"
  c_maskedFill :: Ptr CTHDoubleTensor -> Ptr CTHByteTensor -> CDouble -> IO ()

-- | c_maskedCopy :  tensor mask src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_maskedCopy"
  c_maskedCopy :: Ptr CTHDoubleTensor -> Ptr CTHByteTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_maskedSelect :  tensor src mask -> void
foreign import ccall "THTensorMath.h THDoubleTensor_maskedSelect"
  c_maskedSelect :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHByteTensor -> IO ()

-- | c_nonzero :  subscript tensor -> void
foreign import ccall "THTensorMath.h THDoubleTensor_nonzero"
  c_nonzero :: Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_indexSelect :  tensor src dim index -> void
foreign import ccall "THTensorMath.h THDoubleTensor_indexSelect"
  c_indexSelect :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CInt -> Ptr CTHLongTensor -> IO ()

-- | c_indexCopy :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_indexCopy"
  c_indexCopy :: Ptr CTHDoubleTensor -> CInt -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_indexAdd :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_indexAdd"
  c_indexAdd :: Ptr CTHDoubleTensor -> CInt -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_indexFill :  tensor dim index val -> void
foreign import ccall "THTensorMath.h THDoubleTensor_indexFill"
  c_indexFill :: Ptr CTHDoubleTensor -> CInt -> Ptr CTHLongTensor -> CDouble -> IO ()

-- | c_take :  tensor src index -> void
foreign import ccall "THTensorMath.h THDoubleTensor_take"
  c_take :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHLongTensor -> IO ()

-- | c_put :  tensor index src accumulate -> void
foreign import ccall "THTensorMath.h THDoubleTensor_put"
  c_put :: Ptr CTHDoubleTensor -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> CInt -> IO ()

-- | c_gather :  tensor src dim index -> void
foreign import ccall "THTensorMath.h THDoubleTensor_gather"
  c_gather :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CInt -> Ptr CTHLongTensor -> IO ()

-- | c_scatter :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_scatter"
  c_scatter :: Ptr CTHDoubleTensor -> CInt -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_scatterAdd :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_scatterAdd"
  c_scatterAdd :: Ptr CTHDoubleTensor -> CInt -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_scatterFill :  tensor dim index val -> void
foreign import ccall "THTensorMath.h THDoubleTensor_scatterFill"
  c_scatterFill :: Ptr CTHDoubleTensor -> CInt -> Ptr CTHLongTensor -> CDouble -> IO ()

-- | c_dot :  t src -> accreal
foreign import ccall "THTensorMath.h THDoubleTensor_dot"
  c_dot :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO CDouble

-- | c_minall :  t -> real
foreign import ccall "THTensorMath.h THDoubleTensor_minall"
  c_minall :: Ptr CTHDoubleTensor -> IO CDouble

-- | c_maxall :  t -> real
foreign import ccall "THTensorMath.h THDoubleTensor_maxall"
  c_maxall :: Ptr CTHDoubleTensor -> IO CDouble

-- | c_medianall :  t -> real
foreign import ccall "THTensorMath.h THDoubleTensor_medianall"
  c_medianall :: Ptr CTHDoubleTensor -> IO CDouble

-- | c_sumall :  t -> accreal
foreign import ccall "THTensorMath.h THDoubleTensor_sumall"
  c_sumall :: Ptr CTHDoubleTensor -> IO CDouble

-- | c_prodall :  t -> accreal
foreign import ccall "THTensorMath.h THDoubleTensor_prodall"
  c_prodall :: Ptr CTHDoubleTensor -> IO CDouble

-- | c_neg :  self src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_neg"
  c_neg :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_cinv :  self src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cinv"
  c_cinv :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_add :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_add"
  c_add :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_sub :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_sub"
  c_sub :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_add_scaled :  r_ t value alpha -> void
foreign import ccall "THTensorMath.h THDoubleTensor_add_scaled"
  c_add_scaled :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> CDouble -> IO ()

-- | c_sub_scaled :  r_ t value alpha -> void
foreign import ccall "THTensorMath.h THDoubleTensor_sub_scaled"
  c_sub_scaled :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> CDouble -> IO ()

-- | c_mul :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_mul"
  c_mul :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_div :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_div"
  c_div :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_lshift :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_lshift"
  c_lshift :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_rshift :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_rshift"
  c_rshift :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_fmod :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_fmod"
  c_fmod :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_remainder :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_remainder"
  c_remainder :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_clamp :  r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_clamp"
  c_clamp :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> CDouble -> IO ()

-- | c_bitand :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_bitand"
  c_bitand :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_bitor :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_bitor"
  c_bitor :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_bitxor :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_bitxor"
  c_bitxor :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_cadd :  r_ t value src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cadd"
  c_cadd :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> IO ()

-- | c_csub :  self src1 value src2 -> void
foreign import ccall "THTensorMath.h THDoubleTensor_csub"
  c_csub :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> IO ()

-- | c_cmul :  r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cmul"
  c_cmul :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_cpow :  r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cpow"
  c_cpow :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_cdiv :  r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cdiv"
  c_cdiv :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_clshift :  r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_clshift"
  c_clshift :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_crshift :  r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_crshift"
  c_crshift :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_cfmod :  r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cfmod"
  c_cfmod :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_cremainder :  r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cremainder"
  c_cremainder :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_cbitand :  r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cbitand"
  c_cbitand :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_cbitor :  r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cbitor"
  c_cbitor :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_cbitxor :  r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cbitxor"
  c_cbitxor :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_addcmul :  r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THDoubleTensor_addcmul"
  c_addcmul :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_addcdiv :  r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THDoubleTensor_addcdiv"
  c_addcdiv :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_addmv :  r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h THDoubleTensor_addmv"
  c_addmv :: Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_addmm :  r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h THDoubleTensor_addmm"
  c_addmm :: Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_addr :  r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h THDoubleTensor_addr"
  c_addr :: Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_addbmm :  r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THDoubleTensor_addbmm"
  c_addbmm :: Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_baddbmm :  r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THDoubleTensor_baddbmm"
  c_baddbmm :: Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_match :  r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h THDoubleTensor_match"
  c_match :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_numel :  t -> ptrdiff_t
foreign import ccall "THTensorMath.h THDoubleTensor_numel"
  c_numel :: Ptr CTHDoubleTensor -> IO CPtrdiff

-- | c_max :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensor_max"
  c_max :: Ptr CTHDoubleTensor -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> CInt -> CInt -> IO ()

-- | c_min :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensor_min"
  c_min :: Ptr CTHDoubleTensor -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> CInt -> CInt -> IO ()

-- | c_kthvalue :  values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensor_kthvalue"
  c_kthvalue :: Ptr CTHDoubleTensor -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> CLLong -> CInt -> CInt -> IO ()

-- | c_mode :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensor_mode"
  c_mode :: Ptr CTHDoubleTensor -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> CInt -> CInt -> IO ()

-- | c_median :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensor_median"
  c_median :: Ptr CTHDoubleTensor -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> CInt -> CInt -> IO ()

-- | c_sum :  r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensor_sum"
  c_sum :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CInt -> CInt -> IO ()

-- | c_prod :  r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensor_prod"
  c_prod :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CInt -> CInt -> IO ()

-- | c_cumsum :  r_ t dimension -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cumsum"
  c_cumsum :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CInt -> IO ()

-- | c_cumprod :  r_ t dimension -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cumprod"
  c_cumprod :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CInt -> IO ()

-- | c_sign :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_sign"
  c_sign :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_trace :  t -> accreal
foreign import ccall "THTensorMath.h THDoubleTensor_trace"
  c_trace :: Ptr CTHDoubleTensor -> IO CDouble

-- | c_cross :  r_ a b dimension -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cross"
  c_cross :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CInt -> IO ()

-- | c_cmax :  r t src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cmax"
  c_cmax :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_cmin :  r t src -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cmin"
  c_cmin :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_cmaxValue :  r t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cmaxValue"
  c_cmaxValue :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_cminValue :  r t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cminValue"
  c_cminValue :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_zeros :  r_ size -> void
foreign import ccall "THTensorMath.h THDoubleTensor_zeros"
  c_zeros :: Ptr CTHDoubleTensor -> Ptr CTHLongStorage -> IO ()

-- | c_zerosLike :  r_ input -> void
foreign import ccall "THTensorMath.h THDoubleTensor_zerosLike"
  c_zerosLike :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_ones :  r_ size -> void
foreign import ccall "THTensorMath.h THDoubleTensor_ones"
  c_ones :: Ptr CTHDoubleTensor -> Ptr CTHLongStorage -> IO ()

-- | c_onesLike :  r_ input -> void
foreign import ccall "THTensorMath.h THDoubleTensor_onesLike"
  c_onesLike :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_diag :  r_ t k -> void
foreign import ccall "THTensorMath.h THDoubleTensor_diag"
  c_diag :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CInt -> IO ()

-- | c_eye :  r_ n m -> void
foreign import ccall "THTensorMath.h THDoubleTensor_eye"
  c_eye :: Ptr CTHDoubleTensor -> CLLong -> CLLong -> IO ()

-- | c_arange :  r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THDoubleTensor_arange"
  c_arange :: Ptr CTHDoubleTensor -> CDouble -> CDouble -> CDouble -> IO ()

-- | c_range :  r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THDoubleTensor_range"
  c_range :: Ptr CTHDoubleTensor -> CDouble -> CDouble -> CDouble -> IO ()

-- | c_randperm :  r_ _generator n -> void
foreign import ccall "THTensorMath.h THDoubleTensor_randperm"
  c_randperm :: Ptr CTHDoubleTensor -> Ptr CTHGenerator -> CLLong -> IO ()

-- | c_reshape :  r_ t size -> void
foreign import ccall "THTensorMath.h THDoubleTensor_reshape"
  c_reshape :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHLongStorage -> IO ()

-- | c_sort :  rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h THDoubleTensor_sort"
  c_sort :: Ptr CTHDoubleTensor -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> CInt -> CInt -> IO ()

-- | c_topk :  rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h THDoubleTensor_topk"
  c_topk :: Ptr CTHDoubleTensor -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> CLLong -> CInt -> CInt -> CInt -> IO ()

-- | c_tril :  r_ t k -> void
foreign import ccall "THTensorMath.h THDoubleTensor_tril"
  c_tril :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CLLong -> IO ()

-- | c_triu :  r_ t k -> void
foreign import ccall "THTensorMath.h THDoubleTensor_triu"
  c_triu :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CLLong -> IO ()

-- | c_cat :  r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cat"
  c_cat :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CInt -> IO ()

-- | c_catArray :  result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h THDoubleTensor_catArray"
  c_catArray :: Ptr CTHDoubleTensor -> Ptr (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- | c_equal :  ta tb -> int
foreign import ccall "THTensorMath.h THDoubleTensor_equal"
  c_equal :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO CInt

-- | c_ltValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_ltValue"
  c_ltValue :: Ptr CTHByteTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_leValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_leValue"
  c_leValue :: Ptr CTHByteTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_gtValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_gtValue"
  c_gtValue :: Ptr CTHByteTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_geValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_geValue"
  c_geValue :: Ptr CTHByteTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_neValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_neValue"
  c_neValue :: Ptr CTHByteTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_eqValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_eqValue"
  c_eqValue :: Ptr CTHByteTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_ltValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_ltValueT"
  c_ltValueT :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_leValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_leValueT"
  c_leValueT :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_gtValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_gtValueT"
  c_gtValueT :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_geValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_geValueT"
  c_geValueT :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_neValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_neValueT"
  c_neValueT :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_eqValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_eqValueT"
  c_eqValueT :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_ltTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensor_ltTensor"
  c_ltTensor :: Ptr CTHByteTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_leTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensor_leTensor"
  c_leTensor :: Ptr CTHByteTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_gtTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensor_gtTensor"
  c_gtTensor :: Ptr CTHByteTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_geTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensor_geTensor"
  c_geTensor :: Ptr CTHByteTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_neTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensor_neTensor"
  c_neTensor :: Ptr CTHByteTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_eqTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensor_eqTensor"
  c_eqTensor :: Ptr CTHByteTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_ltTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensor_ltTensorT"
  c_ltTensorT :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_leTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensor_leTensorT"
  c_leTensorT :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_gtTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensor_gtTensorT"
  c_gtTensorT :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_geTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensor_geTensorT"
  c_geTensorT :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_neTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensor_neTensorT"
  c_neTensorT :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_eqTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensor_eqTensorT"
  c_eqTensorT :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_abs :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_abs"
  c_abs :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_sigmoid :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_sigmoid"
  c_sigmoid :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_log :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_log"
  c_log :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_lgamma :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_lgamma"
  c_lgamma :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_digamma :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_digamma"
  c_digamma :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_trigamma :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_trigamma"
  c_trigamma :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_polygamma :  r_ n t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_polygamma"
  c_polygamma :: Ptr CTHDoubleTensor -> CLLong -> Ptr CTHDoubleTensor -> IO ()

-- | c_log1p :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_log1p"
  c_log1p :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_exp :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_exp"
  c_exp :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_expm1 :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_expm1"
  c_expm1 :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_cos :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cos"
  c_cos :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_acos :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_acos"
  c_acos :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_cosh :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_cosh"
  c_cosh :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_sin :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_sin"
  c_sin :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_asin :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_asin"
  c_asin :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_sinh :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_sinh"
  c_sinh :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_tan :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_tan"
  c_tan :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_atan :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_atan"
  c_atan :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_atan2 :  r_ tx ty -> void
foreign import ccall "THTensorMath.h THDoubleTensor_atan2"
  c_atan2 :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_tanh :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_tanh"
  c_tanh :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_erf :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_erf"
  c_erf :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_erfinv :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_erfinv"
  c_erfinv :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_pow :  r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensor_pow"
  c_pow :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_tpow :  r_ value t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_tpow"
  c_tpow :: Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> IO ()

-- | c_sqrt :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_sqrt"
  c_sqrt :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_rsqrt :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_rsqrt"
  c_rsqrt :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_ceil :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_ceil"
  c_ceil :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_floor :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_floor"
  c_floor :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_round :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_round"
  c_round :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_trunc :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_trunc"
  c_trunc :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_frac :  r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_frac"
  c_frac :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_lerp :  r_ a b weight -> void
foreign import ccall "THTensorMath.h THDoubleTensor_lerp"
  c_lerp :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_mean :  r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensor_mean"
  c_mean :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CInt -> CInt -> IO ()

-- | c_std :  r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensor_std"
  c_std :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CInt -> CInt -> CInt -> IO ()

-- | c_var :  r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensor_var"
  c_var :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CInt -> CInt -> CInt -> IO ()

-- | c_norm :  r_ t value dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensor_norm"
  c_norm :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> CInt -> CInt -> IO ()

-- | c_renorm :  r_ t value dimension maxnorm -> void
foreign import ccall "THTensorMath.h THDoubleTensor_renorm"
  c_renorm :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> CInt -> CDouble -> IO ()

-- | c_dist :  a b value -> accreal
foreign import ccall "THTensorMath.h THDoubleTensor_dist"
  c_dist :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO CDouble

-- | c_histc :  hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THDoubleTensor_histc"
  c_histc :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CLLong -> CDouble -> CDouble -> IO ()

-- | c_bhistc :  hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THDoubleTensor_bhistc"
  c_bhistc :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CLLong -> CDouble -> CDouble -> IO ()

-- | c_meanall :  self -> accreal
foreign import ccall "THTensorMath.h THDoubleTensor_meanall"
  c_meanall :: Ptr CTHDoubleTensor -> IO CDouble

-- | c_varall :  self biased -> accreal
foreign import ccall "THTensorMath.h THDoubleTensor_varall"
  c_varall :: Ptr CTHDoubleTensor -> CInt -> IO CDouble

-- | c_stdall :  self biased -> accreal
foreign import ccall "THTensorMath.h THDoubleTensor_stdall"
  c_stdall :: Ptr CTHDoubleTensor -> CInt -> IO CDouble

-- | c_normall :  t value -> accreal
foreign import ccall "THTensorMath.h THDoubleTensor_normall"
  c_normall :: Ptr CTHDoubleTensor -> CDouble -> IO CDouble

-- | c_linspace :  r_ a b n -> void
foreign import ccall "THTensorMath.h THDoubleTensor_linspace"
  c_linspace :: Ptr CTHDoubleTensor -> CDouble -> CDouble -> CLLong -> IO ()

-- | c_logspace :  r_ a b n -> void
foreign import ccall "THTensorMath.h THDoubleTensor_logspace"
  c_logspace :: Ptr CTHDoubleTensor -> CDouble -> CDouble -> CLLong -> IO ()

-- | c_rand :  r_ _generator size -> void
foreign import ccall "THTensorMath.h THDoubleTensor_rand"
  c_rand :: Ptr CTHDoubleTensor -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()

-- | c_randn :  r_ _generator size -> void
foreign import ccall "THTensorMath.h THDoubleTensor_randn"
  c_randn :: Ptr CTHDoubleTensor -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()

-- | c_dirichlet_grad :  self x alpha total -> void
foreign import ccall "THTensorMath.h THDoubleTensor_dirichlet_grad"
  c_dirichlet_grad :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | p_fill : Pointer to function : r_ value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_fill"
  p_fill :: FunPtr (Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_zero : Pointer to function : r_ -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_zero"
  p_zero :: FunPtr (Ptr CTHDoubleTensor -> IO ())

-- | p_maskedFill : Pointer to function : tensor mask value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_maskedFill"
  p_maskedFill :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHByteTensor -> CDouble -> IO ())

-- | p_maskedCopy : Pointer to function : tensor mask src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_maskedCopy"
  p_maskedCopy :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHByteTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_maskedSelect : Pointer to function : tensor src mask -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_maskedSelect"
  p_maskedSelect :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHByteTensor -> IO ())

-- | p_nonzero : Pointer to function : subscript tensor -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_nonzero"
  p_nonzero :: FunPtr (Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_indexSelect : Pointer to function : tensor src dim index -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_indexSelect"
  p_indexSelect :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CInt -> Ptr CTHLongTensor -> IO ())

-- | p_indexCopy : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_indexCopy"
  p_indexCopy :: FunPtr (Ptr CTHDoubleTensor -> CInt -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_indexAdd : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_indexAdd"
  p_indexAdd :: FunPtr (Ptr CTHDoubleTensor -> CInt -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_indexFill : Pointer to function : tensor dim index val -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_indexFill"
  p_indexFill :: FunPtr (Ptr CTHDoubleTensor -> CInt -> Ptr CTHLongTensor -> CDouble -> IO ())

-- | p_take : Pointer to function : tensor src index -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_take"
  p_take :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHLongTensor -> IO ())

-- | p_put : Pointer to function : tensor index src accumulate -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_put"
  p_put :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> CInt -> IO ())

-- | p_gather : Pointer to function : tensor src dim index -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_gather"
  p_gather :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CInt -> Ptr CTHLongTensor -> IO ())

-- | p_scatter : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_scatter"
  p_scatter :: FunPtr (Ptr CTHDoubleTensor -> CInt -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_scatterAdd : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_scatterAdd"
  p_scatterAdd :: FunPtr (Ptr CTHDoubleTensor -> CInt -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_scatterFill : Pointer to function : tensor dim index val -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_scatterFill"
  p_scatterFill :: FunPtr (Ptr CTHDoubleTensor -> CInt -> Ptr CTHLongTensor -> CDouble -> IO ())

-- | p_dot : Pointer to function : t src -> accreal
foreign import ccall "THTensorMath.h &THDoubleTensor_dot"
  p_dot :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO CDouble)

-- | p_minall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THDoubleTensor_minall"
  p_minall :: FunPtr (Ptr CTHDoubleTensor -> IO CDouble)

-- | p_maxall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THDoubleTensor_maxall"
  p_maxall :: FunPtr (Ptr CTHDoubleTensor -> IO CDouble)

-- | p_medianall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THDoubleTensor_medianall"
  p_medianall :: FunPtr (Ptr CTHDoubleTensor -> IO CDouble)

-- | p_sumall : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THDoubleTensor_sumall"
  p_sumall :: FunPtr (Ptr CTHDoubleTensor -> IO CDouble)

-- | p_prodall : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THDoubleTensor_prodall"
  p_prodall :: FunPtr (Ptr CTHDoubleTensor -> IO CDouble)

-- | p_neg : Pointer to function : self src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_neg"
  p_neg :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_cinv : Pointer to function : self src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cinv"
  p_cinv :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_add : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_add"
  p_add :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_sub : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_sub"
  p_sub :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_add_scaled : Pointer to function : r_ t value alpha -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_add_scaled"
  p_add_scaled :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> CDouble -> IO ())

-- | p_sub_scaled : Pointer to function : r_ t value alpha -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_sub_scaled"
  p_sub_scaled :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> CDouble -> IO ())

-- | p_mul : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_mul"
  p_mul :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_div : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_div"
  p_div :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_lshift : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_lshift"
  p_lshift :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_rshift : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_rshift"
  p_rshift :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_fmod : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_fmod"
  p_fmod :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_remainder : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_remainder"
  p_remainder :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_clamp : Pointer to function : r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_clamp"
  p_clamp :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> CDouble -> IO ())

-- | p_bitand : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_bitand"
  p_bitand :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_bitor : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_bitor"
  p_bitor :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_bitxor : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_bitxor"
  p_bitxor :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_cadd : Pointer to function : r_ t value src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cadd"
  p_cadd :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> IO ())

-- | p_csub : Pointer to function : self src1 value src2 -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_csub"
  p_csub :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> IO ())

-- | p_cmul : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cmul"
  p_cmul :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_cpow : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cpow"
  p_cpow :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_cdiv : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cdiv"
  p_cdiv :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_clshift : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_clshift"
  p_clshift :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_crshift : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_crshift"
  p_crshift :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_cfmod : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cfmod"
  p_cfmod :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_cremainder : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cremainder"
  p_cremainder :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_cbitand : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cbitand"
  p_cbitand :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_cbitor : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cbitor"
  p_cbitor :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_cbitxor : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cbitxor"
  p_cbitxor :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_addcmul : Pointer to function : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_addcmul"
  p_addcmul :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_addcdiv : Pointer to function : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_addcdiv"
  p_addcdiv :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_addmv : Pointer to function : r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_addmv"
  p_addmv :: FunPtr (Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_addmm : Pointer to function : r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_addmm"
  p_addmm :: FunPtr (Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_addr : Pointer to function : r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_addr"
  p_addr :: FunPtr (Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_addbmm : Pointer to function : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_addbmm"
  p_addbmm :: FunPtr (Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_baddbmm : Pointer to function : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_baddbmm"
  p_baddbmm :: FunPtr (Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_match : Pointer to function : r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_match"
  p_match :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_numel : Pointer to function : t -> ptrdiff_t
foreign import ccall "THTensorMath.h &THDoubleTensor_numel"
  p_numel :: FunPtr (Ptr CTHDoubleTensor -> IO CPtrdiff)

-- | p_max : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_max"
  p_max :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> CInt -> CInt -> IO ())

-- | p_min : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_min"
  p_min :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> CInt -> CInt -> IO ())

-- | p_kthvalue : Pointer to function : values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_kthvalue"
  p_kthvalue :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> CLLong -> CInt -> CInt -> IO ())

-- | p_mode : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_mode"
  p_mode :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> CInt -> CInt -> IO ())

-- | p_median : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_median"
  p_median :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> CInt -> CInt -> IO ())

-- | p_sum : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_sum"
  p_sum :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CInt -> CInt -> IO ())

-- | p_prod : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_prod"
  p_prod :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CInt -> CInt -> IO ())

-- | p_cumsum : Pointer to function : r_ t dimension -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cumsum"
  p_cumsum :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CInt -> IO ())

-- | p_cumprod : Pointer to function : r_ t dimension -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cumprod"
  p_cumprod :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CInt -> IO ())

-- | p_sign : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_sign"
  p_sign :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_trace : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THDoubleTensor_trace"
  p_trace :: FunPtr (Ptr CTHDoubleTensor -> IO CDouble)

-- | p_cross : Pointer to function : r_ a b dimension -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cross"
  p_cross :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CInt -> IO ())

-- | p_cmax : Pointer to function : r t src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cmax"
  p_cmax :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_cmin : Pointer to function : r t src -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cmin"
  p_cmin :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_cmaxValue : Pointer to function : r t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cmaxValue"
  p_cmaxValue :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_cminValue : Pointer to function : r t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cminValue"
  p_cminValue :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_zeros : Pointer to function : r_ size -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_zeros"
  p_zeros :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHLongStorage -> IO ())

-- | p_zerosLike : Pointer to function : r_ input -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_zerosLike"
  p_zerosLike :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_ones : Pointer to function : r_ size -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_ones"
  p_ones :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHLongStorage -> IO ())

-- | p_onesLike : Pointer to function : r_ input -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_onesLike"
  p_onesLike :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_diag : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_diag"
  p_diag :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CInt -> IO ())

-- | p_eye : Pointer to function : r_ n m -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_eye"
  p_eye :: FunPtr (Ptr CTHDoubleTensor -> CLLong -> CLLong -> IO ())

-- | p_arange : Pointer to function : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_arange"
  p_arange :: FunPtr (Ptr CTHDoubleTensor -> CDouble -> CDouble -> CDouble -> IO ())

-- | p_range : Pointer to function : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_range"
  p_range :: FunPtr (Ptr CTHDoubleTensor -> CDouble -> CDouble -> CDouble -> IO ())

-- | p_randperm : Pointer to function : r_ _generator n -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_randperm"
  p_randperm :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHGenerator -> CLLong -> IO ())

-- | p_reshape : Pointer to function : r_ t size -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_reshape"
  p_reshape :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHLongStorage -> IO ())

-- | p_sort : Pointer to function : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_sort"
  p_sort :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> CInt -> CInt -> IO ())

-- | p_topk : Pointer to function : rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_topk"
  p_topk :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> CLLong -> CInt -> CInt -> CInt -> IO ())

-- | p_tril : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_tril"
  p_tril :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CLLong -> IO ())

-- | p_triu : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_triu"
  p_triu :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CLLong -> IO ())

-- | p_cat : Pointer to function : r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cat"
  p_cat :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CInt -> IO ())

-- | p_catArray : Pointer to function : result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_catArray"
  p_catArray :: FunPtr (Ptr CTHDoubleTensor -> Ptr (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())

-- | p_equal : Pointer to function : ta tb -> int
foreign import ccall "THTensorMath.h &THDoubleTensor_equal"
  p_equal :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO CInt)

-- | p_ltValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_ltValue"
  p_ltValue :: FunPtr (Ptr CTHByteTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_leValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_leValue"
  p_leValue :: FunPtr (Ptr CTHByteTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_gtValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_gtValue"
  p_gtValue :: FunPtr (Ptr CTHByteTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_geValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_geValue"
  p_geValue :: FunPtr (Ptr CTHByteTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_neValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_neValue"
  p_neValue :: FunPtr (Ptr CTHByteTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_eqValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_eqValue"
  p_eqValue :: FunPtr (Ptr CTHByteTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_ltValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_ltValueT"
  p_ltValueT :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_leValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_leValueT"
  p_leValueT :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_gtValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_gtValueT"
  p_gtValueT :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_geValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_geValueT"
  p_geValueT :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_neValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_neValueT"
  p_neValueT :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_eqValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_eqValueT"
  p_eqValueT :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_ltTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_ltTensor"
  p_ltTensor :: FunPtr (Ptr CTHByteTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_leTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_leTensor"
  p_leTensor :: FunPtr (Ptr CTHByteTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_gtTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_gtTensor"
  p_gtTensor :: FunPtr (Ptr CTHByteTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_geTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_geTensor"
  p_geTensor :: FunPtr (Ptr CTHByteTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_neTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_neTensor"
  p_neTensor :: FunPtr (Ptr CTHByteTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_eqTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_eqTensor"
  p_eqTensor :: FunPtr (Ptr CTHByteTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_ltTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_ltTensorT"
  p_ltTensorT :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_leTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_leTensorT"
  p_leTensorT :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_gtTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_gtTensorT"
  p_gtTensorT :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_geTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_geTensorT"
  p_geTensorT :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_neTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_neTensorT"
  p_neTensorT :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_eqTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_eqTensorT"
  p_eqTensorT :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_abs : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_abs"
  p_abs :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_sigmoid : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_sigmoid"
  p_sigmoid :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_log : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_log"
  p_log :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_lgamma : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_lgamma"
  p_lgamma :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_digamma : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_digamma"
  p_digamma :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_trigamma : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_trigamma"
  p_trigamma :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_polygamma : Pointer to function : r_ n t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_polygamma"
  p_polygamma :: FunPtr (Ptr CTHDoubleTensor -> CLLong -> Ptr CTHDoubleTensor -> IO ())

-- | p_log1p : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_log1p"
  p_log1p :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_exp : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_exp"
  p_exp :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_expm1 : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_expm1"
  p_expm1 :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_cos : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cos"
  p_cos :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_acos : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_acos"
  p_acos :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_cosh : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_cosh"
  p_cosh :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_sin : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_sin"
  p_sin :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_asin : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_asin"
  p_asin :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_sinh : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_sinh"
  p_sinh :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_tan : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_tan"
  p_tan :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_atan : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_atan"
  p_atan :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_atan2 : Pointer to function : r_ tx ty -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_atan2"
  p_atan2 :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_tanh : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_tanh"
  p_tanh :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_erf : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_erf"
  p_erf :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_erfinv : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_erfinv"
  p_erfinv :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_pow : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_pow"
  p_pow :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_tpow : Pointer to function : r_ value t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_tpow"
  p_tpow :: FunPtr (Ptr CTHDoubleTensor -> CDouble -> Ptr CTHDoubleTensor -> IO ())

-- | p_sqrt : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_sqrt"
  p_sqrt :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_rsqrt : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_rsqrt"
  p_rsqrt :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_ceil : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_ceil"
  p_ceil :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_floor : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_floor"
  p_floor :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_round : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_round"
  p_round :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_trunc : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_trunc"
  p_trunc :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_frac : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_frac"
  p_frac :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_lerp : Pointer to function : r_ a b weight -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_lerp"
  p_lerp :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_mean : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_mean"
  p_mean :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CInt -> CInt -> IO ())

-- | p_std : Pointer to function : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_std"
  p_std :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CInt -> CInt -> CInt -> IO ())

-- | p_var : Pointer to function : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_var"
  p_var :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CInt -> CInt -> CInt -> IO ())

-- | p_norm : Pointer to function : r_ t value dimension keepdim -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_norm"
  p_norm :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> CInt -> CInt -> IO ())

-- | p_renorm : Pointer to function : r_ t value dimension maxnorm -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_renorm"
  p_renorm :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> CInt -> CDouble -> IO ())

-- | p_dist : Pointer to function : a b value -> accreal
foreign import ccall "THTensorMath.h &THDoubleTensor_dist"
  p_dist :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CDouble -> IO CDouble)

-- | p_histc : Pointer to function : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_histc"
  p_histc :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CLLong -> CDouble -> CDouble -> IO ())

-- | p_bhistc : Pointer to function : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_bhistc"
  p_bhistc :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> CLLong -> CDouble -> CDouble -> IO ())

-- | p_meanall : Pointer to function : self -> accreal
foreign import ccall "THTensorMath.h &THDoubleTensor_meanall"
  p_meanall :: FunPtr (Ptr CTHDoubleTensor -> IO CDouble)

-- | p_varall : Pointer to function : self biased -> accreal
foreign import ccall "THTensorMath.h &THDoubleTensor_varall"
  p_varall :: FunPtr (Ptr CTHDoubleTensor -> CInt -> IO CDouble)

-- | p_stdall : Pointer to function : self biased -> accreal
foreign import ccall "THTensorMath.h &THDoubleTensor_stdall"
  p_stdall :: FunPtr (Ptr CTHDoubleTensor -> CInt -> IO CDouble)

-- | p_normall : Pointer to function : t value -> accreal
foreign import ccall "THTensorMath.h &THDoubleTensor_normall"
  p_normall :: FunPtr (Ptr CTHDoubleTensor -> CDouble -> IO CDouble)

-- | p_linspace : Pointer to function : r_ a b n -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_linspace"
  p_linspace :: FunPtr (Ptr CTHDoubleTensor -> CDouble -> CDouble -> CLLong -> IO ())

-- | p_logspace : Pointer to function : r_ a b n -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_logspace"
  p_logspace :: FunPtr (Ptr CTHDoubleTensor -> CDouble -> CDouble -> CLLong -> IO ())

-- | p_rand : Pointer to function : r_ _generator size -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_rand"
  p_rand :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ())

-- | p_randn : Pointer to function : r_ _generator size -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_randn"
  p_randn :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ())

-- | p_dirichlet_grad : Pointer to function : self x alpha total -> void
foreign import ccall "THTensorMath.h &THDoubleTensor_dirichlet_grad"
  p_dirichlet_grad :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())