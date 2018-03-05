{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Float.TensorMath
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
foreign import ccall "THTensorMath.h THFloatTensor_fill"
  c_fill :: Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_zero :  r_ -> void
foreign import ccall "THTensorMath.h THFloatTensor_zero"
  c_zero :: Ptr (CTHFloatTensor) -> IO (())

-- | c_maskedFill :  tensor mask value -> void
foreign import ccall "THTensorMath.h THFloatTensor_maskedFill"
  c_maskedFill :: Ptr (CTHFloatTensor) -> Ptr (CTHByteTensor) -> CFloat -> IO (())

-- | c_maskedCopy :  tensor mask src -> void
foreign import ccall "THTensorMath.h THFloatTensor_maskedCopy"
  c_maskedCopy :: Ptr (CTHFloatTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_maskedSelect :  tensor src mask -> void
foreign import ccall "THTensorMath.h THFloatTensor_maskedSelect"
  c_maskedSelect :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_nonzero :  subscript tensor -> void
foreign import ccall "THTensorMath.h THFloatTensor_nonzero"
  c_nonzero :: Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_indexSelect :  tensor src dim index -> void
foreign import ccall "THTensorMath.h THFloatTensor_indexSelect"
  c_indexSelect :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (())

-- | c_indexCopy :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THFloatTensor_indexCopy"
  c_indexCopy :: Ptr (CTHFloatTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_indexAdd :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THFloatTensor_indexAdd"
  c_indexAdd :: Ptr (CTHFloatTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_indexFill :  tensor dim index val -> void
foreign import ccall "THTensorMath.h THFloatTensor_indexFill"
  c_indexFill :: Ptr (CTHFloatTensor) -> CInt -> Ptr (CTHLongTensor) -> CFloat -> IO (())

-- | c_take :  tensor src index -> void
foreign import ccall "THTensorMath.h THFloatTensor_take"
  c_take :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_put :  tensor index src accumulate -> void
foreign import ccall "THTensorMath.h THFloatTensor_put"
  c_put :: Ptr (CTHFloatTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> CInt -> IO (())

-- | c_gather :  tensor src dim index -> void
foreign import ccall "THTensorMath.h THFloatTensor_gather"
  c_gather :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (())

-- | c_scatter :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THFloatTensor_scatter"
  c_scatter :: Ptr (CTHFloatTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_scatterAdd :  tensor dim index src -> void
foreign import ccall "THTensorMath.h THFloatTensor_scatterAdd"
  c_scatterAdd :: Ptr (CTHFloatTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_scatterFill :  tensor dim index val -> void
foreign import ccall "THTensorMath.h THFloatTensor_scatterFill"
  c_scatterFill :: Ptr (CTHFloatTensor) -> CInt -> Ptr (CTHLongTensor) -> CFloat -> IO (())

-- | c_dot :  t src -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_dot"
  c_dot :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (CDouble)

-- | c_minall :  t -> real
foreign import ccall "THTensorMath.h THFloatTensor_minall"
  c_minall :: Ptr (CTHFloatTensor) -> IO (CFloat)

-- | c_maxall :  t -> real
foreign import ccall "THTensorMath.h THFloatTensor_maxall"
  c_maxall :: Ptr (CTHFloatTensor) -> IO (CFloat)

-- | c_medianall :  t -> real
foreign import ccall "THTensorMath.h THFloatTensor_medianall"
  c_medianall :: Ptr (CTHFloatTensor) -> IO (CFloat)

-- | c_sumall :  t -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_sumall"
  c_sumall :: Ptr (CTHFloatTensor) -> IO (CDouble)

-- | c_prodall :  t -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_prodall"
  c_prodall :: Ptr (CTHFloatTensor) -> IO (CDouble)

-- | c_neg :  self src -> void
foreign import ccall "THTensorMath.h THFloatTensor_neg"
  c_neg :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_cinv :  self src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cinv"
  c_cinv :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_add :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_add"
  c_add :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_sub :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_sub"
  c_sub :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_add_scaled :  r_ t value alpha -> void
foreign import ccall "THTensorMath.h THFloatTensor_add_scaled"
  c_add_scaled :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> CFloat -> IO (())

-- | c_sub_scaled :  r_ t value alpha -> void
foreign import ccall "THTensorMath.h THFloatTensor_sub_scaled"
  c_sub_scaled :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> CFloat -> IO (())

-- | c_mul :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_mul"
  c_mul :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_div :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_div"
  c_div :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_lshift :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_lshift"
  c_lshift :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_rshift :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_rshift"
  c_rshift :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_fmod :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_fmod"
  c_fmod :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_remainder :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_remainder"
  c_remainder :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_clamp :  r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h THFloatTensor_clamp"
  c_clamp :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> CFloat -> IO (())

-- | c_bitand :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_bitand"
  c_bitand :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_bitor :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_bitor"
  c_bitor :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_bitxor :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_bitxor"
  c_bitxor :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_cadd :  r_ t value src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cadd"
  c_cadd :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> IO (())

-- | c_csub :  self src1 value src2 -> void
foreign import ccall "THTensorMath.h THFloatTensor_csub"
  c_csub :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> IO (())

-- | c_cmul :  r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cmul"
  c_cmul :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_cpow :  r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cpow"
  c_cpow :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_cdiv :  r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cdiv"
  c_cdiv :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_clshift :  r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_clshift"
  c_clshift :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_crshift :  r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_crshift"
  c_crshift :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_cfmod :  r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cfmod"
  c_cfmod :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_cremainder :  r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cremainder"
  c_cremainder :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_cbitand :  r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cbitand"
  c_cbitand :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_cbitor :  r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cbitor"
  c_cbitor :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_cbitxor :  r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cbitxor"
  c_cbitxor :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_addcmul :  r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THFloatTensor_addcmul"
  c_addcmul :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_addcdiv :  r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THFloatTensor_addcdiv"
  c_addcdiv :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_addmv :  r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h THFloatTensor_addmv"
  c_addmv :: Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_addmm :  r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h THFloatTensor_addmm"
  c_addmm :: Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_addr :  r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h THFloatTensor_addr"
  c_addr :: Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_addbmm :  r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THFloatTensor_addbmm"
  c_addbmm :: Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_baddbmm :  r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THFloatTensor_baddbmm"
  c_baddbmm :: Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_match :  r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h THFloatTensor_match"
  c_match :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_numel :  t -> ptrdiff_t
foreign import ccall "THTensorMath.h THFloatTensor_numel"
  c_numel :: Ptr (CTHFloatTensor) -> IO (CPtrdiff)

-- | c_max :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_max"
  c_max :: Ptr (CTHFloatTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (())

-- | c_min :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_min"
  c_min :: Ptr (CTHFloatTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (())

-- | c_kthvalue :  values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_kthvalue"
  c_kthvalue :: Ptr (CTHFloatTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> CLLong -> CInt -> CInt -> IO (())

-- | c_mode :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_mode"
  c_mode :: Ptr (CTHFloatTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (())

-- | c_median :  values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_median"
  c_median :: Ptr (CTHFloatTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (())

-- | c_sum :  r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_sum"
  c_sum :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (())

-- | c_prod :  r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_prod"
  c_prod :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (())

-- | c_cumsum :  r_ t dimension -> void
foreign import ccall "THTensorMath.h THFloatTensor_cumsum"
  c_cumsum :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> IO (())

-- | c_cumprod :  r_ t dimension -> void
foreign import ccall "THTensorMath.h THFloatTensor_cumprod"
  c_cumprod :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> IO (())

-- | c_sign :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_sign"
  c_sign :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_trace :  t -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_trace"
  c_trace :: Ptr (CTHFloatTensor) -> IO (CDouble)

-- | c_cross :  r_ a b dimension -> void
foreign import ccall "THTensorMath.h THFloatTensor_cross"
  c_cross :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> IO (())

-- | c_cmax :  r t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cmax"
  c_cmax :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_cmin :  r t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cmin"
  c_cmin :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_cmaxValue :  r t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_cmaxValue"
  c_cmaxValue :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_cminValue :  r t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_cminValue"
  c_cminValue :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_zeros :  r_ size -> void
foreign import ccall "THTensorMath.h THFloatTensor_zeros"
  c_zeros :: Ptr (CTHFloatTensor) -> Ptr (CTHLongStorage) -> IO (())

-- | c_zerosLike :  r_ input -> void
foreign import ccall "THTensorMath.h THFloatTensor_zerosLike"
  c_zerosLike :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_ones :  r_ size -> void
foreign import ccall "THTensorMath.h THFloatTensor_ones"
  c_ones :: Ptr (CTHFloatTensor) -> Ptr (CTHLongStorage) -> IO (())

-- | c_onesLike :  r_ input -> void
foreign import ccall "THTensorMath.h THFloatTensor_onesLike"
  c_onesLike :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_diag :  r_ t k -> void
foreign import ccall "THTensorMath.h THFloatTensor_diag"
  c_diag :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> IO (())

-- | c_eye :  r_ n m -> void
foreign import ccall "THTensorMath.h THFloatTensor_eye"
  c_eye :: Ptr (CTHFloatTensor) -> CLLong -> CLLong -> IO (())

-- | c_arange :  r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THFloatTensor_arange"
  c_arange :: Ptr (CTHFloatTensor) -> CDouble -> CDouble -> CDouble -> IO (())

-- | c_range :  r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THFloatTensor_range"
  c_range :: Ptr (CTHFloatTensor) -> CDouble -> CDouble -> CDouble -> IO (())

-- | c_randperm :  r_ _generator n -> void
foreign import ccall "THTensorMath.h THFloatTensor_randperm"
  c_randperm :: Ptr (CTHFloatTensor) -> Ptr (CTHGenerator) -> CLLong -> IO (())

-- | c_reshape :  r_ t size -> void
foreign import ccall "THTensorMath.h THFloatTensor_reshape"
  c_reshape :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHLongStorage) -> IO (())

-- | c_sort :  rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h THFloatTensor_sort"
  c_sort :: Ptr (CTHFloatTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (())

-- | c_topk :  rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h THFloatTensor_topk"
  c_topk :: Ptr (CTHFloatTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> CLLong -> CInt -> CInt -> CInt -> IO (())

-- | c_tril :  r_ t k -> void
foreign import ccall "THTensorMath.h THFloatTensor_tril"
  c_tril :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CLLong -> IO (())

-- | c_triu :  r_ t k -> void
foreign import ccall "THTensorMath.h THFloatTensor_triu"
  c_triu :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CLLong -> IO (())

-- | c_cat :  r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h THFloatTensor_cat"
  c_cat :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> IO (())

-- | c_catArray :  result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h THFloatTensor_catArray"
  c_catArray :: Ptr (CTHFloatTensor) -> Ptr (Ptr (CTHFloatTensor)) -> CInt -> CInt -> IO (())

-- | c_equal :  ta tb -> int
foreign import ccall "THTensorMath.h THFloatTensor_equal"
  c_equal :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (CInt)

-- | c_ltValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_ltValue"
  c_ltValue :: Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_leValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_leValue"
  c_leValue :: Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_gtValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_gtValue"
  c_gtValue :: Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_geValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_geValue"
  c_geValue :: Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_neValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_neValue"
  c_neValue :: Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_eqValue :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_eqValue"
  c_eqValue :: Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_ltValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_ltValueT"
  c_ltValueT :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_leValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_leValueT"
  c_leValueT :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_gtValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_gtValueT"
  c_gtValueT :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_geValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_geValueT"
  c_geValueT :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_neValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_neValueT"
  c_neValueT :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_eqValueT :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_eqValueT"
  c_eqValueT :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_ltTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_ltTensor"
  c_ltTensor :: Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_leTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_leTensor"
  c_leTensor :: Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_gtTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_gtTensor"
  c_gtTensor :: Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_geTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_geTensor"
  c_geTensor :: Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_neTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_neTensor"
  c_neTensor :: Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_eqTensor :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_eqTensor"
  c_eqTensor :: Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_ltTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_ltTensorT"
  c_ltTensorT :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_leTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_leTensorT"
  c_leTensorT :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_gtTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_gtTensorT"
  c_gtTensorT :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_geTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_geTensorT"
  c_geTensorT :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_neTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_neTensorT"
  c_neTensorT :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_eqTensorT :  r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_eqTensorT"
  c_eqTensorT :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_abs :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_abs"
  c_abs :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_sigmoid :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_sigmoid"
  c_sigmoid :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_log :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_log"
  c_log :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_lgamma :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_lgamma"
  c_lgamma :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_digamma :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_digamma"
  c_digamma :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_trigamma :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_trigamma"
  c_trigamma :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_polygamma :  r_ n t -> void
foreign import ccall "THTensorMath.h THFloatTensor_polygamma"
  c_polygamma :: Ptr (CTHFloatTensor) -> CLLong -> Ptr (CTHFloatTensor) -> IO (())

-- | c_log1p :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_log1p"
  c_log1p :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_exp :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_exp"
  c_exp :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_expm1 :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_expm1"
  c_expm1 :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_cos :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_cos"
  c_cos :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_acos :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_acos"
  c_acos :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_cosh :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_cosh"
  c_cosh :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_sin :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_sin"
  c_sin :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_asin :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_asin"
  c_asin :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_sinh :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_sinh"
  c_sinh :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_tan :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_tan"
  c_tan :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_atan :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_atan"
  c_atan :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_atan2 :  r_ tx ty -> void
foreign import ccall "THTensorMath.h THFloatTensor_atan2"
  c_atan2 :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_tanh :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_tanh"
  c_tanh :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_erf :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_erf"
  c_erf :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_erfinv :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_erfinv"
  c_erfinv :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_pow :  r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_pow"
  c_pow :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_tpow :  r_ value t -> void
foreign import ccall "THTensorMath.h THFloatTensor_tpow"
  c_tpow :: Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> IO (())

-- | c_sqrt :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_sqrt"
  c_sqrt :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_rsqrt :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_rsqrt"
  c_rsqrt :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_ceil :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_ceil"
  c_ceil :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_floor :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_floor"
  c_floor :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_round :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_round"
  c_round :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_trunc :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_trunc"
  c_trunc :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_frac :  r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_frac"
  c_frac :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_lerp :  r_ a b weight -> void
foreign import ccall "THTensorMath.h THFloatTensor_lerp"
  c_lerp :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (())

-- | c_mean :  r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_mean"
  c_mean :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (())

-- | c_std :  r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_std"
  c_std :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> CInt -> IO (())

-- | c_var :  r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_var"
  c_var :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> CInt -> IO (())

-- | c_norm :  r_ t value dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_norm"
  c_norm :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> CInt -> CInt -> IO (())

-- | c_renorm :  r_ t value dimension maxnorm -> void
foreign import ccall "THTensorMath.h THFloatTensor_renorm"
  c_renorm :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> CInt -> CFloat -> IO (())

-- | c_dist :  a b value -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_dist"
  c_dist :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (CDouble)

-- | c_histc :  hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THFloatTensor_histc"
  c_histc :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CLLong -> CFloat -> CFloat -> IO (())

-- | c_bhistc :  hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THFloatTensor_bhistc"
  c_bhistc :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CLLong -> CFloat -> CFloat -> IO (())

-- | c_meanall :  self -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_meanall"
  c_meanall :: Ptr (CTHFloatTensor) -> IO (CDouble)

-- | c_varall :  self biased -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_varall"
  c_varall :: Ptr (CTHFloatTensor) -> CInt -> IO (CDouble)

-- | c_stdall :  self biased -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_stdall"
  c_stdall :: Ptr (CTHFloatTensor) -> CInt -> IO (CDouble)

-- | c_normall :  t value -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_normall"
  c_normall :: Ptr (CTHFloatTensor) -> CFloat -> IO (CDouble)

-- | c_linspace :  r_ a b n -> void
foreign import ccall "THTensorMath.h THFloatTensor_linspace"
  c_linspace :: Ptr (CTHFloatTensor) -> CFloat -> CFloat -> CLLong -> IO (())

-- | c_logspace :  r_ a b n -> void
foreign import ccall "THTensorMath.h THFloatTensor_logspace"
  c_logspace :: Ptr (CTHFloatTensor) -> CFloat -> CFloat -> CLLong -> IO (())

-- | c_rand :  r_ _generator size -> void
foreign import ccall "THTensorMath.h THFloatTensor_rand"
  c_rand :: Ptr (CTHFloatTensor) -> Ptr (CTHGenerator) -> Ptr (CTHLongStorage) -> IO (())

-- | c_randn :  r_ _generator size -> void
foreign import ccall "THTensorMath.h THFloatTensor_randn"
  c_randn :: Ptr (CTHFloatTensor) -> Ptr (CTHGenerator) -> Ptr (CTHLongStorage) -> IO (())

-- | c_dirichlet_grad :  self x alpha total -> void
foreign import ccall "THTensorMath.h THFloatTensor_dirichlet_grad"
  c_dirichlet_grad :: Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | p_fill : Pointer to function : r_ value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_fill"
  p_fill :: FunPtr (Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_zero : Pointer to function : r_ -> void
foreign import ccall "THTensorMath.h &THFloatTensor_zero"
  p_zero :: FunPtr (Ptr (CTHFloatTensor) -> IO (()))

-- | p_maskedFill : Pointer to function : tensor mask value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_maskedFill"
  p_maskedFill :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHByteTensor) -> CFloat -> IO (()))

-- | p_maskedCopy : Pointer to function : tensor mask src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_maskedCopy"
  p_maskedCopy :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_maskedSelect : Pointer to function : tensor src mask -> void
foreign import ccall "THTensorMath.h &THFloatTensor_maskedSelect"
  p_maskedSelect :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_nonzero : Pointer to function : subscript tensor -> void
foreign import ccall "THTensorMath.h &THFloatTensor_nonzero"
  p_nonzero :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_indexSelect : Pointer to function : tensor src dim index -> void
foreign import ccall "THTensorMath.h &THFloatTensor_indexSelect"
  p_indexSelect :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (()))

-- | p_indexCopy : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_indexCopy"
  p_indexCopy :: FunPtr (Ptr (CTHFloatTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_indexAdd : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_indexAdd"
  p_indexAdd :: FunPtr (Ptr (CTHFloatTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_indexFill : Pointer to function : tensor dim index val -> void
foreign import ccall "THTensorMath.h &THFloatTensor_indexFill"
  p_indexFill :: FunPtr (Ptr (CTHFloatTensor) -> CInt -> Ptr (CTHLongTensor) -> CFloat -> IO (()))

-- | p_take : Pointer to function : tensor src index -> void
foreign import ccall "THTensorMath.h &THFloatTensor_take"
  p_take :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_put : Pointer to function : tensor index src accumulate -> void
foreign import ccall "THTensorMath.h &THFloatTensor_put"
  p_put :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> CInt -> IO (()))

-- | p_gather : Pointer to function : tensor src dim index -> void
foreign import ccall "THTensorMath.h &THFloatTensor_gather"
  p_gather :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> Ptr (CTHLongTensor) -> IO (()))

-- | p_scatter : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_scatter"
  p_scatter :: FunPtr (Ptr (CTHFloatTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_scatterAdd : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_scatterAdd"
  p_scatterAdd :: FunPtr (Ptr (CTHFloatTensor) -> CInt -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_scatterFill : Pointer to function : tensor dim index val -> void
foreign import ccall "THTensorMath.h &THFloatTensor_scatterFill"
  p_scatterFill :: FunPtr (Ptr (CTHFloatTensor) -> CInt -> Ptr (CTHLongTensor) -> CFloat -> IO (()))

-- | p_dot : Pointer to function : t src -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_dot"
  p_dot :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (CDouble))

-- | p_minall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THFloatTensor_minall"
  p_minall :: FunPtr (Ptr (CTHFloatTensor) -> IO (CFloat))

-- | p_maxall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THFloatTensor_maxall"
  p_maxall :: FunPtr (Ptr (CTHFloatTensor) -> IO (CFloat))

-- | p_medianall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THFloatTensor_medianall"
  p_medianall :: FunPtr (Ptr (CTHFloatTensor) -> IO (CFloat))

-- | p_sumall : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_sumall"
  p_sumall :: FunPtr (Ptr (CTHFloatTensor) -> IO (CDouble))

-- | p_prodall : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_prodall"
  p_prodall :: FunPtr (Ptr (CTHFloatTensor) -> IO (CDouble))

-- | p_neg : Pointer to function : self src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_neg"
  p_neg :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_cinv : Pointer to function : self src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cinv"
  p_cinv :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_add : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_add"
  p_add :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_sub : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sub"
  p_sub :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_add_scaled : Pointer to function : r_ t value alpha -> void
foreign import ccall "THTensorMath.h &THFloatTensor_add_scaled"
  p_add_scaled :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> CFloat -> IO (()))

-- | p_sub_scaled : Pointer to function : r_ t value alpha -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sub_scaled"
  p_sub_scaled :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> CFloat -> IO (()))

-- | p_mul : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_mul"
  p_mul :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_div : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_div"
  p_div :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_lshift : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_lshift"
  p_lshift :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_rshift : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_rshift"
  p_rshift :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_fmod : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_fmod"
  p_fmod :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_remainder : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_remainder"
  p_remainder :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_clamp : Pointer to function : r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_clamp"
  p_clamp :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> CFloat -> IO (()))

-- | p_bitand : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_bitand"
  p_bitand :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_bitor : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_bitor"
  p_bitor :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_bitxor : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_bitxor"
  p_bitxor :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_cadd : Pointer to function : r_ t value src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cadd"
  p_cadd :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_csub : Pointer to function : self src1 value src2 -> void
foreign import ccall "THTensorMath.h &THFloatTensor_csub"
  p_csub :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_cmul : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cmul"
  p_cmul :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_cpow : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cpow"
  p_cpow :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_cdiv : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cdiv"
  p_cdiv :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_clshift : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_clshift"
  p_clshift :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_crshift : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_crshift"
  p_crshift :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_cfmod : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cfmod"
  p_cfmod :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_cremainder : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cremainder"
  p_cremainder :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_cbitand : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cbitand"
  p_cbitand :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_cbitor : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cbitor"
  p_cbitor :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_cbitxor : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cbitxor"
  p_cbitxor :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_addcmul : Pointer to function : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h &THFloatTensor_addcmul"
  p_addcmul :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_addcdiv : Pointer to function : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h &THFloatTensor_addcdiv"
  p_addcdiv :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_addmv : Pointer to function : r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h &THFloatTensor_addmv"
  p_addmv :: FunPtr (Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_addmm : Pointer to function : r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h &THFloatTensor_addmm"
  p_addmm :: FunPtr (Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_addr : Pointer to function : r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h &THFloatTensor_addr"
  p_addr :: FunPtr (Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_addbmm : Pointer to function : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h &THFloatTensor_addbmm"
  p_addbmm :: FunPtr (Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_baddbmm : Pointer to function : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h &THFloatTensor_baddbmm"
  p_baddbmm :: FunPtr (Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_match : Pointer to function : r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h &THFloatTensor_match"
  p_match :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_numel : Pointer to function : t -> ptrdiff_t
foreign import ccall "THTensorMath.h &THFloatTensor_numel"
  p_numel :: FunPtr (Ptr (CTHFloatTensor) -> IO (CPtrdiff))

-- | p_max : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_max"
  p_max :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (()))

-- | p_min : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_min"
  p_min :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (()))

-- | p_kthvalue : Pointer to function : values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_kthvalue"
  p_kthvalue :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> CLLong -> CInt -> CInt -> IO (()))

-- | p_mode : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_mode"
  p_mode :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (()))

-- | p_median : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_median"
  p_median :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (()))

-- | p_sum : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sum"
  p_sum :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (()))

-- | p_prod : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_prod"
  p_prod :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (()))

-- | p_cumsum : Pointer to function : r_ t dimension -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cumsum"
  p_cumsum :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> IO (()))

-- | p_cumprod : Pointer to function : r_ t dimension -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cumprod"
  p_cumprod :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> IO (()))

-- | p_sign : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sign"
  p_sign :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_trace : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_trace"
  p_trace :: FunPtr (Ptr (CTHFloatTensor) -> IO (CDouble))

-- | p_cross : Pointer to function : r_ a b dimension -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cross"
  p_cross :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> IO (()))

-- | p_cmax : Pointer to function : r t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cmax"
  p_cmax :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_cmin : Pointer to function : r t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cmin"
  p_cmin :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_cmaxValue : Pointer to function : r t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cmaxValue"
  p_cmaxValue :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_cminValue : Pointer to function : r t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cminValue"
  p_cminValue :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_zeros : Pointer to function : r_ size -> void
foreign import ccall "THTensorMath.h &THFloatTensor_zeros"
  p_zeros :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_zerosLike : Pointer to function : r_ input -> void
foreign import ccall "THTensorMath.h &THFloatTensor_zerosLike"
  p_zerosLike :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_ones : Pointer to function : r_ size -> void
foreign import ccall "THTensorMath.h &THFloatTensor_ones"
  p_ones :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_onesLike : Pointer to function : r_ input -> void
foreign import ccall "THTensorMath.h &THFloatTensor_onesLike"
  p_onesLike :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_diag : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THFloatTensor_diag"
  p_diag :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> IO (()))

-- | p_eye : Pointer to function : r_ n m -> void
foreign import ccall "THTensorMath.h &THFloatTensor_eye"
  p_eye :: FunPtr (Ptr (CTHFloatTensor) -> CLLong -> CLLong -> IO (()))

-- | p_arange : Pointer to function : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h &THFloatTensor_arange"
  p_arange :: FunPtr (Ptr (CTHFloatTensor) -> CDouble -> CDouble -> CDouble -> IO (()))

-- | p_range : Pointer to function : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h &THFloatTensor_range"
  p_range :: FunPtr (Ptr (CTHFloatTensor) -> CDouble -> CDouble -> CDouble -> IO (()))

-- | p_randperm : Pointer to function : r_ _generator n -> void
foreign import ccall "THTensorMath.h &THFloatTensor_randperm"
  p_randperm :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHGenerator) -> CLLong -> IO (()))

-- | p_reshape : Pointer to function : r_ t size -> void
foreign import ccall "THTensorMath.h &THFloatTensor_reshape"
  p_reshape :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_sort : Pointer to function : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sort"
  p_sort :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (()))

-- | p_topk : Pointer to function : rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h &THFloatTensor_topk"
  p_topk :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> CLLong -> CInt -> CInt -> CInt -> IO (()))

-- | p_tril : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THFloatTensor_tril"
  p_tril :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CLLong -> IO (()))

-- | p_triu : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THFloatTensor_triu"
  p_triu :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CLLong -> IO (()))

-- | p_cat : Pointer to function : r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cat"
  p_cat :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> IO (()))

-- | p_catArray : Pointer to function : result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h &THFloatTensor_catArray"
  p_catArray :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (Ptr (CTHFloatTensor)) -> CInt -> CInt -> IO (()))

-- | p_equal : Pointer to function : ta tb -> int
foreign import ccall "THTensorMath.h &THFloatTensor_equal"
  p_equal :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (CInt))

-- | p_ltValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_ltValue"
  p_ltValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_leValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_leValue"
  p_leValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_gtValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_gtValue"
  p_gtValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_geValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_geValue"
  p_geValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_neValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_neValue"
  p_neValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_eqValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_eqValue"
  p_eqValue :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_ltValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_ltValueT"
  p_ltValueT :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_leValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_leValueT"
  p_leValueT :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_gtValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_gtValueT"
  p_gtValueT :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_geValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_geValueT"
  p_geValueT :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_neValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_neValueT"
  p_neValueT :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_eqValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_eqValueT"
  p_eqValueT :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_ltTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_ltTensor"
  p_ltTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_leTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_leTensor"
  p_leTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_gtTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_gtTensor"
  p_gtTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_geTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_geTensor"
  p_geTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_neTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_neTensor"
  p_neTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_eqTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_eqTensor"
  p_eqTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_ltTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_ltTensorT"
  p_ltTensorT :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_leTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_leTensorT"
  p_leTensorT :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_gtTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_gtTensorT"
  p_gtTensorT :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_geTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_geTensorT"
  p_geTensorT :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_neTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_neTensorT"
  p_neTensorT :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_eqTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_eqTensorT"
  p_eqTensorT :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_abs : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_abs"
  p_abs :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_sigmoid : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sigmoid"
  p_sigmoid :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_log : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_log"
  p_log :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_lgamma : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_lgamma"
  p_lgamma :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_digamma : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_digamma"
  p_digamma :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_trigamma : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_trigamma"
  p_trigamma :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_polygamma : Pointer to function : r_ n t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_polygamma"
  p_polygamma :: FunPtr (Ptr (CTHFloatTensor) -> CLLong -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_log1p : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_log1p"
  p_log1p :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_exp : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_exp"
  p_exp :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_expm1 : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_expm1"
  p_expm1 :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_cos : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cos"
  p_cos :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_acos : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_acos"
  p_acos :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_cosh : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cosh"
  p_cosh :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_sin : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sin"
  p_sin :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_asin : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_asin"
  p_asin :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_sinh : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sinh"
  p_sinh :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_tan : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_tan"
  p_tan :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_atan : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_atan"
  p_atan :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_atan2 : Pointer to function : r_ tx ty -> void
foreign import ccall "THTensorMath.h &THFloatTensor_atan2"
  p_atan2 :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_tanh : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_tanh"
  p_tanh :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_erf : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_erf"
  p_erf :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_erfinv : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_erfinv"
  p_erfinv :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_pow : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_pow"
  p_pow :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_tpow : Pointer to function : r_ value t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_tpow"
  p_tpow :: FunPtr (Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_sqrt : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sqrt"
  p_sqrt :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_rsqrt : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_rsqrt"
  p_rsqrt :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_ceil : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_ceil"
  p_ceil :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_floor : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_floor"
  p_floor :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_round : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_round"
  p_round :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_trunc : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_trunc"
  p_trunc :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_frac : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_frac"
  p_frac :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_lerp : Pointer to function : r_ a b weight -> void
foreign import ccall "THTensorMath.h &THFloatTensor_lerp"
  p_lerp :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (()))

-- | p_mean : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_mean"
  p_mean :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (()))

-- | p_std : Pointer to function : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_std"
  p_std :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> CInt -> IO (()))

-- | p_var : Pointer to function : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_var"
  p_var :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> CInt -> IO (()))

-- | p_norm : Pointer to function : r_ t value dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_norm"
  p_norm :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> CInt -> CInt -> IO (()))

-- | p_renorm : Pointer to function : r_ t value dimension maxnorm -> void
foreign import ccall "THTensorMath.h &THFloatTensor_renorm"
  p_renorm :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> CInt -> CFloat -> IO (()))

-- | p_dist : Pointer to function : a b value -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_dist"
  p_dist :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (CDouble))

-- | p_histc : Pointer to function : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h &THFloatTensor_histc"
  p_histc :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CLLong -> CFloat -> CFloat -> IO (()))

-- | p_bhistc : Pointer to function : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h &THFloatTensor_bhistc"
  p_bhistc :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CLLong -> CFloat -> CFloat -> IO (()))

-- | p_meanall : Pointer to function : self -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_meanall"
  p_meanall :: FunPtr (Ptr (CTHFloatTensor) -> IO (CDouble))

-- | p_varall : Pointer to function : self biased -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_varall"
  p_varall :: FunPtr (Ptr (CTHFloatTensor) -> CInt -> IO (CDouble))

-- | p_stdall : Pointer to function : self biased -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_stdall"
  p_stdall :: FunPtr (Ptr (CTHFloatTensor) -> CInt -> IO (CDouble))

-- | p_normall : Pointer to function : t value -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_normall"
  p_normall :: FunPtr (Ptr (CTHFloatTensor) -> CFloat -> IO (CDouble))

-- | p_linspace : Pointer to function : r_ a b n -> void
foreign import ccall "THTensorMath.h &THFloatTensor_linspace"
  p_linspace :: FunPtr (Ptr (CTHFloatTensor) -> CFloat -> CFloat -> CLLong -> IO (()))

-- | p_logspace : Pointer to function : r_ a b n -> void
foreign import ccall "THTensorMath.h &THFloatTensor_logspace"
  p_logspace :: FunPtr (Ptr (CTHFloatTensor) -> CFloat -> CFloat -> CLLong -> IO (()))

-- | p_rand : Pointer to function : r_ _generator size -> void
foreign import ccall "THTensorMath.h &THFloatTensor_rand"
  p_rand :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHGenerator) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_randn : Pointer to function : r_ _generator size -> void
foreign import ccall "THTensorMath.h &THFloatTensor_randn"
  p_randn :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHGenerator) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_dirichlet_grad : Pointer to function : self x alpha total -> void
foreign import ccall "THTensorMath.h &THFloatTensor_dirichlet_grad"
  p_dirichlet_grad :: FunPtr (Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))