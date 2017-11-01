{-# LANGUAGE ForeignFunctionInterface #-}

module THDoubleTensorMath (
    c_THDoubleTensor_fill,
    c_THDoubleTensor_zero,
    c_THDoubleTensor_maskedFill,
    c_THDoubleTensor_maskedCopy,
    c_THDoubleTensor_maskedSelect,
    c_THDoubleTensor_nonzero,
    c_THDoubleTensor_indexSelect,
    c_THDoubleTensor_indexCopy,
    c_THDoubleTensor_indexAdd,
    c_THDoubleTensor_indexFill,
    c_THDoubleTensor_gather,
    c_THDoubleTensor_scatter,
    c_THDoubleTensor_scatterAdd,
    c_THDoubleTensor_scatterFill,
    c_THDoubleTensor_dot,
    c_THDoubleTensor_minall,
    c_THDoubleTensor_maxall,
    c_THDoubleTensor_medianall,
    c_THDoubleTensor_sumall,
    c_THDoubleTensor_prodall,
    c_THDoubleTensor_neg,
    c_THDoubleTensor_cinv,
    c_THDoubleTensor_add,
    c_THDoubleTensor_sub,
    c_THDoubleTensor_mul,
    c_THDoubleTensor_div,
    c_THDoubleTensor_lshift,
    c_THDoubleTensor_rshift,
    c_THDoubleTensor_fmod,
    c_THDoubleTensor_remainder,
    c_THDoubleTensor_clamp,
    c_THDoubleTensor_bitand,
    c_THDoubleTensor_bitor,
    c_THDoubleTensor_bitxor,
    c_THDoubleTensor_cadd,
    c_THDoubleTensor_csub,
    c_THDoubleTensor_cmul,
    c_THDoubleTensor_cpow,
    c_THDoubleTensor_cdiv,
    c_THDoubleTensor_clshift,
    c_THDoubleTensor_crshift,
    c_THDoubleTensor_cfmod,
    c_THDoubleTensor_cremainder,
    c_THDoubleTensor_cbitand,
    c_THDoubleTensor_cbitor,
    c_THDoubleTensor_cbitxor,
    c_THDoubleTensor_addcmul,
    c_THDoubleTensor_addcdiv,
    c_THDoubleTensor_addmv,
    c_THDoubleTensor_addmm,
    c_THDoubleTensor_addr,
    c_THDoubleTensor_addbmm,
    c_THDoubleTensor_baddbmm,
    c_THDoubleTensor_match,
    c_THDoubleTensor_numel,
    c_THDoubleTensor_max,
    c_THDoubleTensor_min,
    c_THDoubleTensor_kthvalue,
    c_THDoubleTensor_mode,
    c_THDoubleTensor_median,
    c_THDoubleTensor_sum,
    c_THDoubleTensor_prod,
    c_THDoubleTensor_cumsum,
    c_THDoubleTensor_cumprod,
    c_THDoubleTensor_sign,
    c_THDoubleTensor_trace,
    c_THDoubleTensor_cross,
    c_THDoubleTensor_cmax,
    c_THDoubleTensor_cmin,
    c_THDoubleTensor_cmaxValue,
    c_THDoubleTensor_cminValue,
    c_THDoubleTensor_zeros,
    c_THDoubleTensor_zerosLike,
    c_THDoubleTensor_ones,
    c_THDoubleTensor_onesLike,
    c_THDoubleTensor_diag,
    c_THDoubleTensor_eye,
    c_THDoubleTensor_arange,
    c_THDoubleTensor_range,
    c_THDoubleTensor_randperm,
    c_THDoubleTensor_reshape,
    c_THDoubleTensor_sort,
    c_THDoubleTensor_topk,
    c_THDoubleTensor_tril,
    c_THDoubleTensor_triu,
    c_THDoubleTensor_cat,
    c_THDoubleTensor_catArray,
    c_THDoubleTensor_equal,
    c_THDoubleTensor_ltValue,
    c_THDoubleTensor_leValue,
    c_THDoubleTensor_gtValue,
    c_THDoubleTensor_geValue,
    c_THDoubleTensor_neValue,
    c_THDoubleTensor_eqValue,
    c_THDoubleTensor_ltValueT,
    c_THDoubleTensor_leValueT,
    c_THDoubleTensor_gtValueT,
    c_THDoubleTensor_geValueT,
    c_THDoubleTensor_neValueT,
    c_THDoubleTensor_eqValueT,
    c_THDoubleTensor_ltTensor,
    c_THDoubleTensor_leTensor,
    c_THDoubleTensor_gtTensor,
    c_THDoubleTensor_geTensor,
    c_THDoubleTensor_neTensor,
    c_THDoubleTensor_eqTensor,
    c_THDoubleTensor_ltTensorT,
    c_THDoubleTensor_leTensorT,
    c_THDoubleTensor_gtTensorT,
    c_THDoubleTensor_geTensorT,
    c_THDoubleTensor_neTensorT,
    c_THDoubleTensor_eqTensorT,
    c_THDoubleTensor_abs,
    c_THDoubleTensor_sigmoid,
    c_THDoubleTensor_log,
    c_THDoubleTensor_lgamma,
    c_THDoubleTensor_log1p,
    c_THDoubleTensor_exp,
    c_THDoubleTensor_cos,
    c_THDoubleTensor_acos,
    c_THDoubleTensor_cosh,
    c_THDoubleTensor_sin,
    c_THDoubleTensor_asin,
    c_THDoubleTensor_sinh,
    c_THDoubleTensor_tan,
    c_THDoubleTensor_atan,
    c_THDoubleTensor_atan2,
    c_THDoubleTensor_tanh,
    c_THDoubleTensor_pow,
    c_THDoubleTensor_tpow,
    c_THDoubleTensor_sqrt,
    c_THDoubleTensor_rsqrt,
    c_THDoubleTensor_ceil,
    c_THDoubleTensor_floor,
    c_THDoubleTensor_round,
    c_THDoubleTensor_trunc,
    c_THDoubleTensor_frac,
    c_THDoubleTensor_lerp,
    c_THDoubleTensor_mean,
    c_THDoubleTensor_std,
    c_THDoubleTensor_var,
    c_THDoubleTensor_norm,
    c_THDoubleTensor_renorm,
    c_THDoubleTensor_dist,
    c_THDoubleTensor_histc,
    c_THDoubleTensor_bhistc,
    c_THDoubleTensor_meanall,
    c_THDoubleTensor_varall,
    c_THDoubleTensor_stdall,
    c_THDoubleTensor_normall,
    c_THDoubleTensor_linspace,
    c_THDoubleTensor_logspace,
    c_THDoubleTensor_rand,
    c_THDoubleTensor_randn,
    p_THDoubleTensor_fill,
    p_THDoubleTensor_zero,
    p_THDoubleTensor_maskedFill,
    p_THDoubleTensor_maskedCopy,
    p_THDoubleTensor_maskedSelect,
    p_THDoubleTensor_nonzero,
    p_THDoubleTensor_indexSelect,
    p_THDoubleTensor_indexCopy,
    p_THDoubleTensor_indexAdd,
    p_THDoubleTensor_indexFill,
    p_THDoubleTensor_gather,
    p_THDoubleTensor_scatter,
    p_THDoubleTensor_scatterAdd,
    p_THDoubleTensor_scatterFill,
    p_THDoubleTensor_dot,
    p_THDoubleTensor_minall,
    p_THDoubleTensor_maxall,
    p_THDoubleTensor_medianall,
    p_THDoubleTensor_sumall,
    p_THDoubleTensor_prodall,
    p_THDoubleTensor_neg,
    p_THDoubleTensor_cinv,
    p_THDoubleTensor_add,
    p_THDoubleTensor_sub,
    p_THDoubleTensor_mul,
    p_THDoubleTensor_div,
    p_THDoubleTensor_lshift,
    p_THDoubleTensor_rshift,
    p_THDoubleTensor_fmod,
    p_THDoubleTensor_remainder,
    p_THDoubleTensor_clamp,
    p_THDoubleTensor_bitand,
    p_THDoubleTensor_bitor,
    p_THDoubleTensor_bitxor,
    p_THDoubleTensor_cadd,
    p_THDoubleTensor_csub,
    p_THDoubleTensor_cmul,
    p_THDoubleTensor_cpow,
    p_THDoubleTensor_cdiv,
    p_THDoubleTensor_clshift,
    p_THDoubleTensor_crshift,
    p_THDoubleTensor_cfmod,
    p_THDoubleTensor_cremainder,
    p_THDoubleTensor_cbitand,
    p_THDoubleTensor_cbitor,
    p_THDoubleTensor_cbitxor,
    p_THDoubleTensor_addcmul,
    p_THDoubleTensor_addcdiv,
    p_THDoubleTensor_addmv,
    p_THDoubleTensor_addmm,
    p_THDoubleTensor_addr,
    p_THDoubleTensor_addbmm,
    p_THDoubleTensor_baddbmm,
    p_THDoubleTensor_match,
    p_THDoubleTensor_numel,
    p_THDoubleTensor_max,
    p_THDoubleTensor_min,
    p_THDoubleTensor_kthvalue,
    p_THDoubleTensor_mode,
    p_THDoubleTensor_median,
    p_THDoubleTensor_sum,
    p_THDoubleTensor_prod,
    p_THDoubleTensor_cumsum,
    p_THDoubleTensor_cumprod,
    p_THDoubleTensor_sign,
    p_THDoubleTensor_trace,
    p_THDoubleTensor_cross,
    p_THDoubleTensor_cmax,
    p_THDoubleTensor_cmin,
    p_THDoubleTensor_cmaxValue,
    p_THDoubleTensor_cminValue,
    p_THDoubleTensor_zeros,
    p_THDoubleTensor_zerosLike,
    p_THDoubleTensor_ones,
    p_THDoubleTensor_onesLike,
    p_THDoubleTensor_diag,
    p_THDoubleTensor_eye,
    p_THDoubleTensor_arange,
    p_THDoubleTensor_range,
    p_THDoubleTensor_randperm,
    p_THDoubleTensor_reshape,
    p_THDoubleTensor_sort,
    p_THDoubleTensor_topk,
    p_THDoubleTensor_tril,
    p_THDoubleTensor_triu,
    p_THDoubleTensor_cat,
    p_THDoubleTensor_catArray,
    p_THDoubleTensor_equal,
    p_THDoubleTensor_ltValue,
    p_THDoubleTensor_leValue,
    p_THDoubleTensor_gtValue,
    p_THDoubleTensor_geValue,
    p_THDoubleTensor_neValue,
    p_THDoubleTensor_eqValue,
    p_THDoubleTensor_ltValueT,
    p_THDoubleTensor_leValueT,
    p_THDoubleTensor_gtValueT,
    p_THDoubleTensor_geValueT,
    p_THDoubleTensor_neValueT,
    p_THDoubleTensor_eqValueT,
    p_THDoubleTensor_ltTensor,
    p_THDoubleTensor_leTensor,
    p_THDoubleTensor_gtTensor,
    p_THDoubleTensor_geTensor,
    p_THDoubleTensor_neTensor,
    p_THDoubleTensor_eqTensor,
    p_THDoubleTensor_ltTensorT,
    p_THDoubleTensor_leTensorT,
    p_THDoubleTensor_gtTensorT,
    p_THDoubleTensor_geTensorT,
    p_THDoubleTensor_neTensorT,
    p_THDoubleTensor_eqTensorT,
    p_THDoubleTensor_abs,
    p_THDoubleTensor_sigmoid,
    p_THDoubleTensor_log,
    p_THDoubleTensor_lgamma,
    p_THDoubleTensor_log1p,
    p_THDoubleTensor_exp,
    p_THDoubleTensor_cos,
    p_THDoubleTensor_acos,
    p_THDoubleTensor_cosh,
    p_THDoubleTensor_sin,
    p_THDoubleTensor_asin,
    p_THDoubleTensor_sinh,
    p_THDoubleTensor_tan,
    p_THDoubleTensor_atan,
    p_THDoubleTensor_atan2,
    p_THDoubleTensor_tanh,
    p_THDoubleTensor_pow,
    p_THDoubleTensor_tpow,
    p_THDoubleTensor_sqrt,
    p_THDoubleTensor_rsqrt,
    p_THDoubleTensor_ceil,
    p_THDoubleTensor_floor,
    p_THDoubleTensor_round,
    p_THDoubleTensor_trunc,
    p_THDoubleTensor_frac,
    p_THDoubleTensor_lerp,
    p_THDoubleTensor_mean,
    p_THDoubleTensor_std,
    p_THDoubleTensor_var,
    p_THDoubleTensor_norm,
    p_THDoubleTensor_renorm,
    p_THDoubleTensor_dist,
    p_THDoubleTensor_histc,
    p_THDoubleTensor_bhistc,
    p_THDoubleTensor_meanall,
    p_THDoubleTensor_varall,
    p_THDoubleTensor_stdall,
    p_THDoubleTensor_normall,
    p_THDoubleTensor_linspace,
    p_THDoubleTensor_logspace,
    p_THDoubleTensor_rand,
    p_THDoubleTensor_randn) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THDoubleTensor_fill : r_ value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_fill"
  c_THDoubleTensor_fill :: (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_zero : r_ -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_zero"
  c_THDoubleTensor_zero :: (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_maskedFill : tensor mask value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_maskedFill"
  c_THDoubleTensor_maskedFill :: (Ptr CTHDoubleTensor) -> Ptr CTHByteTensor -> CDouble -> IO ()

-- |c_THDoubleTensor_maskedCopy : tensor mask src -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_maskedCopy"
  c_THDoubleTensor_maskedCopy :: (Ptr CTHDoubleTensor) -> Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_maskedSelect : tensor src mask -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_maskedSelect"
  c_THDoubleTensor_maskedSelect :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THDoubleTensor_nonzero : subscript tensor -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_nonzero"
  c_THDoubleTensor_nonzero :: Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_indexSelect : tensor src dim index -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_indexSelect"
  c_THDoubleTensor_indexSelect :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THDoubleTensor_indexCopy : tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_indexCopy"
  c_THDoubleTensor_indexCopy :: (Ptr CTHDoubleTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_indexAdd : tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_indexAdd"
  c_THDoubleTensor_indexAdd :: (Ptr CTHDoubleTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_indexFill : tensor dim index val -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_indexFill"
  c_THDoubleTensor_indexFill :: (Ptr CTHDoubleTensor) -> CInt -> Ptr CTHLongTensor -> CDouble -> IO ()

-- |c_THDoubleTensor_gather : tensor src dim index -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_gather"
  c_THDoubleTensor_gather :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THDoubleTensor_scatter : tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_scatter"
  c_THDoubleTensor_scatter :: (Ptr CTHDoubleTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_scatterAdd : tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_scatterAdd"
  c_THDoubleTensor_scatterAdd :: (Ptr CTHDoubleTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_scatterFill : tensor dim index val -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_scatterFill"
  c_THDoubleTensor_scatterFill :: (Ptr CTHDoubleTensor) -> CInt -> Ptr CTHLongTensor -> CDouble -> IO ()

-- |c_THDoubleTensor_dot : t src -> accreal
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_dot"
  c_THDoubleTensor_dot :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble

-- |c_THDoubleTensor_minall : t -> real
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_minall"
  c_THDoubleTensor_minall :: (Ptr CTHDoubleTensor) -> CDouble

-- |c_THDoubleTensor_maxall : t -> real
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_maxall"
  c_THDoubleTensor_maxall :: (Ptr CTHDoubleTensor) -> CDouble

-- |c_THDoubleTensor_medianall : t -> real
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_medianall"
  c_THDoubleTensor_medianall :: (Ptr CTHDoubleTensor) -> CDouble

-- |c_THDoubleTensor_sumall : t -> accreal
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_sumall"
  c_THDoubleTensor_sumall :: (Ptr CTHDoubleTensor) -> CDouble

-- |c_THDoubleTensor_prodall : t -> accreal
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_prodall"
  c_THDoubleTensor_prodall :: (Ptr CTHDoubleTensor) -> CDouble

-- |c_THDoubleTensor_neg : self src -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_neg"
  c_THDoubleTensor_neg :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_cinv : self src -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_cinv"
  c_THDoubleTensor_cinv :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_add : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_add"
  c_THDoubleTensor_add :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_sub : self src value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_sub"
  c_THDoubleTensor_sub :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_mul : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_mul"
  c_THDoubleTensor_mul :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_div : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_div"
  c_THDoubleTensor_div :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_lshift : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_lshift"
  c_THDoubleTensor_lshift :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_rshift : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_rshift"
  c_THDoubleTensor_rshift :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_fmod : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_fmod"
  c_THDoubleTensor_fmod :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_remainder : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_remainder"
  c_THDoubleTensor_remainder :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_clamp : r_ t min_value max_value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_clamp"
  c_THDoubleTensor_clamp :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ()

-- |c_THDoubleTensor_bitand : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_bitand"
  c_THDoubleTensor_bitand :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_bitor : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_bitor"
  c_THDoubleTensor_bitor :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_bitxor : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_bitxor"
  c_THDoubleTensor_bitxor :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_cadd : r_ t value src -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_cadd"
  c_THDoubleTensor_cadd :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_csub : self src1 value src2 -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_csub"
  c_THDoubleTensor_csub :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_cmul : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_cmul"
  c_THDoubleTensor_cmul :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_cpow : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_cpow"
  c_THDoubleTensor_cpow :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_cdiv : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_cdiv"
  c_THDoubleTensor_cdiv :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_clshift : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_clshift"
  c_THDoubleTensor_clshift :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_crshift : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_crshift"
  c_THDoubleTensor_crshift :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_cfmod : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_cfmod"
  c_THDoubleTensor_cfmod :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_cremainder : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_cremainder"
  c_THDoubleTensor_cremainder :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_cbitand : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_cbitand"
  c_THDoubleTensor_cbitand :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_cbitor : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_cbitor"
  c_THDoubleTensor_cbitor :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_cbitxor : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_cbitxor"
  c_THDoubleTensor_cbitxor :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_addcmul : r_ t value src1 src2 -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_addcmul"
  c_THDoubleTensor_addcmul :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_addcdiv : r_ t value src1 src2 -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_addcdiv"
  c_THDoubleTensor_addcdiv :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_addmv : r_ beta t alpha mat vec -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_addmv"
  c_THDoubleTensor_addmv :: (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_addmm : r_ beta t alpha mat1 mat2 -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_addmm"
  c_THDoubleTensor_addmm :: (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_addr : r_ beta t alpha vec1 vec2 -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_addr"
  c_THDoubleTensor_addr :: (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_addbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_addbmm"
  c_THDoubleTensor_addbmm :: (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_baddbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_baddbmm"
  c_THDoubleTensor_baddbmm :: (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_match : r_ m1 m2 gain -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_match"
  c_THDoubleTensor_match :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_numel : t -> ptrdiff_t
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_numel"
  c_THDoubleTensor_numel :: (Ptr CTHDoubleTensor) -> CPtrdiff

-- |c_THDoubleTensor_max : values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_max"
  c_THDoubleTensor_max :: (Ptr CTHDoubleTensor) -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleTensor_min : values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_min"
  c_THDoubleTensor_min :: (Ptr CTHDoubleTensor) -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleTensor_kthvalue : values_ indices_ t k dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_kthvalue"
  c_THDoubleTensor_kthvalue :: (Ptr CTHDoubleTensor) -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> CLong -> CInt -> CInt -> IO ()

-- |c_THDoubleTensor_mode : values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_mode"
  c_THDoubleTensor_mode :: (Ptr CTHDoubleTensor) -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleTensor_median : values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_median"
  c_THDoubleTensor_median :: (Ptr CTHDoubleTensor) -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleTensor_sum : r_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_sum"
  c_THDoubleTensor_sum :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleTensor_prod : r_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_prod"
  c_THDoubleTensor_prod :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleTensor_cumsum : r_ t dimension -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_cumsum"
  c_THDoubleTensor_cumsum :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleTensor_cumprod : r_ t dimension -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_cumprod"
  c_THDoubleTensor_cumprod :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleTensor_sign : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_sign"
  c_THDoubleTensor_sign :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_trace : t -> accreal
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_trace"
  c_THDoubleTensor_trace :: (Ptr CTHDoubleTensor) -> CDouble

-- |c_THDoubleTensor_cross : r_ a b dimension -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_cross"
  c_THDoubleTensor_cross :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleTensor_cmax : r t src -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_cmax"
  c_THDoubleTensor_cmax :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_cmin : r t src -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_cmin"
  c_THDoubleTensor_cmin :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_cmaxValue : r t value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_cmaxValue"
  c_THDoubleTensor_cmaxValue :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_cminValue : r t value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_cminValue"
  c_THDoubleTensor_cminValue :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_zeros : r_ size -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_zeros"
  c_THDoubleTensor_zeros :: (Ptr CTHDoubleTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THDoubleTensor_zerosLike : r_ input -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_zerosLike"
  c_THDoubleTensor_zerosLike :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_ones : r_ size -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_ones"
  c_THDoubleTensor_ones :: (Ptr CTHDoubleTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THDoubleTensor_onesLike : r_ input -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_onesLike"
  c_THDoubleTensor_onesLike :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_diag : r_ t k -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_diag"
  c_THDoubleTensor_diag :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleTensor_eye : r_ n m -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_eye"
  c_THDoubleTensor_eye :: (Ptr CTHDoubleTensor) -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_arange : r_ xmin xmax step -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_arange"
  c_THDoubleTensor_arange :: (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CDouble -> IO ()

-- |c_THDoubleTensor_range : r_ xmin xmax step -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_range"
  c_THDoubleTensor_range :: (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CDouble -> IO ()

-- |c_THDoubleTensor_randperm : r_ _generator n -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_randperm"
  c_THDoubleTensor_randperm :: (Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> CLong -> IO ()

-- |c_THDoubleTensor_reshape : r_ t size -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_reshape"
  c_THDoubleTensor_reshape :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THDoubleTensor_sort : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_sort"
  c_THDoubleTensor_sort :: (Ptr CTHDoubleTensor) -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleTensor_topk : rt_ ri_ t k dim dir sorted -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_topk"
  c_THDoubleTensor_topk :: (Ptr CTHDoubleTensor) -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> CLong -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleTensor_tril : r_ t k -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_tril"
  c_THDoubleTensor_tril :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> IO ()

-- |c_THDoubleTensor_triu : r_ t k -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_triu"
  c_THDoubleTensor_triu :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> IO ()

-- |c_THDoubleTensor_cat : r_ ta tb dimension -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_cat"
  c_THDoubleTensor_cat :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleTensor_catArray : result inputs numInputs dimension -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_catArray"
  c_THDoubleTensor_catArray :: (Ptr CTHDoubleTensor) -> Ptr (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleTensor_equal : ta tb -> int
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_equal"
  c_THDoubleTensor_equal :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt

-- |c_THDoubleTensor_ltValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_ltValue"
  c_THDoubleTensor_ltValue :: Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_leValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_leValue"
  c_THDoubleTensor_leValue :: Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_gtValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_gtValue"
  c_THDoubleTensor_gtValue :: Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_geValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_geValue"
  c_THDoubleTensor_geValue :: Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_neValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_neValue"
  c_THDoubleTensor_neValue :: Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_eqValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_eqValue"
  c_THDoubleTensor_eqValue :: Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_ltValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_ltValueT"
  c_THDoubleTensor_ltValueT :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_leValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_leValueT"
  c_THDoubleTensor_leValueT :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_gtValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_gtValueT"
  c_THDoubleTensor_gtValueT :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_geValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_geValueT"
  c_THDoubleTensor_geValueT :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_neValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_neValueT"
  c_THDoubleTensor_neValueT :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_eqValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_eqValueT"
  c_THDoubleTensor_eqValueT :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_ltTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_ltTensor"
  c_THDoubleTensor_ltTensor :: Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_leTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_leTensor"
  c_THDoubleTensor_leTensor :: Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_gtTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_gtTensor"
  c_THDoubleTensor_gtTensor :: Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_geTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_geTensor"
  c_THDoubleTensor_geTensor :: Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_neTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_neTensor"
  c_THDoubleTensor_neTensor :: Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_eqTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_eqTensor"
  c_THDoubleTensor_eqTensor :: Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_ltTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_ltTensorT"
  c_THDoubleTensor_ltTensorT :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_leTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_leTensorT"
  c_THDoubleTensor_leTensorT :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_gtTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_gtTensorT"
  c_THDoubleTensor_gtTensorT :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_geTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_geTensorT"
  c_THDoubleTensor_geTensorT :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_neTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_neTensorT"
  c_THDoubleTensor_neTensorT :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_eqTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_eqTensorT"
  c_THDoubleTensor_eqTensorT :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_abs : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_abs"
  c_THDoubleTensor_abs :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_sigmoid : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_sigmoid"
  c_THDoubleTensor_sigmoid :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_log : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_log"
  c_THDoubleTensor_log :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_lgamma : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_lgamma"
  c_THDoubleTensor_lgamma :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_log1p : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_log1p"
  c_THDoubleTensor_log1p :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_exp : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_exp"
  c_THDoubleTensor_exp :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_cos : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_cos"
  c_THDoubleTensor_cos :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_acos : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_acos"
  c_THDoubleTensor_acos :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_cosh : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_cosh"
  c_THDoubleTensor_cosh :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_sin : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_sin"
  c_THDoubleTensor_sin :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_asin : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_asin"
  c_THDoubleTensor_asin :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_sinh : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_sinh"
  c_THDoubleTensor_sinh :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_tan : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_tan"
  c_THDoubleTensor_tan :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_atan : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_atan"
  c_THDoubleTensor_atan :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_atan2 : r_ tx ty -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_atan2"
  c_THDoubleTensor_atan2 :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_tanh : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_tanh"
  c_THDoubleTensor_tanh :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_pow : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_pow"
  c_THDoubleTensor_pow :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_tpow : r_ value t -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_tpow"
  c_THDoubleTensor_tpow :: (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_sqrt : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_sqrt"
  c_THDoubleTensor_sqrt :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_rsqrt : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_rsqrt"
  c_THDoubleTensor_rsqrt :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_ceil : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_ceil"
  c_THDoubleTensor_ceil :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_floor : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_floor"
  c_THDoubleTensor_floor :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_round : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_round"
  c_THDoubleTensor_round :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_trunc : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_trunc"
  c_THDoubleTensor_trunc :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_frac : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_frac"
  c_THDoubleTensor_frac :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_lerp : r_ a b weight -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_lerp"
  c_THDoubleTensor_lerp :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_mean : r_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_mean"
  c_THDoubleTensor_mean :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleTensor_std : r_ t dimension biased keepdim -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_std"
  c_THDoubleTensor_std :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleTensor_var : r_ t dimension biased keepdim -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_var"
  c_THDoubleTensor_var :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleTensor_norm : r_ t value dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_norm"
  c_THDoubleTensor_norm :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CInt -> CInt -> IO ()

-- |c_THDoubleTensor_renorm : r_ t value dimension maxnorm -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_renorm"
  c_THDoubleTensor_renorm :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CInt -> CDouble -> IO ()

-- |c_THDoubleTensor_dist : a b value -> accreal
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_dist"
  c_THDoubleTensor_dist :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble

-- |c_THDoubleTensor_histc : hist tensor nbins minvalue maxvalue -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_histc"
  c_THDoubleTensor_histc :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CDouble -> CDouble -> IO ()

-- |c_THDoubleTensor_bhistc : hist tensor nbins minvalue maxvalue -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_bhistc"
  c_THDoubleTensor_bhistc :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CDouble -> CDouble -> IO ()

-- |c_THDoubleTensor_meanall : self -> accreal
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_meanall"
  c_THDoubleTensor_meanall :: (Ptr CTHDoubleTensor) -> CDouble

-- |c_THDoubleTensor_varall : self biased -> accreal
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_varall"
  c_THDoubleTensor_varall :: (Ptr CTHDoubleTensor) -> CInt -> CDouble

-- |c_THDoubleTensor_stdall : self biased -> accreal
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_stdall"
  c_THDoubleTensor_stdall :: (Ptr CTHDoubleTensor) -> CInt -> CDouble

-- |c_THDoubleTensor_normall : t value -> accreal
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_normall"
  c_THDoubleTensor_normall :: (Ptr CTHDoubleTensor) -> CDouble -> CDouble

-- |c_THDoubleTensor_linspace : r_ a b n -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_linspace"
  c_THDoubleTensor_linspace :: (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CLong -> IO ()

-- |c_THDoubleTensor_logspace : r_ a b n -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_logspace"
  c_THDoubleTensor_logspace :: (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CLong -> IO ()

-- |c_THDoubleTensor_rand : r_ _generator size -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_rand"
  c_THDoubleTensor_rand :: (Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()

-- |c_THDoubleTensor_randn : r_ _generator size -> void
foreign import ccall unsafe "THTensorMath.h THDoubleTensor_randn"
  c_THDoubleTensor_randn :: (Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()

-- |p_THDoubleTensor_fill : Pointer to function r_ value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_fill"
  p_THDoubleTensor_fill :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_zero : Pointer to function r_ -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_zero"
  p_THDoubleTensor_zero :: FunPtr ((Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_maskedFill : Pointer to function tensor mask value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_maskedFill"
  p_THDoubleTensor_maskedFill :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHByteTensor -> CDouble -> IO ())

-- |p_THDoubleTensor_maskedCopy : Pointer to function tensor mask src -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_maskedCopy"
  p_THDoubleTensor_maskedCopy :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_maskedSelect : Pointer to function tensor src mask -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_maskedSelect"
  p_THDoubleTensor_maskedSelect :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CTHByteTensor -> IO ())

-- |p_THDoubleTensor_nonzero : Pointer to function subscript tensor -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_nonzero"
  p_THDoubleTensor_nonzero :: FunPtr (Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_indexSelect : Pointer to function tensor src dim index -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_indexSelect"
  p_THDoubleTensor_indexSelect :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> Ptr CTHLongTensor -> IO ())

-- |p_THDoubleTensor_indexCopy : Pointer to function tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_indexCopy"
  p_THDoubleTensor_indexCopy :: FunPtr ((Ptr CTHDoubleTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_indexAdd : Pointer to function tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_indexAdd"
  p_THDoubleTensor_indexAdd :: FunPtr ((Ptr CTHDoubleTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_indexFill : Pointer to function tensor dim index val -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_indexFill"
  p_THDoubleTensor_indexFill :: FunPtr ((Ptr CTHDoubleTensor) -> CInt -> Ptr CTHLongTensor -> CDouble -> IO ())

-- |p_THDoubleTensor_gather : Pointer to function tensor src dim index -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_gather"
  p_THDoubleTensor_gather :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> Ptr CTHLongTensor -> IO ())

-- |p_THDoubleTensor_scatter : Pointer to function tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_scatter"
  p_THDoubleTensor_scatter :: FunPtr ((Ptr CTHDoubleTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_scatterAdd : Pointer to function tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_scatterAdd"
  p_THDoubleTensor_scatterAdd :: FunPtr ((Ptr CTHDoubleTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_scatterFill : Pointer to function tensor dim index val -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_scatterFill"
  p_THDoubleTensor_scatterFill :: FunPtr ((Ptr CTHDoubleTensor) -> CInt -> Ptr CTHLongTensor -> CDouble -> IO ())

-- |p_THDoubleTensor_dot : Pointer to function t src -> accreal
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_dot"
  p_THDoubleTensor_dot :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble)

-- |p_THDoubleTensor_minall : Pointer to function t -> real
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_minall"
  p_THDoubleTensor_minall :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble)

-- |p_THDoubleTensor_maxall : Pointer to function t -> real
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_maxall"
  p_THDoubleTensor_maxall :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble)

-- |p_THDoubleTensor_medianall : Pointer to function t -> real
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_medianall"
  p_THDoubleTensor_medianall :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble)

-- |p_THDoubleTensor_sumall : Pointer to function t -> accreal
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_sumall"
  p_THDoubleTensor_sumall :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble)

-- |p_THDoubleTensor_prodall : Pointer to function t -> accreal
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_prodall"
  p_THDoubleTensor_prodall :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble)

-- |p_THDoubleTensor_neg : Pointer to function self src -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_neg"
  p_THDoubleTensor_neg :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_cinv : Pointer to function self src -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_cinv"
  p_THDoubleTensor_cinv :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_add : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_add"
  p_THDoubleTensor_add :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_sub : Pointer to function self src value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_sub"
  p_THDoubleTensor_sub :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_mul : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_mul"
  p_THDoubleTensor_mul :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_div : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_div"
  p_THDoubleTensor_div :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_lshift : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_lshift"
  p_THDoubleTensor_lshift :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_rshift : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_rshift"
  p_THDoubleTensor_rshift :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_fmod : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_fmod"
  p_THDoubleTensor_fmod :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_remainder : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_remainder"
  p_THDoubleTensor_remainder :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_clamp : Pointer to function r_ t min_value max_value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_clamp"
  p_THDoubleTensor_clamp :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ())

-- |p_THDoubleTensor_bitand : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_bitand"
  p_THDoubleTensor_bitand :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_bitor : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_bitor"
  p_THDoubleTensor_bitor :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_bitxor : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_bitxor"
  p_THDoubleTensor_bitxor :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_cadd : Pointer to function r_ t value src -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_cadd"
  p_THDoubleTensor_cadd :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_csub : Pointer to function self src1 value src2 -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_csub"
  p_THDoubleTensor_csub :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_cmul : Pointer to function r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_cmul"
  p_THDoubleTensor_cmul :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_cpow : Pointer to function r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_cpow"
  p_THDoubleTensor_cpow :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_cdiv : Pointer to function r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_cdiv"
  p_THDoubleTensor_cdiv :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_clshift : Pointer to function r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_clshift"
  p_THDoubleTensor_clshift :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_crshift : Pointer to function r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_crshift"
  p_THDoubleTensor_crshift :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_cfmod : Pointer to function r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_cfmod"
  p_THDoubleTensor_cfmod :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_cremainder : Pointer to function r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_cremainder"
  p_THDoubleTensor_cremainder :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_cbitand : Pointer to function r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_cbitand"
  p_THDoubleTensor_cbitand :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_cbitor : Pointer to function r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_cbitor"
  p_THDoubleTensor_cbitor :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_cbitxor : Pointer to function r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_cbitxor"
  p_THDoubleTensor_cbitxor :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_addcmul : Pointer to function r_ t value src1 src2 -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_addcmul"
  p_THDoubleTensor_addcmul :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_addcdiv : Pointer to function r_ t value src1 src2 -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_addcdiv"
  p_THDoubleTensor_addcdiv :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_addmv : Pointer to function r_ beta t alpha mat vec -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_addmv"
  p_THDoubleTensor_addmv :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_addmm : Pointer to function r_ beta t alpha mat1 mat2 -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_addmm"
  p_THDoubleTensor_addmm :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_addr : Pointer to function r_ beta t alpha vec1 vec2 -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_addr"
  p_THDoubleTensor_addr :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_addbmm : Pointer to function r_ beta t alpha batch1 batch2 -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_addbmm"
  p_THDoubleTensor_addbmm :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_baddbmm : Pointer to function r_ beta t alpha batch1 batch2 -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_baddbmm"
  p_THDoubleTensor_baddbmm :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_match : Pointer to function r_ m1 m2 gain -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_match"
  p_THDoubleTensor_match :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_numel : Pointer to function t -> ptrdiff_t
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_numel"
  p_THDoubleTensor_numel :: FunPtr ((Ptr CTHDoubleTensor) -> CPtrdiff)

-- |p_THDoubleTensor_max : Pointer to function values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_max"
  p_THDoubleTensor_max :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())

-- |p_THDoubleTensor_min : Pointer to function values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_min"
  p_THDoubleTensor_min :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())

-- |p_THDoubleTensor_kthvalue : Pointer to function values_ indices_ t k dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_kthvalue"
  p_THDoubleTensor_kthvalue :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> CLong -> CInt -> CInt -> IO ())

-- |p_THDoubleTensor_mode : Pointer to function values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_mode"
  p_THDoubleTensor_mode :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())

-- |p_THDoubleTensor_median : Pointer to function values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_median"
  p_THDoubleTensor_median :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())

-- |p_THDoubleTensor_sum : Pointer to function r_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_sum"
  p_THDoubleTensor_sum :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())

-- |p_THDoubleTensor_prod : Pointer to function r_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_prod"
  p_THDoubleTensor_prod :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())

-- |p_THDoubleTensor_cumsum : Pointer to function r_ t dimension -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_cumsum"
  p_THDoubleTensor_cumsum :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THDoubleTensor_cumprod : Pointer to function r_ t dimension -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_cumprod"
  p_THDoubleTensor_cumprod :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THDoubleTensor_sign : Pointer to function r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_sign"
  p_THDoubleTensor_sign :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_trace : Pointer to function t -> accreal
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_trace"
  p_THDoubleTensor_trace :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble)

-- |p_THDoubleTensor_cross : Pointer to function r_ a b dimension -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_cross"
  p_THDoubleTensor_cross :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THDoubleTensor_cmax : Pointer to function r t src -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_cmax"
  p_THDoubleTensor_cmax :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_cmin : Pointer to function r t src -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_cmin"
  p_THDoubleTensor_cmin :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_cmaxValue : Pointer to function r t value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_cmaxValue"
  p_THDoubleTensor_cmaxValue :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_cminValue : Pointer to function r t value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_cminValue"
  p_THDoubleTensor_cminValue :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_zeros : Pointer to function r_ size -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_zeros"
  p_THDoubleTensor_zeros :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHLongStorage -> IO ())

-- |p_THDoubleTensor_zerosLike : Pointer to function r_ input -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_zerosLike"
  p_THDoubleTensor_zerosLike :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_ones : Pointer to function r_ size -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_ones"
  p_THDoubleTensor_ones :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHLongStorage -> IO ())

-- |p_THDoubleTensor_onesLike : Pointer to function r_ input -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_onesLike"
  p_THDoubleTensor_onesLike :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_diag : Pointer to function r_ t k -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_diag"
  p_THDoubleTensor_diag :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THDoubleTensor_eye : Pointer to function r_ n m -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_eye"
  p_THDoubleTensor_eye :: FunPtr ((Ptr CTHDoubleTensor) -> CLong -> CLong -> IO ())

-- |p_THDoubleTensor_arange : Pointer to function r_ xmin xmax step -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_arange"
  p_THDoubleTensor_arange :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CDouble -> IO ())

-- |p_THDoubleTensor_range : Pointer to function r_ xmin xmax step -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_range"
  p_THDoubleTensor_range :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CDouble -> IO ())

-- |p_THDoubleTensor_randperm : Pointer to function r_ _generator n -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_randperm"
  p_THDoubleTensor_randperm :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> CLong -> IO ())

-- |p_THDoubleTensor_reshape : Pointer to function r_ t size -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_reshape"
  p_THDoubleTensor_reshape :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CTHLongStorage -> IO ())

-- |p_THDoubleTensor_sort : Pointer to function rt_ ri_ t dimension descendingOrder -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_sort"
  p_THDoubleTensor_sort :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())

-- |p_THDoubleTensor_topk : Pointer to function rt_ ri_ t k dim dir sorted -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_topk"
  p_THDoubleTensor_topk :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> CLong -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleTensor_tril : Pointer to function r_ t k -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_tril"
  p_THDoubleTensor_tril :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> IO ())

-- |p_THDoubleTensor_triu : Pointer to function r_ t k -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_triu"
  p_THDoubleTensor_triu :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> IO ())

-- |p_THDoubleTensor_cat : Pointer to function r_ ta tb dimension -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_cat"
  p_THDoubleTensor_cat :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ())

-- |p_THDoubleTensor_catArray : Pointer to function result inputs numInputs dimension -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_catArray"
  p_THDoubleTensor_catArray :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())

-- |p_THDoubleTensor_equal : Pointer to function ta tb -> int
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_equal"
  p_THDoubleTensor_equal :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt)

-- |p_THDoubleTensor_ltValue : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_ltValue"
  p_THDoubleTensor_ltValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_leValue : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_leValue"
  p_THDoubleTensor_leValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_gtValue : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_gtValue"
  p_THDoubleTensor_gtValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_geValue : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_geValue"
  p_THDoubleTensor_geValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_neValue : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_neValue"
  p_THDoubleTensor_neValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_eqValue : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_eqValue"
  p_THDoubleTensor_eqValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_ltValueT : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_ltValueT"
  p_THDoubleTensor_ltValueT :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_leValueT : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_leValueT"
  p_THDoubleTensor_leValueT :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_gtValueT : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_gtValueT"
  p_THDoubleTensor_gtValueT :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_geValueT : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_geValueT"
  p_THDoubleTensor_geValueT :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_neValueT : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_neValueT"
  p_THDoubleTensor_neValueT :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_eqValueT : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_eqValueT"
  p_THDoubleTensor_eqValueT :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_ltTensor : Pointer to function r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_ltTensor"
  p_THDoubleTensor_ltTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_leTensor : Pointer to function r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_leTensor"
  p_THDoubleTensor_leTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_gtTensor : Pointer to function r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_gtTensor"
  p_THDoubleTensor_gtTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_geTensor : Pointer to function r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_geTensor"
  p_THDoubleTensor_geTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_neTensor : Pointer to function r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_neTensor"
  p_THDoubleTensor_neTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_eqTensor : Pointer to function r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_eqTensor"
  p_THDoubleTensor_eqTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_ltTensorT : Pointer to function r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_ltTensorT"
  p_THDoubleTensor_ltTensorT :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_leTensorT : Pointer to function r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_leTensorT"
  p_THDoubleTensor_leTensorT :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_gtTensorT : Pointer to function r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_gtTensorT"
  p_THDoubleTensor_gtTensorT :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_geTensorT : Pointer to function r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_geTensorT"
  p_THDoubleTensor_geTensorT :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_neTensorT : Pointer to function r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_neTensorT"
  p_THDoubleTensor_neTensorT :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_eqTensorT : Pointer to function r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_eqTensorT"
  p_THDoubleTensor_eqTensorT :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_abs : Pointer to function r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_abs"
  p_THDoubleTensor_abs :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_sigmoid : Pointer to function r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_sigmoid"
  p_THDoubleTensor_sigmoid :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_log : Pointer to function r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_log"
  p_THDoubleTensor_log :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_lgamma : Pointer to function r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_lgamma"
  p_THDoubleTensor_lgamma :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_log1p : Pointer to function r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_log1p"
  p_THDoubleTensor_log1p :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_exp : Pointer to function r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_exp"
  p_THDoubleTensor_exp :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_cos : Pointer to function r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_cos"
  p_THDoubleTensor_cos :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_acos : Pointer to function r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_acos"
  p_THDoubleTensor_acos :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_cosh : Pointer to function r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_cosh"
  p_THDoubleTensor_cosh :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_sin : Pointer to function r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_sin"
  p_THDoubleTensor_sin :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_asin : Pointer to function r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_asin"
  p_THDoubleTensor_asin :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_sinh : Pointer to function r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_sinh"
  p_THDoubleTensor_sinh :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_tan : Pointer to function r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_tan"
  p_THDoubleTensor_tan :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_atan : Pointer to function r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_atan"
  p_THDoubleTensor_atan :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_atan2 : Pointer to function r_ tx ty -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_atan2"
  p_THDoubleTensor_atan2 :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_tanh : Pointer to function r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_tanh"
  p_THDoubleTensor_tanh :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_pow : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_pow"
  p_THDoubleTensor_pow :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_tpow : Pointer to function r_ value t -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_tpow"
  p_THDoubleTensor_tpow :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_sqrt : Pointer to function r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_sqrt"
  p_THDoubleTensor_sqrt :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_rsqrt : Pointer to function r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_rsqrt"
  p_THDoubleTensor_rsqrt :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_ceil : Pointer to function r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_ceil"
  p_THDoubleTensor_ceil :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_floor : Pointer to function r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_floor"
  p_THDoubleTensor_floor :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_round : Pointer to function r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_round"
  p_THDoubleTensor_round :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_trunc : Pointer to function r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_trunc"
  p_THDoubleTensor_trunc :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_frac : Pointer to function r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_frac"
  p_THDoubleTensor_frac :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_lerp : Pointer to function r_ a b weight -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_lerp"
  p_THDoubleTensor_lerp :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_mean : Pointer to function r_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_mean"
  p_THDoubleTensor_mean :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())

-- |p_THDoubleTensor_std : Pointer to function r_ t dimension biased keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_std"
  p_THDoubleTensor_std :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleTensor_var : Pointer to function r_ t dimension biased keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_var"
  p_THDoubleTensor_var :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> IO ())

-- |p_THDoubleTensor_norm : Pointer to function r_ t value dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_norm"
  p_THDoubleTensor_norm :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CInt -> CInt -> IO ())

-- |p_THDoubleTensor_renorm : Pointer to function r_ t value dimension maxnorm -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_renorm"
  p_THDoubleTensor_renorm :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CInt -> CDouble -> IO ())

-- |p_THDoubleTensor_dist : Pointer to function a b value -> accreal
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_dist"
  p_THDoubleTensor_dist :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble)

-- |p_THDoubleTensor_histc : Pointer to function hist tensor nbins minvalue maxvalue -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_histc"
  p_THDoubleTensor_histc :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CDouble -> CDouble -> IO ())

-- |p_THDoubleTensor_bhistc : Pointer to function hist tensor nbins minvalue maxvalue -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_bhistc"
  p_THDoubleTensor_bhistc :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CDouble -> CDouble -> IO ())

-- |p_THDoubleTensor_meanall : Pointer to function self -> accreal
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_meanall"
  p_THDoubleTensor_meanall :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble)

-- |p_THDoubleTensor_varall : Pointer to function self biased -> accreal
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_varall"
  p_THDoubleTensor_varall :: FunPtr ((Ptr CTHDoubleTensor) -> CInt -> CDouble)

-- |p_THDoubleTensor_stdall : Pointer to function self biased -> accreal
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_stdall"
  p_THDoubleTensor_stdall :: FunPtr ((Ptr CTHDoubleTensor) -> CInt -> CDouble)

-- |p_THDoubleTensor_normall : Pointer to function t value -> accreal
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_normall"
  p_THDoubleTensor_normall :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble -> CDouble)

-- |p_THDoubleTensor_linspace : Pointer to function r_ a b n -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_linspace"
  p_THDoubleTensor_linspace :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CLong -> IO ())

-- |p_THDoubleTensor_logspace : Pointer to function r_ a b n -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_logspace"
  p_THDoubleTensor_logspace :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CLong -> IO ())

-- |p_THDoubleTensor_rand : Pointer to function r_ _generator size -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_rand"
  p_THDoubleTensor_rand :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ())

-- |p_THDoubleTensor_randn : Pointer to function r_ _generator size -> void
foreign import ccall unsafe "THTensorMath.h &THDoubleTensor_randn"
  p_THDoubleTensor_randn :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ())
