{-# LANGUAGE ForeignFunctionInterface #-}

module THFloatTensorMath (
    c_THFloatTensor_fill,
    c_THFloatTensor_zero,
    c_THFloatTensor_maskedFill,
    c_THFloatTensor_maskedCopy,
    c_THFloatTensor_maskedSelect,
    c_THFloatTensor_nonzero,
    c_THFloatTensor_indexSelect,
    c_THFloatTensor_indexCopy,
    c_THFloatTensor_indexAdd,
    c_THFloatTensor_indexFill,
    c_THFloatTensor_take,
    c_THFloatTensor_put,
    c_THFloatTensor_gather,
    c_THFloatTensor_scatter,
    c_THFloatTensor_scatterAdd,
    c_THFloatTensor_scatterFill,
    c_THFloatTensor_dot,
    c_THFloatTensor_minall,
    c_THFloatTensor_maxall,
    c_THFloatTensor_medianall,
    c_THFloatTensor_sumall,
    c_THFloatTensor_prodall,
    c_THFloatTensor_neg,
    c_THFloatTensor_cinv,
    c_THFloatTensor_add,
    c_THFloatTensor_sub,
    c_THFloatTensor_add_scaled,
    c_THFloatTensor_sub_scaled,
    c_THFloatTensor_mul,
    c_THFloatTensor_div,
    c_THFloatTensor_lshift,
    c_THFloatTensor_rshift,
    c_THFloatTensor_fmod,
    c_THFloatTensor_remainder,
    c_THFloatTensor_clamp,
    c_THFloatTensor_bitand,
    c_THFloatTensor_bitor,
    c_THFloatTensor_bitxor,
    c_THFloatTensor_cadd,
    c_THFloatTensor_csub,
    c_THFloatTensor_cmul,
    c_THFloatTensor_cpow,
    c_THFloatTensor_cdiv,
    c_THFloatTensor_clshift,
    c_THFloatTensor_crshift,
    c_THFloatTensor_cfmod,
    c_THFloatTensor_cremainder,
    c_THFloatTensor_cbitand,
    c_THFloatTensor_cbitor,
    c_THFloatTensor_cbitxor,
    c_THFloatTensor_addcmul,
    c_THFloatTensor_addcdiv,
    c_THFloatTensor_addmv,
    c_THFloatTensor_addmm,
    c_THFloatTensor_addr,
    c_THFloatTensor_addbmm,
    c_THFloatTensor_baddbmm,
    c_THFloatTensor_match,
    c_THFloatTensor_numel,
    c_THFloatTensor_max,
    c_THFloatTensor_min,
    c_THFloatTensor_kthvalue,
    c_THFloatTensor_mode,
    c_THFloatTensor_median,
    c_THFloatTensor_sum,
    c_THFloatTensor_prod,
    c_THFloatTensor_cumsum,
    c_THFloatTensor_cumprod,
    c_THFloatTensor_sign,
    c_THFloatTensor_trace,
    c_THFloatTensor_cross,
    c_THFloatTensor_cmax,
    c_THFloatTensor_cmin,
    c_THFloatTensor_cmaxValue,
    c_THFloatTensor_cminValue,
    c_THFloatTensor_zeros,
    c_THFloatTensor_zerosLike,
    c_THFloatTensor_ones,
    c_THFloatTensor_onesLike,
    c_THFloatTensor_diag,
    c_THFloatTensor_eye,
    c_THFloatTensor_arange,
    c_THFloatTensor_range,
    c_THFloatTensor_randperm,
    c_THFloatTensor_reshape,
    c_THFloatTensor_sort,
    c_THFloatTensor_topk,
    c_THFloatTensor_tril,
    c_THFloatTensor_triu,
    c_THFloatTensor_cat,
    c_THFloatTensor_catArray,
    c_THFloatTensor_equal,
    c_THFloatTensor_ltValue,
    c_THFloatTensor_leValue,
    c_THFloatTensor_gtValue,
    c_THFloatTensor_geValue,
    c_THFloatTensor_neValue,
    c_THFloatTensor_eqValue,
    c_THFloatTensor_ltValueT,
    c_THFloatTensor_leValueT,
    c_THFloatTensor_gtValueT,
    c_THFloatTensor_geValueT,
    c_THFloatTensor_neValueT,
    c_THFloatTensor_eqValueT,
    c_THFloatTensor_ltTensor,
    c_THFloatTensor_leTensor,
    c_THFloatTensor_gtTensor,
    c_THFloatTensor_geTensor,
    c_THFloatTensor_neTensor,
    c_THFloatTensor_eqTensor,
    c_THFloatTensor_ltTensorT,
    c_THFloatTensor_leTensorT,
    c_THFloatTensor_gtTensorT,
    c_THFloatTensor_geTensorT,
    c_THFloatTensor_neTensorT,
    c_THFloatTensor_eqTensorT,
    c_THFloatTensor_abs,
    c_THFloatTensor_sigmoid,
    c_THFloatTensor_log,
    c_THFloatTensor_lgamma,
    c_THFloatTensor_log1p,
    c_THFloatTensor_exp,
    c_THFloatTensor_cos,
    c_THFloatTensor_acos,
    c_THFloatTensor_cosh,
    c_THFloatTensor_sin,
    c_THFloatTensor_asin,
    c_THFloatTensor_sinh,
    c_THFloatTensor_tan,
    c_THFloatTensor_atan,
    c_THFloatTensor_atan2,
    c_THFloatTensor_tanh,
    c_THFloatTensor_erf,
    c_THFloatTensor_erfinv,
    c_THFloatTensor_pow,
    c_THFloatTensor_tpow,
    c_THFloatTensor_sqrt,
    c_THFloatTensor_rsqrt,
    c_THFloatTensor_ceil,
    c_THFloatTensor_floor,
    c_THFloatTensor_round,
    c_THFloatTensor_trunc,
    c_THFloatTensor_frac,
    c_THFloatTensor_lerp,
    c_THFloatTensor_mean,
    c_THFloatTensor_std,
    c_THFloatTensor_var,
    c_THFloatTensor_norm,
    c_THFloatTensor_renorm,
    c_THFloatTensor_dist,
    c_THFloatTensor_histc,
    c_THFloatTensor_bhistc,
    c_THFloatTensor_meanall,
    c_THFloatTensor_varall,
    c_THFloatTensor_stdall,
    c_THFloatTensor_normall,
    c_THFloatTensor_linspace,
    c_THFloatTensor_logspace,
    c_THFloatTensor_rand,
    c_THFloatTensor_randn,
    p_THFloatTensor_fill,
    p_THFloatTensor_zero,
    p_THFloatTensor_maskedFill,
    p_THFloatTensor_maskedCopy,
    p_THFloatTensor_maskedSelect,
    p_THFloatTensor_nonzero,
    p_THFloatTensor_indexSelect,
    p_THFloatTensor_indexCopy,
    p_THFloatTensor_indexAdd,
    p_THFloatTensor_indexFill,
    p_THFloatTensor_take,
    p_THFloatTensor_put,
    p_THFloatTensor_gather,
    p_THFloatTensor_scatter,
    p_THFloatTensor_scatterAdd,
    p_THFloatTensor_scatterFill,
    p_THFloatTensor_dot,
    p_THFloatTensor_minall,
    p_THFloatTensor_maxall,
    p_THFloatTensor_medianall,
    p_THFloatTensor_sumall,
    p_THFloatTensor_prodall,
    p_THFloatTensor_neg,
    p_THFloatTensor_cinv,
    p_THFloatTensor_add,
    p_THFloatTensor_sub,
    p_THFloatTensor_add_scaled,
    p_THFloatTensor_sub_scaled,
    p_THFloatTensor_mul,
    p_THFloatTensor_div,
    p_THFloatTensor_lshift,
    p_THFloatTensor_rshift,
    p_THFloatTensor_fmod,
    p_THFloatTensor_remainder,
    p_THFloatTensor_clamp,
    p_THFloatTensor_bitand,
    p_THFloatTensor_bitor,
    p_THFloatTensor_bitxor,
    p_THFloatTensor_cadd,
    p_THFloatTensor_csub,
    p_THFloatTensor_cmul,
    p_THFloatTensor_cpow,
    p_THFloatTensor_cdiv,
    p_THFloatTensor_clshift,
    p_THFloatTensor_crshift,
    p_THFloatTensor_cfmod,
    p_THFloatTensor_cremainder,
    p_THFloatTensor_cbitand,
    p_THFloatTensor_cbitor,
    p_THFloatTensor_cbitxor,
    p_THFloatTensor_addcmul,
    p_THFloatTensor_addcdiv,
    p_THFloatTensor_addmv,
    p_THFloatTensor_addmm,
    p_THFloatTensor_addr,
    p_THFloatTensor_addbmm,
    p_THFloatTensor_baddbmm,
    p_THFloatTensor_match,
    p_THFloatTensor_numel,
    p_THFloatTensor_max,
    p_THFloatTensor_min,
    p_THFloatTensor_kthvalue,
    p_THFloatTensor_mode,
    p_THFloatTensor_median,
    p_THFloatTensor_sum,
    p_THFloatTensor_prod,
    p_THFloatTensor_cumsum,
    p_THFloatTensor_cumprod,
    p_THFloatTensor_sign,
    p_THFloatTensor_trace,
    p_THFloatTensor_cross,
    p_THFloatTensor_cmax,
    p_THFloatTensor_cmin,
    p_THFloatTensor_cmaxValue,
    p_THFloatTensor_cminValue,
    p_THFloatTensor_zeros,
    p_THFloatTensor_zerosLike,
    p_THFloatTensor_ones,
    p_THFloatTensor_onesLike,
    p_THFloatTensor_diag,
    p_THFloatTensor_eye,
    p_THFloatTensor_arange,
    p_THFloatTensor_range,
    p_THFloatTensor_randperm,
    p_THFloatTensor_reshape,
    p_THFloatTensor_sort,
    p_THFloatTensor_topk,
    p_THFloatTensor_tril,
    p_THFloatTensor_triu,
    p_THFloatTensor_cat,
    p_THFloatTensor_catArray,
    p_THFloatTensor_equal,
    p_THFloatTensor_ltValue,
    p_THFloatTensor_leValue,
    p_THFloatTensor_gtValue,
    p_THFloatTensor_geValue,
    p_THFloatTensor_neValue,
    p_THFloatTensor_eqValue,
    p_THFloatTensor_ltValueT,
    p_THFloatTensor_leValueT,
    p_THFloatTensor_gtValueT,
    p_THFloatTensor_geValueT,
    p_THFloatTensor_neValueT,
    p_THFloatTensor_eqValueT,
    p_THFloatTensor_ltTensor,
    p_THFloatTensor_leTensor,
    p_THFloatTensor_gtTensor,
    p_THFloatTensor_geTensor,
    p_THFloatTensor_neTensor,
    p_THFloatTensor_eqTensor,
    p_THFloatTensor_ltTensorT,
    p_THFloatTensor_leTensorT,
    p_THFloatTensor_gtTensorT,
    p_THFloatTensor_geTensorT,
    p_THFloatTensor_neTensorT,
    p_THFloatTensor_eqTensorT,
    p_THFloatTensor_abs,
    p_THFloatTensor_sigmoid,
    p_THFloatTensor_log,
    p_THFloatTensor_lgamma,
    p_THFloatTensor_log1p,
    p_THFloatTensor_exp,
    p_THFloatTensor_cos,
    p_THFloatTensor_acos,
    p_THFloatTensor_cosh,
    p_THFloatTensor_sin,
    p_THFloatTensor_asin,
    p_THFloatTensor_sinh,
    p_THFloatTensor_tan,
    p_THFloatTensor_atan,
    p_THFloatTensor_atan2,
    p_THFloatTensor_tanh,
    p_THFloatTensor_erf,
    p_THFloatTensor_erfinv,
    p_THFloatTensor_pow,
    p_THFloatTensor_tpow,
    p_THFloatTensor_sqrt,
    p_THFloatTensor_rsqrt,
    p_THFloatTensor_ceil,
    p_THFloatTensor_floor,
    p_THFloatTensor_round,
    p_THFloatTensor_trunc,
    p_THFloatTensor_frac,
    p_THFloatTensor_lerp,
    p_THFloatTensor_mean,
    p_THFloatTensor_std,
    p_THFloatTensor_var,
    p_THFloatTensor_norm,
    p_THFloatTensor_renorm,
    p_THFloatTensor_dist,
    p_THFloatTensor_histc,
    p_THFloatTensor_bhistc,
    p_THFloatTensor_meanall,
    p_THFloatTensor_varall,
    p_THFloatTensor_stdall,
    p_THFloatTensor_normall,
    p_THFloatTensor_linspace,
    p_THFloatTensor_logspace,
    p_THFloatTensor_rand,
    p_THFloatTensor_randn) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THFloatTensor_fill : r_ value -> void
foreign import ccall "THTensorMath.h THFloatTensor_fill"
  c_THFloatTensor_fill :: (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_zero : r_ -> void
foreign import ccall "THTensorMath.h THFloatTensor_zero"
  c_THFloatTensor_zero :: (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_maskedFill : tensor mask value -> void
foreign import ccall "THTensorMath.h THFloatTensor_maskedFill"
  c_THFloatTensor_maskedFill :: (Ptr CTHFloatTensor) -> Ptr CTHByteTensor -> CFloat -> IO ()

-- |c_THFloatTensor_maskedCopy : tensor mask src -> void
foreign import ccall "THTensorMath.h THFloatTensor_maskedCopy"
  c_THFloatTensor_maskedCopy :: (Ptr CTHFloatTensor) -> Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_maskedSelect : tensor src mask -> void
foreign import ccall "THTensorMath.h THFloatTensor_maskedSelect"
  c_THFloatTensor_maskedSelect :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THFloatTensor_nonzero : subscript tensor -> void
foreign import ccall "THTensorMath.h THFloatTensor_nonzero"
  c_THFloatTensor_nonzero :: Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_indexSelect : tensor src dim index -> void
foreign import ccall "THTensorMath.h THFloatTensor_indexSelect"
  c_THFloatTensor_indexSelect :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THFloatTensor_indexCopy : tensor dim index src -> void
foreign import ccall "THTensorMath.h THFloatTensor_indexCopy"
  c_THFloatTensor_indexCopy :: (Ptr CTHFloatTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_indexAdd : tensor dim index src -> void
foreign import ccall "THTensorMath.h THFloatTensor_indexAdd"
  c_THFloatTensor_indexAdd :: (Ptr CTHFloatTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_indexFill : tensor dim index val -> void
foreign import ccall "THTensorMath.h THFloatTensor_indexFill"
  c_THFloatTensor_indexFill :: (Ptr CTHFloatTensor) -> CInt -> Ptr CTHLongTensor -> CFloat -> IO ()

-- |c_THFloatTensor_take : tensor src index -> void
foreign import ccall "THTensorMath.h THFloatTensor_take"
  c_THFloatTensor_take :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> IO ()

-- |c_THFloatTensor_put : tensor index src accumulate -> void
foreign import ccall "THTensorMath.h THFloatTensor_put"
  c_THFloatTensor_put :: (Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatTensor_gather : tensor src dim index -> void
foreign import ccall "THTensorMath.h THFloatTensor_gather"
  c_THFloatTensor_gather :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THFloatTensor_scatter : tensor dim index src -> void
foreign import ccall "THTensorMath.h THFloatTensor_scatter"
  c_THFloatTensor_scatter :: (Ptr CTHFloatTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_scatterAdd : tensor dim index src -> void
foreign import ccall "THTensorMath.h THFloatTensor_scatterAdd"
  c_THFloatTensor_scatterAdd :: (Ptr CTHFloatTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_scatterFill : tensor dim index val -> void
foreign import ccall "THTensorMath.h THFloatTensor_scatterFill"
  c_THFloatTensor_scatterFill :: (Ptr CTHFloatTensor) -> CInt -> Ptr CTHLongTensor -> CFloat -> IO ()

-- |c_THFloatTensor_dot : t src -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_dot"
  c_THFloatTensor_dot :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble

-- |c_THFloatTensor_minall : t -> real
foreign import ccall "THTensorMath.h THFloatTensor_minall"
  c_THFloatTensor_minall :: (Ptr CTHFloatTensor) -> CFloat

-- |c_THFloatTensor_maxall : t -> real
foreign import ccall "THTensorMath.h THFloatTensor_maxall"
  c_THFloatTensor_maxall :: (Ptr CTHFloatTensor) -> CFloat

-- |c_THFloatTensor_medianall : t -> real
foreign import ccall "THTensorMath.h THFloatTensor_medianall"
  c_THFloatTensor_medianall :: (Ptr CTHFloatTensor) -> CFloat

-- |c_THFloatTensor_sumall : t -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_sumall"
  c_THFloatTensor_sumall :: (Ptr CTHFloatTensor) -> CDouble

-- |c_THFloatTensor_prodall : t -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_prodall"
  c_THFloatTensor_prodall :: (Ptr CTHFloatTensor) -> CDouble

-- |c_THFloatTensor_neg : self src -> void
foreign import ccall "THTensorMath.h THFloatTensor_neg"
  c_THFloatTensor_neg :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_cinv : self src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cinv"
  c_THFloatTensor_cinv :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_add : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_add"
  c_THFloatTensor_add :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_sub : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_sub"
  c_THFloatTensor_sub :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_add_scaled : r_ t value alpha -> void
foreign import ccall "THTensorMath.h THFloatTensor_add_scaled"
  c_THFloatTensor_add_scaled :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> CFloat -> IO ()

-- |c_THFloatTensor_sub_scaled : r_ t value alpha -> void
foreign import ccall "THTensorMath.h THFloatTensor_sub_scaled"
  c_THFloatTensor_sub_scaled :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> CFloat -> IO ()

-- |c_THFloatTensor_mul : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_mul"
  c_THFloatTensor_mul :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_div : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_div"
  c_THFloatTensor_div :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_lshift : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_lshift"
  c_THFloatTensor_lshift :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_rshift : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_rshift"
  c_THFloatTensor_rshift :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_fmod : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_fmod"
  c_THFloatTensor_fmod :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_remainder : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_remainder"
  c_THFloatTensor_remainder :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_clamp : r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h THFloatTensor_clamp"
  c_THFloatTensor_clamp :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> CFloat -> IO ()

-- |c_THFloatTensor_bitand : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_bitand"
  c_THFloatTensor_bitand :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_bitor : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_bitor"
  c_THFloatTensor_bitor :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_bitxor : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_bitxor"
  c_THFloatTensor_bitxor :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_cadd : r_ t value src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cadd"
  c_THFloatTensor_cadd :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_csub : self src1 value src2 -> void
foreign import ccall "THTensorMath.h THFloatTensor_csub"
  c_THFloatTensor_csub :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_cmul : r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cmul"
  c_THFloatTensor_cmul :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_cpow : r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cpow"
  c_THFloatTensor_cpow :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_cdiv : r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cdiv"
  c_THFloatTensor_cdiv :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_clshift : r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_clshift"
  c_THFloatTensor_clshift :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_crshift : r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_crshift"
  c_THFloatTensor_crshift :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_cfmod : r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cfmod"
  c_THFloatTensor_cfmod :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_cremainder : r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cremainder"
  c_THFloatTensor_cremainder :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_cbitand : r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cbitand"
  c_THFloatTensor_cbitand :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_cbitor : r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cbitor"
  c_THFloatTensor_cbitor :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_cbitxor : r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cbitxor"
  c_THFloatTensor_cbitxor :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_addcmul : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THFloatTensor_addcmul"
  c_THFloatTensor_addcmul :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_addcdiv : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THFloatTensor_addcdiv"
  c_THFloatTensor_addcdiv :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_addmv : r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h THFloatTensor_addmv"
  c_THFloatTensor_addmv :: (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_addmm : r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h THFloatTensor_addmm"
  c_THFloatTensor_addmm :: (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_addr : r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h THFloatTensor_addr"
  c_THFloatTensor_addr :: (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_addbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THFloatTensor_addbmm"
  c_THFloatTensor_addbmm :: (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_baddbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THFloatTensor_baddbmm"
  c_THFloatTensor_baddbmm :: (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_match : r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h THFloatTensor_match"
  c_THFloatTensor_match :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_numel : t -> ptrdiff_t
foreign import ccall "THTensorMath.h THFloatTensor_numel"
  c_THFloatTensor_numel :: (Ptr CTHFloatTensor) -> CPtrdiff

-- |c_THFloatTensor_max : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_max"
  c_THFloatTensor_max :: (Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatTensor_min : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_min"
  c_THFloatTensor_min :: (Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatTensor_kthvalue : values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_kthvalue"
  c_THFloatTensor_kthvalue :: (Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> CLLong -> CInt -> CInt -> IO ()

-- |c_THFloatTensor_mode : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_mode"
  c_THFloatTensor_mode :: (Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatTensor_median : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_median"
  c_THFloatTensor_median :: (Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatTensor_sum : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_sum"
  c_THFloatTensor_sum :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatTensor_prod : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_prod"
  c_THFloatTensor_prod :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatTensor_cumsum : r_ t dimension -> void
foreign import ccall "THTensorMath.h THFloatTensor_cumsum"
  c_THFloatTensor_cumsum :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatTensor_cumprod : r_ t dimension -> void
foreign import ccall "THTensorMath.h THFloatTensor_cumprod"
  c_THFloatTensor_cumprod :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatTensor_sign : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_sign"
  c_THFloatTensor_sign :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_trace : t -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_trace"
  c_THFloatTensor_trace :: (Ptr CTHFloatTensor) -> CDouble

-- |c_THFloatTensor_cross : r_ a b dimension -> void
foreign import ccall "THTensorMath.h THFloatTensor_cross"
  c_THFloatTensor_cross :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatTensor_cmax : r t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cmax"
  c_THFloatTensor_cmax :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_cmin : r t src -> void
foreign import ccall "THTensorMath.h THFloatTensor_cmin"
  c_THFloatTensor_cmin :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_cmaxValue : r t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_cmaxValue"
  c_THFloatTensor_cmaxValue :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_cminValue : r t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_cminValue"
  c_THFloatTensor_cminValue :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_zeros : r_ size -> void
foreign import ccall "THTensorMath.h THFloatTensor_zeros"
  c_THFloatTensor_zeros :: (Ptr CTHFloatTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THFloatTensor_zerosLike : r_ input -> void
foreign import ccall "THTensorMath.h THFloatTensor_zerosLike"
  c_THFloatTensor_zerosLike :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_ones : r_ size -> void
foreign import ccall "THTensorMath.h THFloatTensor_ones"
  c_THFloatTensor_ones :: (Ptr CTHFloatTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THFloatTensor_onesLike : r_ input -> void
foreign import ccall "THTensorMath.h THFloatTensor_onesLike"
  c_THFloatTensor_onesLike :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_diag : r_ t k -> void
foreign import ccall "THTensorMath.h THFloatTensor_diag"
  c_THFloatTensor_diag :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatTensor_eye : r_ n m -> void
foreign import ccall "THTensorMath.h THFloatTensor_eye"
  c_THFloatTensor_eye :: (Ptr CTHFloatTensor) -> CLLong -> CLLong -> IO ()

-- |c_THFloatTensor_arange : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THFloatTensor_arange"
  c_THFloatTensor_arange :: (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CDouble -> IO ()

-- |c_THFloatTensor_range : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THFloatTensor_range"
  c_THFloatTensor_range :: (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CDouble -> IO ()

-- |c_THFloatTensor_randperm : r_ _generator n -> void
foreign import ccall "THTensorMath.h THFloatTensor_randperm"
  c_THFloatTensor_randperm :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CLLong -> IO ()

-- |c_THFloatTensor_reshape : r_ t size -> void
foreign import ccall "THTensorMath.h THFloatTensor_reshape"
  c_THFloatTensor_reshape :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THFloatTensor_sort : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h THFloatTensor_sort"
  c_THFloatTensor_sort :: (Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatTensor_topk : rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h THFloatTensor_topk"
  c_THFloatTensor_topk :: (Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> CLLong -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatTensor_tril : r_ t k -> void
foreign import ccall "THTensorMath.h THFloatTensor_tril"
  c_THFloatTensor_tril :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLLong -> IO ()

-- |c_THFloatTensor_triu : r_ t k -> void
foreign import ccall "THTensorMath.h THFloatTensor_triu"
  c_THFloatTensor_triu :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLLong -> IO ()

-- |c_THFloatTensor_cat : r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h THFloatTensor_cat"
  c_THFloatTensor_cat :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatTensor_catArray : result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h THFloatTensor_catArray"
  c_THFloatTensor_catArray :: (Ptr CTHFloatTensor) -> Ptr (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatTensor_equal : ta tb -> int
foreign import ccall "THTensorMath.h THFloatTensor_equal"
  c_THFloatTensor_equal :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt

-- |c_THFloatTensor_ltValue : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_ltValue"
  c_THFloatTensor_ltValue :: Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_leValue : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_leValue"
  c_THFloatTensor_leValue :: Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_gtValue : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_gtValue"
  c_THFloatTensor_gtValue :: Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_geValue : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_geValue"
  c_THFloatTensor_geValue :: Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_neValue : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_neValue"
  c_THFloatTensor_neValue :: Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_eqValue : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_eqValue"
  c_THFloatTensor_eqValue :: Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_ltValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_ltValueT"
  c_THFloatTensor_ltValueT :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_leValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_leValueT"
  c_THFloatTensor_leValueT :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_gtValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_gtValueT"
  c_THFloatTensor_gtValueT :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_geValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_geValueT"
  c_THFloatTensor_geValueT :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_neValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_neValueT"
  c_THFloatTensor_neValueT :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_eqValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_eqValueT"
  c_THFloatTensor_eqValueT :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_ltTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_ltTensor"
  c_THFloatTensor_ltTensor :: Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_leTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_leTensor"
  c_THFloatTensor_leTensor :: Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_gtTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_gtTensor"
  c_THFloatTensor_gtTensor :: Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_geTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_geTensor"
  c_THFloatTensor_geTensor :: Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_neTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_neTensor"
  c_THFloatTensor_neTensor :: Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_eqTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_eqTensor"
  c_THFloatTensor_eqTensor :: Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_ltTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_ltTensorT"
  c_THFloatTensor_ltTensorT :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_leTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_leTensorT"
  c_THFloatTensor_leTensorT :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_gtTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_gtTensorT"
  c_THFloatTensor_gtTensorT :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_geTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_geTensorT"
  c_THFloatTensor_geTensorT :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_neTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_neTensorT"
  c_THFloatTensor_neTensorT :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_eqTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensor_eqTensorT"
  c_THFloatTensor_eqTensorT :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_abs : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_abs"
  c_THFloatTensor_abs :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_sigmoid : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_sigmoid"
  c_THFloatTensor_sigmoid :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_log : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_log"
  c_THFloatTensor_log :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_lgamma : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_lgamma"
  c_THFloatTensor_lgamma :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_log1p : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_log1p"
  c_THFloatTensor_log1p :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_exp : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_exp"
  c_THFloatTensor_exp :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_cos : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_cos"
  c_THFloatTensor_cos :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_acos : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_acos"
  c_THFloatTensor_acos :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_cosh : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_cosh"
  c_THFloatTensor_cosh :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_sin : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_sin"
  c_THFloatTensor_sin :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_asin : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_asin"
  c_THFloatTensor_asin :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_sinh : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_sinh"
  c_THFloatTensor_sinh :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_tan : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_tan"
  c_THFloatTensor_tan :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_atan : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_atan"
  c_THFloatTensor_atan :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_atan2 : r_ tx ty -> void
foreign import ccall "THTensorMath.h THFloatTensor_atan2"
  c_THFloatTensor_atan2 :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_tanh : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_tanh"
  c_THFloatTensor_tanh :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_erf : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_erf"
  c_THFloatTensor_erf :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_erfinv : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_erfinv"
  c_THFloatTensor_erfinv :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_pow : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensor_pow"
  c_THFloatTensor_pow :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_tpow : r_ value t -> void
foreign import ccall "THTensorMath.h THFloatTensor_tpow"
  c_THFloatTensor_tpow :: (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_sqrt : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_sqrt"
  c_THFloatTensor_sqrt :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_rsqrt : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_rsqrt"
  c_THFloatTensor_rsqrt :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_ceil : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_ceil"
  c_THFloatTensor_ceil :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_floor : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_floor"
  c_THFloatTensor_floor :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_round : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_round"
  c_THFloatTensor_round :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_trunc : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_trunc"
  c_THFloatTensor_trunc :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_frac : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensor_frac"
  c_THFloatTensor_frac :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_lerp : r_ a b weight -> void
foreign import ccall "THTensorMath.h THFloatTensor_lerp"
  c_THFloatTensor_lerp :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensor_mean : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_mean"
  c_THFloatTensor_mean :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatTensor_std : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_std"
  c_THFloatTensor_std :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatTensor_var : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_var"
  c_THFloatTensor_var :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatTensor_norm : r_ t value dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensor_norm"
  c_THFloatTensor_norm :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> CInt -> CInt -> IO ()

-- |c_THFloatTensor_renorm : r_ t value dimension maxnorm -> void
foreign import ccall "THTensorMath.h THFloatTensor_renorm"
  c_THFloatTensor_renorm :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> CInt -> CFloat -> IO ()

-- |c_THFloatTensor_dist : a b value -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_dist"
  c_THFloatTensor_dist :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> CDouble

-- |c_THFloatTensor_histc : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THFloatTensor_histc"
  c_THFloatTensor_histc :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLLong -> CFloat -> CFloat -> IO ()

-- |c_THFloatTensor_bhistc : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THFloatTensor_bhistc"
  c_THFloatTensor_bhistc :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLLong -> CFloat -> CFloat -> IO ()

-- |c_THFloatTensor_meanall : self -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_meanall"
  c_THFloatTensor_meanall :: (Ptr CTHFloatTensor) -> CDouble

-- |c_THFloatTensor_varall : self biased -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_varall"
  c_THFloatTensor_varall :: (Ptr CTHFloatTensor) -> CInt -> CDouble

-- |c_THFloatTensor_stdall : self biased -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_stdall"
  c_THFloatTensor_stdall :: (Ptr CTHFloatTensor) -> CInt -> CDouble

-- |c_THFloatTensor_normall : t value -> accreal
foreign import ccall "THTensorMath.h THFloatTensor_normall"
  c_THFloatTensor_normall :: (Ptr CTHFloatTensor) -> CFloat -> CDouble

-- |c_THFloatTensor_linspace : r_ a b n -> void
foreign import ccall "THTensorMath.h THFloatTensor_linspace"
  c_THFloatTensor_linspace :: (Ptr CTHFloatTensor) -> CFloat -> CFloat -> CLLong -> IO ()

-- |c_THFloatTensor_logspace : r_ a b n -> void
foreign import ccall "THTensorMath.h THFloatTensor_logspace"
  c_THFloatTensor_logspace :: (Ptr CTHFloatTensor) -> CFloat -> CFloat -> CLLong -> IO ()

-- |c_THFloatTensor_rand : r_ _generator size -> void
foreign import ccall "THTensorMath.h THFloatTensor_rand"
  c_THFloatTensor_rand :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()

-- |c_THFloatTensor_randn : r_ _generator size -> void
foreign import ccall "THTensorMath.h THFloatTensor_randn"
  c_THFloatTensor_randn :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()

-- |p_THFloatTensor_fill : Pointer to function : r_ value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_fill"
  p_THFloatTensor_fill :: FunPtr ((Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_zero : Pointer to function : r_ -> void
foreign import ccall "THTensorMath.h &THFloatTensor_zero"
  p_THFloatTensor_zero :: FunPtr ((Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_maskedFill : Pointer to function : tensor mask value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_maskedFill"
  p_THFloatTensor_maskedFill :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHByteTensor -> CFloat -> IO ())

-- |p_THFloatTensor_maskedCopy : Pointer to function : tensor mask src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_maskedCopy"
  p_THFloatTensor_maskedCopy :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_maskedSelect : Pointer to function : tensor src mask -> void
foreign import ccall "THTensorMath.h &THFloatTensor_maskedSelect"
  p_THFloatTensor_maskedSelect :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CTHByteTensor -> IO ())

-- |p_THFloatTensor_nonzero : Pointer to function : subscript tensor -> void
foreign import ccall "THTensorMath.h &THFloatTensor_nonzero"
  p_THFloatTensor_nonzero :: FunPtr (Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_indexSelect : Pointer to function : tensor src dim index -> void
foreign import ccall "THTensorMath.h &THFloatTensor_indexSelect"
  p_THFloatTensor_indexSelect :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> Ptr CTHLongTensor -> IO ())

-- |p_THFloatTensor_indexCopy : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_indexCopy"
  p_THFloatTensor_indexCopy :: FunPtr ((Ptr CTHFloatTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_indexAdd : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_indexAdd"
  p_THFloatTensor_indexAdd :: FunPtr ((Ptr CTHFloatTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_indexFill : Pointer to function : tensor dim index val -> void
foreign import ccall "THTensorMath.h &THFloatTensor_indexFill"
  p_THFloatTensor_indexFill :: FunPtr ((Ptr CTHFloatTensor) -> CInt -> Ptr CTHLongTensor -> CFloat -> IO ())

-- |p_THFloatTensor_take : Pointer to function : tensor src index -> void
foreign import ccall "THTensorMath.h &THFloatTensor_take"
  p_THFloatTensor_take :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> IO ())

-- |p_THFloatTensor_put : Pointer to function : tensor index src accumulate -> void
foreign import ccall "THTensorMath.h &THFloatTensor_put"
  p_THFloatTensor_put :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THFloatTensor_gather : Pointer to function : tensor src dim index -> void
foreign import ccall "THTensorMath.h &THFloatTensor_gather"
  p_THFloatTensor_gather :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> Ptr CTHLongTensor -> IO ())

-- |p_THFloatTensor_scatter : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_scatter"
  p_THFloatTensor_scatter :: FunPtr ((Ptr CTHFloatTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_scatterAdd : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_scatterAdd"
  p_THFloatTensor_scatterAdd :: FunPtr ((Ptr CTHFloatTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_scatterFill : Pointer to function : tensor dim index val -> void
foreign import ccall "THTensorMath.h &THFloatTensor_scatterFill"
  p_THFloatTensor_scatterFill :: FunPtr ((Ptr CTHFloatTensor) -> CInt -> Ptr CTHLongTensor -> CFloat -> IO ())

-- |p_THFloatTensor_dot : Pointer to function : t src -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_dot"
  p_THFloatTensor_dot :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble)

-- |p_THFloatTensor_minall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THFloatTensor_minall"
  p_THFloatTensor_minall :: FunPtr ((Ptr CTHFloatTensor) -> CFloat)

-- |p_THFloatTensor_maxall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THFloatTensor_maxall"
  p_THFloatTensor_maxall :: FunPtr ((Ptr CTHFloatTensor) -> CFloat)

-- |p_THFloatTensor_medianall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THFloatTensor_medianall"
  p_THFloatTensor_medianall :: FunPtr ((Ptr CTHFloatTensor) -> CFloat)

-- |p_THFloatTensor_sumall : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_sumall"
  p_THFloatTensor_sumall :: FunPtr ((Ptr CTHFloatTensor) -> CDouble)

-- |p_THFloatTensor_prodall : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_prodall"
  p_THFloatTensor_prodall :: FunPtr ((Ptr CTHFloatTensor) -> CDouble)

-- |p_THFloatTensor_neg : Pointer to function : self src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_neg"
  p_THFloatTensor_neg :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_cinv : Pointer to function : self src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cinv"
  p_THFloatTensor_cinv :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_add : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_add"
  p_THFloatTensor_add :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_sub : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sub"
  p_THFloatTensor_sub :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_add_scaled : Pointer to function : r_ t value alpha -> void
foreign import ccall "THTensorMath.h &THFloatTensor_add_scaled"
  p_THFloatTensor_add_scaled :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> CFloat -> IO ())

-- |p_THFloatTensor_sub_scaled : Pointer to function : r_ t value alpha -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sub_scaled"
  p_THFloatTensor_sub_scaled :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> CFloat -> IO ())

-- |p_THFloatTensor_mul : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_mul"
  p_THFloatTensor_mul :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_div : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_div"
  p_THFloatTensor_div :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_lshift : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_lshift"
  p_THFloatTensor_lshift :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_rshift : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_rshift"
  p_THFloatTensor_rshift :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_fmod : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_fmod"
  p_THFloatTensor_fmod :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_remainder : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_remainder"
  p_THFloatTensor_remainder :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_clamp : Pointer to function : r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_clamp"
  p_THFloatTensor_clamp :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> CFloat -> IO ())

-- |p_THFloatTensor_bitand : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_bitand"
  p_THFloatTensor_bitand :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_bitor : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_bitor"
  p_THFloatTensor_bitor :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_bitxor : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_bitxor"
  p_THFloatTensor_bitxor :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_cadd : Pointer to function : r_ t value src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cadd"
  p_THFloatTensor_cadd :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_csub : Pointer to function : self src1 value src2 -> void
foreign import ccall "THTensorMath.h &THFloatTensor_csub"
  p_THFloatTensor_csub :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_cmul : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cmul"
  p_THFloatTensor_cmul :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_cpow : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cpow"
  p_THFloatTensor_cpow :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_cdiv : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cdiv"
  p_THFloatTensor_cdiv :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_clshift : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_clshift"
  p_THFloatTensor_clshift :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_crshift : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_crshift"
  p_THFloatTensor_crshift :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_cfmod : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cfmod"
  p_THFloatTensor_cfmod :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_cremainder : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cremainder"
  p_THFloatTensor_cremainder :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_cbitand : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cbitand"
  p_THFloatTensor_cbitand :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_cbitor : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cbitor"
  p_THFloatTensor_cbitor :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_cbitxor : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cbitxor"
  p_THFloatTensor_cbitxor :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_addcmul : Pointer to function : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h &THFloatTensor_addcmul"
  p_THFloatTensor_addcmul :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_addcdiv : Pointer to function : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h &THFloatTensor_addcdiv"
  p_THFloatTensor_addcdiv :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_addmv : Pointer to function : r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h &THFloatTensor_addmv"
  p_THFloatTensor_addmv :: FunPtr ((Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_addmm : Pointer to function : r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h &THFloatTensor_addmm"
  p_THFloatTensor_addmm :: FunPtr ((Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_addr : Pointer to function : r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h &THFloatTensor_addr"
  p_THFloatTensor_addr :: FunPtr ((Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_addbmm : Pointer to function : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h &THFloatTensor_addbmm"
  p_THFloatTensor_addbmm :: FunPtr ((Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_baddbmm : Pointer to function : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h &THFloatTensor_baddbmm"
  p_THFloatTensor_baddbmm :: FunPtr ((Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_match : Pointer to function : r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h &THFloatTensor_match"
  p_THFloatTensor_match :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_numel : Pointer to function : t -> ptrdiff_t
foreign import ccall "THTensorMath.h &THFloatTensor_numel"
  p_THFloatTensor_numel :: FunPtr ((Ptr CTHFloatTensor) -> CPtrdiff)

-- |p_THFloatTensor_max : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_max"
  p_THFloatTensor_max :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ())

-- |p_THFloatTensor_min : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_min"
  p_THFloatTensor_min :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ())

-- |p_THFloatTensor_kthvalue : Pointer to function : values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_kthvalue"
  p_THFloatTensor_kthvalue :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> CLLong -> CInt -> CInt -> IO ())

-- |p_THFloatTensor_mode : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_mode"
  p_THFloatTensor_mode :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ())

-- |p_THFloatTensor_median : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_median"
  p_THFloatTensor_median :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ())

-- |p_THFloatTensor_sum : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sum"
  p_THFloatTensor_sum :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ())

-- |p_THFloatTensor_prod : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_prod"
  p_THFloatTensor_prod :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ())

-- |p_THFloatTensor_cumsum : Pointer to function : r_ t dimension -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cumsum"
  p_THFloatTensor_cumsum :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THFloatTensor_cumprod : Pointer to function : r_ t dimension -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cumprod"
  p_THFloatTensor_cumprod :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THFloatTensor_sign : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sign"
  p_THFloatTensor_sign :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_trace : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_trace"
  p_THFloatTensor_trace :: FunPtr ((Ptr CTHFloatTensor) -> CDouble)

-- |p_THFloatTensor_cross : Pointer to function : r_ a b dimension -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cross"
  p_THFloatTensor_cross :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THFloatTensor_cmax : Pointer to function : r t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cmax"
  p_THFloatTensor_cmax :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_cmin : Pointer to function : r t src -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cmin"
  p_THFloatTensor_cmin :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_cmaxValue : Pointer to function : r t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cmaxValue"
  p_THFloatTensor_cmaxValue :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_cminValue : Pointer to function : r t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cminValue"
  p_THFloatTensor_cminValue :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_zeros : Pointer to function : r_ size -> void
foreign import ccall "THTensorMath.h &THFloatTensor_zeros"
  p_THFloatTensor_zeros :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHLongStorage -> IO ())

-- |p_THFloatTensor_zerosLike : Pointer to function : r_ input -> void
foreign import ccall "THTensorMath.h &THFloatTensor_zerosLike"
  p_THFloatTensor_zerosLike :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_ones : Pointer to function : r_ size -> void
foreign import ccall "THTensorMath.h &THFloatTensor_ones"
  p_THFloatTensor_ones :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHLongStorage -> IO ())

-- |p_THFloatTensor_onesLike : Pointer to function : r_ input -> void
foreign import ccall "THTensorMath.h &THFloatTensor_onesLike"
  p_THFloatTensor_onesLike :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_diag : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THFloatTensor_diag"
  p_THFloatTensor_diag :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THFloatTensor_eye : Pointer to function : r_ n m -> void
foreign import ccall "THTensorMath.h &THFloatTensor_eye"
  p_THFloatTensor_eye :: FunPtr ((Ptr CTHFloatTensor) -> CLLong -> CLLong -> IO ())

-- |p_THFloatTensor_arange : Pointer to function : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h &THFloatTensor_arange"
  p_THFloatTensor_arange :: FunPtr ((Ptr CTHFloatTensor) -> CDouble -> CDouble -> CDouble -> IO ())

-- |p_THFloatTensor_range : Pointer to function : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h &THFloatTensor_range"
  p_THFloatTensor_range :: FunPtr ((Ptr CTHFloatTensor) -> CDouble -> CDouble -> CDouble -> IO ())

-- |p_THFloatTensor_randperm : Pointer to function : r_ _generator n -> void
foreign import ccall "THTensorMath.h &THFloatTensor_randperm"
  p_THFloatTensor_randperm :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CLLong -> IO ())

-- |p_THFloatTensor_reshape : Pointer to function : r_ t size -> void
foreign import ccall "THTensorMath.h &THFloatTensor_reshape"
  p_THFloatTensor_reshape :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CTHLongStorage -> IO ())

-- |p_THFloatTensor_sort : Pointer to function : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sort"
  p_THFloatTensor_sort :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ())

-- |p_THFloatTensor_topk : Pointer to function : rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h &THFloatTensor_topk"
  p_THFloatTensor_topk :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> CLLong -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatTensor_tril : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THFloatTensor_tril"
  p_THFloatTensor_tril :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLLong -> IO ())

-- |p_THFloatTensor_triu : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THFloatTensor_triu"
  p_THFloatTensor_triu :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLLong -> IO ())

-- |p_THFloatTensor_cat : Pointer to function : r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cat"
  p_THFloatTensor_cat :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ())

-- |p_THFloatTensor_catArray : Pointer to function : result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h &THFloatTensor_catArray"
  p_THFloatTensor_catArray :: FunPtr ((Ptr CTHFloatTensor) -> Ptr (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ())

-- |p_THFloatTensor_equal : Pointer to function : ta tb -> int
foreign import ccall "THTensorMath.h &THFloatTensor_equal"
  p_THFloatTensor_equal :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt)

-- |p_THFloatTensor_ltValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_ltValue"
  p_THFloatTensor_ltValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_leValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_leValue"
  p_THFloatTensor_leValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_gtValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_gtValue"
  p_THFloatTensor_gtValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_geValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_geValue"
  p_THFloatTensor_geValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_neValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_neValue"
  p_THFloatTensor_neValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_eqValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_eqValue"
  p_THFloatTensor_eqValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_ltValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_ltValueT"
  p_THFloatTensor_ltValueT :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_leValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_leValueT"
  p_THFloatTensor_leValueT :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_gtValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_gtValueT"
  p_THFloatTensor_gtValueT :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_geValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_geValueT"
  p_THFloatTensor_geValueT :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_neValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_neValueT"
  p_THFloatTensor_neValueT :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_eqValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_eqValueT"
  p_THFloatTensor_eqValueT :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_ltTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_ltTensor"
  p_THFloatTensor_ltTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_leTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_leTensor"
  p_THFloatTensor_leTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_gtTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_gtTensor"
  p_THFloatTensor_gtTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_geTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_geTensor"
  p_THFloatTensor_geTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_neTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_neTensor"
  p_THFloatTensor_neTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_eqTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_eqTensor"
  p_THFloatTensor_eqTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_ltTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_ltTensorT"
  p_THFloatTensor_ltTensorT :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_leTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_leTensorT"
  p_THFloatTensor_leTensorT :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_gtTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_gtTensorT"
  p_THFloatTensor_gtTensorT :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_geTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_geTensorT"
  p_THFloatTensor_geTensorT :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_neTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_neTensorT"
  p_THFloatTensor_neTensorT :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_eqTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THFloatTensor_eqTensorT"
  p_THFloatTensor_eqTensorT :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_abs : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_abs"
  p_THFloatTensor_abs :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_sigmoid : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sigmoid"
  p_THFloatTensor_sigmoid :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_log : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_log"
  p_THFloatTensor_log :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_lgamma : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_lgamma"
  p_THFloatTensor_lgamma :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_log1p : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_log1p"
  p_THFloatTensor_log1p :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_exp : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_exp"
  p_THFloatTensor_exp :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_cos : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cos"
  p_THFloatTensor_cos :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_acos : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_acos"
  p_THFloatTensor_acos :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_cosh : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_cosh"
  p_THFloatTensor_cosh :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_sin : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sin"
  p_THFloatTensor_sin :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_asin : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_asin"
  p_THFloatTensor_asin :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_sinh : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sinh"
  p_THFloatTensor_sinh :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_tan : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_tan"
  p_THFloatTensor_tan :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_atan : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_atan"
  p_THFloatTensor_atan :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_atan2 : Pointer to function : r_ tx ty -> void
foreign import ccall "THTensorMath.h &THFloatTensor_atan2"
  p_THFloatTensor_atan2 :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_tanh : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_tanh"
  p_THFloatTensor_tanh :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_erf : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_erf"
  p_THFloatTensor_erf :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_erfinv : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_erfinv"
  p_THFloatTensor_erfinv :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_pow : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THFloatTensor_pow"
  p_THFloatTensor_pow :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_tpow : Pointer to function : r_ value t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_tpow"
  p_THFloatTensor_tpow :: FunPtr ((Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_sqrt : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_sqrt"
  p_THFloatTensor_sqrt :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_rsqrt : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_rsqrt"
  p_THFloatTensor_rsqrt :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_ceil : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_ceil"
  p_THFloatTensor_ceil :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_floor : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_floor"
  p_THFloatTensor_floor :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_round : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_round"
  p_THFloatTensor_round :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_trunc : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_trunc"
  p_THFloatTensor_trunc :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_frac : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THFloatTensor_frac"
  p_THFloatTensor_frac :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_lerp : Pointer to function : r_ a b weight -> void
foreign import ccall "THTensorMath.h &THFloatTensor_lerp"
  p_THFloatTensor_lerp :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ())

-- |p_THFloatTensor_mean : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_mean"
  p_THFloatTensor_mean :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ())

-- |p_THFloatTensor_std : Pointer to function : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_std"
  p_THFloatTensor_std :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatTensor_var : Pointer to function : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_var"
  p_THFloatTensor_var :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> IO ())

-- |p_THFloatTensor_norm : Pointer to function : r_ t value dimension keepdim -> void
foreign import ccall "THTensorMath.h &THFloatTensor_norm"
  p_THFloatTensor_norm :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> CInt -> CInt -> IO ())

-- |p_THFloatTensor_renorm : Pointer to function : r_ t value dimension maxnorm -> void
foreign import ccall "THTensorMath.h &THFloatTensor_renorm"
  p_THFloatTensor_renorm :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> CInt -> CFloat -> IO ())

-- |p_THFloatTensor_dist : Pointer to function : a b value -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_dist"
  p_THFloatTensor_dist :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> CDouble)

-- |p_THFloatTensor_histc : Pointer to function : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h &THFloatTensor_histc"
  p_THFloatTensor_histc :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLLong -> CFloat -> CFloat -> IO ())

-- |p_THFloatTensor_bhistc : Pointer to function : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h &THFloatTensor_bhistc"
  p_THFloatTensor_bhistc :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLLong -> CFloat -> CFloat -> IO ())

-- |p_THFloatTensor_meanall : Pointer to function : self -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_meanall"
  p_THFloatTensor_meanall :: FunPtr ((Ptr CTHFloatTensor) -> CDouble)

-- |p_THFloatTensor_varall : Pointer to function : self biased -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_varall"
  p_THFloatTensor_varall :: FunPtr ((Ptr CTHFloatTensor) -> CInt -> CDouble)

-- |p_THFloatTensor_stdall : Pointer to function : self biased -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_stdall"
  p_THFloatTensor_stdall :: FunPtr ((Ptr CTHFloatTensor) -> CInt -> CDouble)

-- |p_THFloatTensor_normall : Pointer to function : t value -> accreal
foreign import ccall "THTensorMath.h &THFloatTensor_normall"
  p_THFloatTensor_normall :: FunPtr ((Ptr CTHFloatTensor) -> CFloat -> CDouble)

-- |p_THFloatTensor_linspace : Pointer to function : r_ a b n -> void
foreign import ccall "THTensorMath.h &THFloatTensor_linspace"
  p_THFloatTensor_linspace :: FunPtr ((Ptr CTHFloatTensor) -> CFloat -> CFloat -> CLLong -> IO ())

-- |p_THFloatTensor_logspace : Pointer to function : r_ a b n -> void
foreign import ccall "THTensorMath.h &THFloatTensor_logspace"
  p_THFloatTensor_logspace :: FunPtr ((Ptr CTHFloatTensor) -> CFloat -> CFloat -> CLLong -> IO ())

-- |p_THFloatTensor_rand : Pointer to function : r_ _generator size -> void
foreign import ccall "THTensorMath.h &THFloatTensor_rand"
  p_THFloatTensor_rand :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ())

-- |p_THFloatTensor_randn : Pointer to function : r_ _generator size -> void
foreign import ccall "THTensorMath.h &THFloatTensor_randn"
  p_THFloatTensor_randn :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ())