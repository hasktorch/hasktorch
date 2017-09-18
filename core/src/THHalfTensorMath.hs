{-# LANGUAGE ForeignFunctionInterface#-}

module THHalfTensorMath (
    c_THHalfTensor_fill,
    c_THHalfTensor_zero,
    c_THHalfTensor_maskedFill,
    c_THHalfTensor_maskedCopy,
    c_THHalfTensor_maskedSelect,
    c_THHalfTensor_nonzero,
    c_THHalfTensor_indexSelect,
    c_THHalfTensor_indexCopy,
    c_THHalfTensor_indexAdd,
    c_THHalfTensor_indexFill,
    c_THHalfTensor_gather,
    c_THHalfTensor_scatter,
    c_THHalfTensor_scatterAdd,
    c_THHalfTensor_scatterFill,
    c_THHalfTensor_dot,
    c_THHalfTensor_minall,
    c_THHalfTensor_maxall,
    c_THHalfTensor_medianall,
    c_THHalfTensor_sumall,
    c_THHalfTensor_prodall,
    c_THHalfTensor_neg,
    c_THHalfTensor_cinv,
    c_THHalfTensor_add,
    c_THHalfTensor_sub,
    c_THHalfTensor_mul,
    c_THHalfTensor_div,
    c_THHalfTensor_lshift,
    c_THHalfTensor_rshift,
    c_THHalfTensor_fmod,
    c_THHalfTensor_remainder,
    c_THHalfTensor_clamp,
    c_THHalfTensor_bitand,
    c_THHalfTensor_bitor,
    c_THHalfTensor_bitxor,
    c_THHalfTensor_cadd,
    c_THHalfTensor_csub,
    c_THHalfTensor_cmul,
    c_THHalfTensor_cpow,
    c_THHalfTensor_cdiv,
    c_THHalfTensor_clshift,
    c_THHalfTensor_crshift,
    c_THHalfTensor_cfmod,
    c_THHalfTensor_cremainder,
    c_THHalfTensor_cbitand,
    c_THHalfTensor_cbitor,
    c_THHalfTensor_cbitxor,
    c_THHalfTensor_addcmul,
    c_THHalfTensor_addcdiv,
    c_THHalfTensor_addmv,
    c_THHalfTensor_addmm,
    c_THHalfTensor_addr,
    c_THHalfTensor_addbmm,
    c_THHalfTensor_baddbmm,
    c_THHalfTensor_match,
    c_THHalfTensor_numel,
    c_THHalfTensor_max,
    c_THHalfTensor_min,
    c_THHalfTensor_kthvalue,
    c_THHalfTensor_mode,
    c_THHalfTensor_median,
    c_THHalfTensor_sum,
    c_THHalfTensor_prod,
    c_THHalfTensor_cumsum,
    c_THHalfTensor_cumprod,
    c_THHalfTensor_sign,
    c_THHalfTensor_trace,
    c_THHalfTensor_cross,
    c_THHalfTensor_cmax,
    c_THHalfTensor_cmin,
    c_THHalfTensor_cmaxValue,
    c_THHalfTensor_cminValue,
    c_THHalfTensor_zeros,
    c_THHalfTensor_zerosLike,
    c_THHalfTensor_ones,
    c_THHalfTensor_onesLike,
    c_THHalfTensor_diag,
    c_THHalfTensor_eye,
    c_THHalfTensor_arange,
    c_THHalfTensor_range,
    c_THHalfTensor_randperm,
    c_THHalfTensor_reshape,
    c_THHalfTensor_sort,
    c_THHalfTensor_topk,
    c_THHalfTensor_tril,
    c_THHalfTensor_triu,
    c_THHalfTensor_cat,
    c_THHalfTensor_catArray,
    c_THHalfTensor_equal,
    c_THHalfTensor_ltValue,
    c_THHalfTensor_leValue,
    c_THHalfTensor_gtValue,
    c_THHalfTensor_geValue,
    c_THHalfTensor_neValue,
    c_THHalfTensor_eqValue,
    c_THHalfTensor_ltValueT,
    c_THHalfTensor_leValueT,
    c_THHalfTensor_gtValueT,
    c_THHalfTensor_geValueT,
    c_THHalfTensor_neValueT,
    c_THHalfTensor_eqValueT,
    c_THHalfTensor_ltTensor,
    c_THHalfTensor_leTensor,
    c_THHalfTensor_gtTensor,
    c_THHalfTensor_geTensor,
    c_THHalfTensor_neTensor,
    c_THHalfTensor_eqTensor,
    c_THHalfTensor_ltTensorT,
    c_THHalfTensor_leTensorT,
    c_THHalfTensor_gtTensorT,
    c_THHalfTensor_geTensorT,
    c_THHalfTensor_neTensorT,
    c_THHalfTensor_eqTensorT,
    c_THHalfTensor_abs,
    c_THHalfTensor_sigmoid,
    c_THHalfTensor_log,
    c_THHalfTensor_lgamma,
    c_THHalfTensor_log1p,
    c_THHalfTensor_exp,
    c_THHalfTensor_cos,
    c_THHalfTensor_acos,
    c_THHalfTensor_cosh,
    c_THHalfTensor_sin,
    c_THHalfTensor_asin,
    c_THHalfTensor_sinh,
    c_THHalfTensor_tan,
    c_THHalfTensor_atan,
    c_THHalfTensor_atan2,
    c_THHalfTensor_tanh,
    c_THHalfTensor_pow,
    c_THHalfTensor_tpow,
    c_THHalfTensor_sqrt,
    c_THHalfTensor_rsqrt,
    c_THHalfTensor_ceil,
    c_THHalfTensor_floor,
    c_THHalfTensor_round,
    c_THHalfTensor_trunc,
    c_THHalfTensor_frac,
    c_THHalfTensor_lerp,
    c_THHalfTensor_mean,
    c_THHalfTensor_std,
    c_THHalfTensor_var,
    c_THHalfTensor_norm,
    c_THHalfTensor_renorm,
    c_THHalfTensor_dist,
    c_THHalfTensor_histc,
    c_THHalfTensor_bhistc,
    c_THHalfTensor_meanall,
    c_THHalfTensor_varall,
    c_THHalfTensor_stdall,
    c_THHalfTensor_normall,
    c_THHalfTensor_linspace,
    c_THHalfTensor_logspace,
    c_THHalfTensor_rand,
    c_THHalfTensor_randn,
    c_THHalfTensor_logicalall,
    c_THHalfTensor_logicalany) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THHalfTensor_fill : r_ value -> void
foreign import ccall "THTensorMath.h THHalfTensor_fill"
  c_THHalfTensor_fill :: (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_zero : r_ -> void
foreign import ccall "THTensorMath.h THHalfTensor_zero"
  c_THHalfTensor_zero :: (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_maskedFill : tensor mask value -> void
foreign import ccall "THTensorMath.h THHalfTensor_maskedFill"
  c_THHalfTensor_maskedFill :: (Ptr CTHHalfTensor) -> Ptr CTHByteTensor -> THHalf -> IO ()

-- |c_THHalfTensor_maskedCopy : tensor mask src -> void
foreign import ccall "THTensorMath.h THHalfTensor_maskedCopy"
  c_THHalfTensor_maskedCopy :: (Ptr CTHHalfTensor) -> Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_maskedSelect : tensor src mask -> void
foreign import ccall "THTensorMath.h THHalfTensor_maskedSelect"
  c_THHalfTensor_maskedSelect :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THHalfTensor_nonzero : subscript tensor -> void
foreign import ccall "THTensorMath.h THHalfTensor_nonzero"
  c_THHalfTensor_nonzero :: Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_indexSelect : tensor src dim index -> void
foreign import ccall "THTensorMath.h THHalfTensor_indexSelect"
  c_THHalfTensor_indexSelect :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THHalfTensor_indexCopy : tensor dim index src -> void
foreign import ccall "THTensorMath.h THHalfTensor_indexCopy"
  c_THHalfTensor_indexCopy :: (Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_indexAdd : tensor dim index src -> void
foreign import ccall "THTensorMath.h THHalfTensor_indexAdd"
  c_THHalfTensor_indexAdd :: (Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_indexFill : tensor dim index val -> void
foreign import ccall "THTensorMath.h THHalfTensor_indexFill"
  c_THHalfTensor_indexFill :: (Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> THHalf -> IO ()

-- |c_THHalfTensor_gather : tensor src dim index -> void
foreign import ccall "THTensorMath.h THHalfTensor_gather"
  c_THHalfTensor_gather :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THHalfTensor_scatter : tensor dim index src -> void
foreign import ccall "THTensorMath.h THHalfTensor_scatter"
  c_THHalfTensor_scatter :: (Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_scatterAdd : tensor dim index src -> void
foreign import ccall "THTensorMath.h THHalfTensor_scatterAdd"
  c_THHalfTensor_scatterAdd :: (Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_scatterFill : tensor dim index val -> void
foreign import ccall "THTensorMath.h THHalfTensor_scatterFill"
  c_THHalfTensor_scatterFill :: (Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> THHalf -> IO ()

-- |c_THHalfTensor_dot : t src -> accreal
foreign import ccall "THTensorMath.h THHalfTensor_dot"
  c_THHalfTensor_dot :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CFloat

-- |c_THHalfTensor_minall : t -> real
foreign import ccall "THTensorMath.h THHalfTensor_minall"
  c_THHalfTensor_minall :: (Ptr CTHHalfTensor) -> THHalf

-- |c_THHalfTensor_maxall : t -> real
foreign import ccall "THTensorMath.h THHalfTensor_maxall"
  c_THHalfTensor_maxall :: (Ptr CTHHalfTensor) -> THHalf

-- |c_THHalfTensor_medianall : t -> real
foreign import ccall "THTensorMath.h THHalfTensor_medianall"
  c_THHalfTensor_medianall :: (Ptr CTHHalfTensor) -> THHalf

-- |c_THHalfTensor_sumall : t -> accreal
foreign import ccall "THTensorMath.h THHalfTensor_sumall"
  c_THHalfTensor_sumall :: (Ptr CTHHalfTensor) -> CFloat

-- |c_THHalfTensor_prodall : t -> accreal
foreign import ccall "THTensorMath.h THHalfTensor_prodall"
  c_THHalfTensor_prodall :: (Ptr CTHHalfTensor) -> CFloat

-- |c_THHalfTensor_neg : self src -> void
foreign import ccall "THTensorMath.h THHalfTensor_neg"
  c_THHalfTensor_neg :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_cinv : self src -> void
foreign import ccall "THTensorMath.h THHalfTensor_cinv"
  c_THHalfTensor_cinv :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_add : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_add"
  c_THHalfTensor_add :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_sub : self src value -> void
foreign import ccall "THTensorMath.h THHalfTensor_sub"
  c_THHalfTensor_sub :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_mul : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_mul"
  c_THHalfTensor_mul :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_div : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_div"
  c_THHalfTensor_div :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_lshift : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_lshift"
  c_THHalfTensor_lshift :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_rshift : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_rshift"
  c_THHalfTensor_rshift :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_fmod : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_fmod"
  c_THHalfTensor_fmod :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_remainder : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_remainder"
  c_THHalfTensor_remainder :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_clamp : r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h THHalfTensor_clamp"
  c_THHalfTensor_clamp :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> THHalf -> IO ()

-- |c_THHalfTensor_bitand : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_bitand"
  c_THHalfTensor_bitand :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_bitor : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_bitor"
  c_THHalfTensor_bitor :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_bitxor : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_bitxor"
  c_THHalfTensor_bitxor :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_cadd : r_ t value src -> void
foreign import ccall "THTensorMath.h THHalfTensor_cadd"
  c_THHalfTensor_cadd :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_csub : self src1 value src2 -> void
foreign import ccall "THTensorMath.h THHalfTensor_csub"
  c_THHalfTensor_csub :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_cmul : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensor_cmul"
  c_THHalfTensor_cmul :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_cpow : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensor_cpow"
  c_THHalfTensor_cpow :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_cdiv : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensor_cdiv"
  c_THHalfTensor_cdiv :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_clshift : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensor_clshift"
  c_THHalfTensor_clshift :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_crshift : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensor_crshift"
  c_THHalfTensor_crshift :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_cfmod : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensor_cfmod"
  c_THHalfTensor_cfmod :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_cremainder : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensor_cremainder"
  c_THHalfTensor_cremainder :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_cbitand : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensor_cbitand"
  c_THHalfTensor_cbitand :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_cbitor : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensor_cbitor"
  c_THHalfTensor_cbitor :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_cbitxor : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensor_cbitxor"
  c_THHalfTensor_cbitxor :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_addcmul : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THHalfTensor_addcmul"
  c_THHalfTensor_addcmul :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_addcdiv : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THHalfTensor_addcdiv"
  c_THHalfTensor_addcdiv :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_addmv : r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h THHalfTensor_addmv"
  c_THHalfTensor_addmv :: (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_addmm : r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h THHalfTensor_addmm"
  c_THHalfTensor_addmm :: (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_addr : r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h THHalfTensor_addr"
  c_THHalfTensor_addr :: (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_addbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THHalfTensor_addbmm"
  c_THHalfTensor_addbmm :: (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_baddbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THHalfTensor_baddbmm"
  c_THHalfTensor_baddbmm :: (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_match : r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h THHalfTensor_match"
  c_THHalfTensor_match :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_numel : t -> THStorage *
foreign import ccall "THTensorMath.h THHalfTensor_numel"
  c_THHalfTensor_numel :: (Ptr CTHHalfTensor) -> IO (Ptr CTHHalfStorage)

-- |c_THHalfTensor_max : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensor_max"
  c_THHalfTensor_max :: (Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_min : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensor_min"
  c_THHalfTensor_min :: (Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_kthvalue : values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensor_kthvalue"
  c_THHalfTensor_kthvalue :: (Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CLong -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_mode : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensor_mode"
  c_THHalfTensor_mode :: (Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_median : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensor_median"
  c_THHalfTensor_median :: (Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_sum : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensor_sum"
  c_THHalfTensor_sum :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_prod : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensor_prod"
  c_THHalfTensor_prod :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_cumsum : r_ t dimension -> void
foreign import ccall "THTensorMath.h THHalfTensor_cumsum"
  c_THHalfTensor_cumsum :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> IO ()

-- |c_THHalfTensor_cumprod : r_ t dimension -> void
foreign import ccall "THTensorMath.h THHalfTensor_cumprod"
  c_THHalfTensor_cumprod :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> IO ()

-- |c_THHalfTensor_sign : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensor_sign"
  c_THHalfTensor_sign :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_trace : t -> accreal
foreign import ccall "THTensorMath.h THHalfTensor_trace"
  c_THHalfTensor_trace :: (Ptr CTHHalfTensor) -> CFloat

-- |c_THHalfTensor_cross : r_ a b dimension -> void
foreign import ccall "THTensorMath.h THHalfTensor_cross"
  c_THHalfTensor_cross :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> IO ()

-- |c_THHalfTensor_cmax : r t src -> void
foreign import ccall "THTensorMath.h THHalfTensor_cmax"
  c_THHalfTensor_cmax :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_cmin : r t src -> void
foreign import ccall "THTensorMath.h THHalfTensor_cmin"
  c_THHalfTensor_cmin :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_cmaxValue : r t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_cmaxValue"
  c_THHalfTensor_cmaxValue :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_cminValue : r t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_cminValue"
  c_THHalfTensor_cminValue :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_zeros : r_ size -> void
foreign import ccall "THTensorMath.h THHalfTensor_zeros"
  c_THHalfTensor_zeros :: (Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THHalfTensor_zerosLike : r_ input -> void
foreign import ccall "THTensorMath.h THHalfTensor_zerosLike"
  c_THHalfTensor_zerosLike :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_ones : r_ size -> void
foreign import ccall "THTensorMath.h THHalfTensor_ones"
  c_THHalfTensor_ones :: (Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THHalfTensor_onesLike : r_ input -> void
foreign import ccall "THTensorMath.h THHalfTensor_onesLike"
  c_THHalfTensor_onesLike :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_diag : r_ t k -> void
foreign import ccall "THTensorMath.h THHalfTensor_diag"
  c_THHalfTensor_diag :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> IO ()

-- |c_THHalfTensor_eye : r_ n m -> void
foreign import ccall "THTensorMath.h THHalfTensor_eye"
  c_THHalfTensor_eye :: (Ptr CTHHalfTensor) -> CLong -> CLong -> IO ()

-- |c_THHalfTensor_arange : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THHalfTensor_arange"
  c_THHalfTensor_arange :: (Ptr CTHHalfTensor) -> CFloat -> CFloat -> CFloat -> IO ()

-- |c_THHalfTensor_range : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THHalfTensor_range"
  c_THHalfTensor_range :: (Ptr CTHHalfTensor) -> CFloat -> CFloat -> CFloat -> IO ()

-- |c_THHalfTensor_randperm : r_ _generator n -> void
foreign import ccall "THTensorMath.h THHalfTensor_randperm"
  c_THHalfTensor_randperm :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> CLong -> IO ()

-- |c_THHalfTensor_reshape : r_ t size -> void
foreign import ccall "THTensorMath.h THHalfTensor_reshape"
  c_THHalfTensor_reshape :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THHalfTensor_sort : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h THHalfTensor_sort"
  c_THHalfTensor_sort :: (Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_topk : rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h THHalfTensor_topk"
  c_THHalfTensor_topk :: (Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CLong -> CInt -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_tril : r_ t k -> void
foreign import ccall "THTensorMath.h THHalfTensor_tril"
  c_THHalfTensor_tril :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> IO ()

-- |c_THHalfTensor_triu : r_ t k -> void
foreign import ccall "THTensorMath.h THHalfTensor_triu"
  c_THHalfTensor_triu :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> IO ()

-- |c_THHalfTensor_cat : r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h THHalfTensor_cat"
  c_THHalfTensor_cat :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> IO ()

-- |c_THHalfTensor_catArray : result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h THHalfTensor_catArray"
  c_THHalfTensor_catArray :: (Ptr CTHHalfTensor) -> Ptr (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_equal : ta tb -> int
foreign import ccall "THTensorMath.h THHalfTensor_equal"
  c_THHalfTensor_equal :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt

-- |c_THHalfTensor_ltValue : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_ltValue"
  c_THHalfTensor_ltValue :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_leValue : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_leValue"
  c_THHalfTensor_leValue :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_gtValue : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_gtValue"
  c_THHalfTensor_gtValue :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_geValue : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_geValue"
  c_THHalfTensor_geValue :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_neValue : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_neValue"
  c_THHalfTensor_neValue :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_eqValue : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_eqValue"
  c_THHalfTensor_eqValue :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_ltValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_ltValueT"
  c_THHalfTensor_ltValueT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_leValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_leValueT"
  c_THHalfTensor_leValueT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_gtValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_gtValueT"
  c_THHalfTensor_gtValueT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_geValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_geValueT"
  c_THHalfTensor_geValueT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_neValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_neValueT"
  c_THHalfTensor_neValueT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_eqValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_eqValueT"
  c_THHalfTensor_eqValueT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_ltTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensor_ltTensor"
  c_THHalfTensor_ltTensor :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_leTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensor_leTensor"
  c_THHalfTensor_leTensor :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_gtTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensor_gtTensor"
  c_THHalfTensor_gtTensor :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_geTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensor_geTensor"
  c_THHalfTensor_geTensor :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_neTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensor_neTensor"
  c_THHalfTensor_neTensor :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_eqTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensor_eqTensor"
  c_THHalfTensor_eqTensor :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_ltTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensor_ltTensorT"
  c_THHalfTensor_ltTensorT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_leTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensor_leTensorT"
  c_THHalfTensor_leTensorT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_gtTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensor_gtTensorT"
  c_THHalfTensor_gtTensorT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_geTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensor_geTensorT"
  c_THHalfTensor_geTensorT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_neTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensor_neTensorT"
  c_THHalfTensor_neTensorT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_eqTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensor_eqTensorT"
  c_THHalfTensor_eqTensorT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_abs : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensor_abs"
  c_THHalfTensor_abs :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_sigmoid : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensor_sigmoid"
  c_THHalfTensor_sigmoid :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_log : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensor_log"
  c_THHalfTensor_log :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_lgamma : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensor_lgamma"
  c_THHalfTensor_lgamma :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_log1p : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensor_log1p"
  c_THHalfTensor_log1p :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_exp : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensor_exp"
  c_THHalfTensor_exp :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_cos : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensor_cos"
  c_THHalfTensor_cos :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_acos : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensor_acos"
  c_THHalfTensor_acos :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_cosh : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensor_cosh"
  c_THHalfTensor_cosh :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_sin : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensor_sin"
  c_THHalfTensor_sin :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_asin : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensor_asin"
  c_THHalfTensor_asin :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_sinh : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensor_sinh"
  c_THHalfTensor_sinh :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_tan : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensor_tan"
  c_THHalfTensor_tan :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_atan : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensor_atan"
  c_THHalfTensor_atan :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_atan2 : r_ tx ty -> void
foreign import ccall "THTensorMath.h THHalfTensor_atan2"
  c_THHalfTensor_atan2 :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_tanh : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensor_tanh"
  c_THHalfTensor_tanh :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_pow : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_pow"
  c_THHalfTensor_pow :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_tpow : r_ value t -> void
foreign import ccall "THTensorMath.h THHalfTensor_tpow"
  c_THHalfTensor_tpow :: (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_sqrt : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensor_sqrt"
  c_THHalfTensor_sqrt :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_rsqrt : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensor_rsqrt"
  c_THHalfTensor_rsqrt :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_ceil : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensor_ceil"
  c_THHalfTensor_ceil :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_floor : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensor_floor"
  c_THHalfTensor_floor :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_round : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensor_round"
  c_THHalfTensor_round :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_trunc : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensor_trunc"
  c_THHalfTensor_trunc :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_frac : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensor_frac"
  c_THHalfTensor_frac :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_lerp : r_ a b weight -> void
foreign import ccall "THTensorMath.h THHalfTensor_lerp"
  c_THHalfTensor_lerp :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_mean : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensor_mean"
  c_THHalfTensor_mean :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_std : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensor_std"
  c_THHalfTensor_std :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_var : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensor_var"
  c_THHalfTensor_var :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_norm : r_ t value dimension keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensor_norm"
  c_THHalfTensor_norm :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_renorm : r_ t value dimension maxnorm -> void
foreign import ccall "THTensorMath.h THHalfTensor_renorm"
  c_THHalfTensor_renorm :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> CInt -> THHalf -> IO ()

-- |c_THHalfTensor_dist : a b value -> accreal
foreign import ccall "THTensorMath.h THHalfTensor_dist"
  c_THHalfTensor_dist :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> CFloat

-- |c_THHalfTensor_histc : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THHalfTensor_histc"
  c_THHalfTensor_histc :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> THHalf -> THHalf -> IO ()

-- |c_THHalfTensor_bhistc : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THHalfTensor_bhistc"
  c_THHalfTensor_bhistc :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> THHalf -> THHalf -> IO ()

-- |c_THHalfTensor_meanall : self -> accreal
foreign import ccall "THTensorMath.h THHalfTensor_meanall"
  c_THHalfTensor_meanall :: (Ptr CTHHalfTensor) -> CFloat

-- |c_THHalfTensor_varall : self biased -> accreal
foreign import ccall "THTensorMath.h THHalfTensor_varall"
  c_THHalfTensor_varall :: (Ptr CTHHalfTensor) -> CInt -> CFloat

-- |c_THHalfTensor_stdall : self biased -> accreal
foreign import ccall "THTensorMath.h THHalfTensor_stdall"
  c_THHalfTensor_stdall :: (Ptr CTHHalfTensor) -> CInt -> CFloat

-- |c_THHalfTensor_normall : t value -> accreal
foreign import ccall "THTensorMath.h THHalfTensor_normall"
  c_THHalfTensor_normall :: (Ptr CTHHalfTensor) -> THHalf -> CFloat

-- |c_THHalfTensor_linspace : r_ a b n -> void
foreign import ccall "THTensorMath.h THHalfTensor_linspace"
  c_THHalfTensor_linspace :: (Ptr CTHHalfTensor) -> THHalf -> THHalf -> CLong -> IO ()

-- |c_THHalfTensor_logspace : r_ a b n -> void
foreign import ccall "THTensorMath.h THHalfTensor_logspace"
  c_THHalfTensor_logspace :: (Ptr CTHHalfTensor) -> THHalf -> THHalf -> CLong -> IO ()

-- |c_THHalfTensor_rand : r_ _generator size -> void
foreign import ccall "THTensorMath.h THHalfTensor_rand"
  c_THHalfTensor_rand :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()

-- |c_THHalfTensor_randn : r_ _generator size -> void
foreign import ccall "THTensorMath.h THHalfTensor_randn"
  c_THHalfTensor_randn :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()

-- |c_THHalfTensor_logicalall : self -> int
foreign import ccall "THTensorMath.h THHalfTensor_logicalall"
  c_THHalfTensor_logicalall :: (Ptr CTHHalfTensor) -> CInt

-- |c_THHalfTensor_logicalany : self -> int
foreign import ccall "THTensorMath.h THHalfTensor_logicalany"
  c_THHalfTensor_logicalany :: (Ptr CTHHalfTensor) -> CInt