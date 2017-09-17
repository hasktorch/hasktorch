{-# LANGUAGE ForeignFunctionInterface#-}

module THHalfTensorMath (
    c_THHalfTensorMath_fill,
    c_THHalfTensorMath_zero,
    c_THHalfTensorMath_maskedFill,
    c_THHalfTensorMath_maskedCopy,
    c_THHalfTensorMath_maskedSelect,
    c_THHalfTensorMath_nonzero,
    c_THHalfTensorMath_indexSelect,
    c_THHalfTensorMath_indexCopy,
    c_THHalfTensorMath_indexAdd,
    c_THHalfTensorMath_indexFill,
    c_THHalfTensorMath_gather,
    c_THHalfTensorMath_scatter,
    c_THHalfTensorMath_scatterAdd,
    c_THHalfTensorMath_scatterFill,
    c_THHalfTensorMath_dot,
    c_THHalfTensorMath_minall,
    c_THHalfTensorMath_maxall,
    c_THHalfTensorMath_medianall,
    c_THHalfTensorMath_sumall,
    c_THHalfTensorMath_prodall,
    c_THHalfTensorMath_neg,
    c_THHalfTensorMath_cinv,
    c_THHalfTensorMath_add,
    c_THHalfTensorMath_sub,
    c_THHalfTensorMath_mul,
    c_THHalfTensorMath_div,
    c_THHalfTensorMath_lshift,
    c_THHalfTensorMath_rshift,
    c_THHalfTensorMath_fmod,
    c_THHalfTensorMath_remainder,
    c_THHalfTensorMath_clamp,
    c_THHalfTensorMath_bitand,
    c_THHalfTensorMath_bitor,
    c_THHalfTensorMath_bitxor,
    c_THHalfTensorMath_cadd,
    c_THHalfTensorMath_csub,
    c_THHalfTensorMath_cmul,
    c_THHalfTensorMath_cpow,
    c_THHalfTensorMath_cdiv,
    c_THHalfTensorMath_clshift,
    c_THHalfTensorMath_crshift,
    c_THHalfTensorMath_cfmod,
    c_THHalfTensorMath_cremainder,
    c_THHalfTensorMath_cbitand,
    c_THHalfTensorMath_cbitor,
    c_THHalfTensorMath_cbitxor,
    c_THHalfTensorMath_addcmul,
    c_THHalfTensorMath_addcdiv,
    c_THHalfTensorMath_addmv,
    c_THHalfTensorMath_addmm,
    c_THHalfTensorMath_addr,
    c_THHalfTensorMath_addbmm,
    c_THHalfTensorMath_baddbmm,
    c_THHalfTensorMath_match,
    c_THHalfTensorMath_numel,
    c_THHalfTensorMath_max,
    c_THHalfTensorMath_min,
    c_THHalfTensorMath_kthvalue,
    c_THHalfTensorMath_mode,
    c_THHalfTensorMath_median,
    c_THHalfTensorMath_sum,
    c_THHalfTensorMath_prod,
    c_THHalfTensorMath_cumsum,
    c_THHalfTensorMath_cumprod,
    c_THHalfTensorMath_sign,
    c_THHalfTensorMath_trace,
    c_THHalfTensorMath_cross,
    c_THHalfTensorMath_cmax,
    c_THHalfTensorMath_cmin,
    c_THHalfTensorMath_cmaxValue,
    c_THHalfTensorMath_cminValue,
    c_THHalfTensorMath_zeros,
    c_THHalfTensorMath_zerosLike,
    c_THHalfTensorMath_ones,
    c_THHalfTensorMath_onesLike,
    c_THHalfTensorMath_diag,
    c_THHalfTensorMath_eye,
    c_THHalfTensorMath_arange,
    c_THHalfTensorMath_range,
    c_THHalfTensorMath_randperm,
    c_THHalfTensorMath_reshape,
    c_THHalfTensorMath_sort,
    c_THHalfTensorMath_topk,
    c_THHalfTensorMath_tril,
    c_THHalfTensorMath_triu,
    c_THHalfTensorMath_cat,
    c_THHalfTensorMath_catArray,
    c_THHalfTensorMath_equal,
    c_THHalfTensorMath_ltValue,
    c_THHalfTensorMath_leValue,
    c_THHalfTensorMath_gtValue,
    c_THHalfTensorMath_geValue,
    c_THHalfTensorMath_neValue,
    c_THHalfTensorMath_eqValue,
    c_THHalfTensorMath_ltValueT,
    c_THHalfTensorMath_leValueT,
    c_THHalfTensorMath_gtValueT,
    c_THHalfTensorMath_geValueT,
    c_THHalfTensorMath_neValueT,
    c_THHalfTensorMath_eqValueT,
    c_THHalfTensorMath_ltTensor,
    c_THHalfTensorMath_leTensor,
    c_THHalfTensorMath_gtTensor,
    c_THHalfTensorMath_geTensor,
    c_THHalfTensorMath_neTensor,
    c_THHalfTensorMath_eqTensor,
    c_THHalfTensorMath_ltTensorT,
    c_THHalfTensorMath_leTensorT,
    c_THHalfTensorMath_gtTensorT,
    c_THHalfTensorMath_geTensorT,
    c_THHalfTensorMath_neTensorT,
    c_THHalfTensorMath_eqTensorT,
    c_THHalfTensorMath_abs,
    c_THHalfTensorMath_sigmoid,
    c_THHalfTensorMath_log,
    c_THHalfTensorMath_lgamma,
    c_THHalfTensorMath_log1p,
    c_THHalfTensorMath_exp,
    c_THHalfTensorMath_cos,
    c_THHalfTensorMath_acos,
    c_THHalfTensorMath_cosh,
    c_THHalfTensorMath_sin,
    c_THHalfTensorMath_asin,
    c_THHalfTensorMath_sinh,
    c_THHalfTensorMath_tan,
    c_THHalfTensorMath_atan,
    c_THHalfTensorMath_atan2,
    c_THHalfTensorMath_tanh,
    c_THHalfTensorMath_pow,
    c_THHalfTensorMath_tpow,
    c_THHalfTensorMath_sqrt,
    c_THHalfTensorMath_rsqrt,
    c_THHalfTensorMath_ceil,
    c_THHalfTensorMath_floor,
    c_THHalfTensorMath_round,
    c_THHalfTensorMath_trunc,
    c_THHalfTensorMath_frac,
    c_THHalfTensorMath_lerp,
    c_THHalfTensorMath_mean,
    c_THHalfTensorMath_std,
    c_THHalfTensorMath_var,
    c_THHalfTensorMath_norm,
    c_THHalfTensorMath_renorm,
    c_THHalfTensorMath_dist,
    c_THHalfTensorMath_histc,
    c_THHalfTensorMath_bhistc,
    c_THHalfTensorMath_meanall,
    c_THHalfTensorMath_varall,
    c_THHalfTensorMath_stdall,
    c_THHalfTensorMath_normall,
    c_THHalfTensorMath_linspace,
    c_THHalfTensorMath_logspace,
    c_THHalfTensorMath_rand,
    c_THHalfTensorMath_randn,
    c_THHalfTensorMath_logicalall,
    c_THHalfTensorMath_logicalany) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THHalfTensorMath_fill : r_ value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_fill"
  c_THHalfTensorMath_fill :: (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_zero : r_ -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_zero"
  c_THHalfTensorMath_zero :: (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_maskedFill : tensor mask value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_maskedFill"
  c_THHalfTensorMath_maskedFill :: (Ptr CTHHalfTensor) -> Ptr CTHByteTensor -> THHalf -> IO ()

-- |c_THHalfTensorMath_maskedCopy : tensor mask src -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_maskedCopy"
  c_THHalfTensorMath_maskedCopy :: (Ptr CTHHalfTensor) -> Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_maskedSelect : tensor src mask -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_maskedSelect"
  c_THHalfTensorMath_maskedSelect :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THHalfTensorMath_nonzero : subscript tensor -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_nonzero"
  c_THHalfTensorMath_nonzero :: Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_indexSelect : tensor src dim index -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_indexSelect"
  c_THHalfTensorMath_indexSelect :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THHalfTensorMath_indexCopy : tensor dim index src -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_indexCopy"
  c_THHalfTensorMath_indexCopy :: (Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_indexAdd : tensor dim index src -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_indexAdd"
  c_THHalfTensorMath_indexAdd :: (Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_indexFill : tensor dim index val -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_indexFill"
  c_THHalfTensorMath_indexFill :: (Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> THHalf -> IO ()

-- |c_THHalfTensorMath_gather : tensor src dim index -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_gather"
  c_THHalfTensorMath_gather :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THHalfTensorMath_scatter : tensor dim index src -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_scatter"
  c_THHalfTensorMath_scatter :: (Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_scatterAdd : tensor dim index src -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_scatterAdd"
  c_THHalfTensorMath_scatterAdd :: (Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_scatterFill : tensor dim index val -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_scatterFill"
  c_THHalfTensorMath_scatterFill :: (Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> THHalf -> IO ()

-- |c_THHalfTensorMath_dot : t src -> accreal
foreign import ccall "THTensorMath.h THHalfTensorMath_dot"
  c_THHalfTensorMath_dot :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CFloat

-- |c_THHalfTensorMath_minall : t -> real
foreign import ccall "THTensorMath.h THHalfTensorMath_minall"
  c_THHalfTensorMath_minall :: (Ptr CTHHalfTensor) -> THHalf

-- |c_THHalfTensorMath_maxall : t -> real
foreign import ccall "THTensorMath.h THHalfTensorMath_maxall"
  c_THHalfTensorMath_maxall :: (Ptr CTHHalfTensor) -> THHalf

-- |c_THHalfTensorMath_medianall : t -> real
foreign import ccall "THTensorMath.h THHalfTensorMath_medianall"
  c_THHalfTensorMath_medianall :: (Ptr CTHHalfTensor) -> THHalf

-- |c_THHalfTensorMath_sumall : t -> accreal
foreign import ccall "THTensorMath.h THHalfTensorMath_sumall"
  c_THHalfTensorMath_sumall :: (Ptr CTHHalfTensor) -> CFloat

-- |c_THHalfTensorMath_prodall : t -> accreal
foreign import ccall "THTensorMath.h THHalfTensorMath_prodall"
  c_THHalfTensorMath_prodall :: (Ptr CTHHalfTensor) -> CFloat

-- |c_THHalfTensorMath_neg : self src -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_neg"
  c_THHalfTensorMath_neg :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_cinv : self src -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_cinv"
  c_THHalfTensorMath_cinv :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_add : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_add"
  c_THHalfTensorMath_add :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_sub : self src value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_sub"
  c_THHalfTensorMath_sub :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_mul : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_mul"
  c_THHalfTensorMath_mul :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_div : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_div"
  c_THHalfTensorMath_div :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_lshift : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_lshift"
  c_THHalfTensorMath_lshift :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_rshift : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_rshift"
  c_THHalfTensorMath_rshift :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_fmod : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_fmod"
  c_THHalfTensorMath_fmod :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_remainder : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_remainder"
  c_THHalfTensorMath_remainder :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_clamp : r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_clamp"
  c_THHalfTensorMath_clamp :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> THHalf -> IO ()

-- |c_THHalfTensorMath_bitand : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_bitand"
  c_THHalfTensorMath_bitand :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_bitor : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_bitor"
  c_THHalfTensorMath_bitor :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_bitxor : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_bitxor"
  c_THHalfTensorMath_bitxor :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_cadd : r_ t value src -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_cadd"
  c_THHalfTensorMath_cadd :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_csub : self src1 value src2 -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_csub"
  c_THHalfTensorMath_csub :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_cmul : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_cmul"
  c_THHalfTensorMath_cmul :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_cpow : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_cpow"
  c_THHalfTensorMath_cpow :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_cdiv : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_cdiv"
  c_THHalfTensorMath_cdiv :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_clshift : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_clshift"
  c_THHalfTensorMath_clshift :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_crshift : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_crshift"
  c_THHalfTensorMath_crshift :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_cfmod : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_cfmod"
  c_THHalfTensorMath_cfmod :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_cremainder : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_cremainder"
  c_THHalfTensorMath_cremainder :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_cbitand : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_cbitand"
  c_THHalfTensorMath_cbitand :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_cbitor : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_cbitor"
  c_THHalfTensorMath_cbitor :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_cbitxor : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_cbitxor"
  c_THHalfTensorMath_cbitxor :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_addcmul : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_addcmul"
  c_THHalfTensorMath_addcmul :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_addcdiv : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_addcdiv"
  c_THHalfTensorMath_addcdiv :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_addmv : r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_addmv"
  c_THHalfTensorMath_addmv :: (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_addmm : r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_addmm"
  c_THHalfTensorMath_addmm :: (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_addr : r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_addr"
  c_THHalfTensorMath_addr :: (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_addbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_addbmm"
  c_THHalfTensorMath_addbmm :: (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_baddbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_baddbmm"
  c_THHalfTensorMath_baddbmm :: (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_match : r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_match"
  c_THHalfTensorMath_match :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_numel : t -> THStorage *
foreign import ccall "THTensorMath.h THHalfTensorMath_numel"
  c_THHalfTensorMath_numel :: (Ptr CTHHalfTensor) -> IO (Ptr CTHHalfStorage)

-- |c_THHalfTensorMath_max : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_max"
  c_THHalfTensorMath_max :: (Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensorMath_min : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_min"
  c_THHalfTensorMath_min :: (Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensorMath_kthvalue : values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_kthvalue"
  c_THHalfTensorMath_kthvalue :: (Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CLong -> CInt -> CInt -> IO ()

-- |c_THHalfTensorMath_mode : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_mode"
  c_THHalfTensorMath_mode :: (Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensorMath_median : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_median"
  c_THHalfTensorMath_median :: (Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensorMath_sum : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_sum"
  c_THHalfTensorMath_sum :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensorMath_prod : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_prod"
  c_THHalfTensorMath_prod :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensorMath_cumsum : r_ t dimension -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_cumsum"
  c_THHalfTensorMath_cumsum :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> IO ()

-- |c_THHalfTensorMath_cumprod : r_ t dimension -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_cumprod"
  c_THHalfTensorMath_cumprod :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> IO ()

-- |c_THHalfTensorMath_sign : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_sign"
  c_THHalfTensorMath_sign :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_trace : t -> accreal
foreign import ccall "THTensorMath.h THHalfTensorMath_trace"
  c_THHalfTensorMath_trace :: (Ptr CTHHalfTensor) -> CFloat

-- |c_THHalfTensorMath_cross : r_ a b dimension -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_cross"
  c_THHalfTensorMath_cross :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> IO ()

-- |c_THHalfTensorMath_cmax : r t src -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_cmax"
  c_THHalfTensorMath_cmax :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_cmin : r t src -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_cmin"
  c_THHalfTensorMath_cmin :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_cmaxValue : r t value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_cmaxValue"
  c_THHalfTensorMath_cmaxValue :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_cminValue : r t value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_cminValue"
  c_THHalfTensorMath_cminValue :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_zeros : r_ size -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_zeros"
  c_THHalfTensorMath_zeros :: (Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THHalfTensorMath_zerosLike : r_ input -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_zerosLike"
  c_THHalfTensorMath_zerosLike :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_ones : r_ size -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_ones"
  c_THHalfTensorMath_ones :: (Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THHalfTensorMath_onesLike : r_ input -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_onesLike"
  c_THHalfTensorMath_onesLike :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_diag : r_ t k -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_diag"
  c_THHalfTensorMath_diag :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> IO ()

-- |c_THHalfTensorMath_eye : r_ n m -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_eye"
  c_THHalfTensorMath_eye :: (Ptr CTHHalfTensor) -> CLong -> CLong -> IO ()

-- |c_THHalfTensorMath_arange : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_arange"
  c_THHalfTensorMath_arange :: (Ptr CTHHalfTensor) -> CFloat -> CFloat -> CFloat -> IO ()

-- |c_THHalfTensorMath_range : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_range"
  c_THHalfTensorMath_range :: (Ptr CTHHalfTensor) -> CFloat -> CFloat -> CFloat -> IO ()

-- |c_THHalfTensorMath_randperm : r_ _generator n -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_randperm"
  c_THHalfTensorMath_randperm :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> CLong -> IO ()

-- |c_THHalfTensorMath_reshape : r_ t size -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_reshape"
  c_THHalfTensorMath_reshape :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THHalfTensorMath_sort : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_sort"
  c_THHalfTensorMath_sort :: (Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensorMath_topk : rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_topk"
  c_THHalfTensorMath_topk :: (Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CLong -> CInt -> CInt -> CInt -> IO ()

-- |c_THHalfTensorMath_tril : r_ t k -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_tril"
  c_THHalfTensorMath_tril :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> IO ()

-- |c_THHalfTensorMath_triu : r_ t k -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_triu"
  c_THHalfTensorMath_triu :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> IO ()

-- |c_THHalfTensorMath_cat : r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_cat"
  c_THHalfTensorMath_cat :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> IO ()

-- |c_THHalfTensorMath_catArray : result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_catArray"
  c_THHalfTensorMath_catArray :: (Ptr CTHHalfTensor) -> Ptr (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensorMath_equal : ta tb -> int
foreign import ccall "THTensorMath.h THHalfTensorMath_equal"
  c_THHalfTensorMath_equal :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt

-- |c_THHalfTensorMath_ltValue : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_ltValue"
  c_THHalfTensorMath_ltValue :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_leValue : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_leValue"
  c_THHalfTensorMath_leValue :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_gtValue : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_gtValue"
  c_THHalfTensorMath_gtValue :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_geValue : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_geValue"
  c_THHalfTensorMath_geValue :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_neValue : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_neValue"
  c_THHalfTensorMath_neValue :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_eqValue : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_eqValue"
  c_THHalfTensorMath_eqValue :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_ltValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_ltValueT"
  c_THHalfTensorMath_ltValueT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_leValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_leValueT"
  c_THHalfTensorMath_leValueT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_gtValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_gtValueT"
  c_THHalfTensorMath_gtValueT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_geValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_geValueT"
  c_THHalfTensorMath_geValueT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_neValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_neValueT"
  c_THHalfTensorMath_neValueT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_eqValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_eqValueT"
  c_THHalfTensorMath_eqValueT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_ltTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_ltTensor"
  c_THHalfTensorMath_ltTensor :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_leTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_leTensor"
  c_THHalfTensorMath_leTensor :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_gtTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_gtTensor"
  c_THHalfTensorMath_gtTensor :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_geTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_geTensor"
  c_THHalfTensorMath_geTensor :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_neTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_neTensor"
  c_THHalfTensorMath_neTensor :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_eqTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_eqTensor"
  c_THHalfTensorMath_eqTensor :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_ltTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_ltTensorT"
  c_THHalfTensorMath_ltTensorT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_leTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_leTensorT"
  c_THHalfTensorMath_leTensorT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_gtTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_gtTensorT"
  c_THHalfTensorMath_gtTensorT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_geTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_geTensorT"
  c_THHalfTensorMath_geTensorT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_neTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_neTensorT"
  c_THHalfTensorMath_neTensorT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_eqTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_eqTensorT"
  c_THHalfTensorMath_eqTensorT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_abs : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_abs"
  c_THHalfTensorMath_abs :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_sigmoid : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_sigmoid"
  c_THHalfTensorMath_sigmoid :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_log : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_log"
  c_THHalfTensorMath_log :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_lgamma : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_lgamma"
  c_THHalfTensorMath_lgamma :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_log1p : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_log1p"
  c_THHalfTensorMath_log1p :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_exp : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_exp"
  c_THHalfTensorMath_exp :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_cos : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_cos"
  c_THHalfTensorMath_cos :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_acos : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_acos"
  c_THHalfTensorMath_acos :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_cosh : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_cosh"
  c_THHalfTensorMath_cosh :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_sin : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_sin"
  c_THHalfTensorMath_sin :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_asin : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_asin"
  c_THHalfTensorMath_asin :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_sinh : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_sinh"
  c_THHalfTensorMath_sinh :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_tan : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_tan"
  c_THHalfTensorMath_tan :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_atan : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_atan"
  c_THHalfTensorMath_atan :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_atan2 : r_ tx ty -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_atan2"
  c_THHalfTensorMath_atan2 :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_tanh : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_tanh"
  c_THHalfTensorMath_tanh :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_pow : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_pow"
  c_THHalfTensorMath_pow :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_tpow : r_ value t -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_tpow"
  c_THHalfTensorMath_tpow :: (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_sqrt : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_sqrt"
  c_THHalfTensorMath_sqrt :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_rsqrt : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_rsqrt"
  c_THHalfTensorMath_rsqrt :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_ceil : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_ceil"
  c_THHalfTensorMath_ceil :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_floor : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_floor"
  c_THHalfTensorMath_floor :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_round : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_round"
  c_THHalfTensorMath_round :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_trunc : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_trunc"
  c_THHalfTensorMath_trunc :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_frac : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_frac"
  c_THHalfTensorMath_frac :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensorMath_lerp : r_ a b weight -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_lerp"
  c_THHalfTensorMath_lerp :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensorMath_mean : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_mean"
  c_THHalfTensorMath_mean :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensorMath_std : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_std"
  c_THHalfTensorMath_std :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THHalfTensorMath_var : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_var"
  c_THHalfTensorMath_var :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THHalfTensorMath_norm : r_ t value dimension keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_norm"
  c_THHalfTensorMath_norm :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> CInt -> CInt -> IO ()

-- |c_THHalfTensorMath_renorm : r_ t value dimension maxnorm -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_renorm"
  c_THHalfTensorMath_renorm :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> CInt -> THHalf -> IO ()

-- |c_THHalfTensorMath_dist : a b value -> accreal
foreign import ccall "THTensorMath.h THHalfTensorMath_dist"
  c_THHalfTensorMath_dist :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> CFloat

-- |c_THHalfTensorMath_histc : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_histc"
  c_THHalfTensorMath_histc :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> THHalf -> THHalf -> IO ()

-- |c_THHalfTensorMath_bhistc : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_bhistc"
  c_THHalfTensorMath_bhistc :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> THHalf -> THHalf -> IO ()

-- |c_THHalfTensorMath_meanall : self -> accreal
foreign import ccall "THTensorMath.h THHalfTensorMath_meanall"
  c_THHalfTensorMath_meanall :: (Ptr CTHHalfTensor) -> CFloat

-- |c_THHalfTensorMath_varall : self biased -> accreal
foreign import ccall "THTensorMath.h THHalfTensorMath_varall"
  c_THHalfTensorMath_varall :: (Ptr CTHHalfTensor) -> CInt -> CFloat

-- |c_THHalfTensorMath_stdall : self biased -> accreal
foreign import ccall "THTensorMath.h THHalfTensorMath_stdall"
  c_THHalfTensorMath_stdall :: (Ptr CTHHalfTensor) -> CInt -> CFloat

-- |c_THHalfTensorMath_normall : t value -> accreal
foreign import ccall "THTensorMath.h THHalfTensorMath_normall"
  c_THHalfTensorMath_normall :: (Ptr CTHHalfTensor) -> THHalf -> CFloat

-- |c_THHalfTensorMath_linspace : r_ a b n -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_linspace"
  c_THHalfTensorMath_linspace :: (Ptr CTHHalfTensor) -> THHalf -> THHalf -> CLong -> IO ()

-- |c_THHalfTensorMath_logspace : r_ a b n -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_logspace"
  c_THHalfTensorMath_logspace :: (Ptr CTHHalfTensor) -> THHalf -> THHalf -> CLong -> IO ()

-- |c_THHalfTensorMath_rand : r_ _generator size -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_rand"
  c_THHalfTensorMath_rand :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()

-- |c_THHalfTensorMath_randn : r_ _generator size -> void
foreign import ccall "THTensorMath.h THHalfTensorMath_randn"
  c_THHalfTensorMath_randn :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()

-- |c_THHalfTensorMath_logicalall : self -> int
foreign import ccall "THTensorMath.h THHalfTensorMath_logicalall"
  c_THHalfTensorMath_logicalall :: (Ptr CTHHalfTensor) -> CInt

-- |c_THHalfTensorMath_logicalany : self -> int
foreign import ccall "THTensorMath.h THHalfTensorMath_logicalany"
  c_THHalfTensorMath_logicalany :: (Ptr CTHHalfTensor) -> CInt