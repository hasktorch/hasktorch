{-# LANGUAGE ForeignFunctionInterface#-}

module THFloatTensorMath (
    c_THFloatTensorMath_fill,
    c_THFloatTensorMath_zero,
    c_THFloatTensorMath_maskedFill,
    c_THFloatTensorMath_maskedCopy,
    c_THFloatTensorMath_maskedSelect,
    c_THFloatTensorMath_nonzero,
    c_THFloatTensorMath_indexSelect,
    c_THFloatTensorMath_indexCopy,
    c_THFloatTensorMath_indexAdd,
    c_THFloatTensorMath_indexFill,
    c_THFloatTensorMath_gather,
    c_THFloatTensorMath_scatter,
    c_THFloatTensorMath_scatterAdd,
    c_THFloatTensorMath_scatterFill,
    c_THFloatTensorMath_dot,
    c_THFloatTensorMath_minall,
    c_THFloatTensorMath_maxall,
    c_THFloatTensorMath_medianall,
    c_THFloatTensorMath_sumall,
    c_THFloatTensorMath_prodall,
    c_THFloatTensorMath_neg,
    c_THFloatTensorMath_cinv,
    c_THFloatTensorMath_add,
    c_THFloatTensorMath_sub,
    c_THFloatTensorMath_mul,
    c_THFloatTensorMath_div,
    c_THFloatTensorMath_lshift,
    c_THFloatTensorMath_rshift,
    c_THFloatTensorMath_fmod,
    c_THFloatTensorMath_remainder,
    c_THFloatTensorMath_clamp,
    c_THFloatTensorMath_bitand,
    c_THFloatTensorMath_bitor,
    c_THFloatTensorMath_bitxor,
    c_THFloatTensorMath_cadd,
    c_THFloatTensorMath_csub,
    c_THFloatTensorMath_cmul,
    c_THFloatTensorMath_cpow,
    c_THFloatTensorMath_cdiv,
    c_THFloatTensorMath_clshift,
    c_THFloatTensorMath_crshift,
    c_THFloatTensorMath_cfmod,
    c_THFloatTensorMath_cremainder,
    c_THFloatTensorMath_cbitand,
    c_THFloatTensorMath_cbitor,
    c_THFloatTensorMath_cbitxor,
    c_THFloatTensorMath_addcmul,
    c_THFloatTensorMath_addcdiv,
    c_THFloatTensorMath_addmv,
    c_THFloatTensorMath_addmm,
    c_THFloatTensorMath_addr,
    c_THFloatTensorMath_addbmm,
    c_THFloatTensorMath_baddbmm,
    c_THFloatTensorMath_match,
    c_THFloatTensorMath_numel,
    c_THFloatTensorMath_max,
    c_THFloatTensorMath_min,
    c_THFloatTensorMath_kthvalue,
    c_THFloatTensorMath_mode,
    c_THFloatTensorMath_median,
    c_THFloatTensorMath_sum,
    c_THFloatTensorMath_prod,
    c_THFloatTensorMath_cumsum,
    c_THFloatTensorMath_cumprod,
    c_THFloatTensorMath_sign,
    c_THFloatTensorMath_trace,
    c_THFloatTensorMath_cross,
    c_THFloatTensorMath_cmax,
    c_THFloatTensorMath_cmin,
    c_THFloatTensorMath_cmaxValue,
    c_THFloatTensorMath_cminValue,
    c_THFloatTensorMath_zeros,
    c_THFloatTensorMath_zerosLike,
    c_THFloatTensorMath_ones,
    c_THFloatTensorMath_onesLike,
    c_THFloatTensorMath_diag,
    c_THFloatTensorMath_eye,
    c_THFloatTensorMath_arange,
    c_THFloatTensorMath_range,
    c_THFloatTensorMath_randperm,
    c_THFloatTensorMath_reshape,
    c_THFloatTensorMath_sort,
    c_THFloatTensorMath_topk,
    c_THFloatTensorMath_tril,
    c_THFloatTensorMath_triu,
    c_THFloatTensorMath_cat,
    c_THFloatTensorMath_catArray,
    c_THFloatTensorMath_equal,
    c_THFloatTensorMath_ltValue,
    c_THFloatTensorMath_leValue,
    c_THFloatTensorMath_gtValue,
    c_THFloatTensorMath_geValue,
    c_THFloatTensorMath_neValue,
    c_THFloatTensorMath_eqValue,
    c_THFloatTensorMath_ltValueT,
    c_THFloatTensorMath_leValueT,
    c_THFloatTensorMath_gtValueT,
    c_THFloatTensorMath_geValueT,
    c_THFloatTensorMath_neValueT,
    c_THFloatTensorMath_eqValueT,
    c_THFloatTensorMath_ltTensor,
    c_THFloatTensorMath_leTensor,
    c_THFloatTensorMath_gtTensor,
    c_THFloatTensorMath_geTensor,
    c_THFloatTensorMath_neTensor,
    c_THFloatTensorMath_eqTensor,
    c_THFloatTensorMath_ltTensorT,
    c_THFloatTensorMath_leTensorT,
    c_THFloatTensorMath_gtTensorT,
    c_THFloatTensorMath_geTensorT,
    c_THFloatTensorMath_neTensorT,
    c_THFloatTensorMath_eqTensorT,
    c_THFloatTensorMath_abs,
    c_THFloatTensorMath_sigmoid,
    c_THFloatTensorMath_log,
    c_THFloatTensorMath_lgamma,
    c_THFloatTensorMath_log1p,
    c_THFloatTensorMath_exp,
    c_THFloatTensorMath_cos,
    c_THFloatTensorMath_acos,
    c_THFloatTensorMath_cosh,
    c_THFloatTensorMath_sin,
    c_THFloatTensorMath_asin,
    c_THFloatTensorMath_sinh,
    c_THFloatTensorMath_tan,
    c_THFloatTensorMath_atan,
    c_THFloatTensorMath_atan2,
    c_THFloatTensorMath_tanh,
    c_THFloatTensorMath_pow,
    c_THFloatTensorMath_tpow,
    c_THFloatTensorMath_sqrt,
    c_THFloatTensorMath_rsqrt,
    c_THFloatTensorMath_ceil,
    c_THFloatTensorMath_floor,
    c_THFloatTensorMath_round,
    c_THFloatTensorMath_trunc,
    c_THFloatTensorMath_frac,
    c_THFloatTensorMath_lerp,
    c_THFloatTensorMath_mean,
    c_THFloatTensorMath_std,
    c_THFloatTensorMath_var,
    c_THFloatTensorMath_norm,
    c_THFloatTensorMath_renorm,
    c_THFloatTensorMath_dist,
    c_THFloatTensorMath_histc,
    c_THFloatTensorMath_bhistc,
    c_THFloatTensorMath_meanall,
    c_THFloatTensorMath_varall,
    c_THFloatTensorMath_stdall,
    c_THFloatTensorMath_normall,
    c_THFloatTensorMath_linspace,
    c_THFloatTensorMath_logspace,
    c_THFloatTensorMath_rand,
    c_THFloatTensorMath_randn,
    c_THFloatTensorMath_logicalall,
    c_THFloatTensorMath_logicalany) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THFloatTensorMath_fill : r_ value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_fill"
  c_THFloatTensorMath_fill :: (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_zero : r_ -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_zero"
  c_THFloatTensorMath_zero :: (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_maskedFill : tensor mask value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_maskedFill"
  c_THFloatTensorMath_maskedFill :: (Ptr CTHFloatTensor) -> Ptr CTHByteTensor -> CFloat -> IO ()

-- |c_THFloatTensorMath_maskedCopy : tensor mask src -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_maskedCopy"
  c_THFloatTensorMath_maskedCopy :: (Ptr CTHFloatTensor) -> Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_maskedSelect : tensor src mask -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_maskedSelect"
  c_THFloatTensorMath_maskedSelect :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THFloatTensorMath_nonzero : subscript tensor -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_nonzero"
  c_THFloatTensorMath_nonzero :: Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_indexSelect : tensor src dim index -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_indexSelect"
  c_THFloatTensorMath_indexSelect :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THFloatTensorMath_indexCopy : tensor dim index src -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_indexCopy"
  c_THFloatTensorMath_indexCopy :: (Ptr CTHFloatTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_indexAdd : tensor dim index src -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_indexAdd"
  c_THFloatTensorMath_indexAdd :: (Ptr CTHFloatTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_indexFill : tensor dim index val -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_indexFill"
  c_THFloatTensorMath_indexFill :: (Ptr CTHFloatTensor) -> CInt -> Ptr CTHLongTensor -> CFloat -> IO ()

-- |c_THFloatTensorMath_gather : tensor src dim index -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_gather"
  c_THFloatTensorMath_gather :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THFloatTensorMath_scatter : tensor dim index src -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_scatter"
  c_THFloatTensorMath_scatter :: (Ptr CTHFloatTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_scatterAdd : tensor dim index src -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_scatterAdd"
  c_THFloatTensorMath_scatterAdd :: (Ptr CTHFloatTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_scatterFill : tensor dim index val -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_scatterFill"
  c_THFloatTensorMath_scatterFill :: (Ptr CTHFloatTensor) -> CInt -> Ptr CTHLongTensor -> CFloat -> IO ()

-- |c_THFloatTensorMath_dot : t src -> accreal
foreign import ccall "THTensorMath.h THFloatTensorMath_dot"
  c_THFloatTensorMath_dot :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CDouble

-- |c_THFloatTensorMath_minall : t -> real
foreign import ccall "THTensorMath.h THFloatTensorMath_minall"
  c_THFloatTensorMath_minall :: (Ptr CTHFloatTensor) -> CFloat

-- |c_THFloatTensorMath_maxall : t -> real
foreign import ccall "THTensorMath.h THFloatTensorMath_maxall"
  c_THFloatTensorMath_maxall :: (Ptr CTHFloatTensor) -> CFloat

-- |c_THFloatTensorMath_medianall : t -> real
foreign import ccall "THTensorMath.h THFloatTensorMath_medianall"
  c_THFloatTensorMath_medianall :: (Ptr CTHFloatTensor) -> CFloat

-- |c_THFloatTensorMath_sumall : t -> accreal
foreign import ccall "THTensorMath.h THFloatTensorMath_sumall"
  c_THFloatTensorMath_sumall :: (Ptr CTHFloatTensor) -> CDouble

-- |c_THFloatTensorMath_prodall : t -> accreal
foreign import ccall "THTensorMath.h THFloatTensorMath_prodall"
  c_THFloatTensorMath_prodall :: (Ptr CTHFloatTensor) -> CDouble

-- |c_THFloatTensorMath_neg : self src -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_neg"
  c_THFloatTensorMath_neg :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_cinv : self src -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_cinv"
  c_THFloatTensorMath_cinv :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_add : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_add"
  c_THFloatTensorMath_add :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_sub : self src value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_sub"
  c_THFloatTensorMath_sub :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_mul : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_mul"
  c_THFloatTensorMath_mul :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_div : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_div"
  c_THFloatTensorMath_div :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_lshift : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_lshift"
  c_THFloatTensorMath_lshift :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_rshift : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_rshift"
  c_THFloatTensorMath_rshift :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_fmod : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_fmod"
  c_THFloatTensorMath_fmod :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_remainder : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_remainder"
  c_THFloatTensorMath_remainder :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_clamp : r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_clamp"
  c_THFloatTensorMath_clamp :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> CFloat -> IO ()

-- |c_THFloatTensorMath_bitand : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_bitand"
  c_THFloatTensorMath_bitand :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_bitor : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_bitor"
  c_THFloatTensorMath_bitor :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_bitxor : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_bitxor"
  c_THFloatTensorMath_bitxor :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_cadd : r_ t value src -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_cadd"
  c_THFloatTensorMath_cadd :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_csub : self src1 value src2 -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_csub"
  c_THFloatTensorMath_csub :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_cmul : r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_cmul"
  c_THFloatTensorMath_cmul :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_cpow : r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_cpow"
  c_THFloatTensorMath_cpow :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_cdiv : r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_cdiv"
  c_THFloatTensorMath_cdiv :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_clshift : r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_clshift"
  c_THFloatTensorMath_clshift :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_crshift : r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_crshift"
  c_THFloatTensorMath_crshift :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_cfmod : r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_cfmod"
  c_THFloatTensorMath_cfmod :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_cremainder : r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_cremainder"
  c_THFloatTensorMath_cremainder :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_cbitand : r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_cbitand"
  c_THFloatTensorMath_cbitand :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_cbitor : r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_cbitor"
  c_THFloatTensorMath_cbitor :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_cbitxor : r_ t src -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_cbitxor"
  c_THFloatTensorMath_cbitxor :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_addcmul : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_addcmul"
  c_THFloatTensorMath_addcmul :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_addcdiv : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_addcdiv"
  c_THFloatTensorMath_addcdiv :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_addmv : r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_addmv"
  c_THFloatTensorMath_addmv :: (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_addmm : r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_addmm"
  c_THFloatTensorMath_addmm :: (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_addr : r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_addr"
  c_THFloatTensorMath_addr :: (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_addbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_addbmm"
  c_THFloatTensorMath_addbmm :: (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_baddbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_baddbmm"
  c_THFloatTensorMath_baddbmm :: (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_match : r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_match"
  c_THFloatTensorMath_match :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_numel : t -> THStorage *
foreign import ccall "THTensorMath.h THFloatTensorMath_numel"
  c_THFloatTensorMath_numel :: (Ptr CTHFloatTensor) -> IO (Ptr CTHFloatStorage)

-- |c_THFloatTensorMath_max : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_max"
  c_THFloatTensorMath_max :: (Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatTensorMath_min : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_min"
  c_THFloatTensorMath_min :: (Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatTensorMath_kthvalue : values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_kthvalue"
  c_THFloatTensorMath_kthvalue :: (Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> CLong -> CInt -> CInt -> IO ()

-- |c_THFloatTensorMath_mode : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_mode"
  c_THFloatTensorMath_mode :: (Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatTensorMath_median : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_median"
  c_THFloatTensorMath_median :: (Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatTensorMath_sum : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_sum"
  c_THFloatTensorMath_sum :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatTensorMath_prod : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_prod"
  c_THFloatTensorMath_prod :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatTensorMath_cumsum : r_ t dimension -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_cumsum"
  c_THFloatTensorMath_cumsum :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatTensorMath_cumprod : r_ t dimension -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_cumprod"
  c_THFloatTensorMath_cumprod :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatTensorMath_sign : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_sign"
  c_THFloatTensorMath_sign :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_trace : t -> accreal
foreign import ccall "THTensorMath.h THFloatTensorMath_trace"
  c_THFloatTensorMath_trace :: (Ptr CTHFloatTensor) -> CDouble

-- |c_THFloatTensorMath_cross : r_ a b dimension -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_cross"
  c_THFloatTensorMath_cross :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatTensorMath_cmax : r t src -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_cmax"
  c_THFloatTensorMath_cmax :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_cmin : r t src -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_cmin"
  c_THFloatTensorMath_cmin :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_cmaxValue : r t value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_cmaxValue"
  c_THFloatTensorMath_cmaxValue :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_cminValue : r t value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_cminValue"
  c_THFloatTensorMath_cminValue :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_zeros : r_ size -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_zeros"
  c_THFloatTensorMath_zeros :: (Ptr CTHFloatTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THFloatTensorMath_zerosLike : r_ input -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_zerosLike"
  c_THFloatTensorMath_zerosLike :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_ones : r_ size -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_ones"
  c_THFloatTensorMath_ones :: (Ptr CTHFloatTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THFloatTensorMath_onesLike : r_ input -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_onesLike"
  c_THFloatTensorMath_onesLike :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_diag : r_ t k -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_diag"
  c_THFloatTensorMath_diag :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatTensorMath_eye : r_ n m -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_eye"
  c_THFloatTensorMath_eye :: (Ptr CTHFloatTensor) -> CLong -> CLong -> IO ()

-- |c_THFloatTensorMath_arange : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_arange"
  c_THFloatTensorMath_arange :: (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CDouble -> IO ()

-- |c_THFloatTensorMath_range : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_range"
  c_THFloatTensorMath_range :: (Ptr CTHFloatTensor) -> CDouble -> CDouble -> CDouble -> IO ()

-- |c_THFloatTensorMath_randperm : r_ _generator n -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_randperm"
  c_THFloatTensorMath_randperm :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CLong -> IO ()

-- |c_THFloatTensorMath_reshape : r_ t size -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_reshape"
  c_THFloatTensorMath_reshape :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THFloatTensorMath_sort : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_sort"
  c_THFloatTensorMath_sort :: (Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatTensorMath_topk : rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_topk"
  c_THFloatTensorMath_topk :: (Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> CLong -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatTensorMath_tril : r_ t k -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_tril"
  c_THFloatTensorMath_tril :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> IO ()

-- |c_THFloatTensorMath_triu : r_ t k -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_triu"
  c_THFloatTensorMath_triu :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> IO ()

-- |c_THFloatTensorMath_cat : r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_cat"
  c_THFloatTensorMath_cat :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> IO ()

-- |c_THFloatTensorMath_catArray : result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_catArray"
  c_THFloatTensorMath_catArray :: (Ptr CTHFloatTensor) -> Ptr (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatTensorMath_equal : ta tb -> int
foreign import ccall "THTensorMath.h THFloatTensorMath_equal"
  c_THFloatTensorMath_equal :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt

-- |c_THFloatTensorMath_ltValue : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_ltValue"
  c_THFloatTensorMath_ltValue :: Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_leValue : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_leValue"
  c_THFloatTensorMath_leValue :: Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_gtValue : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_gtValue"
  c_THFloatTensorMath_gtValue :: Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_geValue : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_geValue"
  c_THFloatTensorMath_geValue :: Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_neValue : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_neValue"
  c_THFloatTensorMath_neValue :: Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_eqValue : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_eqValue"
  c_THFloatTensorMath_eqValue :: Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_ltValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_ltValueT"
  c_THFloatTensorMath_ltValueT :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_leValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_leValueT"
  c_THFloatTensorMath_leValueT :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_gtValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_gtValueT"
  c_THFloatTensorMath_gtValueT :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_geValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_geValueT"
  c_THFloatTensorMath_geValueT :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_neValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_neValueT"
  c_THFloatTensorMath_neValueT :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_eqValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_eqValueT"
  c_THFloatTensorMath_eqValueT :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_ltTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_ltTensor"
  c_THFloatTensorMath_ltTensor :: Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_leTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_leTensor"
  c_THFloatTensorMath_leTensor :: Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_gtTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_gtTensor"
  c_THFloatTensorMath_gtTensor :: Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_geTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_geTensor"
  c_THFloatTensorMath_geTensor :: Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_neTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_neTensor"
  c_THFloatTensorMath_neTensor :: Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_eqTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_eqTensor"
  c_THFloatTensorMath_eqTensor :: Ptr CTHByteTensor -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_ltTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_ltTensorT"
  c_THFloatTensorMath_ltTensorT :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_leTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_leTensorT"
  c_THFloatTensorMath_leTensorT :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_gtTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_gtTensorT"
  c_THFloatTensorMath_gtTensorT :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_geTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_geTensorT"
  c_THFloatTensorMath_geTensorT :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_neTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_neTensorT"
  c_THFloatTensorMath_neTensorT :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_eqTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_eqTensorT"
  c_THFloatTensorMath_eqTensorT :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_abs : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_abs"
  c_THFloatTensorMath_abs :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_sigmoid : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_sigmoid"
  c_THFloatTensorMath_sigmoid :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_log : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_log"
  c_THFloatTensorMath_log :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_lgamma : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_lgamma"
  c_THFloatTensorMath_lgamma :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_log1p : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_log1p"
  c_THFloatTensorMath_log1p :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_exp : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_exp"
  c_THFloatTensorMath_exp :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_cos : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_cos"
  c_THFloatTensorMath_cos :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_acos : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_acos"
  c_THFloatTensorMath_acos :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_cosh : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_cosh"
  c_THFloatTensorMath_cosh :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_sin : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_sin"
  c_THFloatTensorMath_sin :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_asin : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_asin"
  c_THFloatTensorMath_asin :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_sinh : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_sinh"
  c_THFloatTensorMath_sinh :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_tan : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_tan"
  c_THFloatTensorMath_tan :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_atan : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_atan"
  c_THFloatTensorMath_atan :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_atan2 : r_ tx ty -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_atan2"
  c_THFloatTensorMath_atan2 :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_tanh : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_tanh"
  c_THFloatTensorMath_tanh :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_pow : r_ t value -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_pow"
  c_THFloatTensorMath_pow :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_tpow : r_ value t -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_tpow"
  c_THFloatTensorMath_tpow :: (Ptr CTHFloatTensor) -> CFloat -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_sqrt : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_sqrt"
  c_THFloatTensorMath_sqrt :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_rsqrt : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_rsqrt"
  c_THFloatTensorMath_rsqrt :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_ceil : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_ceil"
  c_THFloatTensorMath_ceil :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_floor : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_floor"
  c_THFloatTensorMath_floor :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_round : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_round"
  c_THFloatTensorMath_round :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_trunc : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_trunc"
  c_THFloatTensorMath_trunc :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_frac : r_ t -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_frac"
  c_THFloatTensorMath_frac :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensorMath_lerp : r_ a b weight -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_lerp"
  c_THFloatTensorMath_lerp :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> IO ()

-- |c_THFloatTensorMath_mean : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_mean"
  c_THFloatTensorMath_mean :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatTensorMath_std : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_std"
  c_THFloatTensorMath_std :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatTensorMath_var : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_var"
  c_THFloatTensorMath_var :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THFloatTensorMath_norm : r_ t value dimension keepdim -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_norm"
  c_THFloatTensorMath_norm :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> CInt -> CInt -> IO ()

-- |c_THFloatTensorMath_renorm : r_ t value dimension maxnorm -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_renorm"
  c_THFloatTensorMath_renorm :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> CInt -> CFloat -> IO ()

-- |c_THFloatTensorMath_dist : a b value -> accreal
foreign import ccall "THTensorMath.h THFloatTensorMath_dist"
  c_THFloatTensorMath_dist :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CFloat -> CDouble

-- |c_THFloatTensorMath_histc : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_histc"
  c_THFloatTensorMath_histc :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> CFloat -> CFloat -> IO ()

-- |c_THFloatTensorMath_bhistc : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_bhistc"
  c_THFloatTensorMath_bhistc :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> CFloat -> CFloat -> IO ()

-- |c_THFloatTensorMath_meanall : self -> accreal
foreign import ccall "THTensorMath.h THFloatTensorMath_meanall"
  c_THFloatTensorMath_meanall :: (Ptr CTHFloatTensor) -> CDouble

-- |c_THFloatTensorMath_varall : self biased -> accreal
foreign import ccall "THTensorMath.h THFloatTensorMath_varall"
  c_THFloatTensorMath_varall :: (Ptr CTHFloatTensor) -> CInt -> CDouble

-- |c_THFloatTensorMath_stdall : self biased -> accreal
foreign import ccall "THTensorMath.h THFloatTensorMath_stdall"
  c_THFloatTensorMath_stdall :: (Ptr CTHFloatTensor) -> CInt -> CDouble

-- |c_THFloatTensorMath_normall : t value -> accreal
foreign import ccall "THTensorMath.h THFloatTensorMath_normall"
  c_THFloatTensorMath_normall :: (Ptr CTHFloatTensor) -> CFloat -> CDouble

-- |c_THFloatTensorMath_linspace : r_ a b n -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_linspace"
  c_THFloatTensorMath_linspace :: (Ptr CTHFloatTensor) -> CFloat -> CFloat -> CLong -> IO ()

-- |c_THFloatTensorMath_logspace : r_ a b n -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_logspace"
  c_THFloatTensorMath_logspace :: (Ptr CTHFloatTensor) -> CFloat -> CFloat -> CLong -> IO ()

-- |c_THFloatTensorMath_rand : r_ _generator size -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_rand"
  c_THFloatTensorMath_rand :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()

-- |c_THFloatTensorMath_randn : r_ _generator size -> void
foreign import ccall "THTensorMath.h THFloatTensorMath_randn"
  c_THFloatTensorMath_randn :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()

-- |c_THFloatTensorMath_logicalall : self -> int
foreign import ccall "THTensorMath.h THFloatTensorMath_logicalall"
  c_THFloatTensorMath_logicalall :: (Ptr CTHFloatTensor) -> CInt

-- |c_THFloatTensorMath_logicalany : self -> int
foreign import ccall "THTensorMath.h THFloatTensorMath_logicalany"
  c_THFloatTensorMath_logicalany :: (Ptr CTHFloatTensor) -> CInt