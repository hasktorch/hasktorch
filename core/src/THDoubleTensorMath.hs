{-# LANGUAGE ForeignFunctionInterface#-}

module THDoubleTensorMath (
    c_THDoubleTensorMath_fill,
    c_THDoubleTensorMath_zero,
    c_THDoubleTensorMath_maskedFill,
    c_THDoubleTensorMath_maskedCopy,
    c_THDoubleTensorMath_maskedSelect,
    c_THDoubleTensorMath_nonzero,
    c_THDoubleTensorMath_indexSelect,
    c_THDoubleTensorMath_indexCopy,
    c_THDoubleTensorMath_indexAdd,
    c_THDoubleTensorMath_indexFill,
    c_THDoubleTensorMath_gather,
    c_THDoubleTensorMath_scatter,
    c_THDoubleTensorMath_scatterAdd,
    c_THDoubleTensorMath_scatterFill,
    c_THDoubleTensorMath_dot,
    c_THDoubleTensorMath_minall,
    c_THDoubleTensorMath_maxall,
    c_THDoubleTensorMath_medianall,
    c_THDoubleTensorMath_sumall,
    c_THDoubleTensorMath_prodall,
    c_THDoubleTensorMath_neg,
    c_THDoubleTensorMath_cinv,
    c_THDoubleTensorMath_add,
    c_THDoubleTensorMath_sub,
    c_THDoubleTensorMath_mul,
    c_THDoubleTensorMath_div,
    c_THDoubleTensorMath_lshift,
    c_THDoubleTensorMath_rshift,
    c_THDoubleTensorMath_fmod,
    c_THDoubleTensorMath_remainder,
    c_THDoubleTensorMath_clamp,
    c_THDoubleTensorMath_bitand,
    c_THDoubleTensorMath_bitor,
    c_THDoubleTensorMath_bitxor,
    c_THDoubleTensorMath_cadd,
    c_THDoubleTensorMath_csub,
    c_THDoubleTensorMath_cmul,
    c_THDoubleTensorMath_cpow,
    c_THDoubleTensorMath_cdiv,
    c_THDoubleTensorMath_clshift,
    c_THDoubleTensorMath_crshift,
    c_THDoubleTensorMath_cfmod,
    c_THDoubleTensorMath_cremainder,
    c_THDoubleTensorMath_cbitand,
    c_THDoubleTensorMath_cbitor,
    c_THDoubleTensorMath_cbitxor,
    c_THDoubleTensorMath_addcmul,
    c_THDoubleTensorMath_addcdiv,
    c_THDoubleTensorMath_addmv,
    c_THDoubleTensorMath_addmm,
    c_THDoubleTensorMath_addr,
    c_THDoubleTensorMath_addbmm,
    c_THDoubleTensorMath_baddbmm,
    c_THDoubleTensorMath_match,
    c_THDoubleTensorMath_numel,
    c_THDoubleTensorMath_max,
    c_THDoubleTensorMath_min,
    c_THDoubleTensorMath_kthvalue,
    c_THDoubleTensorMath_mode,
    c_THDoubleTensorMath_median,
    c_THDoubleTensorMath_sum,
    c_THDoubleTensorMath_prod,
    c_THDoubleTensorMath_cumsum,
    c_THDoubleTensorMath_cumprod,
    c_THDoubleTensorMath_sign,
    c_THDoubleTensorMath_trace,
    c_THDoubleTensorMath_cross,
    c_THDoubleTensorMath_cmax,
    c_THDoubleTensorMath_cmin,
    c_THDoubleTensorMath_cmaxValue,
    c_THDoubleTensorMath_cminValue,
    c_THDoubleTensorMath_zeros,
    c_THDoubleTensorMath_zerosLike,
    c_THDoubleTensorMath_ones,
    c_THDoubleTensorMath_onesLike,
    c_THDoubleTensorMath_diag,
    c_THDoubleTensorMath_eye,
    c_THDoubleTensorMath_arange,
    c_THDoubleTensorMath_range,
    c_THDoubleTensorMath_randperm,
    c_THDoubleTensorMath_reshape,
    c_THDoubleTensorMath_sort,
    c_THDoubleTensorMath_topk,
    c_THDoubleTensorMath_tril,
    c_THDoubleTensorMath_triu,
    c_THDoubleTensorMath_cat,
    c_THDoubleTensorMath_catArray,
    c_THDoubleTensorMath_equal,
    c_THDoubleTensorMath_ltValue,
    c_THDoubleTensorMath_leValue,
    c_THDoubleTensorMath_gtValue,
    c_THDoubleTensorMath_geValue,
    c_THDoubleTensorMath_neValue,
    c_THDoubleTensorMath_eqValue,
    c_THDoubleTensorMath_ltValueT,
    c_THDoubleTensorMath_leValueT,
    c_THDoubleTensorMath_gtValueT,
    c_THDoubleTensorMath_geValueT,
    c_THDoubleTensorMath_neValueT,
    c_THDoubleTensorMath_eqValueT,
    c_THDoubleTensorMath_ltTensor,
    c_THDoubleTensorMath_leTensor,
    c_THDoubleTensorMath_gtTensor,
    c_THDoubleTensorMath_geTensor,
    c_THDoubleTensorMath_neTensor,
    c_THDoubleTensorMath_eqTensor,
    c_THDoubleTensorMath_ltTensorT,
    c_THDoubleTensorMath_leTensorT,
    c_THDoubleTensorMath_gtTensorT,
    c_THDoubleTensorMath_geTensorT,
    c_THDoubleTensorMath_neTensorT,
    c_THDoubleTensorMath_eqTensorT,
    c_THDoubleTensorMath_abs,
    c_THDoubleTensorMath_sigmoid,
    c_THDoubleTensorMath_log,
    c_THDoubleTensorMath_lgamma,
    c_THDoubleTensorMath_log1p,
    c_THDoubleTensorMath_exp,
    c_THDoubleTensorMath_cos,
    c_THDoubleTensorMath_acos,
    c_THDoubleTensorMath_cosh,
    c_THDoubleTensorMath_sin,
    c_THDoubleTensorMath_asin,
    c_THDoubleTensorMath_sinh,
    c_THDoubleTensorMath_tan,
    c_THDoubleTensorMath_atan,
    c_THDoubleTensorMath_atan2,
    c_THDoubleTensorMath_tanh,
    c_THDoubleTensorMath_pow,
    c_THDoubleTensorMath_tpow,
    c_THDoubleTensorMath_sqrt,
    c_THDoubleTensorMath_rsqrt,
    c_THDoubleTensorMath_ceil,
    c_THDoubleTensorMath_floor,
    c_THDoubleTensorMath_round,
    c_THDoubleTensorMath_trunc,
    c_THDoubleTensorMath_frac,
    c_THDoubleTensorMath_lerp,
    c_THDoubleTensorMath_mean,
    c_THDoubleTensorMath_std,
    c_THDoubleTensorMath_var,
    c_THDoubleTensorMath_norm,
    c_THDoubleTensorMath_renorm,
    c_THDoubleTensorMath_dist,
    c_THDoubleTensorMath_histc,
    c_THDoubleTensorMath_bhistc,
    c_THDoubleTensorMath_meanall,
    c_THDoubleTensorMath_varall,
    c_THDoubleTensorMath_stdall,
    c_THDoubleTensorMath_normall,
    c_THDoubleTensorMath_linspace,
    c_THDoubleTensorMath_logspace,
    c_THDoubleTensorMath_rand,
    c_THDoubleTensorMath_randn,
    c_THDoubleTensorMath_logicalall,
    c_THDoubleTensorMath_logicalany) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THDoubleTensorMath_fill : r_ value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_fill"
  c_THDoubleTensorMath_fill :: (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_zero : r_ -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_zero"
  c_THDoubleTensorMath_zero :: (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_maskedFill : tensor mask value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_maskedFill"
  c_THDoubleTensorMath_maskedFill :: (Ptr CTHDoubleTensor) -> Ptr CTHByteTensor -> CDouble -> IO ()

-- |c_THDoubleTensorMath_maskedCopy : tensor mask src -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_maskedCopy"
  c_THDoubleTensorMath_maskedCopy :: (Ptr CTHDoubleTensor) -> Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_maskedSelect : tensor src mask -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_maskedSelect"
  c_THDoubleTensorMath_maskedSelect :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THDoubleTensorMath_nonzero : subscript tensor -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_nonzero"
  c_THDoubleTensorMath_nonzero :: Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_indexSelect : tensor src dim index -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_indexSelect"
  c_THDoubleTensorMath_indexSelect :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THDoubleTensorMath_indexCopy : tensor dim index src -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_indexCopy"
  c_THDoubleTensorMath_indexCopy :: (Ptr CTHDoubleTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_indexAdd : tensor dim index src -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_indexAdd"
  c_THDoubleTensorMath_indexAdd :: (Ptr CTHDoubleTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_indexFill : tensor dim index val -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_indexFill"
  c_THDoubleTensorMath_indexFill :: (Ptr CTHDoubleTensor) -> CInt -> Ptr CTHLongTensor -> CDouble -> IO ()

-- |c_THDoubleTensorMath_gather : tensor src dim index -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_gather"
  c_THDoubleTensorMath_gather :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THDoubleTensorMath_scatter : tensor dim index src -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_scatter"
  c_THDoubleTensorMath_scatter :: (Ptr CTHDoubleTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_scatterAdd : tensor dim index src -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_scatterAdd"
  c_THDoubleTensorMath_scatterAdd :: (Ptr CTHDoubleTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_scatterFill : tensor dim index val -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_scatterFill"
  c_THDoubleTensorMath_scatterFill :: (Ptr CTHDoubleTensor) -> CInt -> Ptr CTHLongTensor -> CDouble -> IO ()

-- |c_THDoubleTensorMath_dot : t src -> accreal
foreign import ccall "THTensorMath.h THDoubleTensorMath_dot"
  c_THDoubleTensorMath_dot :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble

-- |c_THDoubleTensorMath_minall : t -> real
foreign import ccall "THTensorMath.h THDoubleTensorMath_minall"
  c_THDoubleTensorMath_minall :: (Ptr CTHDoubleTensor) -> CDouble

-- |c_THDoubleTensorMath_maxall : t -> real
foreign import ccall "THTensorMath.h THDoubleTensorMath_maxall"
  c_THDoubleTensorMath_maxall :: (Ptr CTHDoubleTensor) -> CDouble

-- |c_THDoubleTensorMath_medianall : t -> real
foreign import ccall "THTensorMath.h THDoubleTensorMath_medianall"
  c_THDoubleTensorMath_medianall :: (Ptr CTHDoubleTensor) -> CDouble

-- |c_THDoubleTensorMath_sumall : t -> accreal
foreign import ccall "THTensorMath.h THDoubleTensorMath_sumall"
  c_THDoubleTensorMath_sumall :: (Ptr CTHDoubleTensor) -> CDouble

-- |c_THDoubleTensorMath_prodall : t -> accreal
foreign import ccall "THTensorMath.h THDoubleTensorMath_prodall"
  c_THDoubleTensorMath_prodall :: (Ptr CTHDoubleTensor) -> CDouble

-- |c_THDoubleTensorMath_neg : self src -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_neg"
  c_THDoubleTensorMath_neg :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_cinv : self src -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_cinv"
  c_THDoubleTensorMath_cinv :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_add : r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_add"
  c_THDoubleTensorMath_add :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_sub : self src value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_sub"
  c_THDoubleTensorMath_sub :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_mul : r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_mul"
  c_THDoubleTensorMath_mul :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_div : r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_div"
  c_THDoubleTensorMath_div :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_lshift : r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_lshift"
  c_THDoubleTensorMath_lshift :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_rshift : r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_rshift"
  c_THDoubleTensorMath_rshift :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_fmod : r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_fmod"
  c_THDoubleTensorMath_fmod :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_remainder : r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_remainder"
  c_THDoubleTensorMath_remainder :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_clamp : r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_clamp"
  c_THDoubleTensorMath_clamp :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> IO ()

-- |c_THDoubleTensorMath_bitand : r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_bitand"
  c_THDoubleTensorMath_bitand :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_bitor : r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_bitor"
  c_THDoubleTensorMath_bitor :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_bitxor : r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_bitxor"
  c_THDoubleTensorMath_bitxor :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_cadd : r_ t value src -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_cadd"
  c_THDoubleTensorMath_cadd :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_csub : self src1 value src2 -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_csub"
  c_THDoubleTensorMath_csub :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_cmul : r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_cmul"
  c_THDoubleTensorMath_cmul :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_cpow : r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_cpow"
  c_THDoubleTensorMath_cpow :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_cdiv : r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_cdiv"
  c_THDoubleTensorMath_cdiv :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_clshift : r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_clshift"
  c_THDoubleTensorMath_clshift :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_crshift : r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_crshift"
  c_THDoubleTensorMath_crshift :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_cfmod : r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_cfmod"
  c_THDoubleTensorMath_cfmod :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_cremainder : r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_cremainder"
  c_THDoubleTensorMath_cremainder :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_cbitand : r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_cbitand"
  c_THDoubleTensorMath_cbitand :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_cbitor : r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_cbitor"
  c_THDoubleTensorMath_cbitor :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_cbitxor : r_ t src -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_cbitxor"
  c_THDoubleTensorMath_cbitxor :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_addcmul : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_addcmul"
  c_THDoubleTensorMath_addcmul :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_addcdiv : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_addcdiv"
  c_THDoubleTensorMath_addcdiv :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_addmv : r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_addmv"
  c_THDoubleTensorMath_addmv :: (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_addmm : r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_addmm"
  c_THDoubleTensorMath_addmm :: (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_addr : r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_addr"
  c_THDoubleTensorMath_addr :: (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_addbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_addbmm"
  c_THDoubleTensorMath_addbmm :: (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_baddbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_baddbmm"
  c_THDoubleTensorMath_baddbmm :: (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_match : r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_match"
  c_THDoubleTensorMath_match :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_numel : t -> THStorage *
foreign import ccall "THTensorMath.h THDoubleTensorMath_numel"
  c_THDoubleTensorMath_numel :: (Ptr CTHDoubleTensor) -> IO (Ptr CTHDoubleStorage)

-- |c_THDoubleTensorMath_max : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_max"
  c_THDoubleTensorMath_max :: (Ptr CTHDoubleTensor) -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleTensorMath_min : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_min"
  c_THDoubleTensorMath_min :: (Ptr CTHDoubleTensor) -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleTensorMath_kthvalue : values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_kthvalue"
  c_THDoubleTensorMath_kthvalue :: (Ptr CTHDoubleTensor) -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> CLong -> CInt -> CInt -> IO ()

-- |c_THDoubleTensorMath_mode : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_mode"
  c_THDoubleTensorMath_mode :: (Ptr CTHDoubleTensor) -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleTensorMath_median : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_median"
  c_THDoubleTensorMath_median :: (Ptr CTHDoubleTensor) -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleTensorMath_sum : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_sum"
  c_THDoubleTensorMath_sum :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleTensorMath_prod : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_prod"
  c_THDoubleTensorMath_prod :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleTensorMath_cumsum : r_ t dimension -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_cumsum"
  c_THDoubleTensorMath_cumsum :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleTensorMath_cumprod : r_ t dimension -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_cumprod"
  c_THDoubleTensorMath_cumprod :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleTensorMath_sign : r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_sign"
  c_THDoubleTensorMath_sign :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_trace : t -> accreal
foreign import ccall "THTensorMath.h THDoubleTensorMath_trace"
  c_THDoubleTensorMath_trace :: (Ptr CTHDoubleTensor) -> CDouble

-- |c_THDoubleTensorMath_cross : r_ a b dimension -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_cross"
  c_THDoubleTensorMath_cross :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleTensorMath_cmax : r t src -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_cmax"
  c_THDoubleTensorMath_cmax :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_cmin : r t src -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_cmin"
  c_THDoubleTensorMath_cmin :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_cmaxValue : r t value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_cmaxValue"
  c_THDoubleTensorMath_cmaxValue :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_cminValue : r t value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_cminValue"
  c_THDoubleTensorMath_cminValue :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_zeros : r_ size -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_zeros"
  c_THDoubleTensorMath_zeros :: (Ptr CTHDoubleTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THDoubleTensorMath_zerosLike : r_ input -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_zerosLike"
  c_THDoubleTensorMath_zerosLike :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_ones : r_ size -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_ones"
  c_THDoubleTensorMath_ones :: (Ptr CTHDoubleTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THDoubleTensorMath_onesLike : r_ input -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_onesLike"
  c_THDoubleTensorMath_onesLike :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_diag : r_ t k -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_diag"
  c_THDoubleTensorMath_diag :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleTensorMath_eye : r_ n m -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_eye"
  c_THDoubleTensorMath_eye :: (Ptr CTHDoubleTensor) -> CLong -> CLong -> IO ()

-- |c_THDoubleTensorMath_arange : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_arange"
  c_THDoubleTensorMath_arange :: (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CDouble -> IO ()

-- |c_THDoubleTensorMath_range : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_range"
  c_THDoubleTensorMath_range :: (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CDouble -> IO ()

-- |c_THDoubleTensorMath_randperm : r_ _generator n -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_randperm"
  c_THDoubleTensorMath_randperm :: (Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> CLong -> IO ()

-- |c_THDoubleTensorMath_reshape : r_ t size -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_reshape"
  c_THDoubleTensorMath_reshape :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THDoubleTensorMath_sort : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_sort"
  c_THDoubleTensorMath_sort :: (Ptr CTHDoubleTensor) -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleTensorMath_topk : rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_topk"
  c_THDoubleTensorMath_topk :: (Ptr CTHDoubleTensor) -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> CLong -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleTensorMath_tril : r_ t k -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_tril"
  c_THDoubleTensorMath_tril :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> IO ()

-- |c_THDoubleTensorMath_triu : r_ t k -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_triu"
  c_THDoubleTensorMath_triu :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> IO ()

-- |c_THDoubleTensorMath_cat : r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_cat"
  c_THDoubleTensorMath_cat :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> IO ()

-- |c_THDoubleTensorMath_catArray : result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_catArray"
  c_THDoubleTensorMath_catArray :: (Ptr CTHDoubleTensor) -> Ptr (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleTensorMath_equal : ta tb -> int
foreign import ccall "THTensorMath.h THDoubleTensorMath_equal"
  c_THDoubleTensorMath_equal :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt

-- |c_THDoubleTensorMath_ltValue : r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_ltValue"
  c_THDoubleTensorMath_ltValue :: Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_leValue : r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_leValue"
  c_THDoubleTensorMath_leValue :: Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_gtValue : r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_gtValue"
  c_THDoubleTensorMath_gtValue :: Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_geValue : r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_geValue"
  c_THDoubleTensorMath_geValue :: Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_neValue : r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_neValue"
  c_THDoubleTensorMath_neValue :: Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_eqValue : r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_eqValue"
  c_THDoubleTensorMath_eqValue :: Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_ltValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_ltValueT"
  c_THDoubleTensorMath_ltValueT :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_leValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_leValueT"
  c_THDoubleTensorMath_leValueT :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_gtValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_gtValueT"
  c_THDoubleTensorMath_gtValueT :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_geValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_geValueT"
  c_THDoubleTensorMath_geValueT :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_neValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_neValueT"
  c_THDoubleTensorMath_neValueT :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_eqValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_eqValueT"
  c_THDoubleTensorMath_eqValueT :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_ltTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_ltTensor"
  c_THDoubleTensorMath_ltTensor :: Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_leTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_leTensor"
  c_THDoubleTensorMath_leTensor :: Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_gtTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_gtTensor"
  c_THDoubleTensorMath_gtTensor :: Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_geTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_geTensor"
  c_THDoubleTensorMath_geTensor :: Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_neTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_neTensor"
  c_THDoubleTensorMath_neTensor :: Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_eqTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_eqTensor"
  c_THDoubleTensorMath_eqTensor :: Ptr CTHByteTensor -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_ltTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_ltTensorT"
  c_THDoubleTensorMath_ltTensorT :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_leTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_leTensorT"
  c_THDoubleTensorMath_leTensorT :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_gtTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_gtTensorT"
  c_THDoubleTensorMath_gtTensorT :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_geTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_geTensorT"
  c_THDoubleTensorMath_geTensorT :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_neTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_neTensorT"
  c_THDoubleTensorMath_neTensorT :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_eqTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_eqTensorT"
  c_THDoubleTensorMath_eqTensorT :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_abs : r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_abs"
  c_THDoubleTensorMath_abs :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_sigmoid : r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_sigmoid"
  c_THDoubleTensorMath_sigmoid :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_log : r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_log"
  c_THDoubleTensorMath_log :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_lgamma : r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_lgamma"
  c_THDoubleTensorMath_lgamma :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_log1p : r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_log1p"
  c_THDoubleTensorMath_log1p :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_exp : r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_exp"
  c_THDoubleTensorMath_exp :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_cos : r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_cos"
  c_THDoubleTensorMath_cos :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_acos : r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_acos"
  c_THDoubleTensorMath_acos :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_cosh : r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_cosh"
  c_THDoubleTensorMath_cosh :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_sin : r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_sin"
  c_THDoubleTensorMath_sin :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_asin : r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_asin"
  c_THDoubleTensorMath_asin :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_sinh : r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_sinh"
  c_THDoubleTensorMath_sinh :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_tan : r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_tan"
  c_THDoubleTensorMath_tan :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_atan : r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_atan"
  c_THDoubleTensorMath_atan :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_atan2 : r_ tx ty -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_atan2"
  c_THDoubleTensorMath_atan2 :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_tanh : r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_tanh"
  c_THDoubleTensorMath_tanh :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_pow : r_ t value -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_pow"
  c_THDoubleTensorMath_pow :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_tpow : r_ value t -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_tpow"
  c_THDoubleTensorMath_tpow :: (Ptr CTHDoubleTensor) -> CDouble -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_sqrt : r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_sqrt"
  c_THDoubleTensorMath_sqrt :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_rsqrt : r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_rsqrt"
  c_THDoubleTensorMath_rsqrt :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_ceil : r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_ceil"
  c_THDoubleTensorMath_ceil :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_floor : r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_floor"
  c_THDoubleTensorMath_floor :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_round : r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_round"
  c_THDoubleTensorMath_round :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_trunc : r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_trunc"
  c_THDoubleTensorMath_trunc :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_frac : r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_frac"
  c_THDoubleTensorMath_frac :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensorMath_lerp : r_ a b weight -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_lerp"
  c_THDoubleTensorMath_lerp :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensorMath_mean : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_mean"
  c_THDoubleTensorMath_mean :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleTensorMath_std : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_std"
  c_THDoubleTensorMath_std :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleTensorMath_var : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_var"
  c_THDoubleTensorMath_var :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THDoubleTensorMath_norm : r_ t value dimension keepdim -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_norm"
  c_THDoubleTensorMath_norm :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CInt -> CInt -> IO ()

-- |c_THDoubleTensorMath_renorm : r_ t value dimension maxnorm -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_renorm"
  c_THDoubleTensorMath_renorm :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CInt -> CDouble -> IO ()

-- |c_THDoubleTensorMath_dist : a b value -> accreal
foreign import ccall "THTensorMath.h THDoubleTensorMath_dist"
  c_THDoubleTensorMath_dist :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CDouble -> CDouble

-- |c_THDoubleTensorMath_histc : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_histc"
  c_THDoubleTensorMath_histc :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CDouble -> CDouble -> IO ()

-- |c_THDoubleTensorMath_bhistc : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_bhistc"
  c_THDoubleTensorMath_bhistc :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CDouble -> CDouble -> IO ()

-- |c_THDoubleTensorMath_meanall : self -> accreal
foreign import ccall "THTensorMath.h THDoubleTensorMath_meanall"
  c_THDoubleTensorMath_meanall :: (Ptr CTHDoubleTensor) -> CDouble

-- |c_THDoubleTensorMath_varall : self biased -> accreal
foreign import ccall "THTensorMath.h THDoubleTensorMath_varall"
  c_THDoubleTensorMath_varall :: (Ptr CTHDoubleTensor) -> CInt -> CDouble

-- |c_THDoubleTensorMath_stdall : self biased -> accreal
foreign import ccall "THTensorMath.h THDoubleTensorMath_stdall"
  c_THDoubleTensorMath_stdall :: (Ptr CTHDoubleTensor) -> CInt -> CDouble

-- |c_THDoubleTensorMath_normall : t value -> accreal
foreign import ccall "THTensorMath.h THDoubleTensorMath_normall"
  c_THDoubleTensorMath_normall :: (Ptr CTHDoubleTensor) -> CDouble -> CDouble

-- |c_THDoubleTensorMath_linspace : r_ a b n -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_linspace"
  c_THDoubleTensorMath_linspace :: (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CLong -> IO ()

-- |c_THDoubleTensorMath_logspace : r_ a b n -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_logspace"
  c_THDoubleTensorMath_logspace :: (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> CLong -> IO ()

-- |c_THDoubleTensorMath_rand : r_ _generator size -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_rand"
  c_THDoubleTensorMath_rand :: (Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()

-- |c_THDoubleTensorMath_randn : r_ _generator size -> void
foreign import ccall "THTensorMath.h THDoubleTensorMath_randn"
  c_THDoubleTensorMath_randn :: (Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()

-- |c_THDoubleTensorMath_logicalall : self -> int
foreign import ccall "THTensorMath.h THDoubleTensorMath_logicalall"
  c_THDoubleTensorMath_logicalall :: (Ptr CTHDoubleTensor) -> CInt

-- |c_THDoubleTensorMath_logicalany : self -> int
foreign import ccall "THTensorMath.h THDoubleTensorMath_logicalany"
  c_THDoubleTensorMath_logicalany :: (Ptr CTHDoubleTensor) -> CInt