{-# LANGUAGE ForeignFunctionInterface#-}

module THLongTensorMath (
    c_THLongTensorMath_fill,
    c_THLongTensorMath_zero,
    c_THLongTensorMath_maskedFill,
    c_THLongTensorMath_maskedCopy,
    c_THLongTensorMath_maskedSelect,
    c_THLongTensorMath_nonzero,
    c_THLongTensorMath_indexSelect,
    c_THLongTensorMath_indexCopy,
    c_THLongTensorMath_indexAdd,
    c_THLongTensorMath_indexFill,
    c_THLongTensorMath_gather,
    c_THLongTensorMath_scatter,
    c_THLongTensorMath_scatterAdd,
    c_THLongTensorMath_scatterFill,
    c_THLongTensorMath_dot,
    c_THLongTensorMath_minall,
    c_THLongTensorMath_maxall,
    c_THLongTensorMath_medianall,
    c_THLongTensorMath_sumall,
    c_THLongTensorMath_prodall,
    c_THLongTensorMath_neg,
    c_THLongTensorMath_cinv,
    c_THLongTensorMath_add,
    c_THLongTensorMath_sub,
    c_THLongTensorMath_mul,
    c_THLongTensorMath_div,
    c_THLongTensorMath_lshift,
    c_THLongTensorMath_rshift,
    c_THLongTensorMath_fmod,
    c_THLongTensorMath_remainder,
    c_THLongTensorMath_clamp,
    c_THLongTensorMath_bitand,
    c_THLongTensorMath_bitor,
    c_THLongTensorMath_bitxor,
    c_THLongTensorMath_cadd,
    c_THLongTensorMath_csub,
    c_THLongTensorMath_cmul,
    c_THLongTensorMath_cpow,
    c_THLongTensorMath_cdiv,
    c_THLongTensorMath_clshift,
    c_THLongTensorMath_crshift,
    c_THLongTensorMath_cfmod,
    c_THLongTensorMath_cremainder,
    c_THLongTensorMath_cbitand,
    c_THLongTensorMath_cbitor,
    c_THLongTensorMath_cbitxor,
    c_THLongTensorMath_addcmul,
    c_THLongTensorMath_addcdiv,
    c_THLongTensorMath_addmv,
    c_THLongTensorMath_addmm,
    c_THLongTensorMath_addr,
    c_THLongTensorMath_addbmm,
    c_THLongTensorMath_baddbmm,
    c_THLongTensorMath_match,
    c_THLongTensorMath_numel,
    c_THLongTensorMath_max,
    c_THLongTensorMath_min,
    c_THLongTensorMath_kthvalue,
    c_THLongTensorMath_mode,
    c_THLongTensorMath_median,
    c_THLongTensorMath_sum,
    c_THLongTensorMath_prod,
    c_THLongTensorMath_cumsum,
    c_THLongTensorMath_cumprod,
    c_THLongTensorMath_sign,
    c_THLongTensorMath_trace,
    c_THLongTensorMath_cross,
    c_THLongTensorMath_cmax,
    c_THLongTensorMath_cmin,
    c_THLongTensorMath_cmaxValue,
    c_THLongTensorMath_cminValue,
    c_THLongTensorMath_zeros,
    c_THLongTensorMath_zerosLike,
    c_THLongTensorMath_ones,
    c_THLongTensorMath_onesLike,
    c_THLongTensorMath_diag,
    c_THLongTensorMath_eye,
    c_THLongTensorMath_arange,
    c_THLongTensorMath_range,
    c_THLongTensorMath_randperm,
    c_THLongTensorMath_reshape,
    c_THLongTensorMath_sort,
    c_THLongTensorMath_topk,
    c_THLongTensorMath_tril,
    c_THLongTensorMath_triu,
    c_THLongTensorMath_cat,
    c_THLongTensorMath_catArray,
    c_THLongTensorMath_equal,
    c_THLongTensorMath_ltValue,
    c_THLongTensorMath_leValue,
    c_THLongTensorMath_gtValue,
    c_THLongTensorMath_geValue,
    c_THLongTensorMath_neValue,
    c_THLongTensorMath_eqValue,
    c_THLongTensorMath_ltValueT,
    c_THLongTensorMath_leValueT,
    c_THLongTensorMath_gtValueT,
    c_THLongTensorMath_geValueT,
    c_THLongTensorMath_neValueT,
    c_THLongTensorMath_eqValueT,
    c_THLongTensorMath_ltTensor,
    c_THLongTensorMath_leTensor,
    c_THLongTensorMath_gtTensor,
    c_THLongTensorMath_geTensor,
    c_THLongTensorMath_neTensor,
    c_THLongTensorMath_eqTensor,
    c_THLongTensorMath_ltTensorT,
    c_THLongTensorMath_leTensorT,
    c_THLongTensorMath_gtTensorT,
    c_THLongTensorMath_geTensorT,
    c_THLongTensorMath_neTensorT,
    c_THLongTensorMath_eqTensorT,
    c_THLongTensorMath_abs,
    c_THLongTensorMath_sigmoid,
    c_THLongTensorMath_log,
    c_THLongTensorMath_lgamma,
    c_THLongTensorMath_log1p,
    c_THLongTensorMath_exp,
    c_THLongTensorMath_cos,
    c_THLongTensorMath_acos,
    c_THLongTensorMath_cosh,
    c_THLongTensorMath_sin,
    c_THLongTensorMath_asin,
    c_THLongTensorMath_sinh,
    c_THLongTensorMath_tan,
    c_THLongTensorMath_atan,
    c_THLongTensorMath_atan2,
    c_THLongTensorMath_tanh,
    c_THLongTensorMath_pow,
    c_THLongTensorMath_tpow,
    c_THLongTensorMath_sqrt,
    c_THLongTensorMath_rsqrt,
    c_THLongTensorMath_ceil,
    c_THLongTensorMath_floor,
    c_THLongTensorMath_round,
    c_THLongTensorMath_trunc,
    c_THLongTensorMath_frac,
    c_THLongTensorMath_lerp,
    c_THLongTensorMath_mean,
    c_THLongTensorMath_std,
    c_THLongTensorMath_var,
    c_THLongTensorMath_norm,
    c_THLongTensorMath_renorm,
    c_THLongTensorMath_dist,
    c_THLongTensorMath_histc,
    c_THLongTensorMath_bhistc,
    c_THLongTensorMath_meanall,
    c_THLongTensorMath_varall,
    c_THLongTensorMath_stdall,
    c_THLongTensorMath_normall,
    c_THLongTensorMath_linspace,
    c_THLongTensorMath_logspace,
    c_THLongTensorMath_rand,
    c_THLongTensorMath_randn,
    c_THLongTensorMath_logicalall,
    c_THLongTensorMath_logicalany) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THLongTensorMath_fill : r_ value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_fill"
  c_THLongTensorMath_fill :: (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_zero : r_ -> void
foreign import ccall "THTensorMath.h THLongTensorMath_zero"
  c_THLongTensorMath_zero :: (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_maskedFill : tensor mask value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_maskedFill"
  c_THLongTensorMath_maskedFill :: (Ptr CTHLongTensor) -> Ptr CTHByteTensor -> CLong -> IO ()

-- |c_THLongTensorMath_maskedCopy : tensor mask src -> void
foreign import ccall "THTensorMath.h THLongTensorMath_maskedCopy"
  c_THLongTensorMath_maskedCopy :: (Ptr CTHLongTensor) -> Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_maskedSelect : tensor src mask -> void
foreign import ccall "THTensorMath.h THLongTensorMath_maskedSelect"
  c_THLongTensorMath_maskedSelect :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THLongTensorMath_nonzero : subscript tensor -> void
foreign import ccall "THTensorMath.h THLongTensorMath_nonzero"
  c_THLongTensorMath_nonzero :: Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_indexSelect : tensor src dim index -> void
foreign import ccall "THTensorMath.h THLongTensorMath_indexSelect"
  c_THLongTensorMath_indexSelect :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THLongTensorMath_indexCopy : tensor dim index src -> void
foreign import ccall "THTensorMath.h THLongTensorMath_indexCopy"
  c_THLongTensorMath_indexCopy :: (Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_indexAdd : tensor dim index src -> void
foreign import ccall "THTensorMath.h THLongTensorMath_indexAdd"
  c_THLongTensorMath_indexAdd :: (Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_indexFill : tensor dim index val -> void
foreign import ccall "THTensorMath.h THLongTensorMath_indexFill"
  c_THLongTensorMath_indexFill :: (Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> CLong -> IO ()

-- |c_THLongTensorMath_gather : tensor src dim index -> void
foreign import ccall "THTensorMath.h THLongTensorMath_gather"
  c_THLongTensorMath_gather :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THLongTensorMath_scatter : tensor dim index src -> void
foreign import ccall "THTensorMath.h THLongTensorMath_scatter"
  c_THLongTensorMath_scatter :: (Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_scatterAdd : tensor dim index src -> void
foreign import ccall "THTensorMath.h THLongTensorMath_scatterAdd"
  c_THLongTensorMath_scatterAdd :: (Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_scatterFill : tensor dim index val -> void
foreign import ccall "THTensorMath.h THLongTensorMath_scatterFill"
  c_THLongTensorMath_scatterFill :: (Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> CLong -> IO ()

-- |c_THLongTensorMath_dot : t src -> accreal
foreign import ccall "THTensorMath.h THLongTensorMath_dot"
  c_THLongTensorMath_dot :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong

-- |c_THLongTensorMath_minall : t -> real
foreign import ccall "THTensorMath.h THLongTensorMath_minall"
  c_THLongTensorMath_minall :: (Ptr CTHLongTensor) -> CLong

-- |c_THLongTensorMath_maxall : t -> real
foreign import ccall "THTensorMath.h THLongTensorMath_maxall"
  c_THLongTensorMath_maxall :: (Ptr CTHLongTensor) -> CLong

-- |c_THLongTensorMath_medianall : t -> real
foreign import ccall "THTensorMath.h THLongTensorMath_medianall"
  c_THLongTensorMath_medianall :: (Ptr CTHLongTensor) -> CLong

-- |c_THLongTensorMath_sumall : t -> accreal
foreign import ccall "THTensorMath.h THLongTensorMath_sumall"
  c_THLongTensorMath_sumall :: (Ptr CTHLongTensor) -> CLong

-- |c_THLongTensorMath_prodall : t -> accreal
foreign import ccall "THTensorMath.h THLongTensorMath_prodall"
  c_THLongTensorMath_prodall :: (Ptr CTHLongTensor) -> CLong

-- |c_THLongTensorMath_neg : self src -> void
foreign import ccall "THTensorMath.h THLongTensorMath_neg"
  c_THLongTensorMath_neg :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_cinv : self src -> void
foreign import ccall "THTensorMath.h THLongTensorMath_cinv"
  c_THLongTensorMath_cinv :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_add : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_add"
  c_THLongTensorMath_add :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_sub : self src value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_sub"
  c_THLongTensorMath_sub :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_mul : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_mul"
  c_THLongTensorMath_mul :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_div : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_div"
  c_THLongTensorMath_div :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_lshift : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_lshift"
  c_THLongTensorMath_lshift :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_rshift : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_rshift"
  c_THLongTensorMath_rshift :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_fmod : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_fmod"
  c_THLongTensorMath_fmod :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_remainder : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_remainder"
  c_THLongTensorMath_remainder :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_clamp : r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_clamp"
  c_THLongTensorMath_clamp :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> IO ()

-- |c_THLongTensorMath_bitand : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_bitand"
  c_THLongTensorMath_bitand :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_bitor : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_bitor"
  c_THLongTensorMath_bitor :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_bitxor : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_bitxor"
  c_THLongTensorMath_bitxor :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_cadd : r_ t value src -> void
foreign import ccall "THTensorMath.h THLongTensorMath_cadd"
  c_THLongTensorMath_cadd :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_csub : self src1 value src2 -> void
foreign import ccall "THTensorMath.h THLongTensorMath_csub"
  c_THLongTensorMath_csub :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_cmul : r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensorMath_cmul"
  c_THLongTensorMath_cmul :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_cpow : r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensorMath_cpow"
  c_THLongTensorMath_cpow :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_cdiv : r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensorMath_cdiv"
  c_THLongTensorMath_cdiv :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_clshift : r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensorMath_clshift"
  c_THLongTensorMath_clshift :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_crshift : r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensorMath_crshift"
  c_THLongTensorMath_crshift :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_cfmod : r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensorMath_cfmod"
  c_THLongTensorMath_cfmod :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_cremainder : r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensorMath_cremainder"
  c_THLongTensorMath_cremainder :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_cbitand : r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensorMath_cbitand"
  c_THLongTensorMath_cbitand :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_cbitor : r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensorMath_cbitor"
  c_THLongTensorMath_cbitor :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_cbitxor : r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensorMath_cbitxor"
  c_THLongTensorMath_cbitxor :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_addcmul : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THLongTensorMath_addcmul"
  c_THLongTensorMath_addcmul :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_addcdiv : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THLongTensorMath_addcdiv"
  c_THLongTensorMath_addcdiv :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_addmv : r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h THLongTensorMath_addmv"
  c_THLongTensorMath_addmv :: (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_addmm : r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h THLongTensorMath_addmm"
  c_THLongTensorMath_addmm :: (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_addr : r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h THLongTensorMath_addr"
  c_THLongTensorMath_addr :: (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_addbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THLongTensorMath_addbmm"
  c_THLongTensorMath_addbmm :: (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_baddbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THLongTensorMath_baddbmm"
  c_THLongTensorMath_baddbmm :: (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_match : r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h THLongTensorMath_match"
  c_THLongTensorMath_match :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_numel : t -> THStorage *
foreign import ccall "THTensorMath.h THLongTensorMath_numel"
  c_THLongTensorMath_numel :: (Ptr CTHLongTensor) -> IO (Ptr CTHLongStorage)

-- |c_THLongTensorMath_max : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THLongTensorMath_max"
  c_THLongTensorMath_max :: (Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensorMath_min : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THLongTensorMath_min"
  c_THLongTensorMath_min :: (Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensorMath_kthvalue : values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h THLongTensorMath_kthvalue"
  c_THLongTensorMath_kthvalue :: (Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CLong -> CInt -> CInt -> IO ()

-- |c_THLongTensorMath_mode : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THLongTensorMath_mode"
  c_THLongTensorMath_mode :: (Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensorMath_median : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THLongTensorMath_median"
  c_THLongTensorMath_median :: (Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensorMath_sum : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THLongTensorMath_sum"
  c_THLongTensorMath_sum :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensorMath_prod : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THLongTensorMath_prod"
  c_THLongTensorMath_prod :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensorMath_cumsum : r_ t dimension -> void
foreign import ccall "THTensorMath.h THLongTensorMath_cumsum"
  c_THLongTensorMath_cumsum :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> IO ()

-- |c_THLongTensorMath_cumprod : r_ t dimension -> void
foreign import ccall "THTensorMath.h THLongTensorMath_cumprod"
  c_THLongTensorMath_cumprod :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> IO ()

-- |c_THLongTensorMath_sign : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensorMath_sign"
  c_THLongTensorMath_sign :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_trace : t -> accreal
foreign import ccall "THTensorMath.h THLongTensorMath_trace"
  c_THLongTensorMath_trace :: (Ptr CTHLongTensor) -> CLong

-- |c_THLongTensorMath_cross : r_ a b dimension -> void
foreign import ccall "THTensorMath.h THLongTensorMath_cross"
  c_THLongTensorMath_cross :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> IO ()

-- |c_THLongTensorMath_cmax : r t src -> void
foreign import ccall "THTensorMath.h THLongTensorMath_cmax"
  c_THLongTensorMath_cmax :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_cmin : r t src -> void
foreign import ccall "THTensorMath.h THLongTensorMath_cmin"
  c_THLongTensorMath_cmin :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_cmaxValue : r t value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_cmaxValue"
  c_THLongTensorMath_cmaxValue :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_cminValue : r t value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_cminValue"
  c_THLongTensorMath_cminValue :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_zeros : r_ size -> void
foreign import ccall "THTensorMath.h THLongTensorMath_zeros"
  c_THLongTensorMath_zeros :: (Ptr CTHLongTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THLongTensorMath_zerosLike : r_ input -> void
foreign import ccall "THTensorMath.h THLongTensorMath_zerosLike"
  c_THLongTensorMath_zerosLike :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_ones : r_ size -> void
foreign import ccall "THTensorMath.h THLongTensorMath_ones"
  c_THLongTensorMath_ones :: (Ptr CTHLongTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THLongTensorMath_onesLike : r_ input -> void
foreign import ccall "THTensorMath.h THLongTensorMath_onesLike"
  c_THLongTensorMath_onesLike :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_diag : r_ t k -> void
foreign import ccall "THTensorMath.h THLongTensorMath_diag"
  c_THLongTensorMath_diag :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> IO ()

-- |c_THLongTensorMath_eye : r_ n m -> void
foreign import ccall "THTensorMath.h THLongTensorMath_eye"
  c_THLongTensorMath_eye :: (Ptr CTHLongTensor) -> CLong -> CLong -> IO ()

-- |c_THLongTensorMath_arange : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THLongTensorMath_arange"
  c_THLongTensorMath_arange :: (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensorMath_range : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THLongTensorMath_range"
  c_THLongTensorMath_range :: (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensorMath_randperm : r_ _generator n -> void
foreign import ccall "THTensorMath.h THLongTensorMath_randperm"
  c_THLongTensorMath_randperm :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> CLong -> IO ()

-- |c_THLongTensorMath_reshape : r_ t size -> void
foreign import ccall "THTensorMath.h THLongTensorMath_reshape"
  c_THLongTensorMath_reshape :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THLongTensorMath_sort : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h THLongTensorMath_sort"
  c_THLongTensorMath_sort :: (Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensorMath_topk : rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h THLongTensorMath_topk"
  c_THLongTensorMath_topk :: (Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CLong -> CInt -> CInt -> CInt -> IO ()

-- |c_THLongTensorMath_tril : r_ t k -> void
foreign import ccall "THTensorMath.h THLongTensorMath_tril"
  c_THLongTensorMath_tril :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_triu : r_ t k -> void
foreign import ccall "THTensorMath.h THLongTensorMath_triu"
  c_THLongTensorMath_triu :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_cat : r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h THLongTensorMath_cat"
  c_THLongTensorMath_cat :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> IO ()

-- |c_THLongTensorMath_catArray : result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h THLongTensorMath_catArray"
  c_THLongTensorMath_catArray :: (Ptr CTHLongTensor) -> Ptr (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensorMath_equal : ta tb -> int
foreign import ccall "THTensorMath.h THLongTensorMath_equal"
  c_THLongTensorMath_equal :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt

-- |c_THLongTensorMath_ltValue : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_ltValue"
  c_THLongTensorMath_ltValue :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_leValue : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_leValue"
  c_THLongTensorMath_leValue :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_gtValue : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_gtValue"
  c_THLongTensorMath_gtValue :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_geValue : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_geValue"
  c_THLongTensorMath_geValue :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_neValue : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_neValue"
  c_THLongTensorMath_neValue :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_eqValue : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_eqValue"
  c_THLongTensorMath_eqValue :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_ltValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_ltValueT"
  c_THLongTensorMath_ltValueT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_leValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_leValueT"
  c_THLongTensorMath_leValueT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_gtValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_gtValueT"
  c_THLongTensorMath_gtValueT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_geValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_geValueT"
  c_THLongTensorMath_geValueT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_neValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_neValueT"
  c_THLongTensorMath_neValueT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_eqValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_eqValueT"
  c_THLongTensorMath_eqValueT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_ltTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensorMath_ltTensor"
  c_THLongTensorMath_ltTensor :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_leTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensorMath_leTensor"
  c_THLongTensorMath_leTensor :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_gtTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensorMath_gtTensor"
  c_THLongTensorMath_gtTensor :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_geTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensorMath_geTensor"
  c_THLongTensorMath_geTensor :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_neTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensorMath_neTensor"
  c_THLongTensorMath_neTensor :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_eqTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensorMath_eqTensor"
  c_THLongTensorMath_eqTensor :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_ltTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensorMath_ltTensorT"
  c_THLongTensorMath_ltTensorT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_leTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensorMath_leTensorT"
  c_THLongTensorMath_leTensorT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_gtTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensorMath_gtTensorT"
  c_THLongTensorMath_gtTensorT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_geTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensorMath_geTensorT"
  c_THLongTensorMath_geTensorT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_neTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensorMath_neTensorT"
  c_THLongTensorMath_neTensorT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_eqTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensorMath_eqTensorT"
  c_THLongTensorMath_eqTensorT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_abs : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensorMath_abs"
  c_THLongTensorMath_abs :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_sigmoid : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensorMath_sigmoid"
  c_THLongTensorMath_sigmoid :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_log : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensorMath_log"
  c_THLongTensorMath_log :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_lgamma : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensorMath_lgamma"
  c_THLongTensorMath_lgamma :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_log1p : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensorMath_log1p"
  c_THLongTensorMath_log1p :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_exp : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensorMath_exp"
  c_THLongTensorMath_exp :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_cos : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensorMath_cos"
  c_THLongTensorMath_cos :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_acos : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensorMath_acos"
  c_THLongTensorMath_acos :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_cosh : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensorMath_cosh"
  c_THLongTensorMath_cosh :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_sin : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensorMath_sin"
  c_THLongTensorMath_sin :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_asin : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensorMath_asin"
  c_THLongTensorMath_asin :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_sinh : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensorMath_sinh"
  c_THLongTensorMath_sinh :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_tan : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensorMath_tan"
  c_THLongTensorMath_tan :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_atan : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensorMath_atan"
  c_THLongTensorMath_atan :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_atan2 : r_ tx ty -> void
foreign import ccall "THTensorMath.h THLongTensorMath_atan2"
  c_THLongTensorMath_atan2 :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_tanh : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensorMath_tanh"
  c_THLongTensorMath_tanh :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_pow : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensorMath_pow"
  c_THLongTensorMath_pow :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_tpow : r_ value t -> void
foreign import ccall "THTensorMath.h THLongTensorMath_tpow"
  c_THLongTensorMath_tpow :: (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_sqrt : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensorMath_sqrt"
  c_THLongTensorMath_sqrt :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_rsqrt : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensorMath_rsqrt"
  c_THLongTensorMath_rsqrt :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_ceil : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensorMath_ceil"
  c_THLongTensorMath_ceil :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_floor : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensorMath_floor"
  c_THLongTensorMath_floor :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_round : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensorMath_round"
  c_THLongTensorMath_round :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_trunc : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensorMath_trunc"
  c_THLongTensorMath_trunc :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_frac : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensorMath_frac"
  c_THLongTensorMath_frac :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensorMath_lerp : r_ a b weight -> void
foreign import ccall "THTensorMath.h THLongTensorMath_lerp"
  c_THLongTensorMath_lerp :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensorMath_mean : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THLongTensorMath_mean"
  c_THLongTensorMath_mean :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensorMath_std : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THLongTensorMath_std"
  c_THLongTensorMath_std :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THLongTensorMath_var : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THLongTensorMath_var"
  c_THLongTensorMath_var :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THLongTensorMath_norm : r_ t value dimension keepdim -> void
foreign import ccall "THTensorMath.h THLongTensorMath_norm"
  c_THLongTensorMath_norm :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CInt -> CInt -> IO ()

-- |c_THLongTensorMath_renorm : r_ t value dimension maxnorm -> void
foreign import ccall "THTensorMath.h THLongTensorMath_renorm"
  c_THLongTensorMath_renorm :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CInt -> CLong -> IO ()

-- |c_THLongTensorMath_dist : a b value -> accreal
foreign import ccall "THTensorMath.h THLongTensorMath_dist"
  c_THLongTensorMath_dist :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong

-- |c_THLongTensorMath_histc : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THLongTensorMath_histc"
  c_THLongTensorMath_histc :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensorMath_bhistc : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THLongTensorMath_bhistc"
  c_THLongTensorMath_bhistc :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensorMath_meanall : self -> accreal
foreign import ccall "THTensorMath.h THLongTensorMath_meanall"
  c_THLongTensorMath_meanall :: (Ptr CTHLongTensor) -> CLong

-- |c_THLongTensorMath_varall : self biased -> accreal
foreign import ccall "THTensorMath.h THLongTensorMath_varall"
  c_THLongTensorMath_varall :: (Ptr CTHLongTensor) -> CInt -> CLong

-- |c_THLongTensorMath_stdall : self biased -> accreal
foreign import ccall "THTensorMath.h THLongTensorMath_stdall"
  c_THLongTensorMath_stdall :: (Ptr CTHLongTensor) -> CInt -> CLong

-- |c_THLongTensorMath_normall : t value -> accreal
foreign import ccall "THTensorMath.h THLongTensorMath_normall"
  c_THLongTensorMath_normall :: (Ptr CTHLongTensor) -> CLong -> CLong

-- |c_THLongTensorMath_linspace : r_ a b n -> void
foreign import ccall "THTensorMath.h THLongTensorMath_linspace"
  c_THLongTensorMath_linspace :: (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensorMath_logspace : r_ a b n -> void
foreign import ccall "THTensorMath.h THLongTensorMath_logspace"
  c_THLongTensorMath_logspace :: (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensorMath_rand : r_ _generator size -> void
foreign import ccall "THTensorMath.h THLongTensorMath_rand"
  c_THLongTensorMath_rand :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()

-- |c_THLongTensorMath_randn : r_ _generator size -> void
foreign import ccall "THTensorMath.h THLongTensorMath_randn"
  c_THLongTensorMath_randn :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()

-- |c_THLongTensorMath_logicalall : self -> int
foreign import ccall "THTensorMath.h THLongTensorMath_logicalall"
  c_THLongTensorMath_logicalall :: (Ptr CTHLongTensor) -> CInt

-- |c_THLongTensorMath_logicalany : self -> int
foreign import ccall "THTensorMath.h THLongTensorMath_logicalany"
  c_THLongTensorMath_logicalany :: (Ptr CTHLongTensor) -> CInt