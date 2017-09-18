{-# LANGUAGE ForeignFunctionInterface#-}

module THLongTensorMath (
    c_THLongTensor_fill,
    c_THLongTensor_zero,
    c_THLongTensor_maskedFill,
    c_THLongTensor_maskedCopy,
    c_THLongTensor_maskedSelect,
    c_THLongTensor_nonzero,
    c_THLongTensor_indexSelect,
    c_THLongTensor_indexCopy,
    c_THLongTensor_indexAdd,
    c_THLongTensor_indexFill,
    c_THLongTensor_gather,
    c_THLongTensor_scatter,
    c_THLongTensor_scatterAdd,
    c_THLongTensor_scatterFill,
    c_THLongTensor_dot,
    c_THLongTensor_minall,
    c_THLongTensor_maxall,
    c_THLongTensor_medianall,
    c_THLongTensor_sumall,
    c_THLongTensor_prodall,
    c_THLongTensor_neg,
    c_THLongTensor_cinv,
    c_THLongTensor_add,
    c_THLongTensor_sub,
    c_THLongTensor_mul,
    c_THLongTensor_div,
    c_THLongTensor_lshift,
    c_THLongTensor_rshift,
    c_THLongTensor_fmod,
    c_THLongTensor_remainder,
    c_THLongTensor_clamp,
    c_THLongTensor_bitand,
    c_THLongTensor_bitor,
    c_THLongTensor_bitxor,
    c_THLongTensor_cadd,
    c_THLongTensor_csub,
    c_THLongTensor_cmul,
    c_THLongTensor_cpow,
    c_THLongTensor_cdiv,
    c_THLongTensor_clshift,
    c_THLongTensor_crshift,
    c_THLongTensor_cfmod,
    c_THLongTensor_cremainder,
    c_THLongTensor_cbitand,
    c_THLongTensor_cbitor,
    c_THLongTensor_cbitxor,
    c_THLongTensor_addcmul,
    c_THLongTensor_addcdiv,
    c_THLongTensor_addmv,
    c_THLongTensor_addmm,
    c_THLongTensor_addr,
    c_THLongTensor_addbmm,
    c_THLongTensor_baddbmm,
    c_THLongTensor_match,
    c_THLongTensor_numel,
    c_THLongTensor_max,
    c_THLongTensor_min,
    c_THLongTensor_kthvalue,
    c_THLongTensor_mode,
    c_THLongTensor_median,
    c_THLongTensor_sum,
    c_THLongTensor_prod,
    c_THLongTensor_cumsum,
    c_THLongTensor_cumprod,
    c_THLongTensor_sign,
    c_THLongTensor_trace,
    c_THLongTensor_cross,
    c_THLongTensor_cmax,
    c_THLongTensor_cmin,
    c_THLongTensor_cmaxValue,
    c_THLongTensor_cminValue,
    c_THLongTensor_zeros,
    c_THLongTensor_zerosLike,
    c_THLongTensor_ones,
    c_THLongTensor_onesLike,
    c_THLongTensor_diag,
    c_THLongTensor_eye,
    c_THLongTensor_arange,
    c_THLongTensor_range,
    c_THLongTensor_randperm,
    c_THLongTensor_reshape,
    c_THLongTensor_sort,
    c_THLongTensor_topk,
    c_THLongTensor_tril,
    c_THLongTensor_triu,
    c_THLongTensor_cat,
    c_THLongTensor_catArray,
    c_THLongTensor_equal,
    c_THLongTensor_ltValue,
    c_THLongTensor_leValue,
    c_THLongTensor_gtValue,
    c_THLongTensor_geValue,
    c_THLongTensor_neValue,
    c_THLongTensor_eqValue,
    c_THLongTensor_ltValueT,
    c_THLongTensor_leValueT,
    c_THLongTensor_gtValueT,
    c_THLongTensor_geValueT,
    c_THLongTensor_neValueT,
    c_THLongTensor_eqValueT,
    c_THLongTensor_ltTensor,
    c_THLongTensor_leTensor,
    c_THLongTensor_gtTensor,
    c_THLongTensor_geTensor,
    c_THLongTensor_neTensor,
    c_THLongTensor_eqTensor,
    c_THLongTensor_ltTensorT,
    c_THLongTensor_leTensorT,
    c_THLongTensor_gtTensorT,
    c_THLongTensor_geTensorT,
    c_THLongTensor_neTensorT,
    c_THLongTensor_eqTensorT,
    c_THLongTensor_abs,
    c_THLongTensor_sigmoid,
    c_THLongTensor_log,
    c_THLongTensor_lgamma,
    c_THLongTensor_log1p,
    c_THLongTensor_exp,
    c_THLongTensor_cos,
    c_THLongTensor_acos,
    c_THLongTensor_cosh,
    c_THLongTensor_sin,
    c_THLongTensor_asin,
    c_THLongTensor_sinh,
    c_THLongTensor_tan,
    c_THLongTensor_atan,
    c_THLongTensor_atan2,
    c_THLongTensor_tanh,
    c_THLongTensor_pow,
    c_THLongTensor_tpow,
    c_THLongTensor_sqrt,
    c_THLongTensor_rsqrt,
    c_THLongTensor_ceil,
    c_THLongTensor_floor,
    c_THLongTensor_round,
    c_THLongTensor_trunc,
    c_THLongTensor_frac,
    c_THLongTensor_lerp,
    c_THLongTensor_mean,
    c_THLongTensor_std,
    c_THLongTensor_var,
    c_THLongTensor_norm,
    c_THLongTensor_renorm,
    c_THLongTensor_dist,
    c_THLongTensor_histc,
    c_THLongTensor_bhistc,
    c_THLongTensor_meanall,
    c_THLongTensor_varall,
    c_THLongTensor_stdall,
    c_THLongTensor_normall,
    c_THLongTensor_linspace,
    c_THLongTensor_logspace,
    c_THLongTensor_rand,
    c_THLongTensor_randn,
    c_THLongTensor_logicalall,
    c_THLongTensor_logicalany) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THLongTensor_fill : r_ value -> void
foreign import ccall "THTensorMath.h THLongTensor_fill"
  c_THLongTensor_fill :: (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_zero : r_ -> void
foreign import ccall "THTensorMath.h THLongTensor_zero"
  c_THLongTensor_zero :: (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_maskedFill : tensor mask value -> void
foreign import ccall "THTensorMath.h THLongTensor_maskedFill"
  c_THLongTensor_maskedFill :: (Ptr CTHLongTensor) -> Ptr CTHByteTensor -> CLong -> IO ()

-- |c_THLongTensor_maskedCopy : tensor mask src -> void
foreign import ccall "THTensorMath.h THLongTensor_maskedCopy"
  c_THLongTensor_maskedCopy :: (Ptr CTHLongTensor) -> Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_maskedSelect : tensor src mask -> void
foreign import ccall "THTensorMath.h THLongTensor_maskedSelect"
  c_THLongTensor_maskedSelect :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THLongTensor_nonzero : subscript tensor -> void
foreign import ccall "THTensorMath.h THLongTensor_nonzero"
  c_THLongTensor_nonzero :: Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_indexSelect : tensor src dim index -> void
foreign import ccall "THTensorMath.h THLongTensor_indexSelect"
  c_THLongTensor_indexSelect :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THLongTensor_indexCopy : tensor dim index src -> void
foreign import ccall "THTensorMath.h THLongTensor_indexCopy"
  c_THLongTensor_indexCopy :: (Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_indexAdd : tensor dim index src -> void
foreign import ccall "THTensorMath.h THLongTensor_indexAdd"
  c_THLongTensor_indexAdd :: (Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_indexFill : tensor dim index val -> void
foreign import ccall "THTensorMath.h THLongTensor_indexFill"
  c_THLongTensor_indexFill :: (Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> CLong -> IO ()

-- |c_THLongTensor_gather : tensor src dim index -> void
foreign import ccall "THTensorMath.h THLongTensor_gather"
  c_THLongTensor_gather :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THLongTensor_scatter : tensor dim index src -> void
foreign import ccall "THTensorMath.h THLongTensor_scatter"
  c_THLongTensor_scatter :: (Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_scatterAdd : tensor dim index src -> void
foreign import ccall "THTensorMath.h THLongTensor_scatterAdd"
  c_THLongTensor_scatterAdd :: (Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_scatterFill : tensor dim index val -> void
foreign import ccall "THTensorMath.h THLongTensor_scatterFill"
  c_THLongTensor_scatterFill :: (Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> CLong -> IO ()

-- |c_THLongTensor_dot : t src -> accreal
foreign import ccall "THTensorMath.h THLongTensor_dot"
  c_THLongTensor_dot :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong

-- |c_THLongTensor_minall : t -> real
foreign import ccall "THTensorMath.h THLongTensor_minall"
  c_THLongTensor_minall :: (Ptr CTHLongTensor) -> CLong

-- |c_THLongTensor_maxall : t -> real
foreign import ccall "THTensorMath.h THLongTensor_maxall"
  c_THLongTensor_maxall :: (Ptr CTHLongTensor) -> CLong

-- |c_THLongTensor_medianall : t -> real
foreign import ccall "THTensorMath.h THLongTensor_medianall"
  c_THLongTensor_medianall :: (Ptr CTHLongTensor) -> CLong

-- |c_THLongTensor_sumall : t -> accreal
foreign import ccall "THTensorMath.h THLongTensor_sumall"
  c_THLongTensor_sumall :: (Ptr CTHLongTensor) -> CLong

-- |c_THLongTensor_prodall : t -> accreal
foreign import ccall "THTensorMath.h THLongTensor_prodall"
  c_THLongTensor_prodall :: (Ptr CTHLongTensor) -> CLong

-- |c_THLongTensor_neg : self src -> void
foreign import ccall "THTensorMath.h THLongTensor_neg"
  c_THLongTensor_neg :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_cinv : self src -> void
foreign import ccall "THTensorMath.h THLongTensor_cinv"
  c_THLongTensor_cinv :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_add : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_add"
  c_THLongTensor_add :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_sub : self src value -> void
foreign import ccall "THTensorMath.h THLongTensor_sub"
  c_THLongTensor_sub :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_mul : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_mul"
  c_THLongTensor_mul :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_div : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_div"
  c_THLongTensor_div :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_lshift : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_lshift"
  c_THLongTensor_lshift :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_rshift : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_rshift"
  c_THLongTensor_rshift :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_fmod : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_fmod"
  c_THLongTensor_fmod :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_remainder : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_remainder"
  c_THLongTensor_remainder :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_clamp : r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h THLongTensor_clamp"
  c_THLongTensor_clamp :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> IO ()

-- |c_THLongTensor_bitand : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_bitand"
  c_THLongTensor_bitand :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_bitor : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_bitor"
  c_THLongTensor_bitor :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_bitxor : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_bitxor"
  c_THLongTensor_bitxor :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_cadd : r_ t value src -> void
foreign import ccall "THTensorMath.h THLongTensor_cadd"
  c_THLongTensor_cadd :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_csub : self src1 value src2 -> void
foreign import ccall "THTensorMath.h THLongTensor_csub"
  c_THLongTensor_csub :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_cmul : r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensor_cmul"
  c_THLongTensor_cmul :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_cpow : r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensor_cpow"
  c_THLongTensor_cpow :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_cdiv : r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensor_cdiv"
  c_THLongTensor_cdiv :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_clshift : r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensor_clshift"
  c_THLongTensor_clshift :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_crshift : r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensor_crshift"
  c_THLongTensor_crshift :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_cfmod : r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensor_cfmod"
  c_THLongTensor_cfmod :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_cremainder : r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensor_cremainder"
  c_THLongTensor_cremainder :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_cbitand : r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensor_cbitand"
  c_THLongTensor_cbitand :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_cbitor : r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensor_cbitor"
  c_THLongTensor_cbitor :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_cbitxor : r_ t src -> void
foreign import ccall "THTensorMath.h THLongTensor_cbitxor"
  c_THLongTensor_cbitxor :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_addcmul : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THLongTensor_addcmul"
  c_THLongTensor_addcmul :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_addcdiv : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THLongTensor_addcdiv"
  c_THLongTensor_addcdiv :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_addmv : r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h THLongTensor_addmv"
  c_THLongTensor_addmv :: (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_addmm : r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h THLongTensor_addmm"
  c_THLongTensor_addmm :: (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_addr : r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h THLongTensor_addr"
  c_THLongTensor_addr :: (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_addbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THLongTensor_addbmm"
  c_THLongTensor_addbmm :: (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_baddbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THLongTensor_baddbmm"
  c_THLongTensor_baddbmm :: (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_match : r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h THLongTensor_match"
  c_THLongTensor_match :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_numel : t -> THStorage *
foreign import ccall "THTensorMath.h THLongTensor_numel"
  c_THLongTensor_numel :: (Ptr CTHLongTensor) -> IO (Ptr CTHLongStorage)

-- |c_THLongTensor_max : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THLongTensor_max"
  c_THLongTensor_max :: (Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensor_min : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THLongTensor_min"
  c_THLongTensor_min :: (Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensor_kthvalue : values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h THLongTensor_kthvalue"
  c_THLongTensor_kthvalue :: (Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CLong -> CInt -> CInt -> IO ()

-- |c_THLongTensor_mode : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THLongTensor_mode"
  c_THLongTensor_mode :: (Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensor_median : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THLongTensor_median"
  c_THLongTensor_median :: (Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensor_sum : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THLongTensor_sum"
  c_THLongTensor_sum :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensor_prod : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THLongTensor_prod"
  c_THLongTensor_prod :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensor_cumsum : r_ t dimension -> void
foreign import ccall "THTensorMath.h THLongTensor_cumsum"
  c_THLongTensor_cumsum :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> IO ()

-- |c_THLongTensor_cumprod : r_ t dimension -> void
foreign import ccall "THTensorMath.h THLongTensor_cumprod"
  c_THLongTensor_cumprod :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> IO ()

-- |c_THLongTensor_sign : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensor_sign"
  c_THLongTensor_sign :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_trace : t -> accreal
foreign import ccall "THTensorMath.h THLongTensor_trace"
  c_THLongTensor_trace :: (Ptr CTHLongTensor) -> CLong

-- |c_THLongTensor_cross : r_ a b dimension -> void
foreign import ccall "THTensorMath.h THLongTensor_cross"
  c_THLongTensor_cross :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> IO ()

-- |c_THLongTensor_cmax : r t src -> void
foreign import ccall "THTensorMath.h THLongTensor_cmax"
  c_THLongTensor_cmax :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_cmin : r t src -> void
foreign import ccall "THTensorMath.h THLongTensor_cmin"
  c_THLongTensor_cmin :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_cmaxValue : r t value -> void
foreign import ccall "THTensorMath.h THLongTensor_cmaxValue"
  c_THLongTensor_cmaxValue :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_cminValue : r t value -> void
foreign import ccall "THTensorMath.h THLongTensor_cminValue"
  c_THLongTensor_cminValue :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_zeros : r_ size -> void
foreign import ccall "THTensorMath.h THLongTensor_zeros"
  c_THLongTensor_zeros :: (Ptr CTHLongTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THLongTensor_zerosLike : r_ input -> void
foreign import ccall "THTensorMath.h THLongTensor_zerosLike"
  c_THLongTensor_zerosLike :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_ones : r_ size -> void
foreign import ccall "THTensorMath.h THLongTensor_ones"
  c_THLongTensor_ones :: (Ptr CTHLongTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THLongTensor_onesLike : r_ input -> void
foreign import ccall "THTensorMath.h THLongTensor_onesLike"
  c_THLongTensor_onesLike :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_diag : r_ t k -> void
foreign import ccall "THTensorMath.h THLongTensor_diag"
  c_THLongTensor_diag :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> IO ()

-- |c_THLongTensor_eye : r_ n m -> void
foreign import ccall "THTensorMath.h THLongTensor_eye"
  c_THLongTensor_eye :: (Ptr CTHLongTensor) -> CLong -> CLong -> IO ()

-- |c_THLongTensor_arange : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THLongTensor_arange"
  c_THLongTensor_arange :: (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_range : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THLongTensor_range"
  c_THLongTensor_range :: (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_randperm : r_ _generator n -> void
foreign import ccall "THTensorMath.h THLongTensor_randperm"
  c_THLongTensor_randperm :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> CLong -> IO ()

-- |c_THLongTensor_reshape : r_ t size -> void
foreign import ccall "THTensorMath.h THLongTensor_reshape"
  c_THLongTensor_reshape :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THLongTensor_sort : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h THLongTensor_sort"
  c_THLongTensor_sort :: (Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensor_topk : rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h THLongTensor_topk"
  c_THLongTensor_topk :: (Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CLong -> CInt -> CInt -> CInt -> IO ()

-- |c_THLongTensor_tril : r_ t k -> void
foreign import ccall "THTensorMath.h THLongTensor_tril"
  c_THLongTensor_tril :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_triu : r_ t k -> void
foreign import ccall "THTensorMath.h THLongTensor_triu"
  c_THLongTensor_triu :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_cat : r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h THLongTensor_cat"
  c_THLongTensor_cat :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> IO ()

-- |c_THLongTensor_catArray : result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h THLongTensor_catArray"
  c_THLongTensor_catArray :: (Ptr CTHLongTensor) -> Ptr (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensor_equal : ta tb -> int
foreign import ccall "THTensorMath.h THLongTensor_equal"
  c_THLongTensor_equal :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt

-- |c_THLongTensor_ltValue : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_ltValue"
  c_THLongTensor_ltValue :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_leValue : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_leValue"
  c_THLongTensor_leValue :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_gtValue : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_gtValue"
  c_THLongTensor_gtValue :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_geValue : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_geValue"
  c_THLongTensor_geValue :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_neValue : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_neValue"
  c_THLongTensor_neValue :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_eqValue : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_eqValue"
  c_THLongTensor_eqValue :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_ltValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_ltValueT"
  c_THLongTensor_ltValueT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_leValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_leValueT"
  c_THLongTensor_leValueT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_gtValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_gtValueT"
  c_THLongTensor_gtValueT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_geValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_geValueT"
  c_THLongTensor_geValueT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_neValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_neValueT"
  c_THLongTensor_neValueT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_eqValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_eqValueT"
  c_THLongTensor_eqValueT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_ltTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensor_ltTensor"
  c_THLongTensor_ltTensor :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_leTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensor_leTensor"
  c_THLongTensor_leTensor :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_gtTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensor_gtTensor"
  c_THLongTensor_gtTensor :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_geTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensor_geTensor"
  c_THLongTensor_geTensor :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_neTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensor_neTensor"
  c_THLongTensor_neTensor :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_eqTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensor_eqTensor"
  c_THLongTensor_eqTensor :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_ltTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensor_ltTensorT"
  c_THLongTensor_ltTensorT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_leTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensor_leTensorT"
  c_THLongTensor_leTensorT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_gtTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensor_gtTensorT"
  c_THLongTensor_gtTensorT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_geTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensor_geTensorT"
  c_THLongTensor_geTensorT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_neTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensor_neTensorT"
  c_THLongTensor_neTensorT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_eqTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THLongTensor_eqTensorT"
  c_THLongTensor_eqTensorT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_abs : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensor_abs"
  c_THLongTensor_abs :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_sigmoid : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensor_sigmoid"
  c_THLongTensor_sigmoid :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_log : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensor_log"
  c_THLongTensor_log :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_lgamma : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensor_lgamma"
  c_THLongTensor_lgamma :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_log1p : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensor_log1p"
  c_THLongTensor_log1p :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_exp : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensor_exp"
  c_THLongTensor_exp :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_cos : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensor_cos"
  c_THLongTensor_cos :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_acos : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensor_acos"
  c_THLongTensor_acos :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_cosh : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensor_cosh"
  c_THLongTensor_cosh :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_sin : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensor_sin"
  c_THLongTensor_sin :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_asin : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensor_asin"
  c_THLongTensor_asin :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_sinh : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensor_sinh"
  c_THLongTensor_sinh :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_tan : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensor_tan"
  c_THLongTensor_tan :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_atan : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensor_atan"
  c_THLongTensor_atan :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_atan2 : r_ tx ty -> void
foreign import ccall "THTensorMath.h THLongTensor_atan2"
  c_THLongTensor_atan2 :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_tanh : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensor_tanh"
  c_THLongTensor_tanh :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_pow : r_ t value -> void
foreign import ccall "THTensorMath.h THLongTensor_pow"
  c_THLongTensor_pow :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_tpow : r_ value t -> void
foreign import ccall "THTensorMath.h THLongTensor_tpow"
  c_THLongTensor_tpow :: (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_sqrt : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensor_sqrt"
  c_THLongTensor_sqrt :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_rsqrt : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensor_rsqrt"
  c_THLongTensor_rsqrt :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_ceil : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensor_ceil"
  c_THLongTensor_ceil :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_floor : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensor_floor"
  c_THLongTensor_floor :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_round : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensor_round"
  c_THLongTensor_round :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_trunc : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensor_trunc"
  c_THLongTensor_trunc :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_frac : r_ t -> void
foreign import ccall "THTensorMath.h THLongTensor_frac"
  c_THLongTensor_frac :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_lerp : r_ a b weight -> void
foreign import ccall "THTensorMath.h THLongTensor_lerp"
  c_THLongTensor_lerp :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_mean : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THLongTensor_mean"
  c_THLongTensor_mean :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensor_std : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THLongTensor_std"
  c_THLongTensor_std :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THLongTensor_var : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THLongTensor_var"
  c_THLongTensor_var :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THLongTensor_norm : r_ t value dimension keepdim -> void
foreign import ccall "THTensorMath.h THLongTensor_norm"
  c_THLongTensor_norm :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CInt -> CInt -> IO ()

-- |c_THLongTensor_renorm : r_ t value dimension maxnorm -> void
foreign import ccall "THTensorMath.h THLongTensor_renorm"
  c_THLongTensor_renorm :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CInt -> CLong -> IO ()

-- |c_THLongTensor_dist : a b value -> accreal
foreign import ccall "THTensorMath.h THLongTensor_dist"
  c_THLongTensor_dist :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong

-- |c_THLongTensor_histc : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THLongTensor_histc"
  c_THLongTensor_histc :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_bhistc : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THLongTensor_bhistc"
  c_THLongTensor_bhistc :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_meanall : self -> accreal
foreign import ccall "THTensorMath.h THLongTensor_meanall"
  c_THLongTensor_meanall :: (Ptr CTHLongTensor) -> CLong

-- |c_THLongTensor_varall : self biased -> accreal
foreign import ccall "THTensorMath.h THLongTensor_varall"
  c_THLongTensor_varall :: (Ptr CTHLongTensor) -> CInt -> CLong

-- |c_THLongTensor_stdall : self biased -> accreal
foreign import ccall "THTensorMath.h THLongTensor_stdall"
  c_THLongTensor_stdall :: (Ptr CTHLongTensor) -> CInt -> CLong

-- |c_THLongTensor_normall : t value -> accreal
foreign import ccall "THTensorMath.h THLongTensor_normall"
  c_THLongTensor_normall :: (Ptr CTHLongTensor) -> CLong -> CLong

-- |c_THLongTensor_linspace : r_ a b n -> void
foreign import ccall "THTensorMath.h THLongTensor_linspace"
  c_THLongTensor_linspace :: (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_logspace : r_ a b n -> void
foreign import ccall "THTensorMath.h THLongTensor_logspace"
  c_THLongTensor_logspace :: (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_rand : r_ _generator size -> void
foreign import ccall "THTensorMath.h THLongTensor_rand"
  c_THLongTensor_rand :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()

-- |c_THLongTensor_randn : r_ _generator size -> void
foreign import ccall "THTensorMath.h THLongTensor_randn"
  c_THLongTensor_randn :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()

-- |c_THLongTensor_logicalall : self -> int
foreign import ccall "THTensorMath.h THLongTensor_logicalall"
  c_THLongTensor_logicalall :: (Ptr CTHLongTensor) -> CInt

-- |c_THLongTensor_logicalany : self -> int
foreign import ccall "THTensorMath.h THLongTensor_logicalany"
  c_THLongTensor_logicalany :: (Ptr CTHLongTensor) -> CInt