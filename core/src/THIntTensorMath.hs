{-# LANGUAGE ForeignFunctionInterface#-}

module THIntTensorMath (
    c_THIntTensorMath_fill,
    c_THIntTensorMath_zero,
    c_THIntTensorMath_maskedFill,
    c_THIntTensorMath_maskedCopy,
    c_THIntTensorMath_maskedSelect,
    c_THIntTensorMath_nonzero,
    c_THIntTensorMath_indexSelect,
    c_THIntTensorMath_indexCopy,
    c_THIntTensorMath_indexAdd,
    c_THIntTensorMath_indexFill,
    c_THIntTensorMath_gather,
    c_THIntTensorMath_scatter,
    c_THIntTensorMath_scatterAdd,
    c_THIntTensorMath_scatterFill,
    c_THIntTensorMath_dot,
    c_THIntTensorMath_minall,
    c_THIntTensorMath_maxall,
    c_THIntTensorMath_medianall,
    c_THIntTensorMath_sumall,
    c_THIntTensorMath_prodall,
    c_THIntTensorMath_neg,
    c_THIntTensorMath_cinv,
    c_THIntTensorMath_add,
    c_THIntTensorMath_sub,
    c_THIntTensorMath_mul,
    c_THIntTensorMath_div,
    c_THIntTensorMath_lshift,
    c_THIntTensorMath_rshift,
    c_THIntTensorMath_fmod,
    c_THIntTensorMath_remainder,
    c_THIntTensorMath_clamp,
    c_THIntTensorMath_bitand,
    c_THIntTensorMath_bitor,
    c_THIntTensorMath_bitxor,
    c_THIntTensorMath_cadd,
    c_THIntTensorMath_csub,
    c_THIntTensorMath_cmul,
    c_THIntTensorMath_cpow,
    c_THIntTensorMath_cdiv,
    c_THIntTensorMath_clshift,
    c_THIntTensorMath_crshift,
    c_THIntTensorMath_cfmod,
    c_THIntTensorMath_cremainder,
    c_THIntTensorMath_cbitand,
    c_THIntTensorMath_cbitor,
    c_THIntTensorMath_cbitxor,
    c_THIntTensorMath_addcmul,
    c_THIntTensorMath_addcdiv,
    c_THIntTensorMath_addmv,
    c_THIntTensorMath_addmm,
    c_THIntTensorMath_addr,
    c_THIntTensorMath_addbmm,
    c_THIntTensorMath_baddbmm,
    c_THIntTensorMath_match,
    c_THIntTensorMath_numel,
    c_THIntTensorMath_max,
    c_THIntTensorMath_min,
    c_THIntTensorMath_kthvalue,
    c_THIntTensorMath_mode,
    c_THIntTensorMath_median,
    c_THIntTensorMath_sum,
    c_THIntTensorMath_prod,
    c_THIntTensorMath_cumsum,
    c_THIntTensorMath_cumprod,
    c_THIntTensorMath_sign,
    c_THIntTensorMath_trace,
    c_THIntTensorMath_cross,
    c_THIntTensorMath_cmax,
    c_THIntTensorMath_cmin,
    c_THIntTensorMath_cmaxValue,
    c_THIntTensorMath_cminValue,
    c_THIntTensorMath_zeros,
    c_THIntTensorMath_zerosLike,
    c_THIntTensorMath_ones,
    c_THIntTensorMath_onesLike,
    c_THIntTensorMath_diag,
    c_THIntTensorMath_eye,
    c_THIntTensorMath_arange,
    c_THIntTensorMath_range,
    c_THIntTensorMath_randperm,
    c_THIntTensorMath_reshape,
    c_THIntTensorMath_sort,
    c_THIntTensorMath_topk,
    c_THIntTensorMath_tril,
    c_THIntTensorMath_triu,
    c_THIntTensorMath_cat,
    c_THIntTensorMath_catArray,
    c_THIntTensorMath_equal,
    c_THIntTensorMath_ltValue,
    c_THIntTensorMath_leValue,
    c_THIntTensorMath_gtValue,
    c_THIntTensorMath_geValue,
    c_THIntTensorMath_neValue,
    c_THIntTensorMath_eqValue,
    c_THIntTensorMath_ltValueT,
    c_THIntTensorMath_leValueT,
    c_THIntTensorMath_gtValueT,
    c_THIntTensorMath_geValueT,
    c_THIntTensorMath_neValueT,
    c_THIntTensorMath_eqValueT,
    c_THIntTensorMath_ltTensor,
    c_THIntTensorMath_leTensor,
    c_THIntTensorMath_gtTensor,
    c_THIntTensorMath_geTensor,
    c_THIntTensorMath_neTensor,
    c_THIntTensorMath_eqTensor,
    c_THIntTensorMath_ltTensorT,
    c_THIntTensorMath_leTensorT,
    c_THIntTensorMath_gtTensorT,
    c_THIntTensorMath_geTensorT,
    c_THIntTensorMath_neTensorT,
    c_THIntTensorMath_eqTensorT,
    c_THIntTensorMath_abs,
    c_THIntTensorMath_sigmoid,
    c_THIntTensorMath_log,
    c_THIntTensorMath_lgamma,
    c_THIntTensorMath_log1p,
    c_THIntTensorMath_exp,
    c_THIntTensorMath_cos,
    c_THIntTensorMath_acos,
    c_THIntTensorMath_cosh,
    c_THIntTensorMath_sin,
    c_THIntTensorMath_asin,
    c_THIntTensorMath_sinh,
    c_THIntTensorMath_tan,
    c_THIntTensorMath_atan,
    c_THIntTensorMath_atan2,
    c_THIntTensorMath_tanh,
    c_THIntTensorMath_pow,
    c_THIntTensorMath_tpow,
    c_THIntTensorMath_sqrt,
    c_THIntTensorMath_rsqrt,
    c_THIntTensorMath_ceil,
    c_THIntTensorMath_floor,
    c_THIntTensorMath_round,
    c_THIntTensorMath_trunc,
    c_THIntTensorMath_frac,
    c_THIntTensorMath_lerp,
    c_THIntTensorMath_mean,
    c_THIntTensorMath_std,
    c_THIntTensorMath_var,
    c_THIntTensorMath_norm,
    c_THIntTensorMath_renorm,
    c_THIntTensorMath_dist,
    c_THIntTensorMath_histc,
    c_THIntTensorMath_bhistc,
    c_THIntTensorMath_meanall,
    c_THIntTensorMath_varall,
    c_THIntTensorMath_stdall,
    c_THIntTensorMath_normall,
    c_THIntTensorMath_linspace,
    c_THIntTensorMath_logspace,
    c_THIntTensorMath_rand,
    c_THIntTensorMath_randn,
    c_THIntTensorMath_logicalall,
    c_THIntTensorMath_logicalany) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THIntTensorMath_fill : r_ value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_fill"
  c_THIntTensorMath_fill :: (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_zero : r_ -> void
foreign import ccall "THTensorMath.h THIntTensorMath_zero"
  c_THIntTensorMath_zero :: (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_maskedFill : tensor mask value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_maskedFill"
  c_THIntTensorMath_maskedFill :: (Ptr CTHIntTensor) -> Ptr CTHByteTensor -> CInt -> IO ()

-- |c_THIntTensorMath_maskedCopy : tensor mask src -> void
foreign import ccall "THTensorMath.h THIntTensorMath_maskedCopy"
  c_THIntTensorMath_maskedCopy :: (Ptr CTHIntTensor) -> Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_maskedSelect : tensor src mask -> void
foreign import ccall "THTensorMath.h THIntTensorMath_maskedSelect"
  c_THIntTensorMath_maskedSelect :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THIntTensorMath_nonzero : subscript tensor -> void
foreign import ccall "THTensorMath.h THIntTensorMath_nonzero"
  c_THIntTensorMath_nonzero :: Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_indexSelect : tensor src dim index -> void
foreign import ccall "THTensorMath.h THIntTensorMath_indexSelect"
  c_THIntTensorMath_indexSelect :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THIntTensorMath_indexCopy : tensor dim index src -> void
foreign import ccall "THTensorMath.h THIntTensorMath_indexCopy"
  c_THIntTensorMath_indexCopy :: (Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_indexAdd : tensor dim index src -> void
foreign import ccall "THTensorMath.h THIntTensorMath_indexAdd"
  c_THIntTensorMath_indexAdd :: (Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_indexFill : tensor dim index val -> void
foreign import ccall "THTensorMath.h THIntTensorMath_indexFill"
  c_THIntTensorMath_indexFill :: (Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> CInt -> IO ()

-- |c_THIntTensorMath_gather : tensor src dim index -> void
foreign import ccall "THTensorMath.h THIntTensorMath_gather"
  c_THIntTensorMath_gather :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THIntTensorMath_scatter : tensor dim index src -> void
foreign import ccall "THTensorMath.h THIntTensorMath_scatter"
  c_THIntTensorMath_scatter :: (Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_scatterAdd : tensor dim index src -> void
foreign import ccall "THTensorMath.h THIntTensorMath_scatterAdd"
  c_THIntTensorMath_scatterAdd :: (Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_scatterFill : tensor dim index val -> void
foreign import ccall "THTensorMath.h THIntTensorMath_scatterFill"
  c_THIntTensorMath_scatterFill :: (Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> CInt -> IO ()

-- |c_THIntTensorMath_dot : t src -> accreal
foreign import ccall "THTensorMath.h THIntTensorMath_dot"
  c_THIntTensorMath_dot :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CLong

-- |c_THIntTensorMath_minall : t -> real
foreign import ccall "THTensorMath.h THIntTensorMath_minall"
  c_THIntTensorMath_minall :: (Ptr CTHIntTensor) -> CInt

-- |c_THIntTensorMath_maxall : t -> real
foreign import ccall "THTensorMath.h THIntTensorMath_maxall"
  c_THIntTensorMath_maxall :: (Ptr CTHIntTensor) -> CInt

-- |c_THIntTensorMath_medianall : t -> real
foreign import ccall "THTensorMath.h THIntTensorMath_medianall"
  c_THIntTensorMath_medianall :: (Ptr CTHIntTensor) -> CInt

-- |c_THIntTensorMath_sumall : t -> accreal
foreign import ccall "THTensorMath.h THIntTensorMath_sumall"
  c_THIntTensorMath_sumall :: (Ptr CTHIntTensor) -> CLong

-- |c_THIntTensorMath_prodall : t -> accreal
foreign import ccall "THTensorMath.h THIntTensorMath_prodall"
  c_THIntTensorMath_prodall :: (Ptr CTHIntTensor) -> CLong

-- |c_THIntTensorMath_neg : self src -> void
foreign import ccall "THTensorMath.h THIntTensorMath_neg"
  c_THIntTensorMath_neg :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_cinv : self src -> void
foreign import ccall "THTensorMath.h THIntTensorMath_cinv"
  c_THIntTensorMath_cinv :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_add : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_add"
  c_THIntTensorMath_add :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_sub : self src value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_sub"
  c_THIntTensorMath_sub :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_mul : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_mul"
  c_THIntTensorMath_mul :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_div : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_div"
  c_THIntTensorMath_div :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_lshift : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_lshift"
  c_THIntTensorMath_lshift :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_rshift : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_rshift"
  c_THIntTensorMath_rshift :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_fmod : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_fmod"
  c_THIntTensorMath_fmod :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_remainder : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_remainder"
  c_THIntTensorMath_remainder :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_clamp : r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_clamp"
  c_THIntTensorMath_clamp :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensorMath_bitand : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_bitand"
  c_THIntTensorMath_bitand :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_bitor : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_bitor"
  c_THIntTensorMath_bitor :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_bitxor : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_bitxor"
  c_THIntTensorMath_bitxor :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_cadd : r_ t value src -> void
foreign import ccall "THTensorMath.h THIntTensorMath_cadd"
  c_THIntTensorMath_cadd :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_csub : self src1 value src2 -> void
foreign import ccall "THTensorMath.h THIntTensorMath_csub"
  c_THIntTensorMath_csub :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_cmul : r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensorMath_cmul"
  c_THIntTensorMath_cmul :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_cpow : r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensorMath_cpow"
  c_THIntTensorMath_cpow :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_cdiv : r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensorMath_cdiv"
  c_THIntTensorMath_cdiv :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_clshift : r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensorMath_clshift"
  c_THIntTensorMath_clshift :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_crshift : r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensorMath_crshift"
  c_THIntTensorMath_crshift :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_cfmod : r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensorMath_cfmod"
  c_THIntTensorMath_cfmod :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_cremainder : r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensorMath_cremainder"
  c_THIntTensorMath_cremainder :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_cbitand : r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensorMath_cbitand"
  c_THIntTensorMath_cbitand :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_cbitor : r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensorMath_cbitor"
  c_THIntTensorMath_cbitor :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_cbitxor : r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensorMath_cbitxor"
  c_THIntTensorMath_cbitxor :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_addcmul : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THIntTensorMath_addcmul"
  c_THIntTensorMath_addcmul :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_addcdiv : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THIntTensorMath_addcdiv"
  c_THIntTensorMath_addcdiv :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_addmv : r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h THIntTensorMath_addmv"
  c_THIntTensorMath_addmv :: (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_addmm : r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h THIntTensorMath_addmm"
  c_THIntTensorMath_addmm :: (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_addr : r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h THIntTensorMath_addr"
  c_THIntTensorMath_addr :: (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_addbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THIntTensorMath_addbmm"
  c_THIntTensorMath_addbmm :: (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_baddbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THIntTensorMath_baddbmm"
  c_THIntTensorMath_baddbmm :: (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_match : r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h THIntTensorMath_match"
  c_THIntTensorMath_match :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_numel : t -> THStorage *
foreign import ccall "THTensorMath.h THIntTensorMath_numel"
  c_THIntTensorMath_numel :: (Ptr CTHIntTensor) -> IO (Ptr CTHIntStorage)

-- |c_THIntTensorMath_max : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensorMath_max"
  c_THIntTensorMath_max :: (Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensorMath_min : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensorMath_min"
  c_THIntTensorMath_min :: (Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensorMath_kthvalue : values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensorMath_kthvalue"
  c_THIntTensorMath_kthvalue :: (Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CLong -> CInt -> CInt -> IO ()

-- |c_THIntTensorMath_mode : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensorMath_mode"
  c_THIntTensorMath_mode :: (Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensorMath_median : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensorMath_median"
  c_THIntTensorMath_median :: (Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensorMath_sum : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensorMath_sum"
  c_THIntTensorMath_sum :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensorMath_prod : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensorMath_prod"
  c_THIntTensorMath_prod :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensorMath_cumsum : r_ t dimension -> void
foreign import ccall "THTensorMath.h THIntTensorMath_cumsum"
  c_THIntTensorMath_cumsum :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_cumprod : r_ t dimension -> void
foreign import ccall "THTensorMath.h THIntTensorMath_cumprod"
  c_THIntTensorMath_cumprod :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_sign : r_ t -> void
foreign import ccall "THTensorMath.h THIntTensorMath_sign"
  c_THIntTensorMath_sign :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_trace : t -> accreal
foreign import ccall "THTensorMath.h THIntTensorMath_trace"
  c_THIntTensorMath_trace :: (Ptr CTHIntTensor) -> CLong

-- |c_THIntTensorMath_cross : r_ a b dimension -> void
foreign import ccall "THTensorMath.h THIntTensorMath_cross"
  c_THIntTensorMath_cross :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_cmax : r t src -> void
foreign import ccall "THTensorMath.h THIntTensorMath_cmax"
  c_THIntTensorMath_cmax :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_cmin : r t src -> void
foreign import ccall "THTensorMath.h THIntTensorMath_cmin"
  c_THIntTensorMath_cmin :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_cmaxValue : r t value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_cmaxValue"
  c_THIntTensorMath_cmaxValue :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_cminValue : r t value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_cminValue"
  c_THIntTensorMath_cminValue :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_zeros : r_ size -> void
foreign import ccall "THTensorMath.h THIntTensorMath_zeros"
  c_THIntTensorMath_zeros :: (Ptr CTHIntTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THIntTensorMath_zerosLike : r_ input -> void
foreign import ccall "THTensorMath.h THIntTensorMath_zerosLike"
  c_THIntTensorMath_zerosLike :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_ones : r_ size -> void
foreign import ccall "THTensorMath.h THIntTensorMath_ones"
  c_THIntTensorMath_ones :: (Ptr CTHIntTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THIntTensorMath_onesLike : r_ input -> void
foreign import ccall "THTensorMath.h THIntTensorMath_onesLike"
  c_THIntTensorMath_onesLike :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_diag : r_ t k -> void
foreign import ccall "THTensorMath.h THIntTensorMath_diag"
  c_THIntTensorMath_diag :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_eye : r_ n m -> void
foreign import ccall "THTensorMath.h THIntTensorMath_eye"
  c_THIntTensorMath_eye :: (Ptr CTHIntTensor) -> CLong -> CLong -> IO ()

-- |c_THIntTensorMath_arange : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THIntTensorMath_arange"
  c_THIntTensorMath_arange :: (Ptr CTHIntTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THIntTensorMath_range : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THIntTensorMath_range"
  c_THIntTensorMath_range :: (Ptr CTHIntTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THIntTensorMath_randperm : r_ _generator n -> void
foreign import ccall "THTensorMath.h THIntTensorMath_randperm"
  c_THIntTensorMath_randperm :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> CLong -> IO ()

-- |c_THIntTensorMath_reshape : r_ t size -> void
foreign import ccall "THTensorMath.h THIntTensorMath_reshape"
  c_THIntTensorMath_reshape :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THIntTensorMath_sort : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h THIntTensorMath_sort"
  c_THIntTensorMath_sort :: (Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensorMath_topk : rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h THIntTensorMath_topk"
  c_THIntTensorMath_topk :: (Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CLong -> CInt -> CInt -> CInt -> IO ()

-- |c_THIntTensorMath_tril : r_ t k -> void
foreign import ccall "THTensorMath.h THIntTensorMath_tril"
  c_THIntTensorMath_tril :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CLong -> IO ()

-- |c_THIntTensorMath_triu : r_ t k -> void
foreign import ccall "THTensorMath.h THIntTensorMath_triu"
  c_THIntTensorMath_triu :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CLong -> IO ()

-- |c_THIntTensorMath_cat : r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h THIntTensorMath_cat"
  c_THIntTensorMath_cat :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_catArray : result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h THIntTensorMath_catArray"
  c_THIntTensorMath_catArray :: (Ptr CTHIntTensor) -> Ptr (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensorMath_equal : ta tb -> int
foreign import ccall "THTensorMath.h THIntTensorMath_equal"
  c_THIntTensorMath_equal :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt

-- |c_THIntTensorMath_ltValue : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_ltValue"
  c_THIntTensorMath_ltValue :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_leValue : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_leValue"
  c_THIntTensorMath_leValue :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_gtValue : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_gtValue"
  c_THIntTensorMath_gtValue :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_geValue : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_geValue"
  c_THIntTensorMath_geValue :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_neValue : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_neValue"
  c_THIntTensorMath_neValue :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_eqValue : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_eqValue"
  c_THIntTensorMath_eqValue :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_ltValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_ltValueT"
  c_THIntTensorMath_ltValueT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_leValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_leValueT"
  c_THIntTensorMath_leValueT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_gtValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_gtValueT"
  c_THIntTensorMath_gtValueT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_geValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_geValueT"
  c_THIntTensorMath_geValueT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_neValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_neValueT"
  c_THIntTensorMath_neValueT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_eqValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_eqValueT"
  c_THIntTensorMath_eqValueT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_ltTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensorMath_ltTensor"
  c_THIntTensorMath_ltTensor :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_leTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensorMath_leTensor"
  c_THIntTensorMath_leTensor :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_gtTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensorMath_gtTensor"
  c_THIntTensorMath_gtTensor :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_geTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensorMath_geTensor"
  c_THIntTensorMath_geTensor :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_neTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensorMath_neTensor"
  c_THIntTensorMath_neTensor :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_eqTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensorMath_eqTensor"
  c_THIntTensorMath_eqTensor :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_ltTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensorMath_ltTensorT"
  c_THIntTensorMath_ltTensorT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_leTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensorMath_leTensorT"
  c_THIntTensorMath_leTensorT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_gtTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensorMath_gtTensorT"
  c_THIntTensorMath_gtTensorT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_geTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensorMath_geTensorT"
  c_THIntTensorMath_geTensorT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_neTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensorMath_neTensorT"
  c_THIntTensorMath_neTensorT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_eqTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensorMath_eqTensorT"
  c_THIntTensorMath_eqTensorT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_abs : r_ t -> void
foreign import ccall "THTensorMath.h THIntTensorMath_abs"
  c_THIntTensorMath_abs :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_sigmoid : r_ t -> void
foreign import ccall "THTensorMath.h THIntTensorMath_sigmoid"
  c_THIntTensorMath_sigmoid :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_log : r_ t -> void
foreign import ccall "THTensorMath.h THIntTensorMath_log"
  c_THIntTensorMath_log :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_lgamma : r_ t -> void
foreign import ccall "THTensorMath.h THIntTensorMath_lgamma"
  c_THIntTensorMath_lgamma :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_log1p : r_ t -> void
foreign import ccall "THTensorMath.h THIntTensorMath_log1p"
  c_THIntTensorMath_log1p :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_exp : r_ t -> void
foreign import ccall "THTensorMath.h THIntTensorMath_exp"
  c_THIntTensorMath_exp :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_cos : r_ t -> void
foreign import ccall "THTensorMath.h THIntTensorMath_cos"
  c_THIntTensorMath_cos :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_acos : r_ t -> void
foreign import ccall "THTensorMath.h THIntTensorMath_acos"
  c_THIntTensorMath_acos :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_cosh : r_ t -> void
foreign import ccall "THTensorMath.h THIntTensorMath_cosh"
  c_THIntTensorMath_cosh :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_sin : r_ t -> void
foreign import ccall "THTensorMath.h THIntTensorMath_sin"
  c_THIntTensorMath_sin :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_asin : r_ t -> void
foreign import ccall "THTensorMath.h THIntTensorMath_asin"
  c_THIntTensorMath_asin :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_sinh : r_ t -> void
foreign import ccall "THTensorMath.h THIntTensorMath_sinh"
  c_THIntTensorMath_sinh :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_tan : r_ t -> void
foreign import ccall "THTensorMath.h THIntTensorMath_tan"
  c_THIntTensorMath_tan :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_atan : r_ t -> void
foreign import ccall "THTensorMath.h THIntTensorMath_atan"
  c_THIntTensorMath_atan :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_atan2 : r_ tx ty -> void
foreign import ccall "THTensorMath.h THIntTensorMath_atan2"
  c_THIntTensorMath_atan2 :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_tanh : r_ t -> void
foreign import ccall "THTensorMath.h THIntTensorMath_tanh"
  c_THIntTensorMath_tanh :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_pow : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensorMath_pow"
  c_THIntTensorMath_pow :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_tpow : r_ value t -> void
foreign import ccall "THTensorMath.h THIntTensorMath_tpow"
  c_THIntTensorMath_tpow :: (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_sqrt : r_ t -> void
foreign import ccall "THTensorMath.h THIntTensorMath_sqrt"
  c_THIntTensorMath_sqrt :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_rsqrt : r_ t -> void
foreign import ccall "THTensorMath.h THIntTensorMath_rsqrt"
  c_THIntTensorMath_rsqrt :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_ceil : r_ t -> void
foreign import ccall "THTensorMath.h THIntTensorMath_ceil"
  c_THIntTensorMath_ceil :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_floor : r_ t -> void
foreign import ccall "THTensorMath.h THIntTensorMath_floor"
  c_THIntTensorMath_floor :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_round : r_ t -> void
foreign import ccall "THTensorMath.h THIntTensorMath_round"
  c_THIntTensorMath_round :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_trunc : r_ t -> void
foreign import ccall "THTensorMath.h THIntTensorMath_trunc"
  c_THIntTensorMath_trunc :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_frac : r_ t -> void
foreign import ccall "THTensorMath.h THIntTensorMath_frac"
  c_THIntTensorMath_frac :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensorMath_lerp : r_ a b weight -> void
foreign import ccall "THTensorMath.h THIntTensorMath_lerp"
  c_THIntTensorMath_lerp :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensorMath_mean : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensorMath_mean"
  c_THIntTensorMath_mean :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensorMath_std : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THIntTensorMath_std"
  c_THIntTensorMath_std :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THIntTensorMath_var : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THIntTensorMath_var"
  c_THIntTensorMath_var :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THIntTensorMath_norm : r_ t value dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensorMath_norm"
  c_THIntTensorMath_norm :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THIntTensorMath_renorm : r_ t value dimension maxnorm -> void
foreign import ccall "THTensorMath.h THIntTensorMath_renorm"
  c_THIntTensorMath_renorm :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THIntTensorMath_dist : a b value -> accreal
foreign import ccall "THTensorMath.h THIntTensorMath_dist"
  c_THIntTensorMath_dist :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CLong

-- |c_THIntTensorMath_histc : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THIntTensorMath_histc"
  c_THIntTensorMath_histc :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CLong -> CInt -> CInt -> IO ()

-- |c_THIntTensorMath_bhistc : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THIntTensorMath_bhistc"
  c_THIntTensorMath_bhistc :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CLong -> CInt -> CInt -> IO ()

-- |c_THIntTensorMath_meanall : self -> accreal
foreign import ccall "THTensorMath.h THIntTensorMath_meanall"
  c_THIntTensorMath_meanall :: (Ptr CTHIntTensor) -> CLong

-- |c_THIntTensorMath_varall : self biased -> accreal
foreign import ccall "THTensorMath.h THIntTensorMath_varall"
  c_THIntTensorMath_varall :: (Ptr CTHIntTensor) -> CInt -> CLong

-- |c_THIntTensorMath_stdall : self biased -> accreal
foreign import ccall "THTensorMath.h THIntTensorMath_stdall"
  c_THIntTensorMath_stdall :: (Ptr CTHIntTensor) -> CInt -> CLong

-- |c_THIntTensorMath_normall : t value -> accreal
foreign import ccall "THTensorMath.h THIntTensorMath_normall"
  c_THIntTensorMath_normall :: (Ptr CTHIntTensor) -> CInt -> CLong

-- |c_THIntTensorMath_linspace : r_ a b n -> void
foreign import ccall "THTensorMath.h THIntTensorMath_linspace"
  c_THIntTensorMath_linspace :: (Ptr CTHIntTensor) -> CInt -> CInt -> CLong -> IO ()

-- |c_THIntTensorMath_logspace : r_ a b n -> void
foreign import ccall "THTensorMath.h THIntTensorMath_logspace"
  c_THIntTensorMath_logspace :: (Ptr CTHIntTensor) -> CInt -> CInt -> CLong -> IO ()

-- |c_THIntTensorMath_rand : r_ _generator size -> void
foreign import ccall "THTensorMath.h THIntTensorMath_rand"
  c_THIntTensorMath_rand :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()

-- |c_THIntTensorMath_randn : r_ _generator size -> void
foreign import ccall "THTensorMath.h THIntTensorMath_randn"
  c_THIntTensorMath_randn :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()

-- |c_THIntTensorMath_logicalall : self -> int
foreign import ccall "THTensorMath.h THIntTensorMath_logicalall"
  c_THIntTensorMath_logicalall :: (Ptr CTHIntTensor) -> CInt

-- |c_THIntTensorMath_logicalany : self -> int
foreign import ccall "THTensorMath.h THIntTensorMath_logicalany"
  c_THIntTensorMath_logicalany :: (Ptr CTHIntTensor) -> CInt