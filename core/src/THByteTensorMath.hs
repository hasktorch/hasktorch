{-# LANGUAGE ForeignFunctionInterface#-}

module THByteTensorMath (
    c_THByteTensorMath_fill,
    c_THByteTensorMath_zero,
    c_THByteTensorMath_maskedFill,
    c_THByteTensorMath_maskedCopy,
    c_THByteTensorMath_maskedSelect,
    c_THByteTensorMath_nonzero,
    c_THByteTensorMath_indexSelect,
    c_THByteTensorMath_indexCopy,
    c_THByteTensorMath_indexAdd,
    c_THByteTensorMath_indexFill,
    c_THByteTensorMath_gather,
    c_THByteTensorMath_scatter,
    c_THByteTensorMath_scatterAdd,
    c_THByteTensorMath_scatterFill,
    c_THByteTensorMath_dot,
    c_THByteTensorMath_minall,
    c_THByteTensorMath_maxall,
    c_THByteTensorMath_medianall,
    c_THByteTensorMath_sumall,
    c_THByteTensorMath_prodall,
    c_THByteTensorMath_neg,
    c_THByteTensorMath_cinv,
    c_THByteTensorMath_add,
    c_THByteTensorMath_sub,
    c_THByteTensorMath_mul,
    c_THByteTensorMath_div,
    c_THByteTensorMath_lshift,
    c_THByteTensorMath_rshift,
    c_THByteTensorMath_fmod,
    c_THByteTensorMath_remainder,
    c_THByteTensorMath_clamp,
    c_THByteTensorMath_bitand,
    c_THByteTensorMath_bitor,
    c_THByteTensorMath_bitxor,
    c_THByteTensorMath_cadd,
    c_THByteTensorMath_csub,
    c_THByteTensorMath_cmul,
    c_THByteTensorMath_cpow,
    c_THByteTensorMath_cdiv,
    c_THByteTensorMath_clshift,
    c_THByteTensorMath_crshift,
    c_THByteTensorMath_cfmod,
    c_THByteTensorMath_cremainder,
    c_THByteTensorMath_cbitand,
    c_THByteTensorMath_cbitor,
    c_THByteTensorMath_cbitxor,
    c_THByteTensorMath_addcmul,
    c_THByteTensorMath_addcdiv,
    c_THByteTensorMath_addmv,
    c_THByteTensorMath_addmm,
    c_THByteTensorMath_addr,
    c_THByteTensorMath_addbmm,
    c_THByteTensorMath_baddbmm,
    c_THByteTensorMath_match,
    c_THByteTensorMath_numel,
    c_THByteTensorMath_max,
    c_THByteTensorMath_min,
    c_THByteTensorMath_kthvalue,
    c_THByteTensorMath_mode,
    c_THByteTensorMath_median,
    c_THByteTensorMath_sum,
    c_THByteTensorMath_prod,
    c_THByteTensorMath_cumsum,
    c_THByteTensorMath_cumprod,
    c_THByteTensorMath_sign,
    c_THByteTensorMath_trace,
    c_THByteTensorMath_cross,
    c_THByteTensorMath_cmax,
    c_THByteTensorMath_cmin,
    c_THByteTensorMath_cmaxValue,
    c_THByteTensorMath_cminValue,
    c_THByteTensorMath_zeros,
    c_THByteTensorMath_zerosLike,
    c_THByteTensorMath_ones,
    c_THByteTensorMath_onesLike,
    c_THByteTensorMath_diag,
    c_THByteTensorMath_eye,
    c_THByteTensorMath_arange,
    c_THByteTensorMath_range,
    c_THByteTensorMath_randperm,
    c_THByteTensorMath_reshape,
    c_THByteTensorMath_sort,
    c_THByteTensorMath_topk,
    c_THByteTensorMath_tril,
    c_THByteTensorMath_triu,
    c_THByteTensorMath_cat,
    c_THByteTensorMath_catArray,
    c_THByteTensorMath_equal,
    c_THByteTensorMath_ltValue,
    c_THByteTensorMath_leValue,
    c_THByteTensorMath_gtValue,
    c_THByteTensorMath_geValue,
    c_THByteTensorMath_neValue,
    c_THByteTensorMath_eqValue,
    c_THByteTensorMath_ltValueT,
    c_THByteTensorMath_leValueT,
    c_THByteTensorMath_gtValueT,
    c_THByteTensorMath_geValueT,
    c_THByteTensorMath_neValueT,
    c_THByteTensorMath_eqValueT,
    c_THByteTensorMath_ltTensor,
    c_THByteTensorMath_leTensor,
    c_THByteTensorMath_gtTensor,
    c_THByteTensorMath_geTensor,
    c_THByteTensorMath_neTensor,
    c_THByteTensorMath_eqTensor,
    c_THByteTensorMath_ltTensorT,
    c_THByteTensorMath_leTensorT,
    c_THByteTensorMath_gtTensorT,
    c_THByteTensorMath_geTensorT,
    c_THByteTensorMath_neTensorT,
    c_THByteTensorMath_eqTensorT,
    c_THByteTensorMath_abs,
    c_THByteTensorMath_sigmoid,
    c_THByteTensorMath_log,
    c_THByteTensorMath_lgamma,
    c_THByteTensorMath_log1p,
    c_THByteTensorMath_exp,
    c_THByteTensorMath_cos,
    c_THByteTensorMath_acos,
    c_THByteTensorMath_cosh,
    c_THByteTensorMath_sin,
    c_THByteTensorMath_asin,
    c_THByteTensorMath_sinh,
    c_THByteTensorMath_tan,
    c_THByteTensorMath_atan,
    c_THByteTensorMath_atan2,
    c_THByteTensorMath_tanh,
    c_THByteTensorMath_pow,
    c_THByteTensorMath_tpow,
    c_THByteTensorMath_sqrt,
    c_THByteTensorMath_rsqrt,
    c_THByteTensorMath_ceil,
    c_THByteTensorMath_floor,
    c_THByteTensorMath_round,
    c_THByteTensorMath_trunc,
    c_THByteTensorMath_frac,
    c_THByteTensorMath_lerp,
    c_THByteTensorMath_mean,
    c_THByteTensorMath_std,
    c_THByteTensorMath_var,
    c_THByteTensorMath_norm,
    c_THByteTensorMath_renorm,
    c_THByteTensorMath_dist,
    c_THByteTensorMath_histc,
    c_THByteTensorMath_bhistc,
    c_THByteTensorMath_meanall,
    c_THByteTensorMath_varall,
    c_THByteTensorMath_stdall,
    c_THByteTensorMath_normall,
    c_THByteTensorMath_linspace,
    c_THByteTensorMath_logspace,
    c_THByteTensorMath_rand,
    c_THByteTensorMath_randn,
    c_THByteTensorMath_logicalall,
    c_THByteTensorMath_logicalany) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THByteTensorMath_fill : r_ value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_fill"
  c_THByteTensorMath_fill :: (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_zero : r_ -> void
foreign import ccall "THTensorMath.h THByteTensorMath_zero"
  c_THByteTensorMath_zero :: (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_maskedFill : tensor mask value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_maskedFill"
  c_THByteTensorMath_maskedFill :: (Ptr CTHByteTensor) -> Ptr CTHByteTensor -> CChar -> IO ()

-- |c_THByteTensorMath_maskedCopy : tensor mask src -> void
foreign import ccall "THTensorMath.h THByteTensorMath_maskedCopy"
  c_THByteTensorMath_maskedCopy :: (Ptr CTHByteTensor) -> Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_maskedSelect : tensor src mask -> void
foreign import ccall "THTensorMath.h THByteTensorMath_maskedSelect"
  c_THByteTensorMath_maskedSelect :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THByteTensorMath_nonzero : subscript tensor -> void
foreign import ccall "THTensorMath.h THByteTensorMath_nonzero"
  c_THByteTensorMath_nonzero :: Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_indexSelect : tensor src dim index -> void
foreign import ccall "THTensorMath.h THByteTensorMath_indexSelect"
  c_THByteTensorMath_indexSelect :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THByteTensorMath_indexCopy : tensor dim index src -> void
foreign import ccall "THTensorMath.h THByteTensorMath_indexCopy"
  c_THByteTensorMath_indexCopy :: (Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_indexAdd : tensor dim index src -> void
foreign import ccall "THTensorMath.h THByteTensorMath_indexAdd"
  c_THByteTensorMath_indexAdd :: (Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_indexFill : tensor dim index val -> void
foreign import ccall "THTensorMath.h THByteTensorMath_indexFill"
  c_THByteTensorMath_indexFill :: (Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> CChar -> IO ()

-- |c_THByteTensorMath_gather : tensor src dim index -> void
foreign import ccall "THTensorMath.h THByteTensorMath_gather"
  c_THByteTensorMath_gather :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THByteTensorMath_scatter : tensor dim index src -> void
foreign import ccall "THTensorMath.h THByteTensorMath_scatter"
  c_THByteTensorMath_scatter :: (Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_scatterAdd : tensor dim index src -> void
foreign import ccall "THTensorMath.h THByteTensorMath_scatterAdd"
  c_THByteTensorMath_scatterAdd :: (Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_scatterFill : tensor dim index val -> void
foreign import ccall "THTensorMath.h THByteTensorMath_scatterFill"
  c_THByteTensorMath_scatterFill :: (Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> CChar -> IO ()

-- |c_THByteTensorMath_dot : t src -> accreal
foreign import ccall "THTensorMath.h THByteTensorMath_dot"
  c_THByteTensorMath_dot :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CLong

-- |c_THByteTensorMath_minall : t -> real
foreign import ccall "THTensorMath.h THByteTensorMath_minall"
  c_THByteTensorMath_minall :: (Ptr CTHByteTensor) -> CChar

-- |c_THByteTensorMath_maxall : t -> real
foreign import ccall "THTensorMath.h THByteTensorMath_maxall"
  c_THByteTensorMath_maxall :: (Ptr CTHByteTensor) -> CChar

-- |c_THByteTensorMath_medianall : t -> real
foreign import ccall "THTensorMath.h THByteTensorMath_medianall"
  c_THByteTensorMath_medianall :: (Ptr CTHByteTensor) -> CChar

-- |c_THByteTensorMath_sumall : t -> accreal
foreign import ccall "THTensorMath.h THByteTensorMath_sumall"
  c_THByteTensorMath_sumall :: (Ptr CTHByteTensor) -> CLong

-- |c_THByteTensorMath_prodall : t -> accreal
foreign import ccall "THTensorMath.h THByteTensorMath_prodall"
  c_THByteTensorMath_prodall :: (Ptr CTHByteTensor) -> CLong

-- |c_THByteTensorMath_neg : self src -> void
foreign import ccall "THTensorMath.h THByteTensorMath_neg"
  c_THByteTensorMath_neg :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_cinv : self src -> void
foreign import ccall "THTensorMath.h THByteTensorMath_cinv"
  c_THByteTensorMath_cinv :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_add : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_add"
  c_THByteTensorMath_add :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_sub : self src value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_sub"
  c_THByteTensorMath_sub :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_mul : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_mul"
  c_THByteTensorMath_mul :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_div : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_div"
  c_THByteTensorMath_div :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_lshift : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_lshift"
  c_THByteTensorMath_lshift :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_rshift : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_rshift"
  c_THByteTensorMath_rshift :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_fmod : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_fmod"
  c_THByteTensorMath_fmod :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_remainder : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_remainder"
  c_THByteTensorMath_remainder :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_clamp : r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_clamp"
  c_THByteTensorMath_clamp :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> CChar -> IO ()

-- |c_THByteTensorMath_bitand : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_bitand"
  c_THByteTensorMath_bitand :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_bitor : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_bitor"
  c_THByteTensorMath_bitor :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_bitxor : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_bitxor"
  c_THByteTensorMath_bitxor :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_cadd : r_ t value src -> void
foreign import ccall "THTensorMath.h THByteTensorMath_cadd"
  c_THByteTensorMath_cadd :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_csub : self src1 value src2 -> void
foreign import ccall "THTensorMath.h THByteTensorMath_csub"
  c_THByteTensorMath_csub :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_cmul : r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensorMath_cmul"
  c_THByteTensorMath_cmul :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_cpow : r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensorMath_cpow"
  c_THByteTensorMath_cpow :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_cdiv : r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensorMath_cdiv"
  c_THByteTensorMath_cdiv :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_clshift : r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensorMath_clshift"
  c_THByteTensorMath_clshift :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_crshift : r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensorMath_crshift"
  c_THByteTensorMath_crshift :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_cfmod : r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensorMath_cfmod"
  c_THByteTensorMath_cfmod :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_cremainder : r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensorMath_cremainder"
  c_THByteTensorMath_cremainder :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_cbitand : r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensorMath_cbitand"
  c_THByteTensorMath_cbitand :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_cbitor : r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensorMath_cbitor"
  c_THByteTensorMath_cbitor :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_cbitxor : r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensorMath_cbitxor"
  c_THByteTensorMath_cbitxor :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_addcmul : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THByteTensorMath_addcmul"
  c_THByteTensorMath_addcmul :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_addcdiv : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THByteTensorMath_addcdiv"
  c_THByteTensorMath_addcdiv :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_addmv : r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h THByteTensorMath_addmv"
  c_THByteTensorMath_addmv :: (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_addmm : r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h THByteTensorMath_addmm"
  c_THByteTensorMath_addmm :: (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_addr : r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h THByteTensorMath_addr"
  c_THByteTensorMath_addr :: (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_addbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THByteTensorMath_addbmm"
  c_THByteTensorMath_addbmm :: (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_baddbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THByteTensorMath_baddbmm"
  c_THByteTensorMath_baddbmm :: (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_match : r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h THByteTensorMath_match"
  c_THByteTensorMath_match :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_numel : t -> THStorage *
foreign import ccall "THTensorMath.h THByteTensorMath_numel"
  c_THByteTensorMath_numel :: (Ptr CTHByteTensor) -> IO (Ptr CTHByteStorage)

-- |c_THByteTensorMath_max : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THByteTensorMath_max"
  c_THByteTensorMath_max :: (Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensorMath_min : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THByteTensorMath_min"
  c_THByteTensorMath_min :: (Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensorMath_kthvalue : values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h THByteTensorMath_kthvalue"
  c_THByteTensorMath_kthvalue :: (Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CLong -> CInt -> CInt -> IO ()

-- |c_THByteTensorMath_mode : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THByteTensorMath_mode"
  c_THByteTensorMath_mode :: (Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensorMath_median : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THByteTensorMath_median"
  c_THByteTensorMath_median :: (Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensorMath_sum : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THByteTensorMath_sum"
  c_THByteTensorMath_sum :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensorMath_prod : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THByteTensorMath_prod"
  c_THByteTensorMath_prod :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensorMath_cumsum : r_ t dimension -> void
foreign import ccall "THTensorMath.h THByteTensorMath_cumsum"
  c_THByteTensorMath_cumsum :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> IO ()

-- |c_THByteTensorMath_cumprod : r_ t dimension -> void
foreign import ccall "THTensorMath.h THByteTensorMath_cumprod"
  c_THByteTensorMath_cumprod :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> IO ()

-- |c_THByteTensorMath_sign : r_ t -> void
foreign import ccall "THTensorMath.h THByteTensorMath_sign"
  c_THByteTensorMath_sign :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_trace : t -> accreal
foreign import ccall "THTensorMath.h THByteTensorMath_trace"
  c_THByteTensorMath_trace :: (Ptr CTHByteTensor) -> CLong

-- |c_THByteTensorMath_cross : r_ a b dimension -> void
foreign import ccall "THTensorMath.h THByteTensorMath_cross"
  c_THByteTensorMath_cross :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> IO ()

-- |c_THByteTensorMath_cmax : r t src -> void
foreign import ccall "THTensorMath.h THByteTensorMath_cmax"
  c_THByteTensorMath_cmax :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_cmin : r t src -> void
foreign import ccall "THTensorMath.h THByteTensorMath_cmin"
  c_THByteTensorMath_cmin :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_cmaxValue : r t value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_cmaxValue"
  c_THByteTensorMath_cmaxValue :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_cminValue : r t value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_cminValue"
  c_THByteTensorMath_cminValue :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_zeros : r_ size -> void
foreign import ccall "THTensorMath.h THByteTensorMath_zeros"
  c_THByteTensorMath_zeros :: (Ptr CTHByteTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THByteTensorMath_zerosLike : r_ input -> void
foreign import ccall "THTensorMath.h THByteTensorMath_zerosLike"
  c_THByteTensorMath_zerosLike :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_ones : r_ size -> void
foreign import ccall "THTensorMath.h THByteTensorMath_ones"
  c_THByteTensorMath_ones :: (Ptr CTHByteTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THByteTensorMath_onesLike : r_ input -> void
foreign import ccall "THTensorMath.h THByteTensorMath_onesLike"
  c_THByteTensorMath_onesLike :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_diag : r_ t k -> void
foreign import ccall "THTensorMath.h THByteTensorMath_diag"
  c_THByteTensorMath_diag :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> IO ()

-- |c_THByteTensorMath_eye : r_ n m -> void
foreign import ccall "THTensorMath.h THByteTensorMath_eye"
  c_THByteTensorMath_eye :: (Ptr CTHByteTensor) -> CLong -> CLong -> IO ()

-- |c_THByteTensorMath_arange : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THByteTensorMath_arange"
  c_THByteTensorMath_arange :: (Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensorMath_range : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THByteTensorMath_range"
  c_THByteTensorMath_range :: (Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensorMath_randperm : r_ _generator n -> void
foreign import ccall "THTensorMath.h THByteTensorMath_randperm"
  c_THByteTensorMath_randperm :: (Ptr CTHByteTensor) -> Ptr CTHGenerator -> CLong -> IO ()

-- |c_THByteTensorMath_reshape : r_ t size -> void
foreign import ccall "THTensorMath.h THByteTensorMath_reshape"
  c_THByteTensorMath_reshape :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THByteTensorMath_sort : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h THByteTensorMath_sort"
  c_THByteTensorMath_sort :: (Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensorMath_topk : rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h THByteTensorMath_topk"
  c_THByteTensorMath_topk :: (Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CLong -> CInt -> CInt -> CInt -> IO ()

-- |c_THByteTensorMath_tril : r_ t k -> void
foreign import ccall "THTensorMath.h THByteTensorMath_tril"
  c_THByteTensorMath_tril :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CLong -> IO ()

-- |c_THByteTensorMath_triu : r_ t k -> void
foreign import ccall "THTensorMath.h THByteTensorMath_triu"
  c_THByteTensorMath_triu :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CLong -> IO ()

-- |c_THByteTensorMath_cat : r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h THByteTensorMath_cat"
  c_THByteTensorMath_cat :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> IO ()

-- |c_THByteTensorMath_catArray : result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h THByteTensorMath_catArray"
  c_THByteTensorMath_catArray :: (Ptr CTHByteTensor) -> Ptr (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensorMath_equal : ta tb -> int
foreign import ccall "THTensorMath.h THByteTensorMath_equal"
  c_THByteTensorMath_equal :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt

-- |c_THByteTensorMath_ltValue : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_ltValue"
  c_THByteTensorMath_ltValue :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_leValue : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_leValue"
  c_THByteTensorMath_leValue :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_gtValue : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_gtValue"
  c_THByteTensorMath_gtValue :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_geValue : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_geValue"
  c_THByteTensorMath_geValue :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_neValue : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_neValue"
  c_THByteTensorMath_neValue :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_eqValue : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_eqValue"
  c_THByteTensorMath_eqValue :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_ltValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_ltValueT"
  c_THByteTensorMath_ltValueT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_leValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_leValueT"
  c_THByteTensorMath_leValueT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_gtValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_gtValueT"
  c_THByteTensorMath_gtValueT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_geValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_geValueT"
  c_THByteTensorMath_geValueT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_neValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_neValueT"
  c_THByteTensorMath_neValueT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_eqValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_eqValueT"
  c_THByteTensorMath_eqValueT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_ltTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensorMath_ltTensor"
  c_THByteTensorMath_ltTensor :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_leTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensorMath_leTensor"
  c_THByteTensorMath_leTensor :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_gtTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensorMath_gtTensor"
  c_THByteTensorMath_gtTensor :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_geTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensorMath_geTensor"
  c_THByteTensorMath_geTensor :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_neTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensorMath_neTensor"
  c_THByteTensorMath_neTensor :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_eqTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensorMath_eqTensor"
  c_THByteTensorMath_eqTensor :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_ltTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensorMath_ltTensorT"
  c_THByteTensorMath_ltTensorT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_leTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensorMath_leTensorT"
  c_THByteTensorMath_leTensorT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_gtTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensorMath_gtTensorT"
  c_THByteTensorMath_gtTensorT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_geTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensorMath_geTensorT"
  c_THByteTensorMath_geTensorT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_neTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensorMath_neTensorT"
  c_THByteTensorMath_neTensorT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_eqTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensorMath_eqTensorT"
  c_THByteTensorMath_eqTensorT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_abs : r_ t -> void
foreign import ccall "THTensorMath.h THByteTensorMath_abs"
  c_THByteTensorMath_abs :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_sigmoid : r_ t -> void
foreign import ccall "THTensorMath.h THByteTensorMath_sigmoid"
  c_THByteTensorMath_sigmoid :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_log : r_ t -> void
foreign import ccall "THTensorMath.h THByteTensorMath_log"
  c_THByteTensorMath_log :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_lgamma : r_ t -> void
foreign import ccall "THTensorMath.h THByteTensorMath_lgamma"
  c_THByteTensorMath_lgamma :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_log1p : r_ t -> void
foreign import ccall "THTensorMath.h THByteTensorMath_log1p"
  c_THByteTensorMath_log1p :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_exp : r_ t -> void
foreign import ccall "THTensorMath.h THByteTensorMath_exp"
  c_THByteTensorMath_exp :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_cos : r_ t -> void
foreign import ccall "THTensorMath.h THByteTensorMath_cos"
  c_THByteTensorMath_cos :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_acos : r_ t -> void
foreign import ccall "THTensorMath.h THByteTensorMath_acos"
  c_THByteTensorMath_acos :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_cosh : r_ t -> void
foreign import ccall "THTensorMath.h THByteTensorMath_cosh"
  c_THByteTensorMath_cosh :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_sin : r_ t -> void
foreign import ccall "THTensorMath.h THByteTensorMath_sin"
  c_THByteTensorMath_sin :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_asin : r_ t -> void
foreign import ccall "THTensorMath.h THByteTensorMath_asin"
  c_THByteTensorMath_asin :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_sinh : r_ t -> void
foreign import ccall "THTensorMath.h THByteTensorMath_sinh"
  c_THByteTensorMath_sinh :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_tan : r_ t -> void
foreign import ccall "THTensorMath.h THByteTensorMath_tan"
  c_THByteTensorMath_tan :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_atan : r_ t -> void
foreign import ccall "THTensorMath.h THByteTensorMath_atan"
  c_THByteTensorMath_atan :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_atan2 : r_ tx ty -> void
foreign import ccall "THTensorMath.h THByteTensorMath_atan2"
  c_THByteTensorMath_atan2 :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_tanh : r_ t -> void
foreign import ccall "THTensorMath.h THByteTensorMath_tanh"
  c_THByteTensorMath_tanh :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_pow : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensorMath_pow"
  c_THByteTensorMath_pow :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_tpow : r_ value t -> void
foreign import ccall "THTensorMath.h THByteTensorMath_tpow"
  c_THByteTensorMath_tpow :: (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_sqrt : r_ t -> void
foreign import ccall "THTensorMath.h THByteTensorMath_sqrt"
  c_THByteTensorMath_sqrt :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_rsqrt : r_ t -> void
foreign import ccall "THTensorMath.h THByteTensorMath_rsqrt"
  c_THByteTensorMath_rsqrt :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_ceil : r_ t -> void
foreign import ccall "THTensorMath.h THByteTensorMath_ceil"
  c_THByteTensorMath_ceil :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_floor : r_ t -> void
foreign import ccall "THTensorMath.h THByteTensorMath_floor"
  c_THByteTensorMath_floor :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_round : r_ t -> void
foreign import ccall "THTensorMath.h THByteTensorMath_round"
  c_THByteTensorMath_round :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_trunc : r_ t -> void
foreign import ccall "THTensorMath.h THByteTensorMath_trunc"
  c_THByteTensorMath_trunc :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_frac : r_ t -> void
foreign import ccall "THTensorMath.h THByteTensorMath_frac"
  c_THByteTensorMath_frac :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensorMath_lerp : r_ a b weight -> void
foreign import ccall "THTensorMath.h THByteTensorMath_lerp"
  c_THByteTensorMath_lerp :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensorMath_mean : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THByteTensorMath_mean"
  c_THByteTensorMath_mean :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensorMath_std : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THByteTensorMath_std"
  c_THByteTensorMath_std :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THByteTensorMath_var : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THByteTensorMath_var"
  c_THByteTensorMath_var :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THByteTensorMath_norm : r_ t value dimension keepdim -> void
foreign import ccall "THTensorMath.h THByteTensorMath_norm"
  c_THByteTensorMath_norm :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> CInt -> CInt -> IO ()

-- |c_THByteTensorMath_renorm : r_ t value dimension maxnorm -> void
foreign import ccall "THTensorMath.h THByteTensorMath_renorm"
  c_THByteTensorMath_renorm :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> CInt -> CChar -> IO ()

-- |c_THByteTensorMath_dist : a b value -> accreal
foreign import ccall "THTensorMath.h THByteTensorMath_dist"
  c_THByteTensorMath_dist :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> CLong

-- |c_THByteTensorMath_histc : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THByteTensorMath_histc"
  c_THByteTensorMath_histc :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CLong -> CChar -> CChar -> IO ()

-- |c_THByteTensorMath_bhistc : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THByteTensorMath_bhistc"
  c_THByteTensorMath_bhistc :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CLong -> CChar -> CChar -> IO ()

-- |c_THByteTensorMath_meanall : self -> accreal
foreign import ccall "THTensorMath.h THByteTensorMath_meanall"
  c_THByteTensorMath_meanall :: (Ptr CTHByteTensor) -> CLong

-- |c_THByteTensorMath_varall : self biased -> accreal
foreign import ccall "THTensorMath.h THByteTensorMath_varall"
  c_THByteTensorMath_varall :: (Ptr CTHByteTensor) -> CInt -> CLong

-- |c_THByteTensorMath_stdall : self biased -> accreal
foreign import ccall "THTensorMath.h THByteTensorMath_stdall"
  c_THByteTensorMath_stdall :: (Ptr CTHByteTensor) -> CInt -> CLong

-- |c_THByteTensorMath_normall : t value -> accreal
foreign import ccall "THTensorMath.h THByteTensorMath_normall"
  c_THByteTensorMath_normall :: (Ptr CTHByteTensor) -> CChar -> CLong

-- |c_THByteTensorMath_linspace : r_ a b n -> void
foreign import ccall "THTensorMath.h THByteTensorMath_linspace"
  c_THByteTensorMath_linspace :: (Ptr CTHByteTensor) -> CChar -> CChar -> CLong -> IO ()

-- |c_THByteTensorMath_logspace : r_ a b n -> void
foreign import ccall "THTensorMath.h THByteTensorMath_logspace"
  c_THByteTensorMath_logspace :: (Ptr CTHByteTensor) -> CChar -> CChar -> CLong -> IO ()

-- |c_THByteTensorMath_rand : r_ _generator size -> void
foreign import ccall "THTensorMath.h THByteTensorMath_rand"
  c_THByteTensorMath_rand :: (Ptr CTHByteTensor) -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()

-- |c_THByteTensorMath_randn : r_ _generator size -> void
foreign import ccall "THTensorMath.h THByteTensorMath_randn"
  c_THByteTensorMath_randn :: (Ptr CTHByteTensor) -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()

-- |c_THByteTensorMath_logicalall : self -> int
foreign import ccall "THTensorMath.h THByteTensorMath_logicalall"
  c_THByteTensorMath_logicalall :: (Ptr CTHByteTensor) -> CInt

-- |c_THByteTensorMath_logicalany : self -> int
foreign import ccall "THTensorMath.h THByteTensorMath_logicalany"
  c_THByteTensorMath_logicalany :: (Ptr CTHByteTensor) -> CInt