{-# LANGUAGE ForeignFunctionInterface#-}

module THShortTensorMath (
    c_THShortTensorMath_fill,
    c_THShortTensorMath_zero,
    c_THShortTensorMath_maskedFill,
    c_THShortTensorMath_maskedCopy,
    c_THShortTensorMath_maskedSelect,
    c_THShortTensorMath_nonzero,
    c_THShortTensorMath_indexSelect,
    c_THShortTensorMath_indexCopy,
    c_THShortTensorMath_indexAdd,
    c_THShortTensorMath_indexFill,
    c_THShortTensorMath_gather,
    c_THShortTensorMath_scatter,
    c_THShortTensorMath_scatterAdd,
    c_THShortTensorMath_scatterFill,
    c_THShortTensorMath_dot,
    c_THShortTensorMath_minall,
    c_THShortTensorMath_maxall,
    c_THShortTensorMath_medianall,
    c_THShortTensorMath_sumall,
    c_THShortTensorMath_prodall,
    c_THShortTensorMath_neg,
    c_THShortTensorMath_cinv,
    c_THShortTensorMath_add,
    c_THShortTensorMath_sub,
    c_THShortTensorMath_mul,
    c_THShortTensorMath_div,
    c_THShortTensorMath_lshift,
    c_THShortTensorMath_rshift,
    c_THShortTensorMath_fmod,
    c_THShortTensorMath_remainder,
    c_THShortTensorMath_clamp,
    c_THShortTensorMath_bitand,
    c_THShortTensorMath_bitor,
    c_THShortTensorMath_bitxor,
    c_THShortTensorMath_cadd,
    c_THShortTensorMath_csub,
    c_THShortTensorMath_cmul,
    c_THShortTensorMath_cpow,
    c_THShortTensorMath_cdiv,
    c_THShortTensorMath_clshift,
    c_THShortTensorMath_crshift,
    c_THShortTensorMath_cfmod,
    c_THShortTensorMath_cremainder,
    c_THShortTensorMath_cbitand,
    c_THShortTensorMath_cbitor,
    c_THShortTensorMath_cbitxor,
    c_THShortTensorMath_addcmul,
    c_THShortTensorMath_addcdiv,
    c_THShortTensorMath_addmv,
    c_THShortTensorMath_addmm,
    c_THShortTensorMath_addr,
    c_THShortTensorMath_addbmm,
    c_THShortTensorMath_baddbmm,
    c_THShortTensorMath_match,
    c_THShortTensorMath_numel,
    c_THShortTensorMath_max,
    c_THShortTensorMath_min,
    c_THShortTensorMath_kthvalue,
    c_THShortTensorMath_mode,
    c_THShortTensorMath_median,
    c_THShortTensorMath_sum,
    c_THShortTensorMath_prod,
    c_THShortTensorMath_cumsum,
    c_THShortTensorMath_cumprod,
    c_THShortTensorMath_sign,
    c_THShortTensorMath_trace,
    c_THShortTensorMath_cross,
    c_THShortTensorMath_cmax,
    c_THShortTensorMath_cmin,
    c_THShortTensorMath_cmaxValue,
    c_THShortTensorMath_cminValue,
    c_THShortTensorMath_zeros,
    c_THShortTensorMath_zerosLike,
    c_THShortTensorMath_ones,
    c_THShortTensorMath_onesLike,
    c_THShortTensorMath_diag,
    c_THShortTensorMath_eye,
    c_THShortTensorMath_arange,
    c_THShortTensorMath_range,
    c_THShortTensorMath_randperm,
    c_THShortTensorMath_reshape,
    c_THShortTensorMath_sort,
    c_THShortTensorMath_topk,
    c_THShortTensorMath_tril,
    c_THShortTensorMath_triu,
    c_THShortTensorMath_cat,
    c_THShortTensorMath_catArray,
    c_THShortTensorMath_equal,
    c_THShortTensorMath_ltValue,
    c_THShortTensorMath_leValue,
    c_THShortTensorMath_gtValue,
    c_THShortTensorMath_geValue,
    c_THShortTensorMath_neValue,
    c_THShortTensorMath_eqValue,
    c_THShortTensorMath_ltValueT,
    c_THShortTensorMath_leValueT,
    c_THShortTensorMath_gtValueT,
    c_THShortTensorMath_geValueT,
    c_THShortTensorMath_neValueT,
    c_THShortTensorMath_eqValueT,
    c_THShortTensorMath_ltTensor,
    c_THShortTensorMath_leTensor,
    c_THShortTensorMath_gtTensor,
    c_THShortTensorMath_geTensor,
    c_THShortTensorMath_neTensor,
    c_THShortTensorMath_eqTensor,
    c_THShortTensorMath_ltTensorT,
    c_THShortTensorMath_leTensorT,
    c_THShortTensorMath_gtTensorT,
    c_THShortTensorMath_geTensorT,
    c_THShortTensorMath_neTensorT,
    c_THShortTensorMath_eqTensorT,
    c_THShortTensorMath_abs,
    c_THShortTensorMath_sigmoid,
    c_THShortTensorMath_log,
    c_THShortTensorMath_lgamma,
    c_THShortTensorMath_log1p,
    c_THShortTensorMath_exp,
    c_THShortTensorMath_cos,
    c_THShortTensorMath_acos,
    c_THShortTensorMath_cosh,
    c_THShortTensorMath_sin,
    c_THShortTensorMath_asin,
    c_THShortTensorMath_sinh,
    c_THShortTensorMath_tan,
    c_THShortTensorMath_atan,
    c_THShortTensorMath_atan2,
    c_THShortTensorMath_tanh,
    c_THShortTensorMath_pow,
    c_THShortTensorMath_tpow,
    c_THShortTensorMath_sqrt,
    c_THShortTensorMath_rsqrt,
    c_THShortTensorMath_ceil,
    c_THShortTensorMath_floor,
    c_THShortTensorMath_round,
    c_THShortTensorMath_trunc,
    c_THShortTensorMath_frac,
    c_THShortTensorMath_lerp,
    c_THShortTensorMath_mean,
    c_THShortTensorMath_std,
    c_THShortTensorMath_var,
    c_THShortTensorMath_norm,
    c_THShortTensorMath_renorm,
    c_THShortTensorMath_dist,
    c_THShortTensorMath_histc,
    c_THShortTensorMath_bhistc,
    c_THShortTensorMath_meanall,
    c_THShortTensorMath_varall,
    c_THShortTensorMath_stdall,
    c_THShortTensorMath_normall,
    c_THShortTensorMath_linspace,
    c_THShortTensorMath_logspace,
    c_THShortTensorMath_rand,
    c_THShortTensorMath_randn,
    c_THShortTensorMath_logicalall,
    c_THShortTensorMath_logicalany) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THShortTensorMath_fill : r_ value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_fill"
  c_THShortTensorMath_fill :: (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_zero : r_ -> void
foreign import ccall "THTensorMath.h THShortTensorMath_zero"
  c_THShortTensorMath_zero :: (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_maskedFill : tensor mask value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_maskedFill"
  c_THShortTensorMath_maskedFill :: (Ptr CTHShortTensor) -> Ptr CTHByteTensor -> CShort -> IO ()

-- |c_THShortTensorMath_maskedCopy : tensor mask src -> void
foreign import ccall "THTensorMath.h THShortTensorMath_maskedCopy"
  c_THShortTensorMath_maskedCopy :: (Ptr CTHShortTensor) -> Ptr CTHByteTensor -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_maskedSelect : tensor src mask -> void
foreign import ccall "THTensorMath.h THShortTensorMath_maskedSelect"
  c_THShortTensorMath_maskedSelect :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THShortTensorMath_nonzero : subscript tensor -> void
foreign import ccall "THTensorMath.h THShortTensorMath_nonzero"
  c_THShortTensorMath_nonzero :: Ptr CTHLongTensor -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_indexSelect : tensor src dim index -> void
foreign import ccall "THTensorMath.h THShortTensorMath_indexSelect"
  c_THShortTensorMath_indexSelect :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THShortTensorMath_indexCopy : tensor dim index src -> void
foreign import ccall "THTensorMath.h THShortTensorMath_indexCopy"
  c_THShortTensorMath_indexCopy :: (Ptr CTHShortTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_indexAdd : tensor dim index src -> void
foreign import ccall "THTensorMath.h THShortTensorMath_indexAdd"
  c_THShortTensorMath_indexAdd :: (Ptr CTHShortTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_indexFill : tensor dim index val -> void
foreign import ccall "THTensorMath.h THShortTensorMath_indexFill"
  c_THShortTensorMath_indexFill :: (Ptr CTHShortTensor) -> CInt -> Ptr CTHLongTensor -> CShort -> IO ()

-- |c_THShortTensorMath_gather : tensor src dim index -> void
foreign import ccall "THTensorMath.h THShortTensorMath_gather"
  c_THShortTensorMath_gather :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THShortTensorMath_scatter : tensor dim index src -> void
foreign import ccall "THTensorMath.h THShortTensorMath_scatter"
  c_THShortTensorMath_scatter :: (Ptr CTHShortTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_scatterAdd : tensor dim index src -> void
foreign import ccall "THTensorMath.h THShortTensorMath_scatterAdd"
  c_THShortTensorMath_scatterAdd :: (Ptr CTHShortTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_scatterFill : tensor dim index val -> void
foreign import ccall "THTensorMath.h THShortTensorMath_scatterFill"
  c_THShortTensorMath_scatterFill :: (Ptr CTHShortTensor) -> CInt -> Ptr CTHLongTensor -> CShort -> IO ()

-- |c_THShortTensorMath_dot : t src -> accreal
foreign import ccall "THTensorMath.h THShortTensorMath_dot"
  c_THShortTensorMath_dot :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CLong

-- |c_THShortTensorMath_minall : t -> real
foreign import ccall "THTensorMath.h THShortTensorMath_minall"
  c_THShortTensorMath_minall :: (Ptr CTHShortTensor) -> CShort

-- |c_THShortTensorMath_maxall : t -> real
foreign import ccall "THTensorMath.h THShortTensorMath_maxall"
  c_THShortTensorMath_maxall :: (Ptr CTHShortTensor) -> CShort

-- |c_THShortTensorMath_medianall : t -> real
foreign import ccall "THTensorMath.h THShortTensorMath_medianall"
  c_THShortTensorMath_medianall :: (Ptr CTHShortTensor) -> CShort

-- |c_THShortTensorMath_sumall : t -> accreal
foreign import ccall "THTensorMath.h THShortTensorMath_sumall"
  c_THShortTensorMath_sumall :: (Ptr CTHShortTensor) -> CLong

-- |c_THShortTensorMath_prodall : t -> accreal
foreign import ccall "THTensorMath.h THShortTensorMath_prodall"
  c_THShortTensorMath_prodall :: (Ptr CTHShortTensor) -> CLong

-- |c_THShortTensorMath_neg : self src -> void
foreign import ccall "THTensorMath.h THShortTensorMath_neg"
  c_THShortTensorMath_neg :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_cinv : self src -> void
foreign import ccall "THTensorMath.h THShortTensorMath_cinv"
  c_THShortTensorMath_cinv :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_add : r_ t value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_add"
  c_THShortTensorMath_add :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_sub : self src value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_sub"
  c_THShortTensorMath_sub :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_mul : r_ t value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_mul"
  c_THShortTensorMath_mul :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_div : r_ t value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_div"
  c_THShortTensorMath_div :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_lshift : r_ t value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_lshift"
  c_THShortTensorMath_lshift :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_rshift : r_ t value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_rshift"
  c_THShortTensorMath_rshift :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_fmod : r_ t value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_fmod"
  c_THShortTensorMath_fmod :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_remainder : r_ t value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_remainder"
  c_THShortTensorMath_remainder :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_clamp : r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_clamp"
  c_THShortTensorMath_clamp :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> CShort -> IO ()

-- |c_THShortTensorMath_bitand : r_ t value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_bitand"
  c_THShortTensorMath_bitand :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_bitor : r_ t value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_bitor"
  c_THShortTensorMath_bitor :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_bitxor : r_ t value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_bitxor"
  c_THShortTensorMath_bitxor :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_cadd : r_ t value src -> void
foreign import ccall "THTensorMath.h THShortTensorMath_cadd"
  c_THShortTensorMath_cadd :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_csub : self src1 value src2 -> void
foreign import ccall "THTensorMath.h THShortTensorMath_csub"
  c_THShortTensorMath_csub :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_cmul : r_ t src -> void
foreign import ccall "THTensorMath.h THShortTensorMath_cmul"
  c_THShortTensorMath_cmul :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_cpow : r_ t src -> void
foreign import ccall "THTensorMath.h THShortTensorMath_cpow"
  c_THShortTensorMath_cpow :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_cdiv : r_ t src -> void
foreign import ccall "THTensorMath.h THShortTensorMath_cdiv"
  c_THShortTensorMath_cdiv :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_clshift : r_ t src -> void
foreign import ccall "THTensorMath.h THShortTensorMath_clshift"
  c_THShortTensorMath_clshift :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_crshift : r_ t src -> void
foreign import ccall "THTensorMath.h THShortTensorMath_crshift"
  c_THShortTensorMath_crshift :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_cfmod : r_ t src -> void
foreign import ccall "THTensorMath.h THShortTensorMath_cfmod"
  c_THShortTensorMath_cfmod :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_cremainder : r_ t src -> void
foreign import ccall "THTensorMath.h THShortTensorMath_cremainder"
  c_THShortTensorMath_cremainder :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_cbitand : r_ t src -> void
foreign import ccall "THTensorMath.h THShortTensorMath_cbitand"
  c_THShortTensorMath_cbitand :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_cbitor : r_ t src -> void
foreign import ccall "THTensorMath.h THShortTensorMath_cbitor"
  c_THShortTensorMath_cbitor :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_cbitxor : r_ t src -> void
foreign import ccall "THTensorMath.h THShortTensorMath_cbitxor"
  c_THShortTensorMath_cbitxor :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_addcmul : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THShortTensorMath_addcmul"
  c_THShortTensorMath_addcmul :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_addcdiv : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THShortTensorMath_addcdiv"
  c_THShortTensorMath_addcdiv :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_addmv : r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h THShortTensorMath_addmv"
  c_THShortTensorMath_addmv :: (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_addmm : r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h THShortTensorMath_addmm"
  c_THShortTensorMath_addmm :: (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_addr : r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h THShortTensorMath_addr"
  c_THShortTensorMath_addr :: (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_addbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THShortTensorMath_addbmm"
  c_THShortTensorMath_addbmm :: (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_baddbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THShortTensorMath_baddbmm"
  c_THShortTensorMath_baddbmm :: (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_match : r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h THShortTensorMath_match"
  c_THShortTensorMath_match :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_numel : t -> THStorage *
foreign import ccall "THTensorMath.h THShortTensorMath_numel"
  c_THShortTensorMath_numel :: (Ptr CTHShortTensor) -> IO (Ptr CTHShortStorage)

-- |c_THShortTensorMath_max : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THShortTensorMath_max"
  c_THShortTensorMath_max :: (Ptr CTHShortTensor) -> Ptr CTHLongTensor -> (Ptr CTHShortTensor) -> CInt -> CInt -> IO ()

-- |c_THShortTensorMath_min : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THShortTensorMath_min"
  c_THShortTensorMath_min :: (Ptr CTHShortTensor) -> Ptr CTHLongTensor -> (Ptr CTHShortTensor) -> CInt -> CInt -> IO ()

-- |c_THShortTensorMath_kthvalue : values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h THShortTensorMath_kthvalue"
  c_THShortTensorMath_kthvalue :: (Ptr CTHShortTensor) -> Ptr CTHLongTensor -> (Ptr CTHShortTensor) -> CLong -> CInt -> CInt -> IO ()

-- |c_THShortTensorMath_mode : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THShortTensorMath_mode"
  c_THShortTensorMath_mode :: (Ptr CTHShortTensor) -> Ptr CTHLongTensor -> (Ptr CTHShortTensor) -> CInt -> CInt -> IO ()

-- |c_THShortTensorMath_median : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THShortTensorMath_median"
  c_THShortTensorMath_median :: (Ptr CTHShortTensor) -> Ptr CTHLongTensor -> (Ptr CTHShortTensor) -> CInt -> CInt -> IO ()

-- |c_THShortTensorMath_sum : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THShortTensorMath_sum"
  c_THShortTensorMath_sum :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> CInt -> IO ()

-- |c_THShortTensorMath_prod : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THShortTensorMath_prod"
  c_THShortTensorMath_prod :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> CInt -> IO ()

-- |c_THShortTensorMath_cumsum : r_ t dimension -> void
foreign import ccall "THTensorMath.h THShortTensorMath_cumsum"
  c_THShortTensorMath_cumsum :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> IO ()

-- |c_THShortTensorMath_cumprod : r_ t dimension -> void
foreign import ccall "THTensorMath.h THShortTensorMath_cumprod"
  c_THShortTensorMath_cumprod :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> IO ()

-- |c_THShortTensorMath_sign : r_ t -> void
foreign import ccall "THTensorMath.h THShortTensorMath_sign"
  c_THShortTensorMath_sign :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_trace : t -> accreal
foreign import ccall "THTensorMath.h THShortTensorMath_trace"
  c_THShortTensorMath_trace :: (Ptr CTHShortTensor) -> CLong

-- |c_THShortTensorMath_cross : r_ a b dimension -> void
foreign import ccall "THTensorMath.h THShortTensorMath_cross"
  c_THShortTensorMath_cross :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> IO ()

-- |c_THShortTensorMath_cmax : r t src -> void
foreign import ccall "THTensorMath.h THShortTensorMath_cmax"
  c_THShortTensorMath_cmax :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_cmin : r t src -> void
foreign import ccall "THTensorMath.h THShortTensorMath_cmin"
  c_THShortTensorMath_cmin :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_cmaxValue : r t value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_cmaxValue"
  c_THShortTensorMath_cmaxValue :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_cminValue : r t value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_cminValue"
  c_THShortTensorMath_cminValue :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_zeros : r_ size -> void
foreign import ccall "THTensorMath.h THShortTensorMath_zeros"
  c_THShortTensorMath_zeros :: (Ptr CTHShortTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THShortTensorMath_zerosLike : r_ input -> void
foreign import ccall "THTensorMath.h THShortTensorMath_zerosLike"
  c_THShortTensorMath_zerosLike :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_ones : r_ size -> void
foreign import ccall "THTensorMath.h THShortTensorMath_ones"
  c_THShortTensorMath_ones :: (Ptr CTHShortTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THShortTensorMath_onesLike : r_ input -> void
foreign import ccall "THTensorMath.h THShortTensorMath_onesLike"
  c_THShortTensorMath_onesLike :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_diag : r_ t k -> void
foreign import ccall "THTensorMath.h THShortTensorMath_diag"
  c_THShortTensorMath_diag :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> IO ()

-- |c_THShortTensorMath_eye : r_ n m -> void
foreign import ccall "THTensorMath.h THShortTensorMath_eye"
  c_THShortTensorMath_eye :: (Ptr CTHShortTensor) -> CLong -> CLong -> IO ()

-- |c_THShortTensorMath_arange : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THShortTensorMath_arange"
  c_THShortTensorMath_arange :: (Ptr CTHShortTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THShortTensorMath_range : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THShortTensorMath_range"
  c_THShortTensorMath_range :: (Ptr CTHShortTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THShortTensorMath_randperm : r_ _generator n -> void
foreign import ccall "THTensorMath.h THShortTensorMath_randperm"
  c_THShortTensorMath_randperm :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> CLong -> IO ()

-- |c_THShortTensorMath_reshape : r_ t size -> void
foreign import ccall "THTensorMath.h THShortTensorMath_reshape"
  c_THShortTensorMath_reshape :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THShortTensorMath_sort : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h THShortTensorMath_sort"
  c_THShortTensorMath_sort :: (Ptr CTHShortTensor) -> Ptr CTHLongTensor -> (Ptr CTHShortTensor) -> CInt -> CInt -> IO ()

-- |c_THShortTensorMath_topk : rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h THShortTensorMath_topk"
  c_THShortTensorMath_topk :: (Ptr CTHShortTensor) -> Ptr CTHLongTensor -> (Ptr CTHShortTensor) -> CLong -> CInt -> CInt -> CInt -> IO ()

-- |c_THShortTensorMath_tril : r_ t k -> void
foreign import ccall "THTensorMath.h THShortTensorMath_tril"
  c_THShortTensorMath_tril :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CLong -> IO ()

-- |c_THShortTensorMath_triu : r_ t k -> void
foreign import ccall "THTensorMath.h THShortTensorMath_triu"
  c_THShortTensorMath_triu :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CLong -> IO ()

-- |c_THShortTensorMath_cat : r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h THShortTensorMath_cat"
  c_THShortTensorMath_cat :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> IO ()

-- |c_THShortTensorMath_catArray : result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h THShortTensorMath_catArray"
  c_THShortTensorMath_catArray :: (Ptr CTHShortTensor) -> Ptr (Ptr CTHShortTensor) -> CInt -> CInt -> IO ()

-- |c_THShortTensorMath_equal : ta tb -> int
foreign import ccall "THTensorMath.h THShortTensorMath_equal"
  c_THShortTensorMath_equal :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt

-- |c_THShortTensorMath_ltValue : r_ t value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_ltValue"
  c_THShortTensorMath_ltValue :: Ptr CTHByteTensor -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_leValue : r_ t value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_leValue"
  c_THShortTensorMath_leValue :: Ptr CTHByteTensor -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_gtValue : r_ t value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_gtValue"
  c_THShortTensorMath_gtValue :: Ptr CTHByteTensor -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_geValue : r_ t value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_geValue"
  c_THShortTensorMath_geValue :: Ptr CTHByteTensor -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_neValue : r_ t value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_neValue"
  c_THShortTensorMath_neValue :: Ptr CTHByteTensor -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_eqValue : r_ t value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_eqValue"
  c_THShortTensorMath_eqValue :: Ptr CTHByteTensor -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_ltValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_ltValueT"
  c_THShortTensorMath_ltValueT :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_leValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_leValueT"
  c_THShortTensorMath_leValueT :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_gtValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_gtValueT"
  c_THShortTensorMath_gtValueT :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_geValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_geValueT"
  c_THShortTensorMath_geValueT :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_neValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_neValueT"
  c_THShortTensorMath_neValueT :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_eqValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_eqValueT"
  c_THShortTensorMath_eqValueT :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_ltTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THShortTensorMath_ltTensor"
  c_THShortTensorMath_ltTensor :: Ptr CTHByteTensor -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_leTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THShortTensorMath_leTensor"
  c_THShortTensorMath_leTensor :: Ptr CTHByteTensor -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_gtTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THShortTensorMath_gtTensor"
  c_THShortTensorMath_gtTensor :: Ptr CTHByteTensor -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_geTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THShortTensorMath_geTensor"
  c_THShortTensorMath_geTensor :: Ptr CTHByteTensor -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_neTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THShortTensorMath_neTensor"
  c_THShortTensorMath_neTensor :: Ptr CTHByteTensor -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_eqTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THShortTensorMath_eqTensor"
  c_THShortTensorMath_eqTensor :: Ptr CTHByteTensor -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_ltTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THShortTensorMath_ltTensorT"
  c_THShortTensorMath_ltTensorT :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_leTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THShortTensorMath_leTensorT"
  c_THShortTensorMath_leTensorT :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_gtTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THShortTensorMath_gtTensorT"
  c_THShortTensorMath_gtTensorT :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_geTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THShortTensorMath_geTensorT"
  c_THShortTensorMath_geTensorT :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_neTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THShortTensorMath_neTensorT"
  c_THShortTensorMath_neTensorT :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_eqTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THShortTensorMath_eqTensorT"
  c_THShortTensorMath_eqTensorT :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_abs : r_ t -> void
foreign import ccall "THTensorMath.h THShortTensorMath_abs"
  c_THShortTensorMath_abs :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_sigmoid : r_ t -> void
foreign import ccall "THTensorMath.h THShortTensorMath_sigmoid"
  c_THShortTensorMath_sigmoid :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_log : r_ t -> void
foreign import ccall "THTensorMath.h THShortTensorMath_log"
  c_THShortTensorMath_log :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_lgamma : r_ t -> void
foreign import ccall "THTensorMath.h THShortTensorMath_lgamma"
  c_THShortTensorMath_lgamma :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_log1p : r_ t -> void
foreign import ccall "THTensorMath.h THShortTensorMath_log1p"
  c_THShortTensorMath_log1p :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_exp : r_ t -> void
foreign import ccall "THTensorMath.h THShortTensorMath_exp"
  c_THShortTensorMath_exp :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_cos : r_ t -> void
foreign import ccall "THTensorMath.h THShortTensorMath_cos"
  c_THShortTensorMath_cos :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_acos : r_ t -> void
foreign import ccall "THTensorMath.h THShortTensorMath_acos"
  c_THShortTensorMath_acos :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_cosh : r_ t -> void
foreign import ccall "THTensorMath.h THShortTensorMath_cosh"
  c_THShortTensorMath_cosh :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_sin : r_ t -> void
foreign import ccall "THTensorMath.h THShortTensorMath_sin"
  c_THShortTensorMath_sin :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_asin : r_ t -> void
foreign import ccall "THTensorMath.h THShortTensorMath_asin"
  c_THShortTensorMath_asin :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_sinh : r_ t -> void
foreign import ccall "THTensorMath.h THShortTensorMath_sinh"
  c_THShortTensorMath_sinh :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_tan : r_ t -> void
foreign import ccall "THTensorMath.h THShortTensorMath_tan"
  c_THShortTensorMath_tan :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_atan : r_ t -> void
foreign import ccall "THTensorMath.h THShortTensorMath_atan"
  c_THShortTensorMath_atan :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_atan2 : r_ tx ty -> void
foreign import ccall "THTensorMath.h THShortTensorMath_atan2"
  c_THShortTensorMath_atan2 :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_tanh : r_ t -> void
foreign import ccall "THTensorMath.h THShortTensorMath_tanh"
  c_THShortTensorMath_tanh :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_pow : r_ t value -> void
foreign import ccall "THTensorMath.h THShortTensorMath_pow"
  c_THShortTensorMath_pow :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_tpow : r_ value t -> void
foreign import ccall "THTensorMath.h THShortTensorMath_tpow"
  c_THShortTensorMath_tpow :: (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_sqrt : r_ t -> void
foreign import ccall "THTensorMath.h THShortTensorMath_sqrt"
  c_THShortTensorMath_sqrt :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_rsqrt : r_ t -> void
foreign import ccall "THTensorMath.h THShortTensorMath_rsqrt"
  c_THShortTensorMath_rsqrt :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_ceil : r_ t -> void
foreign import ccall "THTensorMath.h THShortTensorMath_ceil"
  c_THShortTensorMath_ceil :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_floor : r_ t -> void
foreign import ccall "THTensorMath.h THShortTensorMath_floor"
  c_THShortTensorMath_floor :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_round : r_ t -> void
foreign import ccall "THTensorMath.h THShortTensorMath_round"
  c_THShortTensorMath_round :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_trunc : r_ t -> void
foreign import ccall "THTensorMath.h THShortTensorMath_trunc"
  c_THShortTensorMath_trunc :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_frac : r_ t -> void
foreign import ccall "THTensorMath.h THShortTensorMath_frac"
  c_THShortTensorMath_frac :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensorMath_lerp : r_ a b weight -> void
foreign import ccall "THTensorMath.h THShortTensorMath_lerp"
  c_THShortTensorMath_lerp :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensorMath_mean : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THShortTensorMath_mean"
  c_THShortTensorMath_mean :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> CInt -> IO ()

-- |c_THShortTensorMath_std : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THShortTensorMath_std"
  c_THShortTensorMath_std :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THShortTensorMath_var : r_ t dimension biased keepdim -> void
foreign import ccall "THTensorMath.h THShortTensorMath_var"
  c_THShortTensorMath_var :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> CInt -> CInt -> IO ()

-- |c_THShortTensorMath_norm : r_ t value dimension keepdim -> void
foreign import ccall "THTensorMath.h THShortTensorMath_norm"
  c_THShortTensorMath_norm :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> CInt -> CInt -> IO ()

-- |c_THShortTensorMath_renorm : r_ t value dimension maxnorm -> void
foreign import ccall "THTensorMath.h THShortTensorMath_renorm"
  c_THShortTensorMath_renorm :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> CInt -> CShort -> IO ()

-- |c_THShortTensorMath_dist : a b value -> accreal
foreign import ccall "THTensorMath.h THShortTensorMath_dist"
  c_THShortTensorMath_dist :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> CLong

-- |c_THShortTensorMath_histc : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THShortTensorMath_histc"
  c_THShortTensorMath_histc :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CLong -> CShort -> CShort -> IO ()

-- |c_THShortTensorMath_bhistc : hist tensor nbins minvalue maxvalue -> void
foreign import ccall "THTensorMath.h THShortTensorMath_bhistc"
  c_THShortTensorMath_bhistc :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CLong -> CShort -> CShort -> IO ()

-- |c_THShortTensorMath_meanall : self -> accreal
foreign import ccall "THTensorMath.h THShortTensorMath_meanall"
  c_THShortTensorMath_meanall :: (Ptr CTHShortTensor) -> CLong

-- |c_THShortTensorMath_varall : self biased -> accreal
foreign import ccall "THTensorMath.h THShortTensorMath_varall"
  c_THShortTensorMath_varall :: (Ptr CTHShortTensor) -> CInt -> CLong

-- |c_THShortTensorMath_stdall : self biased -> accreal
foreign import ccall "THTensorMath.h THShortTensorMath_stdall"
  c_THShortTensorMath_stdall :: (Ptr CTHShortTensor) -> CInt -> CLong

-- |c_THShortTensorMath_normall : t value -> accreal
foreign import ccall "THTensorMath.h THShortTensorMath_normall"
  c_THShortTensorMath_normall :: (Ptr CTHShortTensor) -> CShort -> CLong

-- |c_THShortTensorMath_linspace : r_ a b n -> void
foreign import ccall "THTensorMath.h THShortTensorMath_linspace"
  c_THShortTensorMath_linspace :: (Ptr CTHShortTensor) -> CShort -> CShort -> CLong -> IO ()

-- |c_THShortTensorMath_logspace : r_ a b n -> void
foreign import ccall "THTensorMath.h THShortTensorMath_logspace"
  c_THShortTensorMath_logspace :: (Ptr CTHShortTensor) -> CShort -> CShort -> CLong -> IO ()

-- |c_THShortTensorMath_rand : r_ _generator size -> void
foreign import ccall "THTensorMath.h THShortTensorMath_rand"
  c_THShortTensorMath_rand :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()

-- |c_THShortTensorMath_randn : r_ _generator size -> void
foreign import ccall "THTensorMath.h THShortTensorMath_randn"
  c_THShortTensorMath_randn :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()

-- |c_THShortTensorMath_logicalall : self -> int
foreign import ccall "THTensorMath.h THShortTensorMath_logicalall"
  c_THShortTensorMath_logicalall :: (Ptr CTHShortTensor) -> CInt

-- |c_THShortTensorMath_logicalany : self -> int
foreign import ccall "THTensorMath.h THShortTensorMath_logicalany"
  c_THShortTensorMath_logicalany :: (Ptr CTHShortTensor) -> CInt