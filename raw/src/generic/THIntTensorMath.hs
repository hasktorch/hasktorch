{-# LANGUAGE ForeignFunctionInterface #-}

module THIntTensorMath (
    c_THIntTensor_fill,
    c_THIntTensor_zero,
    c_THIntTensor_maskedFill,
    c_THIntTensor_maskedCopy,
    c_THIntTensor_maskedSelect,
    c_THIntTensor_nonzero,
    c_THIntTensor_indexSelect,
    c_THIntTensor_indexCopy,
    c_THIntTensor_indexAdd,
    c_THIntTensor_indexFill,
    c_THIntTensor_gather,
    c_THIntTensor_scatter,
    c_THIntTensor_scatterAdd,
    c_THIntTensor_scatterFill,
    c_THIntTensor_dot,
    c_THIntTensor_minall,
    c_THIntTensor_maxall,
    c_THIntTensor_medianall,
    c_THIntTensor_sumall,
    c_THIntTensor_prodall,
    c_THIntTensor_neg,
    c_THIntTensor_add,
    c_THIntTensor_sub,
    c_THIntTensor_mul,
    c_THIntTensor_div,
    c_THIntTensor_lshift,
    c_THIntTensor_rshift,
    c_THIntTensor_fmod,
    c_THIntTensor_remainder,
    c_THIntTensor_clamp,
    c_THIntTensor_bitand,
    c_THIntTensor_bitor,
    c_THIntTensor_bitxor,
    c_THIntTensor_cadd,
    c_THIntTensor_csub,
    c_THIntTensor_cmul,
    c_THIntTensor_cpow,
    c_THIntTensor_cdiv,
    c_THIntTensor_clshift,
    c_THIntTensor_crshift,
    c_THIntTensor_cfmod,
    c_THIntTensor_cremainder,
    c_THIntTensor_cbitand,
    c_THIntTensor_cbitor,
    c_THIntTensor_cbitxor,
    c_THIntTensor_addcmul,
    c_THIntTensor_addcdiv,
    c_THIntTensor_addmv,
    c_THIntTensor_addmm,
    c_THIntTensor_addr,
    c_THIntTensor_addbmm,
    c_THIntTensor_baddbmm,
    c_THIntTensor_match,
    c_THIntTensor_numel,
    c_THIntTensor_max,
    c_THIntTensor_min,
    c_THIntTensor_kthvalue,
    c_THIntTensor_mode,
    c_THIntTensor_median,
    c_THIntTensor_sum,
    c_THIntTensor_prod,
    c_THIntTensor_cumsum,
    c_THIntTensor_cumprod,
    c_THIntTensor_sign,
    c_THIntTensor_trace,
    c_THIntTensor_cross,
    c_THIntTensor_cmax,
    c_THIntTensor_cmin,
    c_THIntTensor_cmaxValue,
    c_THIntTensor_cminValue,
    c_THIntTensor_zeros,
    c_THIntTensor_zerosLike,
    c_THIntTensor_ones,
    c_THIntTensor_onesLike,
    c_THIntTensor_diag,
    c_THIntTensor_eye,
    c_THIntTensor_arange,
    c_THIntTensor_range,
    c_THIntTensor_randperm,
    c_THIntTensor_reshape,
    c_THIntTensor_sort,
    c_THIntTensor_topk,
    c_THIntTensor_tril,
    c_THIntTensor_triu,
    c_THIntTensor_cat,
    c_THIntTensor_catArray,
    c_THIntTensor_equal,
    c_THIntTensor_ltValue,
    c_THIntTensor_leValue,
    c_THIntTensor_gtValue,
    c_THIntTensor_geValue,
    c_THIntTensor_neValue,
    c_THIntTensor_eqValue,
    c_THIntTensor_ltValueT,
    c_THIntTensor_leValueT,
    c_THIntTensor_gtValueT,
    c_THIntTensor_geValueT,
    c_THIntTensor_neValueT,
    c_THIntTensor_eqValueT,
    c_THIntTensor_ltTensor,
    c_THIntTensor_leTensor,
    c_THIntTensor_gtTensor,
    c_THIntTensor_geTensor,
    c_THIntTensor_neTensor,
    c_THIntTensor_eqTensor,
    c_THIntTensor_ltTensorT,
    c_THIntTensor_leTensorT,
    c_THIntTensor_gtTensorT,
    c_THIntTensor_geTensorT,
    c_THIntTensor_neTensorT,
    c_THIntTensor_eqTensorT,
    c_THIntTensor_abs,
    p_THIntTensor_fill,
    p_THIntTensor_zero,
    p_THIntTensor_maskedFill,
    p_THIntTensor_maskedCopy,
    p_THIntTensor_maskedSelect,
    p_THIntTensor_nonzero,
    p_THIntTensor_indexSelect,
    p_THIntTensor_indexCopy,
    p_THIntTensor_indexAdd,
    p_THIntTensor_indexFill,
    p_THIntTensor_gather,
    p_THIntTensor_scatter,
    p_THIntTensor_scatterAdd,
    p_THIntTensor_scatterFill,
    p_THIntTensor_dot,
    p_THIntTensor_minall,
    p_THIntTensor_maxall,
    p_THIntTensor_medianall,
    p_THIntTensor_sumall,
    p_THIntTensor_prodall,
    p_THIntTensor_neg,
    p_THIntTensor_add,
    p_THIntTensor_sub,
    p_THIntTensor_mul,
    p_THIntTensor_div,
    p_THIntTensor_lshift,
    p_THIntTensor_rshift,
    p_THIntTensor_fmod,
    p_THIntTensor_remainder,
    p_THIntTensor_clamp,
    p_THIntTensor_bitand,
    p_THIntTensor_bitor,
    p_THIntTensor_bitxor,
    p_THIntTensor_cadd,
    p_THIntTensor_csub,
    p_THIntTensor_cmul,
    p_THIntTensor_cpow,
    p_THIntTensor_cdiv,
    p_THIntTensor_clshift,
    p_THIntTensor_crshift,
    p_THIntTensor_cfmod,
    p_THIntTensor_cremainder,
    p_THIntTensor_cbitand,
    p_THIntTensor_cbitor,
    p_THIntTensor_cbitxor,
    p_THIntTensor_addcmul,
    p_THIntTensor_addcdiv,
    p_THIntTensor_addmv,
    p_THIntTensor_addmm,
    p_THIntTensor_addr,
    p_THIntTensor_addbmm,
    p_THIntTensor_baddbmm,
    p_THIntTensor_match,
    p_THIntTensor_numel,
    p_THIntTensor_max,
    p_THIntTensor_min,
    p_THIntTensor_kthvalue,
    p_THIntTensor_mode,
    p_THIntTensor_median,
    p_THIntTensor_sum,
    p_THIntTensor_prod,
    p_THIntTensor_cumsum,
    p_THIntTensor_cumprod,
    p_THIntTensor_sign,
    p_THIntTensor_trace,
    p_THIntTensor_cross,
    p_THIntTensor_cmax,
    p_THIntTensor_cmin,
    p_THIntTensor_cmaxValue,
    p_THIntTensor_cminValue,
    p_THIntTensor_zeros,
    p_THIntTensor_zerosLike,
    p_THIntTensor_ones,
    p_THIntTensor_onesLike,
    p_THIntTensor_diag,
    p_THIntTensor_eye,
    p_THIntTensor_arange,
    p_THIntTensor_range,
    p_THIntTensor_randperm,
    p_THIntTensor_reshape,
    p_THIntTensor_sort,
    p_THIntTensor_topk,
    p_THIntTensor_tril,
    p_THIntTensor_triu,
    p_THIntTensor_cat,
    p_THIntTensor_catArray,
    p_THIntTensor_equal,
    p_THIntTensor_ltValue,
    p_THIntTensor_leValue,
    p_THIntTensor_gtValue,
    p_THIntTensor_geValue,
    p_THIntTensor_neValue,
    p_THIntTensor_eqValue,
    p_THIntTensor_ltValueT,
    p_THIntTensor_leValueT,
    p_THIntTensor_gtValueT,
    p_THIntTensor_geValueT,
    p_THIntTensor_neValueT,
    p_THIntTensor_eqValueT,
    p_THIntTensor_ltTensor,
    p_THIntTensor_leTensor,
    p_THIntTensor_gtTensor,
    p_THIntTensor_geTensor,
    p_THIntTensor_neTensor,
    p_THIntTensor_eqTensor,
    p_THIntTensor_ltTensorT,
    p_THIntTensor_leTensorT,
    p_THIntTensor_gtTensorT,
    p_THIntTensor_geTensorT,
    p_THIntTensor_neTensorT,
    p_THIntTensor_eqTensorT,
    p_THIntTensor_abs) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THIntTensor_fill : r_ value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_fill"
  c_THIntTensor_fill :: (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_zero : r_ -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_zero"
  c_THIntTensor_zero :: (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_maskedFill : tensor mask value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_maskedFill"
  c_THIntTensor_maskedFill :: (Ptr CTHIntTensor) -> Ptr CTHByteTensor -> CInt -> IO ()

-- |c_THIntTensor_maskedCopy : tensor mask src -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_maskedCopy"
  c_THIntTensor_maskedCopy :: (Ptr CTHIntTensor) -> Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_maskedSelect : tensor src mask -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_maskedSelect"
  c_THIntTensor_maskedSelect :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THIntTensor_nonzero : subscript tensor -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_nonzero"
  c_THIntTensor_nonzero :: Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_indexSelect : tensor src dim index -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_indexSelect"
  c_THIntTensor_indexSelect :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THIntTensor_indexCopy : tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_indexCopy"
  c_THIntTensor_indexCopy :: (Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_indexAdd : tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_indexAdd"
  c_THIntTensor_indexAdd :: (Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_indexFill : tensor dim index val -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_indexFill"
  c_THIntTensor_indexFill :: (Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> CInt -> IO ()

-- |c_THIntTensor_gather : tensor src dim index -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_gather"
  c_THIntTensor_gather :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THIntTensor_scatter : tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_scatter"
  c_THIntTensor_scatter :: (Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_scatterAdd : tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_scatterAdd"
  c_THIntTensor_scatterAdd :: (Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_scatterFill : tensor dim index val -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_scatterFill"
  c_THIntTensor_scatterFill :: (Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> CInt -> IO ()

-- |c_THIntTensor_dot : t src -> accreal
foreign import ccall unsafe "THTensorMath.h THIntTensor_dot"
  c_THIntTensor_dot :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CLong

-- |c_THIntTensor_minall : t -> real
foreign import ccall unsafe "THTensorMath.h THIntTensor_minall"
  c_THIntTensor_minall :: (Ptr CTHIntTensor) -> CInt

-- |c_THIntTensor_maxall : t -> real
foreign import ccall unsafe "THTensorMath.h THIntTensor_maxall"
  c_THIntTensor_maxall :: (Ptr CTHIntTensor) -> CInt

-- |c_THIntTensor_medianall : t -> real
foreign import ccall unsafe "THTensorMath.h THIntTensor_medianall"
  c_THIntTensor_medianall :: (Ptr CTHIntTensor) -> CInt

-- |c_THIntTensor_sumall : t -> accreal
foreign import ccall unsafe "THTensorMath.h THIntTensor_sumall"
  c_THIntTensor_sumall :: (Ptr CTHIntTensor) -> CLong

-- |c_THIntTensor_prodall : t -> accreal
foreign import ccall unsafe "THTensorMath.h THIntTensor_prodall"
  c_THIntTensor_prodall :: (Ptr CTHIntTensor) -> CLong

-- |c_THIntTensor_neg : self src -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_neg"
  c_THIntTensor_neg :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_add : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_add"
  c_THIntTensor_add :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_sub : self src value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_sub"
  c_THIntTensor_sub :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_mul : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_mul"
  c_THIntTensor_mul :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_div : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_div"
  c_THIntTensor_div :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_lshift : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_lshift"
  c_THIntTensor_lshift :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_rshift : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_rshift"
  c_THIntTensor_rshift :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_fmod : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_fmod"
  c_THIntTensor_fmod :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_remainder : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_remainder"
  c_THIntTensor_remainder :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_clamp : r_ t min_value max_value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_clamp"
  c_THIntTensor_clamp :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensor_bitand : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_bitand"
  c_THIntTensor_bitand :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_bitor : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_bitor"
  c_THIntTensor_bitor :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_bitxor : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_bitxor"
  c_THIntTensor_bitxor :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_cadd : r_ t value src -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_cadd"
  c_THIntTensor_cadd :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_csub : self src1 value src2 -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_csub"
  c_THIntTensor_csub :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_cmul : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_cmul"
  c_THIntTensor_cmul :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_cpow : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_cpow"
  c_THIntTensor_cpow :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_cdiv : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_cdiv"
  c_THIntTensor_cdiv :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_clshift : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_clshift"
  c_THIntTensor_clshift :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_crshift : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_crshift"
  c_THIntTensor_crshift :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_cfmod : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_cfmod"
  c_THIntTensor_cfmod :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_cremainder : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_cremainder"
  c_THIntTensor_cremainder :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_cbitand : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_cbitand"
  c_THIntTensor_cbitand :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_cbitor : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_cbitor"
  c_THIntTensor_cbitor :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_cbitxor : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_cbitxor"
  c_THIntTensor_cbitxor :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_addcmul : r_ t value src1 src2 -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_addcmul"
  c_THIntTensor_addcmul :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_addcdiv : r_ t value src1 src2 -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_addcdiv"
  c_THIntTensor_addcdiv :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_addmv : r_ beta t alpha mat vec -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_addmv"
  c_THIntTensor_addmv :: (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_addmm : r_ beta t alpha mat1 mat2 -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_addmm"
  c_THIntTensor_addmm :: (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_addr : r_ beta t alpha vec1 vec2 -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_addr"
  c_THIntTensor_addr :: (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_addbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_addbmm"
  c_THIntTensor_addbmm :: (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_baddbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_baddbmm"
  c_THIntTensor_baddbmm :: (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_match : r_ m1 m2 gain -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_match"
  c_THIntTensor_match :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_numel : t -> ptrdiff_t
foreign import ccall unsafe "THTensorMath.h THIntTensor_numel"
  c_THIntTensor_numel :: (Ptr CTHIntTensor) -> CPtrdiff

-- |c_THIntTensor_max : values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_max"
  c_THIntTensor_max :: (Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensor_min : values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_min"
  c_THIntTensor_min :: (Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensor_kthvalue : values_ indices_ t k dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_kthvalue"
  c_THIntTensor_kthvalue :: (Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CLong -> CInt -> CInt -> IO ()

-- |c_THIntTensor_mode : values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_mode"
  c_THIntTensor_mode :: (Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensor_median : values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_median"
  c_THIntTensor_median :: (Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensor_sum : r_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_sum"
  c_THIntTensor_sum :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensor_prod : r_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_prod"
  c_THIntTensor_prod :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensor_cumsum : r_ t dimension -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_cumsum"
  c_THIntTensor_cumsum :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_cumprod : r_ t dimension -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_cumprod"
  c_THIntTensor_cumprod :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_sign : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_sign"
  c_THIntTensor_sign :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_trace : t -> accreal
foreign import ccall unsafe "THTensorMath.h THIntTensor_trace"
  c_THIntTensor_trace :: (Ptr CTHIntTensor) -> CLong

-- |c_THIntTensor_cross : r_ a b dimension -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_cross"
  c_THIntTensor_cross :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_cmax : r t src -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_cmax"
  c_THIntTensor_cmax :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_cmin : r t src -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_cmin"
  c_THIntTensor_cmin :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_cmaxValue : r t value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_cmaxValue"
  c_THIntTensor_cmaxValue :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_cminValue : r t value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_cminValue"
  c_THIntTensor_cminValue :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_zeros : r_ size -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_zeros"
  c_THIntTensor_zeros :: (Ptr CTHIntTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THIntTensor_zerosLike : r_ input -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_zerosLike"
  c_THIntTensor_zerosLike :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_ones : r_ size -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_ones"
  c_THIntTensor_ones :: (Ptr CTHIntTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THIntTensor_onesLike : r_ input -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_onesLike"
  c_THIntTensor_onesLike :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_diag : r_ t k -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_diag"
  c_THIntTensor_diag :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_eye : r_ n m -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_eye"
  c_THIntTensor_eye :: (Ptr CTHIntTensor) -> CLong -> CLong -> IO ()

-- |c_THIntTensor_arange : r_ xmin xmax step -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_arange"
  c_THIntTensor_arange :: (Ptr CTHIntTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THIntTensor_range : r_ xmin xmax step -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_range"
  c_THIntTensor_range :: (Ptr CTHIntTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THIntTensor_randperm : r_ _generator n -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_randperm"
  c_THIntTensor_randperm :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> CLong -> IO ()

-- |c_THIntTensor_reshape : r_ t size -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_reshape"
  c_THIntTensor_reshape :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THIntTensor_sort : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_sort"
  c_THIntTensor_sort :: (Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensor_topk : rt_ ri_ t k dim dir sorted -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_topk"
  c_THIntTensor_topk :: (Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CLong -> CInt -> CInt -> CInt -> IO ()

-- |c_THIntTensor_tril : r_ t k -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_tril"
  c_THIntTensor_tril :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CLong -> IO ()

-- |c_THIntTensor_triu : r_ t k -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_triu"
  c_THIntTensor_triu :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CLong -> IO ()

-- |c_THIntTensor_cat : r_ ta tb dimension -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_cat"
  c_THIntTensor_cat :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_catArray : result inputs numInputs dimension -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_catArray"
  c_THIntTensor_catArray :: (Ptr CTHIntTensor) -> Ptr (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensor_equal : ta tb -> int
foreign import ccall unsafe "THTensorMath.h THIntTensor_equal"
  c_THIntTensor_equal :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt

-- |c_THIntTensor_ltValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_ltValue"
  c_THIntTensor_ltValue :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_leValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_leValue"
  c_THIntTensor_leValue :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_gtValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_gtValue"
  c_THIntTensor_gtValue :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_geValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_geValue"
  c_THIntTensor_geValue :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_neValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_neValue"
  c_THIntTensor_neValue :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_eqValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_eqValue"
  c_THIntTensor_eqValue :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_ltValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_ltValueT"
  c_THIntTensor_ltValueT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_leValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_leValueT"
  c_THIntTensor_leValueT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_gtValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_gtValueT"
  c_THIntTensor_gtValueT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_geValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_geValueT"
  c_THIntTensor_geValueT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_neValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_neValueT"
  c_THIntTensor_neValueT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_eqValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_eqValueT"
  c_THIntTensor_eqValueT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_ltTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_ltTensor"
  c_THIntTensor_ltTensor :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_leTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_leTensor"
  c_THIntTensor_leTensor :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_gtTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_gtTensor"
  c_THIntTensor_gtTensor :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_geTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_geTensor"
  c_THIntTensor_geTensor :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_neTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_neTensor"
  c_THIntTensor_neTensor :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_eqTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_eqTensor"
  c_THIntTensor_eqTensor :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_ltTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_ltTensorT"
  c_THIntTensor_ltTensorT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_leTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_leTensorT"
  c_THIntTensor_leTensorT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_gtTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_gtTensorT"
  c_THIntTensor_gtTensorT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_geTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_geTensorT"
  c_THIntTensor_geTensorT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_neTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_neTensorT"
  c_THIntTensor_neTensorT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_eqTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_eqTensorT"
  c_THIntTensor_eqTensorT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_abs : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THIntTensor_abs"
  c_THIntTensor_abs :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |p_THIntTensor_fill : Pointer to function r_ value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_fill"
  p_THIntTensor_fill :: FunPtr ((Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_zero : Pointer to function r_ -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_zero"
  p_THIntTensor_zero :: FunPtr ((Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_maskedFill : Pointer to function tensor mask value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_maskedFill"
  p_THIntTensor_maskedFill :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHByteTensor -> CInt -> IO ())

-- |p_THIntTensor_maskedCopy : Pointer to function tensor mask src -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_maskedCopy"
  p_THIntTensor_maskedCopy :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_maskedSelect : Pointer to function tensor src mask -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_maskedSelect"
  p_THIntTensor_maskedSelect :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CTHByteTensor -> IO ())

-- |p_THIntTensor_nonzero : Pointer to function subscript tensor -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_nonzero"
  p_THIntTensor_nonzero :: FunPtr (Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_indexSelect : Pointer to function tensor src dim index -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_indexSelect"
  p_THIntTensor_indexSelect :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> IO ())

-- |p_THIntTensor_indexCopy : Pointer to function tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_indexCopy"
  p_THIntTensor_indexCopy :: FunPtr ((Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_indexAdd : Pointer to function tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_indexAdd"
  p_THIntTensor_indexAdd :: FunPtr ((Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_indexFill : Pointer to function tensor dim index val -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_indexFill"
  p_THIntTensor_indexFill :: FunPtr ((Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> CInt -> IO ())

-- |p_THIntTensor_gather : Pointer to function tensor src dim index -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_gather"
  p_THIntTensor_gather :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> IO ())

-- |p_THIntTensor_scatter : Pointer to function tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_scatter"
  p_THIntTensor_scatter :: FunPtr ((Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_scatterAdd : Pointer to function tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_scatterAdd"
  p_THIntTensor_scatterAdd :: FunPtr ((Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_scatterFill : Pointer to function tensor dim index val -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_scatterFill"
  p_THIntTensor_scatterFill :: FunPtr ((Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> CInt -> IO ())

-- |p_THIntTensor_dot : Pointer to function t src -> accreal
foreign import ccall unsafe "THTensorMath.h &THIntTensor_dot"
  p_THIntTensor_dot :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CLong)

-- |p_THIntTensor_minall : Pointer to function t -> real
foreign import ccall unsafe "THTensorMath.h &THIntTensor_minall"
  p_THIntTensor_minall :: FunPtr ((Ptr CTHIntTensor) -> CInt)

-- |p_THIntTensor_maxall : Pointer to function t -> real
foreign import ccall unsafe "THTensorMath.h &THIntTensor_maxall"
  p_THIntTensor_maxall :: FunPtr ((Ptr CTHIntTensor) -> CInt)

-- |p_THIntTensor_medianall : Pointer to function t -> real
foreign import ccall unsafe "THTensorMath.h &THIntTensor_medianall"
  p_THIntTensor_medianall :: FunPtr ((Ptr CTHIntTensor) -> CInt)

-- |p_THIntTensor_sumall : Pointer to function t -> accreal
foreign import ccall unsafe "THTensorMath.h &THIntTensor_sumall"
  p_THIntTensor_sumall :: FunPtr ((Ptr CTHIntTensor) -> CLong)

-- |p_THIntTensor_prodall : Pointer to function t -> accreal
foreign import ccall unsafe "THTensorMath.h &THIntTensor_prodall"
  p_THIntTensor_prodall :: FunPtr ((Ptr CTHIntTensor) -> CLong)

-- |p_THIntTensor_neg : Pointer to function self src -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_neg"
  p_THIntTensor_neg :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_add : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_add"
  p_THIntTensor_add :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_sub : Pointer to function self src value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_sub"
  p_THIntTensor_sub :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_mul : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_mul"
  p_THIntTensor_mul :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_div : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_div"
  p_THIntTensor_div :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_lshift : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_lshift"
  p_THIntTensor_lshift :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_rshift : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_rshift"
  p_THIntTensor_rshift :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_fmod : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_fmod"
  p_THIntTensor_fmod :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_remainder : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_remainder"
  p_THIntTensor_remainder :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_clamp : Pointer to function r_ t min_value max_value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_clamp"
  p_THIntTensor_clamp :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ())

-- |p_THIntTensor_bitand : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_bitand"
  p_THIntTensor_bitand :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_bitor : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_bitor"
  p_THIntTensor_bitor :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_bitxor : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_bitxor"
  p_THIntTensor_bitxor :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_cadd : Pointer to function r_ t value src -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_cadd"
  p_THIntTensor_cadd :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_csub : Pointer to function self src1 value src2 -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_csub"
  p_THIntTensor_csub :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_cmul : Pointer to function r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_cmul"
  p_THIntTensor_cmul :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_cpow : Pointer to function r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_cpow"
  p_THIntTensor_cpow :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_cdiv : Pointer to function r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_cdiv"
  p_THIntTensor_cdiv :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_clshift : Pointer to function r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_clshift"
  p_THIntTensor_clshift :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_crshift : Pointer to function r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_crshift"
  p_THIntTensor_crshift :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_cfmod : Pointer to function r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_cfmod"
  p_THIntTensor_cfmod :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_cremainder : Pointer to function r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_cremainder"
  p_THIntTensor_cremainder :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_cbitand : Pointer to function r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_cbitand"
  p_THIntTensor_cbitand :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_cbitor : Pointer to function r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_cbitor"
  p_THIntTensor_cbitor :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_cbitxor : Pointer to function r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_cbitxor"
  p_THIntTensor_cbitxor :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_addcmul : Pointer to function r_ t value src1 src2 -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_addcmul"
  p_THIntTensor_addcmul :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_addcdiv : Pointer to function r_ t value src1 src2 -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_addcdiv"
  p_THIntTensor_addcdiv :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_addmv : Pointer to function r_ beta t alpha mat vec -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_addmv"
  p_THIntTensor_addmv :: FunPtr ((Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_addmm : Pointer to function r_ beta t alpha mat1 mat2 -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_addmm"
  p_THIntTensor_addmm :: FunPtr ((Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_addr : Pointer to function r_ beta t alpha vec1 vec2 -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_addr"
  p_THIntTensor_addr :: FunPtr ((Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_addbmm : Pointer to function r_ beta t alpha batch1 batch2 -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_addbmm"
  p_THIntTensor_addbmm :: FunPtr ((Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_baddbmm : Pointer to function r_ beta t alpha batch1 batch2 -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_baddbmm"
  p_THIntTensor_baddbmm :: FunPtr ((Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_match : Pointer to function r_ m1 m2 gain -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_match"
  p_THIntTensor_match :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_numel : Pointer to function t -> ptrdiff_t
foreign import ccall unsafe "THTensorMath.h &THIntTensor_numel"
  p_THIntTensor_numel :: FunPtr ((Ptr CTHIntTensor) -> CPtrdiff)

-- |p_THIntTensor_max : Pointer to function values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_max"
  p_THIntTensor_max :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ())

-- |p_THIntTensor_min : Pointer to function values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_min"
  p_THIntTensor_min :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ())

-- |p_THIntTensor_kthvalue : Pointer to function values_ indices_ t k dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_kthvalue"
  p_THIntTensor_kthvalue :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CLong -> CInt -> CInt -> IO ())

-- |p_THIntTensor_mode : Pointer to function values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_mode"
  p_THIntTensor_mode :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ())

-- |p_THIntTensor_median : Pointer to function values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_median"
  p_THIntTensor_median :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ())

-- |p_THIntTensor_sum : Pointer to function r_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_sum"
  p_THIntTensor_sum :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ())

-- |p_THIntTensor_prod : Pointer to function r_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_prod"
  p_THIntTensor_prod :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ())

-- |p_THIntTensor_cumsum : Pointer to function r_ t dimension -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_cumsum"
  p_THIntTensor_cumsum :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_cumprod : Pointer to function r_ t dimension -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_cumprod"
  p_THIntTensor_cumprod :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_sign : Pointer to function r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_sign"
  p_THIntTensor_sign :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_trace : Pointer to function t -> accreal
foreign import ccall unsafe "THTensorMath.h &THIntTensor_trace"
  p_THIntTensor_trace :: FunPtr ((Ptr CTHIntTensor) -> CLong)

-- |p_THIntTensor_cross : Pointer to function r_ a b dimension -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_cross"
  p_THIntTensor_cross :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_cmax : Pointer to function r t src -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_cmax"
  p_THIntTensor_cmax :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_cmin : Pointer to function r t src -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_cmin"
  p_THIntTensor_cmin :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_cmaxValue : Pointer to function r t value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_cmaxValue"
  p_THIntTensor_cmaxValue :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_cminValue : Pointer to function r t value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_cminValue"
  p_THIntTensor_cminValue :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_zeros : Pointer to function r_ size -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_zeros"
  p_THIntTensor_zeros :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHLongStorage -> IO ())

-- |p_THIntTensor_zerosLike : Pointer to function r_ input -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_zerosLike"
  p_THIntTensor_zerosLike :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_ones : Pointer to function r_ size -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_ones"
  p_THIntTensor_ones :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHLongStorage -> IO ())

-- |p_THIntTensor_onesLike : Pointer to function r_ input -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_onesLike"
  p_THIntTensor_onesLike :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_diag : Pointer to function r_ t k -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_diag"
  p_THIntTensor_diag :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_eye : Pointer to function r_ n m -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_eye"
  p_THIntTensor_eye :: FunPtr ((Ptr CTHIntTensor) -> CLong -> CLong -> IO ())

-- |p_THIntTensor_arange : Pointer to function r_ xmin xmax step -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_arange"
  p_THIntTensor_arange :: FunPtr ((Ptr CTHIntTensor) -> CLong -> CLong -> CLong -> IO ())

-- |p_THIntTensor_range : Pointer to function r_ xmin xmax step -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_range"
  p_THIntTensor_range :: FunPtr ((Ptr CTHIntTensor) -> CLong -> CLong -> CLong -> IO ())

-- |p_THIntTensor_randperm : Pointer to function r_ _generator n -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_randperm"
  p_THIntTensor_randperm :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHGenerator -> CLong -> IO ())

-- |p_THIntTensor_reshape : Pointer to function r_ t size -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_reshape"
  p_THIntTensor_reshape :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CTHLongStorage -> IO ())

-- |p_THIntTensor_sort : Pointer to function rt_ ri_ t dimension descendingOrder -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_sort"
  p_THIntTensor_sort :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ())

-- |p_THIntTensor_topk : Pointer to function rt_ ri_ t k dim dir sorted -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_topk"
  p_THIntTensor_topk :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CLong -> CInt -> CInt -> CInt -> IO ())

-- |p_THIntTensor_tril : Pointer to function r_ t k -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_tril"
  p_THIntTensor_tril :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CLong -> IO ())

-- |p_THIntTensor_triu : Pointer to function r_ t k -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_triu"
  p_THIntTensor_triu :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CLong -> IO ())

-- |p_THIntTensor_cat : Pointer to function r_ ta tb dimension -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_cat"
  p_THIntTensor_cat :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_catArray : Pointer to function result inputs numInputs dimension -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_catArray"
  p_THIntTensor_catArray :: FunPtr ((Ptr CTHIntTensor) -> Ptr (Ptr CTHIntTensor) -> CInt -> CInt -> IO ())

-- |p_THIntTensor_equal : Pointer to function ta tb -> int
foreign import ccall unsafe "THTensorMath.h &THIntTensor_equal"
  p_THIntTensor_equal :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt)

-- |p_THIntTensor_ltValue : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_ltValue"
  p_THIntTensor_ltValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_leValue : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_leValue"
  p_THIntTensor_leValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_gtValue : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_gtValue"
  p_THIntTensor_gtValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_geValue : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_geValue"
  p_THIntTensor_geValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_neValue : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_neValue"
  p_THIntTensor_neValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_eqValue : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_eqValue"
  p_THIntTensor_eqValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_ltValueT : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_ltValueT"
  p_THIntTensor_ltValueT :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_leValueT : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_leValueT"
  p_THIntTensor_leValueT :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_gtValueT : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_gtValueT"
  p_THIntTensor_gtValueT :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_geValueT : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_geValueT"
  p_THIntTensor_geValueT :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_neValueT : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_neValueT"
  p_THIntTensor_neValueT :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_eqValueT : Pointer to function r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_eqValueT"
  p_THIntTensor_eqValueT :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ())

-- |p_THIntTensor_ltTensor : Pointer to function r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_ltTensor"
  p_THIntTensor_ltTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_leTensor : Pointer to function r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_leTensor"
  p_THIntTensor_leTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_gtTensor : Pointer to function r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_gtTensor"
  p_THIntTensor_gtTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_geTensor : Pointer to function r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_geTensor"
  p_THIntTensor_geTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_neTensor : Pointer to function r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_neTensor"
  p_THIntTensor_neTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_eqTensor : Pointer to function r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_eqTensor"
  p_THIntTensor_eqTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_ltTensorT : Pointer to function r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_ltTensorT"
  p_THIntTensor_ltTensorT :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_leTensorT : Pointer to function r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_leTensorT"
  p_THIntTensor_leTensorT :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_gtTensorT : Pointer to function r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_gtTensorT"
  p_THIntTensor_gtTensorT :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_geTensorT : Pointer to function r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_geTensorT"
  p_THIntTensor_geTensorT :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_neTensorT : Pointer to function r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_neTensorT"
  p_THIntTensor_neTensorT :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_eqTensorT : Pointer to function r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_eqTensorT"
  p_THIntTensor_eqTensorT :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_abs : Pointer to function r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THIntTensor_abs"
  p_THIntTensor_abs :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())