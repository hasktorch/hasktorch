{-# LANGUAGE ForeignFunctionInterface #-}

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
    p_THLongTensor_fill,
    p_THLongTensor_zero,
    p_THLongTensor_maskedFill,
    p_THLongTensor_maskedCopy,
    p_THLongTensor_maskedSelect,
    p_THLongTensor_nonzero,
    p_THLongTensor_indexSelect,
    p_THLongTensor_indexCopy,
    p_THLongTensor_indexAdd,
    p_THLongTensor_indexFill,
    p_THLongTensor_gather,
    p_THLongTensor_scatter,
    p_THLongTensor_scatterAdd,
    p_THLongTensor_scatterFill,
    p_THLongTensor_dot,
    p_THLongTensor_minall,
    p_THLongTensor_maxall,
    p_THLongTensor_medianall,
    p_THLongTensor_sumall,
    p_THLongTensor_prodall,
    p_THLongTensor_neg,
    p_THLongTensor_add,
    p_THLongTensor_sub,
    p_THLongTensor_mul,
    p_THLongTensor_div,
    p_THLongTensor_lshift,
    p_THLongTensor_rshift,
    p_THLongTensor_fmod,
    p_THLongTensor_remainder,
    p_THLongTensor_clamp,
    p_THLongTensor_bitand,
    p_THLongTensor_bitor,
    p_THLongTensor_bitxor,
    p_THLongTensor_cadd,
    p_THLongTensor_csub,
    p_THLongTensor_cmul,
    p_THLongTensor_cpow,
    p_THLongTensor_cdiv,
    p_THLongTensor_clshift,
    p_THLongTensor_crshift,
    p_THLongTensor_cfmod,
    p_THLongTensor_cremainder,
    p_THLongTensor_cbitand,
    p_THLongTensor_cbitor,
    p_THLongTensor_cbitxor,
    p_THLongTensor_addcmul,
    p_THLongTensor_addcdiv,
    p_THLongTensor_addmv,
    p_THLongTensor_addmm,
    p_THLongTensor_addr,
    p_THLongTensor_addbmm,
    p_THLongTensor_baddbmm,
    p_THLongTensor_match,
    p_THLongTensor_numel,
    p_THLongTensor_max,
    p_THLongTensor_min,
    p_THLongTensor_kthvalue,
    p_THLongTensor_mode,
    p_THLongTensor_median,
    p_THLongTensor_sum,
    p_THLongTensor_prod,
    p_THLongTensor_cumsum,
    p_THLongTensor_cumprod,
    p_THLongTensor_sign,
    p_THLongTensor_trace,
    p_THLongTensor_cross,
    p_THLongTensor_cmax,
    p_THLongTensor_cmin,
    p_THLongTensor_cmaxValue,
    p_THLongTensor_cminValue,
    p_THLongTensor_zeros,
    p_THLongTensor_zerosLike,
    p_THLongTensor_ones,
    p_THLongTensor_onesLike,
    p_THLongTensor_diag,
    p_THLongTensor_eye,
    p_THLongTensor_arange,
    p_THLongTensor_range,
    p_THLongTensor_randperm,
    p_THLongTensor_reshape,
    p_THLongTensor_sort,
    p_THLongTensor_topk,
    p_THLongTensor_tril,
    p_THLongTensor_triu,
    p_THLongTensor_cat,
    p_THLongTensor_catArray,
    p_THLongTensor_equal,
    p_THLongTensor_ltValue,
    p_THLongTensor_leValue,
    p_THLongTensor_gtValue,
    p_THLongTensor_geValue,
    p_THLongTensor_neValue,
    p_THLongTensor_eqValue,
    p_THLongTensor_ltValueT,
    p_THLongTensor_leValueT,
    p_THLongTensor_gtValueT,
    p_THLongTensor_geValueT,
    p_THLongTensor_neValueT,
    p_THLongTensor_eqValueT,
    p_THLongTensor_ltTensor,
    p_THLongTensor_leTensor,
    p_THLongTensor_gtTensor,
    p_THLongTensor_geTensor,
    p_THLongTensor_neTensor,
    p_THLongTensor_eqTensor,
    p_THLongTensor_ltTensorT,
    p_THLongTensor_leTensorT,
    p_THLongTensor_gtTensorT,
    p_THLongTensor_geTensorT,
    p_THLongTensor_neTensorT,
    p_THLongTensor_eqTensorT,
    p_THLongTensor_abs) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THLongTensor_fill : r_ value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_fill"
  c_THLongTensor_fill :: (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_zero : r_ -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_zero"
  c_THLongTensor_zero :: (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_maskedFill : tensor mask value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_maskedFill"
  c_THLongTensor_maskedFill :: (Ptr CTHLongTensor) -> Ptr CTHByteTensor -> CLong -> IO ()

-- |c_THLongTensor_maskedCopy : tensor mask src -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_maskedCopy"
  c_THLongTensor_maskedCopy :: (Ptr CTHLongTensor) -> Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_maskedSelect : tensor src mask -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_maskedSelect"
  c_THLongTensor_maskedSelect :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THLongTensor_nonzero : subscript tensor -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_nonzero"
  c_THLongTensor_nonzero :: Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_indexSelect : tensor src dim index -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_indexSelect"
  c_THLongTensor_indexSelect :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THLongTensor_indexCopy : tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_indexCopy"
  c_THLongTensor_indexCopy :: (Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_indexAdd : tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_indexAdd"
  c_THLongTensor_indexAdd :: (Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_indexFill : tensor dim index val -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_indexFill"
  c_THLongTensor_indexFill :: (Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> CLong -> IO ()

-- |c_THLongTensor_gather : tensor src dim index -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_gather"
  c_THLongTensor_gather :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THLongTensor_scatter : tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_scatter"
  c_THLongTensor_scatter :: (Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_scatterAdd : tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_scatterAdd"
  c_THLongTensor_scatterAdd :: (Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_scatterFill : tensor dim index val -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_scatterFill"
  c_THLongTensor_scatterFill :: (Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> CLong -> IO ()

-- |c_THLongTensor_dot : t src -> accreal
foreign import ccall unsafe "THTensorMath.h THLongTensor_dot"
  c_THLongTensor_dot :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong

-- |c_THLongTensor_minall : t -> real
foreign import ccall unsafe "THTensorMath.h THLongTensor_minall"
  c_THLongTensor_minall :: (Ptr CTHLongTensor) -> CLong

-- |c_THLongTensor_maxall : t -> real
foreign import ccall unsafe "THTensorMath.h THLongTensor_maxall"
  c_THLongTensor_maxall :: (Ptr CTHLongTensor) -> CLong

-- |c_THLongTensor_medianall : t -> real
foreign import ccall unsafe "THTensorMath.h THLongTensor_medianall"
  c_THLongTensor_medianall :: (Ptr CTHLongTensor) -> CLong

-- |c_THLongTensor_sumall : t -> accreal
foreign import ccall unsafe "THTensorMath.h THLongTensor_sumall"
  c_THLongTensor_sumall :: (Ptr CTHLongTensor) -> CLong

-- |c_THLongTensor_prodall : t -> accreal
foreign import ccall unsafe "THTensorMath.h THLongTensor_prodall"
  c_THLongTensor_prodall :: (Ptr CTHLongTensor) -> CLong

-- |c_THLongTensor_neg : self src -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_neg"
  c_THLongTensor_neg :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_add : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_add"
  c_THLongTensor_add :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_sub : self src value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_sub"
  c_THLongTensor_sub :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_mul : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_mul"
  c_THLongTensor_mul :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_div : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_div"
  c_THLongTensor_div :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_lshift : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_lshift"
  c_THLongTensor_lshift :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_rshift : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_rshift"
  c_THLongTensor_rshift :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_fmod : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_fmod"
  c_THLongTensor_fmod :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_remainder : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_remainder"
  c_THLongTensor_remainder :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_clamp : r_ t min_value max_value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_clamp"
  c_THLongTensor_clamp :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> IO ()

-- |c_THLongTensor_bitand : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_bitand"
  c_THLongTensor_bitand :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_bitor : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_bitor"
  c_THLongTensor_bitor :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_bitxor : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_bitxor"
  c_THLongTensor_bitxor :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_cadd : r_ t value src -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_cadd"
  c_THLongTensor_cadd :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_csub : self src1 value src2 -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_csub"
  c_THLongTensor_csub :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_cmul : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_cmul"
  c_THLongTensor_cmul :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_cpow : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_cpow"
  c_THLongTensor_cpow :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_cdiv : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_cdiv"
  c_THLongTensor_cdiv :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_clshift : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_clshift"
  c_THLongTensor_clshift :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_crshift : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_crshift"
  c_THLongTensor_crshift :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_cfmod : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_cfmod"
  c_THLongTensor_cfmod :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_cremainder : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_cremainder"
  c_THLongTensor_cremainder :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_cbitand : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_cbitand"
  c_THLongTensor_cbitand :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_cbitor : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_cbitor"
  c_THLongTensor_cbitor :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_cbitxor : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_cbitxor"
  c_THLongTensor_cbitxor :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_addcmul : r_ t value src1 src2 -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_addcmul"
  c_THLongTensor_addcmul :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_addcdiv : r_ t value src1 src2 -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_addcdiv"
  c_THLongTensor_addcdiv :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_addmv : r_ beta t alpha mat vec -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_addmv"
  c_THLongTensor_addmv :: (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_addmm : r_ beta t alpha mat1 mat2 -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_addmm"
  c_THLongTensor_addmm :: (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_addr : r_ beta t alpha vec1 vec2 -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_addr"
  c_THLongTensor_addr :: (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_addbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_addbmm"
  c_THLongTensor_addbmm :: (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_baddbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_baddbmm"
  c_THLongTensor_baddbmm :: (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_match : r_ m1 m2 gain -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_match"
  c_THLongTensor_match :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_numel : t -> ptrdiff_t
foreign import ccall unsafe "THTensorMath.h THLongTensor_numel"
  c_THLongTensor_numel :: (Ptr CTHLongTensor) -> CPtrdiff

-- |c_THLongTensor_max : values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_max"
  c_THLongTensor_max :: (Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensor_min : values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_min"
  c_THLongTensor_min :: (Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensor_kthvalue : values_ indices_ t k dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_kthvalue"
  c_THLongTensor_kthvalue :: (Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CLong -> CInt -> CInt -> IO ()

-- |c_THLongTensor_mode : values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_mode"
  c_THLongTensor_mode :: (Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensor_median : values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_median"
  c_THLongTensor_median :: (Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensor_sum : r_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_sum"
  c_THLongTensor_sum :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensor_prod : r_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_prod"
  c_THLongTensor_prod :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensor_cumsum : r_ t dimension -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_cumsum"
  c_THLongTensor_cumsum :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> IO ()

-- |c_THLongTensor_cumprod : r_ t dimension -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_cumprod"
  c_THLongTensor_cumprod :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> IO ()

-- |c_THLongTensor_sign : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_sign"
  c_THLongTensor_sign :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_trace : t -> accreal
foreign import ccall unsafe "THTensorMath.h THLongTensor_trace"
  c_THLongTensor_trace :: (Ptr CTHLongTensor) -> CLong

-- |c_THLongTensor_cross : r_ a b dimension -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_cross"
  c_THLongTensor_cross :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> IO ()

-- |c_THLongTensor_cmax : r t src -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_cmax"
  c_THLongTensor_cmax :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_cmin : r t src -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_cmin"
  c_THLongTensor_cmin :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_cmaxValue : r t value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_cmaxValue"
  c_THLongTensor_cmaxValue :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_cminValue : r t value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_cminValue"
  c_THLongTensor_cminValue :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_zeros : r_ size -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_zeros"
  c_THLongTensor_zeros :: (Ptr CTHLongTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THLongTensor_zerosLike : r_ input -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_zerosLike"
  c_THLongTensor_zerosLike :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_ones : r_ size -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_ones"
  c_THLongTensor_ones :: (Ptr CTHLongTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THLongTensor_onesLike : r_ input -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_onesLike"
  c_THLongTensor_onesLike :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_diag : r_ t k -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_diag"
  c_THLongTensor_diag :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> IO ()

-- |c_THLongTensor_eye : r_ n m -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_eye"
  c_THLongTensor_eye :: (Ptr CTHLongTensor) -> CLong -> CLong -> IO ()

-- |c_THLongTensor_arange : r_ xmin xmax step -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_arange"
  c_THLongTensor_arange :: (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_range : r_ xmin xmax step -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_range"
  c_THLongTensor_range :: (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_randperm : r_ _generator n -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_randperm"
  c_THLongTensor_randperm :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> CLong -> IO ()

-- |c_THLongTensor_reshape : r_ t size -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_reshape"
  c_THLongTensor_reshape :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THLongTensor_sort : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_sort"
  c_THLongTensor_sort :: (Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensor_topk : rt_ ri_ t k dim dir sorted -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_topk"
  c_THLongTensor_topk :: (Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CLong -> CInt -> CInt -> CInt -> IO ()

-- |c_THLongTensor_tril : r_ t k -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_tril"
  c_THLongTensor_tril :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_triu : r_ t k -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_triu"
  c_THLongTensor_triu :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_cat : r_ ta tb dimension -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_cat"
  c_THLongTensor_cat :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> IO ()

-- |c_THLongTensor_catArray : result inputs numInputs dimension -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_catArray"
  c_THLongTensor_catArray :: (Ptr CTHLongTensor) -> Ptr (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensor_equal : ta tb -> int
foreign import ccall unsafe "THTensorMath.h THLongTensor_equal"
  c_THLongTensor_equal :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt

-- |c_THLongTensor_ltValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_ltValue"
  c_THLongTensor_ltValue :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_leValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_leValue"
  c_THLongTensor_leValue :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_gtValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_gtValue"
  c_THLongTensor_gtValue :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_geValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_geValue"
  c_THLongTensor_geValue :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_neValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_neValue"
  c_THLongTensor_neValue :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_eqValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_eqValue"
  c_THLongTensor_eqValue :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_ltValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_ltValueT"
  c_THLongTensor_ltValueT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_leValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_leValueT"
  c_THLongTensor_leValueT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_gtValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_gtValueT"
  c_THLongTensor_gtValueT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_geValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_geValueT"
  c_THLongTensor_geValueT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_neValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_neValueT"
  c_THLongTensor_neValueT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_eqValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_eqValueT"
  c_THLongTensor_eqValueT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ()

-- |c_THLongTensor_ltTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_ltTensor"
  c_THLongTensor_ltTensor :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_leTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_leTensor"
  c_THLongTensor_leTensor :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_gtTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_gtTensor"
  c_THLongTensor_gtTensor :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_geTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_geTensor"
  c_THLongTensor_geTensor :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_neTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_neTensor"
  c_THLongTensor_neTensor :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_eqTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_eqTensor"
  c_THLongTensor_eqTensor :: Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_ltTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_ltTensorT"
  c_THLongTensor_ltTensorT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_leTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_leTensorT"
  c_THLongTensor_leTensorT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_gtTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_gtTensorT"
  c_THLongTensor_gtTensorT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_geTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_geTensorT"
  c_THLongTensor_geTensorT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_neTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_neTensorT"
  c_THLongTensor_neTensorT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_eqTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_eqTensorT"
  c_THLongTensor_eqTensorT :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_abs : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THLongTensor_abs"
  c_THLongTensor_abs :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |p_THLongTensor_fill : Pointer to r_ value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_fill"
  p_THLongTensor_fill :: FunPtr ((Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_zero : Pointer to r_ -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_zero"
  p_THLongTensor_zero :: FunPtr ((Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_maskedFill : Pointer to tensor mask value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_maskedFill"
  p_THLongTensor_maskedFill :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHByteTensor -> CLong -> IO ())

-- |p_THLongTensor_maskedCopy : Pointer to tensor mask src -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_maskedCopy"
  p_THLongTensor_maskedCopy :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_maskedSelect : Pointer to tensor src mask -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_maskedSelect"
  p_THLongTensor_maskedSelect :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CTHByteTensor -> IO ())

-- |p_THLongTensor_nonzero : Pointer to subscript tensor -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_nonzero"
  p_THLongTensor_nonzero :: FunPtr (Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_indexSelect : Pointer to tensor src dim index -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_indexSelect"
  p_THLongTensor_indexSelect :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> IO ())

-- |p_THLongTensor_indexCopy : Pointer to tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_indexCopy"
  p_THLongTensor_indexCopy :: FunPtr ((Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_indexAdd : Pointer to tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_indexAdd"
  p_THLongTensor_indexAdd :: FunPtr ((Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_indexFill : Pointer to tensor dim index val -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_indexFill"
  p_THLongTensor_indexFill :: FunPtr ((Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> CLong -> IO ())

-- |p_THLongTensor_gather : Pointer to tensor src dim index -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_gather"
  p_THLongTensor_gather :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> IO ())

-- |p_THLongTensor_scatter : Pointer to tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_scatter"
  p_THLongTensor_scatter :: FunPtr ((Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_scatterAdd : Pointer to tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_scatterAdd"
  p_THLongTensor_scatterAdd :: FunPtr ((Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_scatterFill : Pointer to tensor dim index val -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_scatterFill"
  p_THLongTensor_scatterFill :: FunPtr ((Ptr CTHLongTensor) -> CInt -> Ptr CTHLongTensor -> CLong -> IO ())

-- |p_THLongTensor_dot : Pointer to t src -> accreal
foreign import ccall unsafe "THTensorMath.h &THLongTensor_dot"
  p_THLongTensor_dot :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong)

-- |p_THLongTensor_minall : Pointer to t -> real
foreign import ccall unsafe "THTensorMath.h &THLongTensor_minall"
  p_THLongTensor_minall :: FunPtr ((Ptr CTHLongTensor) -> CLong)

-- |p_THLongTensor_maxall : Pointer to t -> real
foreign import ccall unsafe "THTensorMath.h &THLongTensor_maxall"
  p_THLongTensor_maxall :: FunPtr ((Ptr CTHLongTensor) -> CLong)

-- |p_THLongTensor_medianall : Pointer to t -> real
foreign import ccall unsafe "THTensorMath.h &THLongTensor_medianall"
  p_THLongTensor_medianall :: FunPtr ((Ptr CTHLongTensor) -> CLong)

-- |p_THLongTensor_sumall : Pointer to t -> accreal
foreign import ccall unsafe "THTensorMath.h &THLongTensor_sumall"
  p_THLongTensor_sumall :: FunPtr ((Ptr CTHLongTensor) -> CLong)

-- |p_THLongTensor_prodall : Pointer to t -> accreal
foreign import ccall unsafe "THTensorMath.h &THLongTensor_prodall"
  p_THLongTensor_prodall :: FunPtr ((Ptr CTHLongTensor) -> CLong)

-- |p_THLongTensor_neg : Pointer to self src -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_neg"
  p_THLongTensor_neg :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_add : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_add"
  p_THLongTensor_add :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_sub : Pointer to self src value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_sub"
  p_THLongTensor_sub :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_mul : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_mul"
  p_THLongTensor_mul :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_div : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_div"
  p_THLongTensor_div :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_lshift : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_lshift"
  p_THLongTensor_lshift :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_rshift : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_rshift"
  p_THLongTensor_rshift :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_fmod : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_fmod"
  p_THLongTensor_fmod :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_remainder : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_remainder"
  p_THLongTensor_remainder :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_clamp : Pointer to r_ t min_value max_value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_clamp"
  p_THLongTensor_clamp :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> IO ())

-- |p_THLongTensor_bitand : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_bitand"
  p_THLongTensor_bitand :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_bitor : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_bitor"
  p_THLongTensor_bitor :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_bitxor : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_bitxor"
  p_THLongTensor_bitxor :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_cadd : Pointer to r_ t value src -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_cadd"
  p_THLongTensor_cadd :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_csub : Pointer to self src1 value src2 -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_csub"
  p_THLongTensor_csub :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_cmul : Pointer to r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_cmul"
  p_THLongTensor_cmul :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_cpow : Pointer to r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_cpow"
  p_THLongTensor_cpow :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_cdiv : Pointer to r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_cdiv"
  p_THLongTensor_cdiv :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_clshift : Pointer to r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_clshift"
  p_THLongTensor_clshift :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_crshift : Pointer to r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_crshift"
  p_THLongTensor_crshift :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_cfmod : Pointer to r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_cfmod"
  p_THLongTensor_cfmod :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_cremainder : Pointer to r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_cremainder"
  p_THLongTensor_cremainder :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_cbitand : Pointer to r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_cbitand"
  p_THLongTensor_cbitand :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_cbitor : Pointer to r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_cbitor"
  p_THLongTensor_cbitor :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_cbitxor : Pointer to r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_cbitxor"
  p_THLongTensor_cbitxor :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_addcmul : Pointer to r_ t value src1 src2 -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_addcmul"
  p_THLongTensor_addcmul :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_addcdiv : Pointer to r_ t value src1 src2 -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_addcdiv"
  p_THLongTensor_addcdiv :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_addmv : Pointer to r_ beta t alpha mat vec -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_addmv"
  p_THLongTensor_addmv :: FunPtr ((Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_addmm : Pointer to r_ beta t alpha mat1 mat2 -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_addmm"
  p_THLongTensor_addmm :: FunPtr ((Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_addr : Pointer to r_ beta t alpha vec1 vec2 -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_addr"
  p_THLongTensor_addr :: FunPtr ((Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_addbmm : Pointer to r_ beta t alpha batch1 batch2 -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_addbmm"
  p_THLongTensor_addbmm :: FunPtr ((Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_baddbmm : Pointer to r_ beta t alpha batch1 batch2 -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_baddbmm"
  p_THLongTensor_baddbmm :: FunPtr ((Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_match : Pointer to r_ m1 m2 gain -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_match"
  p_THLongTensor_match :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_numel : Pointer to t -> ptrdiff_t
foreign import ccall unsafe "THTensorMath.h &THLongTensor_numel"
  p_THLongTensor_numel :: FunPtr ((Ptr CTHLongTensor) -> CPtrdiff)

-- |p_THLongTensor_max : Pointer to values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_max"
  p_THLongTensor_max :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ())

-- |p_THLongTensor_min : Pointer to values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_min"
  p_THLongTensor_min :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ())

-- |p_THLongTensor_kthvalue : Pointer to values_ indices_ t k dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_kthvalue"
  p_THLongTensor_kthvalue :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CLong -> CInt -> CInt -> IO ())

-- |p_THLongTensor_mode : Pointer to values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_mode"
  p_THLongTensor_mode :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ())

-- |p_THLongTensor_median : Pointer to values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_median"
  p_THLongTensor_median :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ())

-- |p_THLongTensor_sum : Pointer to r_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_sum"
  p_THLongTensor_sum :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ())

-- |p_THLongTensor_prod : Pointer to r_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_prod"
  p_THLongTensor_prod :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ())

-- |p_THLongTensor_cumsum : Pointer to r_ t dimension -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_cumsum"
  p_THLongTensor_cumsum :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> IO ())

-- |p_THLongTensor_cumprod : Pointer to r_ t dimension -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_cumprod"
  p_THLongTensor_cumprod :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> IO ())

-- |p_THLongTensor_sign : Pointer to r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_sign"
  p_THLongTensor_sign :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_trace : Pointer to t -> accreal
foreign import ccall unsafe "THTensorMath.h &THLongTensor_trace"
  p_THLongTensor_trace :: FunPtr ((Ptr CTHLongTensor) -> CLong)

-- |p_THLongTensor_cross : Pointer to r_ a b dimension -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_cross"
  p_THLongTensor_cross :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> IO ())

-- |p_THLongTensor_cmax : Pointer to r t src -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_cmax"
  p_THLongTensor_cmax :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_cmin : Pointer to r t src -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_cmin"
  p_THLongTensor_cmin :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_cmaxValue : Pointer to r t value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_cmaxValue"
  p_THLongTensor_cmaxValue :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_cminValue : Pointer to r t value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_cminValue"
  p_THLongTensor_cminValue :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_zeros : Pointer to r_ size -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_zeros"
  p_THLongTensor_zeros :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHLongStorage -> IO ())

-- |p_THLongTensor_zerosLike : Pointer to r_ input -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_zerosLike"
  p_THLongTensor_zerosLike :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_ones : Pointer to r_ size -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_ones"
  p_THLongTensor_ones :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHLongStorage -> IO ())

-- |p_THLongTensor_onesLike : Pointer to r_ input -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_onesLike"
  p_THLongTensor_onesLike :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_diag : Pointer to r_ t k -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_diag"
  p_THLongTensor_diag :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> IO ())

-- |p_THLongTensor_eye : Pointer to r_ n m -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_eye"
  p_THLongTensor_eye :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong -> IO ())

-- |p_THLongTensor_arange : Pointer to r_ xmin xmax step -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_arange"
  p_THLongTensor_arange :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> IO ())

-- |p_THLongTensor_range : Pointer to r_ xmin xmax step -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_range"
  p_THLongTensor_range :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> IO ())

-- |p_THLongTensor_randperm : Pointer to r_ _generator n -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_randperm"
  p_THLongTensor_randperm :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHGenerator -> CLong -> IO ())

-- |p_THLongTensor_reshape : Pointer to r_ t size -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_reshape"
  p_THLongTensor_reshape :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CTHLongStorage -> IO ())

-- |p_THLongTensor_sort : Pointer to rt_ ri_ t dimension descendingOrder -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_sort"
  p_THLongTensor_sort :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ())

-- |p_THLongTensor_topk : Pointer to rt_ ri_ t k dim dir sorted -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_topk"
  p_THLongTensor_topk :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> CLong -> CInt -> CInt -> CInt -> IO ())

-- |p_THLongTensor_tril : Pointer to r_ t k -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_tril"
  p_THLongTensor_tril :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_triu : Pointer to r_ t k -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_triu"
  p_THLongTensor_triu :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_cat : Pointer to r_ ta tb dimension -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_cat"
  p_THLongTensor_cat :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt -> IO ())

-- |p_THLongTensor_catArray : Pointer to result inputs numInputs dimension -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_catArray"
  p_THLongTensor_catArray :: FunPtr ((Ptr CTHLongTensor) -> Ptr (Ptr CTHLongTensor) -> CInt -> CInt -> IO ())

-- |p_THLongTensor_equal : Pointer to ta tb -> int
foreign import ccall unsafe "THTensorMath.h &THLongTensor_equal"
  p_THLongTensor_equal :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CInt)

-- |p_THLongTensor_ltValue : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_ltValue"
  p_THLongTensor_ltValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_leValue : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_leValue"
  p_THLongTensor_leValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_gtValue : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_gtValue"
  p_THLongTensor_gtValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_geValue : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_geValue"
  p_THLongTensor_geValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_neValue : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_neValue"
  p_THLongTensor_neValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_eqValue : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_eqValue"
  p_THLongTensor_eqValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_ltValueT : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_ltValueT"
  p_THLongTensor_ltValueT :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_leValueT : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_leValueT"
  p_THLongTensor_leValueT :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_gtValueT : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_gtValueT"
  p_THLongTensor_gtValueT :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_geValueT : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_geValueT"
  p_THLongTensor_geValueT :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_neValueT : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_neValueT"
  p_THLongTensor_neValueT :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_eqValueT : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_eqValueT"
  p_THLongTensor_eqValueT :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> IO ())

-- |p_THLongTensor_ltTensor : Pointer to r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_ltTensor"
  p_THLongTensor_ltTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_leTensor : Pointer to r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_leTensor"
  p_THLongTensor_leTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_gtTensor : Pointer to r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_gtTensor"
  p_THLongTensor_gtTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_geTensor : Pointer to r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_geTensor"
  p_THLongTensor_geTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_neTensor : Pointer to r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_neTensor"
  p_THLongTensor_neTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_eqTensor : Pointer to r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_eqTensor"
  p_THLongTensor_eqTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_ltTensorT : Pointer to r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_ltTensorT"
  p_THLongTensor_ltTensorT :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_leTensorT : Pointer to r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_leTensorT"
  p_THLongTensor_leTensorT :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_gtTensorT : Pointer to r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_gtTensorT"
  p_THLongTensor_gtTensorT :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_geTensorT : Pointer to r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_geTensorT"
  p_THLongTensor_geTensorT :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_neTensorT : Pointer to r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_neTensorT"
  p_THLongTensor_neTensorT :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_eqTensorT : Pointer to r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_eqTensorT"
  p_THLongTensor_eqTensorT :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_abs : Pointer to r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THLongTensor_abs"
  p_THLongTensor_abs :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())