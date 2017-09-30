{-# LANGUAGE ForeignFunctionInterface #-}

module THByteTensorMath (
    c_THByteTensor_fill,
    c_THByteTensor_zero,
    c_THByteTensor_maskedFill,
    c_THByteTensor_maskedCopy,
    c_THByteTensor_maskedSelect,
    c_THByteTensor_nonzero,
    c_THByteTensor_indexSelect,
    c_THByteTensor_indexCopy,
    c_THByteTensor_indexAdd,
    c_THByteTensor_indexFill,
    c_THByteTensor_gather,
    c_THByteTensor_scatter,
    c_THByteTensor_scatterAdd,
    c_THByteTensor_scatterFill,
    c_THByteTensor_dot,
    c_THByteTensor_minall,
    c_THByteTensor_maxall,
    c_THByteTensor_medianall,
    c_THByteTensor_sumall,
    c_THByteTensor_prodall,
    c_THByteTensor_add,
    c_THByteTensor_sub,
    c_THByteTensor_mul,
    c_THByteTensor_div,
    c_THByteTensor_lshift,
    c_THByteTensor_rshift,
    c_THByteTensor_fmod,
    c_THByteTensor_remainder,
    c_THByteTensor_clamp,
    c_THByteTensor_bitand,
    c_THByteTensor_bitor,
    c_THByteTensor_bitxor,
    c_THByteTensor_cadd,
    c_THByteTensor_csub,
    c_THByteTensor_cmul,
    c_THByteTensor_cpow,
    c_THByteTensor_cdiv,
    c_THByteTensor_clshift,
    c_THByteTensor_crshift,
    c_THByteTensor_cfmod,
    c_THByteTensor_cremainder,
    c_THByteTensor_cbitand,
    c_THByteTensor_cbitor,
    c_THByteTensor_cbitxor,
    c_THByteTensor_addcmul,
    c_THByteTensor_addcdiv,
    c_THByteTensor_addmv,
    c_THByteTensor_addmm,
    c_THByteTensor_addr,
    c_THByteTensor_addbmm,
    c_THByteTensor_baddbmm,
    c_THByteTensor_match,
    c_THByteTensor_numel,
    c_THByteTensor_max,
    c_THByteTensor_min,
    c_THByteTensor_kthvalue,
    c_THByteTensor_mode,
    c_THByteTensor_median,
    c_THByteTensor_sum,
    c_THByteTensor_prod,
    c_THByteTensor_cumsum,
    c_THByteTensor_cumprod,
    c_THByteTensor_sign,
    c_THByteTensor_trace,
    c_THByteTensor_cross,
    c_THByteTensor_cmax,
    c_THByteTensor_cmin,
    c_THByteTensor_cmaxValue,
    c_THByteTensor_cminValue,
    c_THByteTensor_zeros,
    c_THByteTensor_zerosLike,
    c_THByteTensor_ones,
    c_THByteTensor_onesLike,
    c_THByteTensor_diag,
    c_THByteTensor_eye,
    c_THByteTensor_arange,
    c_THByteTensor_range,
    c_THByteTensor_randperm,
    c_THByteTensor_reshape,
    c_THByteTensor_sort,
    c_THByteTensor_topk,
    c_THByteTensor_tril,
    c_THByteTensor_triu,
    c_THByteTensor_cat,
    c_THByteTensor_catArray,
    c_THByteTensor_equal,
    c_THByteTensor_ltValue,
    c_THByteTensor_leValue,
    c_THByteTensor_gtValue,
    c_THByteTensor_geValue,
    c_THByteTensor_neValue,
    c_THByteTensor_eqValue,
    c_THByteTensor_ltValueT,
    c_THByteTensor_leValueT,
    c_THByteTensor_gtValueT,
    c_THByteTensor_geValueT,
    c_THByteTensor_neValueT,
    c_THByteTensor_eqValueT,
    c_THByteTensor_ltTensor,
    c_THByteTensor_leTensor,
    c_THByteTensor_gtTensor,
    c_THByteTensor_geTensor,
    c_THByteTensor_neTensor,
    c_THByteTensor_eqTensor,
    c_THByteTensor_ltTensorT,
    c_THByteTensor_leTensorT,
    c_THByteTensor_gtTensorT,
    c_THByteTensor_geTensorT,
    c_THByteTensor_neTensorT,
    c_THByteTensor_eqTensorT,
    p_THByteTensor_fill,
    p_THByteTensor_zero,
    p_THByteTensor_maskedFill,
    p_THByteTensor_maskedCopy,
    p_THByteTensor_maskedSelect,
    p_THByteTensor_nonzero,
    p_THByteTensor_indexSelect,
    p_THByteTensor_indexCopy,
    p_THByteTensor_indexAdd,
    p_THByteTensor_indexFill,
    p_THByteTensor_gather,
    p_THByteTensor_scatter,
    p_THByteTensor_scatterAdd,
    p_THByteTensor_scatterFill,
    p_THByteTensor_dot,
    p_THByteTensor_minall,
    p_THByteTensor_maxall,
    p_THByteTensor_medianall,
    p_THByteTensor_sumall,
    p_THByteTensor_prodall,
    p_THByteTensor_add,
    p_THByteTensor_sub,
    p_THByteTensor_mul,
    p_THByteTensor_div,
    p_THByteTensor_lshift,
    p_THByteTensor_rshift,
    p_THByteTensor_fmod,
    p_THByteTensor_remainder,
    p_THByteTensor_clamp,
    p_THByteTensor_bitand,
    p_THByteTensor_bitor,
    p_THByteTensor_bitxor,
    p_THByteTensor_cadd,
    p_THByteTensor_csub,
    p_THByteTensor_cmul,
    p_THByteTensor_cpow,
    p_THByteTensor_cdiv,
    p_THByteTensor_clshift,
    p_THByteTensor_crshift,
    p_THByteTensor_cfmod,
    p_THByteTensor_cremainder,
    p_THByteTensor_cbitand,
    p_THByteTensor_cbitor,
    p_THByteTensor_cbitxor,
    p_THByteTensor_addcmul,
    p_THByteTensor_addcdiv,
    p_THByteTensor_addmv,
    p_THByteTensor_addmm,
    p_THByteTensor_addr,
    p_THByteTensor_addbmm,
    p_THByteTensor_baddbmm,
    p_THByteTensor_match,
    p_THByteTensor_numel,
    p_THByteTensor_max,
    p_THByteTensor_min,
    p_THByteTensor_kthvalue,
    p_THByteTensor_mode,
    p_THByteTensor_median,
    p_THByteTensor_sum,
    p_THByteTensor_prod,
    p_THByteTensor_cumsum,
    p_THByteTensor_cumprod,
    p_THByteTensor_sign,
    p_THByteTensor_trace,
    p_THByteTensor_cross,
    p_THByteTensor_cmax,
    p_THByteTensor_cmin,
    p_THByteTensor_cmaxValue,
    p_THByteTensor_cminValue,
    p_THByteTensor_zeros,
    p_THByteTensor_zerosLike,
    p_THByteTensor_ones,
    p_THByteTensor_onesLike,
    p_THByteTensor_diag,
    p_THByteTensor_eye,
    p_THByteTensor_arange,
    p_THByteTensor_range,
    p_THByteTensor_randperm,
    p_THByteTensor_reshape,
    p_THByteTensor_sort,
    p_THByteTensor_topk,
    p_THByteTensor_tril,
    p_THByteTensor_triu,
    p_THByteTensor_cat,
    p_THByteTensor_catArray,
    p_THByteTensor_equal,
    p_THByteTensor_ltValue,
    p_THByteTensor_leValue,
    p_THByteTensor_gtValue,
    p_THByteTensor_geValue,
    p_THByteTensor_neValue,
    p_THByteTensor_eqValue,
    p_THByteTensor_ltValueT,
    p_THByteTensor_leValueT,
    p_THByteTensor_gtValueT,
    p_THByteTensor_geValueT,
    p_THByteTensor_neValueT,
    p_THByteTensor_eqValueT,
    p_THByteTensor_ltTensor,
    p_THByteTensor_leTensor,
    p_THByteTensor_gtTensor,
    p_THByteTensor_geTensor,
    p_THByteTensor_neTensor,
    p_THByteTensor_eqTensor,
    p_THByteTensor_ltTensorT,
    p_THByteTensor_leTensorT,
    p_THByteTensor_gtTensorT,
    p_THByteTensor_geTensorT,
    p_THByteTensor_neTensorT,
    p_THByteTensor_eqTensorT) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THByteTensor_fill : r_ value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_fill"
  c_THByteTensor_fill :: (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_zero : r_ -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_zero"
  c_THByteTensor_zero :: (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_maskedFill : tensor mask value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_maskedFill"
  c_THByteTensor_maskedFill :: (Ptr CTHByteTensor) -> Ptr CTHByteTensor -> CChar -> IO ()

-- |c_THByteTensor_maskedCopy : tensor mask src -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_maskedCopy"
  c_THByteTensor_maskedCopy :: (Ptr CTHByteTensor) -> Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_maskedSelect : tensor src mask -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_maskedSelect"
  c_THByteTensor_maskedSelect :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THByteTensor_nonzero : subscript tensor -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_nonzero"
  c_THByteTensor_nonzero :: Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_indexSelect : tensor src dim index -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_indexSelect"
  c_THByteTensor_indexSelect :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THByteTensor_indexCopy : tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_indexCopy"
  c_THByteTensor_indexCopy :: (Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_indexAdd : tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_indexAdd"
  c_THByteTensor_indexAdd :: (Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_indexFill : tensor dim index val -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_indexFill"
  c_THByteTensor_indexFill :: (Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> CChar -> IO ()

-- |c_THByteTensor_gather : tensor src dim index -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_gather"
  c_THByteTensor_gather :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THByteTensor_scatter : tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_scatter"
  c_THByteTensor_scatter :: (Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_scatterAdd : tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_scatterAdd"
  c_THByteTensor_scatterAdd :: (Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_scatterFill : tensor dim index val -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_scatterFill"
  c_THByteTensor_scatterFill :: (Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> CChar -> IO ()

-- |c_THByteTensor_dot : t src -> accreal
foreign import ccall unsafe "THTensorMath.h THByteTensor_dot"
  c_THByteTensor_dot :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CLong

-- |c_THByteTensor_minall : t -> real
foreign import ccall unsafe "THTensorMath.h THByteTensor_minall"
  c_THByteTensor_minall :: (Ptr CTHByteTensor) -> CChar

-- |c_THByteTensor_maxall : t -> real
foreign import ccall unsafe "THTensorMath.h THByteTensor_maxall"
  c_THByteTensor_maxall :: (Ptr CTHByteTensor) -> CChar

-- |c_THByteTensor_medianall : t -> real
foreign import ccall unsafe "THTensorMath.h THByteTensor_medianall"
  c_THByteTensor_medianall :: (Ptr CTHByteTensor) -> CChar

-- |c_THByteTensor_sumall : t -> accreal
foreign import ccall unsafe "THTensorMath.h THByteTensor_sumall"
  c_THByteTensor_sumall :: (Ptr CTHByteTensor) -> CLong

-- |c_THByteTensor_prodall : t -> accreal
foreign import ccall unsafe "THTensorMath.h THByteTensor_prodall"
  c_THByteTensor_prodall :: (Ptr CTHByteTensor) -> CLong

-- |c_THByteTensor_add : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_add"
  c_THByteTensor_add :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_sub : self src value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_sub"
  c_THByteTensor_sub :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_mul : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_mul"
  c_THByteTensor_mul :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_div : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_div"
  c_THByteTensor_div :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_lshift : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_lshift"
  c_THByteTensor_lshift :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_rshift : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_rshift"
  c_THByteTensor_rshift :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_fmod : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_fmod"
  c_THByteTensor_fmod :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_remainder : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_remainder"
  c_THByteTensor_remainder :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_clamp : r_ t min_value max_value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_clamp"
  c_THByteTensor_clamp :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> CChar -> IO ()

-- |c_THByteTensor_bitand : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_bitand"
  c_THByteTensor_bitand :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_bitor : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_bitor"
  c_THByteTensor_bitor :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_bitxor : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_bitxor"
  c_THByteTensor_bitxor :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_cadd : r_ t value src -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_cadd"
  c_THByteTensor_cadd :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_csub : self src1 value src2 -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_csub"
  c_THByteTensor_csub :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_cmul : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_cmul"
  c_THByteTensor_cmul :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_cpow : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_cpow"
  c_THByteTensor_cpow :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_cdiv : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_cdiv"
  c_THByteTensor_cdiv :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_clshift : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_clshift"
  c_THByteTensor_clshift :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_crshift : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_crshift"
  c_THByteTensor_crshift :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_cfmod : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_cfmod"
  c_THByteTensor_cfmod :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_cremainder : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_cremainder"
  c_THByteTensor_cremainder :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_cbitand : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_cbitand"
  c_THByteTensor_cbitand :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_cbitor : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_cbitor"
  c_THByteTensor_cbitor :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_cbitxor : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_cbitxor"
  c_THByteTensor_cbitxor :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_addcmul : r_ t value src1 src2 -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_addcmul"
  c_THByteTensor_addcmul :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_addcdiv : r_ t value src1 src2 -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_addcdiv"
  c_THByteTensor_addcdiv :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_addmv : r_ beta t alpha mat vec -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_addmv"
  c_THByteTensor_addmv :: (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_addmm : r_ beta t alpha mat1 mat2 -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_addmm"
  c_THByteTensor_addmm :: (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_addr : r_ beta t alpha vec1 vec2 -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_addr"
  c_THByteTensor_addr :: (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_addbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_addbmm"
  c_THByteTensor_addbmm :: (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_baddbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_baddbmm"
  c_THByteTensor_baddbmm :: (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_match : r_ m1 m2 gain -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_match"
  c_THByteTensor_match :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_numel : t -> ptrdiff_t
foreign import ccall unsafe "THTensorMath.h THByteTensor_numel"
  c_THByteTensor_numel :: (Ptr CTHByteTensor) -> CPtrdiff

-- |c_THByteTensor_max : values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_max"
  c_THByteTensor_max :: (Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensor_min : values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_min"
  c_THByteTensor_min :: (Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensor_kthvalue : values_ indices_ t k dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_kthvalue"
  c_THByteTensor_kthvalue :: (Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CLong -> CInt -> CInt -> IO ()

-- |c_THByteTensor_mode : values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_mode"
  c_THByteTensor_mode :: (Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensor_median : values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_median"
  c_THByteTensor_median :: (Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensor_sum : r_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_sum"
  c_THByteTensor_sum :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensor_prod : r_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_prod"
  c_THByteTensor_prod :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensor_cumsum : r_ t dimension -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_cumsum"
  c_THByteTensor_cumsum :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> IO ()

-- |c_THByteTensor_cumprod : r_ t dimension -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_cumprod"
  c_THByteTensor_cumprod :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> IO ()

-- |c_THByteTensor_sign : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_sign"
  c_THByteTensor_sign :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_trace : t -> accreal
foreign import ccall unsafe "THTensorMath.h THByteTensor_trace"
  c_THByteTensor_trace :: (Ptr CTHByteTensor) -> CLong

-- |c_THByteTensor_cross : r_ a b dimension -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_cross"
  c_THByteTensor_cross :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> IO ()

-- |c_THByteTensor_cmax : r t src -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_cmax"
  c_THByteTensor_cmax :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_cmin : r t src -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_cmin"
  c_THByteTensor_cmin :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_cmaxValue : r t value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_cmaxValue"
  c_THByteTensor_cmaxValue :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_cminValue : r t value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_cminValue"
  c_THByteTensor_cminValue :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_zeros : r_ size -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_zeros"
  c_THByteTensor_zeros :: (Ptr CTHByteTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THByteTensor_zerosLike : r_ input -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_zerosLike"
  c_THByteTensor_zerosLike :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_ones : r_ size -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_ones"
  c_THByteTensor_ones :: (Ptr CTHByteTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THByteTensor_onesLike : r_ input -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_onesLike"
  c_THByteTensor_onesLike :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_diag : r_ t k -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_diag"
  c_THByteTensor_diag :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> IO ()

-- |c_THByteTensor_eye : r_ n m -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_eye"
  c_THByteTensor_eye :: (Ptr CTHByteTensor) -> CLong -> CLong -> IO ()

-- |c_THByteTensor_arange : r_ xmin xmax step -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_arange"
  c_THByteTensor_arange :: (Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_range : r_ xmin xmax step -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_range"
  c_THByteTensor_range :: (Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_randperm : r_ _generator n -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_randperm"
  c_THByteTensor_randperm :: (Ptr CTHByteTensor) -> Ptr CTHGenerator -> CLong -> IO ()

-- |c_THByteTensor_reshape : r_ t size -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_reshape"
  c_THByteTensor_reshape :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THByteTensor_sort : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_sort"
  c_THByteTensor_sort :: (Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensor_topk : rt_ ri_ t k dim dir sorted -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_topk"
  c_THByteTensor_topk :: (Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CLong -> CInt -> CInt -> CInt -> IO ()

-- |c_THByteTensor_tril : r_ t k -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_tril"
  c_THByteTensor_tril :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CLong -> IO ()

-- |c_THByteTensor_triu : r_ t k -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_triu"
  c_THByteTensor_triu :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CLong -> IO ()

-- |c_THByteTensor_cat : r_ ta tb dimension -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_cat"
  c_THByteTensor_cat :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> IO ()

-- |c_THByteTensor_catArray : result inputs numInputs dimension -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_catArray"
  c_THByteTensor_catArray :: (Ptr CTHByteTensor) -> Ptr (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensor_equal : ta tb -> int
foreign import ccall unsafe "THTensorMath.h THByteTensor_equal"
  c_THByteTensor_equal :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt

-- |c_THByteTensor_ltValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_ltValue"
  c_THByteTensor_ltValue :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_leValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_leValue"
  c_THByteTensor_leValue :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_gtValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_gtValue"
  c_THByteTensor_gtValue :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_geValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_geValue"
  c_THByteTensor_geValue :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_neValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_neValue"
  c_THByteTensor_neValue :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_eqValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_eqValue"
  c_THByteTensor_eqValue :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_ltValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_ltValueT"
  c_THByteTensor_ltValueT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_leValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_leValueT"
  c_THByteTensor_leValueT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_gtValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_gtValueT"
  c_THByteTensor_gtValueT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_geValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_geValueT"
  c_THByteTensor_geValueT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_neValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_neValueT"
  c_THByteTensor_neValueT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_eqValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_eqValueT"
  c_THByteTensor_eqValueT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_ltTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_ltTensor"
  c_THByteTensor_ltTensor :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_leTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_leTensor"
  c_THByteTensor_leTensor :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_gtTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_gtTensor"
  c_THByteTensor_gtTensor :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_geTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_geTensor"
  c_THByteTensor_geTensor :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_neTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_neTensor"
  c_THByteTensor_neTensor :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_eqTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_eqTensor"
  c_THByteTensor_eqTensor :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_ltTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_ltTensorT"
  c_THByteTensor_ltTensorT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_leTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_leTensorT"
  c_THByteTensor_leTensorT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_gtTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_gtTensorT"
  c_THByteTensor_gtTensorT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_geTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_geTensorT"
  c_THByteTensor_geTensorT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_neTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_neTensorT"
  c_THByteTensor_neTensorT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_eqTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THByteTensor_eqTensorT"
  c_THByteTensor_eqTensorT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |p_THByteTensor_fill : Pointer to r_ value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_fill"
  p_THByteTensor_fill :: FunPtr ((Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_zero : Pointer to r_ -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_zero"
  p_THByteTensor_zero :: FunPtr ((Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_maskedFill : Pointer to tensor mask value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_maskedFill"
  p_THByteTensor_maskedFill :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHByteTensor -> CChar -> IO ())

-- |p_THByteTensor_maskedCopy : Pointer to tensor mask src -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_maskedCopy"
  p_THByteTensor_maskedCopy :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_maskedSelect : Pointer to tensor src mask -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_maskedSelect"
  p_THByteTensor_maskedSelect :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CTHByteTensor -> IO ())

-- |p_THByteTensor_nonzero : Pointer to subscript tensor -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_nonzero"
  p_THByteTensor_nonzero :: FunPtr (Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_indexSelect : Pointer to tensor src dim index -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_indexSelect"
  p_THByteTensor_indexSelect :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> IO ())

-- |p_THByteTensor_indexCopy : Pointer to tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_indexCopy"
  p_THByteTensor_indexCopy :: FunPtr ((Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_indexAdd : Pointer to tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_indexAdd"
  p_THByteTensor_indexAdd :: FunPtr ((Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_indexFill : Pointer to tensor dim index val -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_indexFill"
  p_THByteTensor_indexFill :: FunPtr ((Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> CChar -> IO ())

-- |p_THByteTensor_gather : Pointer to tensor src dim index -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_gather"
  p_THByteTensor_gather :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> IO ())

-- |p_THByteTensor_scatter : Pointer to tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_scatter"
  p_THByteTensor_scatter :: FunPtr ((Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_scatterAdd : Pointer to tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_scatterAdd"
  p_THByteTensor_scatterAdd :: FunPtr ((Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_scatterFill : Pointer to tensor dim index val -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_scatterFill"
  p_THByteTensor_scatterFill :: FunPtr ((Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> CChar -> IO ())

-- |p_THByteTensor_dot : Pointer to t src -> accreal
foreign import ccall unsafe "THTensorMath.h &THByteTensor_dot"
  p_THByteTensor_dot :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CLong)

-- |p_THByteTensor_minall : Pointer to t -> real
foreign import ccall unsafe "THTensorMath.h &THByteTensor_minall"
  p_THByteTensor_minall :: FunPtr ((Ptr CTHByteTensor) -> CChar)

-- |p_THByteTensor_maxall : Pointer to t -> real
foreign import ccall unsafe "THTensorMath.h &THByteTensor_maxall"
  p_THByteTensor_maxall :: FunPtr ((Ptr CTHByteTensor) -> CChar)

-- |p_THByteTensor_medianall : Pointer to t -> real
foreign import ccall unsafe "THTensorMath.h &THByteTensor_medianall"
  p_THByteTensor_medianall :: FunPtr ((Ptr CTHByteTensor) -> CChar)

-- |p_THByteTensor_sumall : Pointer to t -> accreal
foreign import ccall unsafe "THTensorMath.h &THByteTensor_sumall"
  p_THByteTensor_sumall :: FunPtr ((Ptr CTHByteTensor) -> CLong)

-- |p_THByteTensor_prodall : Pointer to t -> accreal
foreign import ccall unsafe "THTensorMath.h &THByteTensor_prodall"
  p_THByteTensor_prodall :: FunPtr ((Ptr CTHByteTensor) -> CLong)

-- |p_THByteTensor_add : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_add"
  p_THByteTensor_add :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_sub : Pointer to self src value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_sub"
  p_THByteTensor_sub :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_mul : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_mul"
  p_THByteTensor_mul :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_div : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_div"
  p_THByteTensor_div :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_lshift : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_lshift"
  p_THByteTensor_lshift :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_rshift : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_rshift"
  p_THByteTensor_rshift :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_fmod : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_fmod"
  p_THByteTensor_fmod :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_remainder : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_remainder"
  p_THByteTensor_remainder :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_clamp : Pointer to r_ t min_value max_value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_clamp"
  p_THByteTensor_clamp :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> CChar -> IO ())

-- |p_THByteTensor_bitand : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_bitand"
  p_THByteTensor_bitand :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_bitor : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_bitor"
  p_THByteTensor_bitor :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_bitxor : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_bitxor"
  p_THByteTensor_bitxor :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_cadd : Pointer to r_ t value src -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_cadd"
  p_THByteTensor_cadd :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_csub : Pointer to self src1 value src2 -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_csub"
  p_THByteTensor_csub :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_cmul : Pointer to r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_cmul"
  p_THByteTensor_cmul :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_cpow : Pointer to r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_cpow"
  p_THByteTensor_cpow :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_cdiv : Pointer to r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_cdiv"
  p_THByteTensor_cdiv :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_clshift : Pointer to r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_clshift"
  p_THByteTensor_clshift :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_crshift : Pointer to r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_crshift"
  p_THByteTensor_crshift :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_cfmod : Pointer to r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_cfmod"
  p_THByteTensor_cfmod :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_cremainder : Pointer to r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_cremainder"
  p_THByteTensor_cremainder :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_cbitand : Pointer to r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_cbitand"
  p_THByteTensor_cbitand :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_cbitor : Pointer to r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_cbitor"
  p_THByteTensor_cbitor :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_cbitxor : Pointer to r_ t src -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_cbitxor"
  p_THByteTensor_cbitxor :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_addcmul : Pointer to r_ t value src1 src2 -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_addcmul"
  p_THByteTensor_addcmul :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_addcdiv : Pointer to r_ t value src1 src2 -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_addcdiv"
  p_THByteTensor_addcdiv :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_addmv : Pointer to r_ beta t alpha mat vec -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_addmv"
  p_THByteTensor_addmv :: FunPtr ((Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_addmm : Pointer to r_ beta t alpha mat1 mat2 -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_addmm"
  p_THByteTensor_addmm :: FunPtr ((Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_addr : Pointer to r_ beta t alpha vec1 vec2 -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_addr"
  p_THByteTensor_addr :: FunPtr ((Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_addbmm : Pointer to r_ beta t alpha batch1 batch2 -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_addbmm"
  p_THByteTensor_addbmm :: FunPtr ((Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_baddbmm : Pointer to r_ beta t alpha batch1 batch2 -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_baddbmm"
  p_THByteTensor_baddbmm :: FunPtr ((Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_match : Pointer to r_ m1 m2 gain -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_match"
  p_THByteTensor_match :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_numel : Pointer to t -> ptrdiff_t
foreign import ccall unsafe "THTensorMath.h &THByteTensor_numel"
  p_THByteTensor_numel :: FunPtr ((Ptr CTHByteTensor) -> CPtrdiff)

-- |p_THByteTensor_max : Pointer to values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_max"
  p_THByteTensor_max :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ())

-- |p_THByteTensor_min : Pointer to values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_min"
  p_THByteTensor_min :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ())

-- |p_THByteTensor_kthvalue : Pointer to values_ indices_ t k dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_kthvalue"
  p_THByteTensor_kthvalue :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CLong -> CInt -> CInt -> IO ())

-- |p_THByteTensor_mode : Pointer to values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_mode"
  p_THByteTensor_mode :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ())

-- |p_THByteTensor_median : Pointer to values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_median"
  p_THByteTensor_median :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ())

-- |p_THByteTensor_sum : Pointer to r_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_sum"
  p_THByteTensor_sum :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ())

-- |p_THByteTensor_prod : Pointer to r_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_prod"
  p_THByteTensor_prod :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ())

-- |p_THByteTensor_cumsum : Pointer to r_ t dimension -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_cumsum"
  p_THByteTensor_cumsum :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> IO ())

-- |p_THByteTensor_cumprod : Pointer to r_ t dimension -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_cumprod"
  p_THByteTensor_cumprod :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> IO ())

-- |p_THByteTensor_sign : Pointer to r_ t -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_sign"
  p_THByteTensor_sign :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_trace : Pointer to t -> accreal
foreign import ccall unsafe "THTensorMath.h &THByteTensor_trace"
  p_THByteTensor_trace :: FunPtr ((Ptr CTHByteTensor) -> CLong)

-- |p_THByteTensor_cross : Pointer to r_ a b dimension -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_cross"
  p_THByteTensor_cross :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> IO ())

-- |p_THByteTensor_cmax : Pointer to r t src -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_cmax"
  p_THByteTensor_cmax :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_cmin : Pointer to r t src -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_cmin"
  p_THByteTensor_cmin :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_cmaxValue : Pointer to r t value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_cmaxValue"
  p_THByteTensor_cmaxValue :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_cminValue : Pointer to r t value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_cminValue"
  p_THByteTensor_cminValue :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_zeros : Pointer to r_ size -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_zeros"
  p_THByteTensor_zeros :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHLongStorage -> IO ())

-- |p_THByteTensor_zerosLike : Pointer to r_ input -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_zerosLike"
  p_THByteTensor_zerosLike :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_ones : Pointer to r_ size -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_ones"
  p_THByteTensor_ones :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHLongStorage -> IO ())

-- |p_THByteTensor_onesLike : Pointer to r_ input -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_onesLike"
  p_THByteTensor_onesLike :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_diag : Pointer to r_ t k -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_diag"
  p_THByteTensor_diag :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> IO ())

-- |p_THByteTensor_eye : Pointer to r_ n m -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_eye"
  p_THByteTensor_eye :: FunPtr ((Ptr CTHByteTensor) -> CLong -> CLong -> IO ())

-- |p_THByteTensor_arange : Pointer to r_ xmin xmax step -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_arange"
  p_THByteTensor_arange :: FunPtr ((Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> IO ())

-- |p_THByteTensor_range : Pointer to r_ xmin xmax step -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_range"
  p_THByteTensor_range :: FunPtr ((Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> IO ())

-- |p_THByteTensor_randperm : Pointer to r_ _generator n -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_randperm"
  p_THByteTensor_randperm :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHGenerator -> CLong -> IO ())

-- |p_THByteTensor_reshape : Pointer to r_ t size -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_reshape"
  p_THByteTensor_reshape :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CTHLongStorage -> IO ())

-- |p_THByteTensor_sort : Pointer to rt_ ri_ t dimension descendingOrder -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_sort"
  p_THByteTensor_sort :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ())

-- |p_THByteTensor_topk : Pointer to rt_ ri_ t k dim dir sorted -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_topk"
  p_THByteTensor_topk :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CLong -> CInt -> CInt -> CInt -> IO ())

-- |p_THByteTensor_tril : Pointer to r_ t k -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_tril"
  p_THByteTensor_tril :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CLong -> IO ())

-- |p_THByteTensor_triu : Pointer to r_ t k -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_triu"
  p_THByteTensor_triu :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CLong -> IO ())

-- |p_THByteTensor_cat : Pointer to r_ ta tb dimension -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_cat"
  p_THByteTensor_cat :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> IO ())

-- |p_THByteTensor_catArray : Pointer to result inputs numInputs dimension -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_catArray"
  p_THByteTensor_catArray :: FunPtr ((Ptr CTHByteTensor) -> Ptr (Ptr CTHByteTensor) -> CInt -> CInt -> IO ())

-- |p_THByteTensor_equal : Pointer to ta tb -> int
foreign import ccall unsafe "THTensorMath.h &THByteTensor_equal"
  p_THByteTensor_equal :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt)

-- |p_THByteTensor_ltValue : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_ltValue"
  p_THByteTensor_ltValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_leValue : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_leValue"
  p_THByteTensor_leValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_gtValue : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_gtValue"
  p_THByteTensor_gtValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_geValue : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_geValue"
  p_THByteTensor_geValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_neValue : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_neValue"
  p_THByteTensor_neValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_eqValue : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_eqValue"
  p_THByteTensor_eqValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_ltValueT : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_ltValueT"
  p_THByteTensor_ltValueT :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_leValueT : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_leValueT"
  p_THByteTensor_leValueT :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_gtValueT : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_gtValueT"
  p_THByteTensor_gtValueT :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_geValueT : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_geValueT"
  p_THByteTensor_geValueT :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_neValueT : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_neValueT"
  p_THByteTensor_neValueT :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_eqValueT : Pointer to r_ t value -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_eqValueT"
  p_THByteTensor_eqValueT :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ())

-- |p_THByteTensor_ltTensor : Pointer to r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_ltTensor"
  p_THByteTensor_ltTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_leTensor : Pointer to r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_leTensor"
  p_THByteTensor_leTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_gtTensor : Pointer to r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_gtTensor"
  p_THByteTensor_gtTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_geTensor : Pointer to r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_geTensor"
  p_THByteTensor_geTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_neTensor : Pointer to r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_neTensor"
  p_THByteTensor_neTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_eqTensor : Pointer to r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_eqTensor"
  p_THByteTensor_eqTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_ltTensorT : Pointer to r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_ltTensorT"
  p_THByteTensor_ltTensorT :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_leTensorT : Pointer to r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_leTensorT"
  p_THByteTensor_leTensorT :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_gtTensorT : Pointer to r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_gtTensorT"
  p_THByteTensor_gtTensorT :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_geTensorT : Pointer to r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_geTensorT"
  p_THByteTensor_geTensorT :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_neTensorT : Pointer to r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_neTensorT"
  p_THByteTensor_neTensorT :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_eqTensorT : Pointer to r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h &THByteTensor_eqTensorT"
  p_THByteTensor_eqTensorT :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())