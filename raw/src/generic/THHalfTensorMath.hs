{-# LANGUAGE ForeignFunctionInterface #-}

module THHalfTensorMath (
    c_THHalfTensor_fill,
    c_THHalfTensor_zero,
    c_THHalfTensor_maskedFill,
    c_THHalfTensor_maskedCopy,
    c_THHalfTensor_maskedSelect,
    c_THHalfTensor_nonzero,
    c_THHalfTensor_indexSelect,
    c_THHalfTensor_indexCopy,
    c_THHalfTensor_indexAdd,
    c_THHalfTensor_indexFill,
    c_THHalfTensor_take,
    c_THHalfTensor_put,
    c_THHalfTensor_gather,
    c_THHalfTensor_scatter,
    c_THHalfTensor_scatterAdd,
    c_THHalfTensor_scatterFill,
    c_THHalfTensor_dot,
    c_THHalfTensor_minall,
    c_THHalfTensor_maxall,
    c_THHalfTensor_medianall,
    c_THHalfTensor_sumall,
    c_THHalfTensor_prodall,
    c_THHalfTensor_add,
    c_THHalfTensor_sub,
    c_THHalfTensor_add_scaled,
    c_THHalfTensor_sub_scaled,
    c_THHalfTensor_mul,
    c_THHalfTensor_div,
    c_THHalfTensor_lshift,
    c_THHalfTensor_rshift,
    c_THHalfTensor_fmod,
    c_THHalfTensor_remainder,
    c_THHalfTensor_clamp,
    c_THHalfTensor_bitand,
    c_THHalfTensor_bitor,
    c_THHalfTensor_bitxor,
    c_THHalfTensor_cadd,
    c_THHalfTensor_csub,
    c_THHalfTensor_cmul,
    c_THHalfTensor_cpow,
    c_THHalfTensor_cdiv,
    c_THHalfTensor_clshift,
    c_THHalfTensor_crshift,
    c_THHalfTensor_cfmod,
    c_THHalfTensor_cremainder,
    c_THHalfTensor_cbitand,
    c_THHalfTensor_cbitor,
    c_THHalfTensor_cbitxor,
    c_THHalfTensor_addcmul,
    c_THHalfTensor_addcdiv,
    c_THHalfTensor_addmv,
    c_THHalfTensor_addmm,
    c_THHalfTensor_addr,
    c_THHalfTensor_addbmm,
    c_THHalfTensor_baddbmm,
    c_THHalfTensor_match,
    c_THHalfTensor_numel,
    c_THHalfTensor_max,
    c_THHalfTensor_min,
    c_THHalfTensor_kthvalue,
    c_THHalfTensor_mode,
    c_THHalfTensor_median,
    c_THHalfTensor_sum,
    c_THHalfTensor_prod,
    c_THHalfTensor_cumsum,
    c_THHalfTensor_cumprod,
    c_THHalfTensor_sign,
    c_THHalfTensor_trace,
    c_THHalfTensor_cross,
    c_THHalfTensor_cmax,
    c_THHalfTensor_cmin,
    c_THHalfTensor_cmaxValue,
    c_THHalfTensor_cminValue,
    c_THHalfTensor_zeros,
    c_THHalfTensor_zerosLike,
    c_THHalfTensor_ones,
    c_THHalfTensor_onesLike,
    c_THHalfTensor_diag,
    c_THHalfTensor_eye,
    c_THHalfTensor_arange,
    c_THHalfTensor_range,
    c_THHalfTensor_randperm,
    c_THHalfTensor_reshape,
    c_THHalfTensor_sort,
    c_THHalfTensor_topk,
    c_THHalfTensor_tril,
    c_THHalfTensor_triu,
    c_THHalfTensor_cat,
    c_THHalfTensor_catArray,
    c_THHalfTensor_equal,
    c_THHalfTensor_ltValue,
    c_THHalfTensor_leValue,
    c_THHalfTensor_gtValue,
    c_THHalfTensor_geValue,
    c_THHalfTensor_neValue,
    c_THHalfTensor_eqValue,
    c_THHalfTensor_ltValueT,
    c_THHalfTensor_leValueT,
    c_THHalfTensor_gtValueT,
    c_THHalfTensor_geValueT,
    c_THHalfTensor_neValueT,
    c_THHalfTensor_eqValueT,
    c_THHalfTensor_ltTensor,
    c_THHalfTensor_leTensor,
    c_THHalfTensor_gtTensor,
    c_THHalfTensor_geTensor,
    c_THHalfTensor_neTensor,
    c_THHalfTensor_eqTensor,
    c_THHalfTensor_ltTensorT,
    c_THHalfTensor_leTensorT,
    c_THHalfTensor_gtTensorT,
    c_THHalfTensor_geTensorT,
    c_THHalfTensor_neTensorT,
    c_THHalfTensor_eqTensorT,
    p_THHalfTensor_fill,
    p_THHalfTensor_zero,
    p_THHalfTensor_maskedFill,
    p_THHalfTensor_maskedCopy,
    p_THHalfTensor_maskedSelect,
    p_THHalfTensor_nonzero,
    p_THHalfTensor_indexSelect,
    p_THHalfTensor_indexCopy,
    p_THHalfTensor_indexAdd,
    p_THHalfTensor_indexFill,
    p_THHalfTensor_take,
    p_THHalfTensor_put,
    p_THHalfTensor_gather,
    p_THHalfTensor_scatter,
    p_THHalfTensor_scatterAdd,
    p_THHalfTensor_scatterFill,
    p_THHalfTensor_dot,
    p_THHalfTensor_minall,
    p_THHalfTensor_maxall,
    p_THHalfTensor_medianall,
    p_THHalfTensor_sumall,
    p_THHalfTensor_prodall,
    p_THHalfTensor_add,
    p_THHalfTensor_sub,
    p_THHalfTensor_add_scaled,
    p_THHalfTensor_sub_scaled,
    p_THHalfTensor_mul,
    p_THHalfTensor_div,
    p_THHalfTensor_lshift,
    p_THHalfTensor_rshift,
    p_THHalfTensor_fmod,
    p_THHalfTensor_remainder,
    p_THHalfTensor_clamp,
    p_THHalfTensor_bitand,
    p_THHalfTensor_bitor,
    p_THHalfTensor_bitxor,
    p_THHalfTensor_cadd,
    p_THHalfTensor_csub,
    p_THHalfTensor_cmul,
    p_THHalfTensor_cpow,
    p_THHalfTensor_cdiv,
    p_THHalfTensor_clshift,
    p_THHalfTensor_crshift,
    p_THHalfTensor_cfmod,
    p_THHalfTensor_cremainder,
    p_THHalfTensor_cbitand,
    p_THHalfTensor_cbitor,
    p_THHalfTensor_cbitxor,
    p_THHalfTensor_addcmul,
    p_THHalfTensor_addcdiv,
    p_THHalfTensor_addmv,
    p_THHalfTensor_addmm,
    p_THHalfTensor_addr,
    p_THHalfTensor_addbmm,
    p_THHalfTensor_baddbmm,
    p_THHalfTensor_match,
    p_THHalfTensor_numel,
    p_THHalfTensor_max,
    p_THHalfTensor_min,
    p_THHalfTensor_kthvalue,
    p_THHalfTensor_mode,
    p_THHalfTensor_median,
    p_THHalfTensor_sum,
    p_THHalfTensor_prod,
    p_THHalfTensor_cumsum,
    p_THHalfTensor_cumprod,
    p_THHalfTensor_sign,
    p_THHalfTensor_trace,
    p_THHalfTensor_cross,
    p_THHalfTensor_cmax,
    p_THHalfTensor_cmin,
    p_THHalfTensor_cmaxValue,
    p_THHalfTensor_cminValue,
    p_THHalfTensor_zeros,
    p_THHalfTensor_zerosLike,
    p_THHalfTensor_ones,
    p_THHalfTensor_onesLike,
    p_THHalfTensor_diag,
    p_THHalfTensor_eye,
    p_THHalfTensor_arange,
    p_THHalfTensor_range,
    p_THHalfTensor_randperm,
    p_THHalfTensor_reshape,
    p_THHalfTensor_sort,
    p_THHalfTensor_topk,
    p_THHalfTensor_tril,
    p_THHalfTensor_triu,
    p_THHalfTensor_cat,
    p_THHalfTensor_catArray,
    p_THHalfTensor_equal,
    p_THHalfTensor_ltValue,
    p_THHalfTensor_leValue,
    p_THHalfTensor_gtValue,
    p_THHalfTensor_geValue,
    p_THHalfTensor_neValue,
    p_THHalfTensor_eqValue,
    p_THHalfTensor_ltValueT,
    p_THHalfTensor_leValueT,
    p_THHalfTensor_gtValueT,
    p_THHalfTensor_geValueT,
    p_THHalfTensor_neValueT,
    p_THHalfTensor_eqValueT,
    p_THHalfTensor_ltTensor,
    p_THHalfTensor_leTensor,
    p_THHalfTensor_gtTensor,
    p_THHalfTensor_geTensor,
    p_THHalfTensor_neTensor,
    p_THHalfTensor_eqTensor,
    p_THHalfTensor_ltTensorT,
    p_THHalfTensor_leTensorT,
    p_THHalfTensor_gtTensorT,
    p_THHalfTensor_geTensorT,
    p_THHalfTensor_neTensorT,
    p_THHalfTensor_eqTensorT) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THHalfTensor_fill : r_ value -> void
foreign import ccall "THTensorMath.h THHalfTensor_fill"
  c_THHalfTensor_fill :: (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_zero : r_ -> void
foreign import ccall "THTensorMath.h THHalfTensor_zero"
  c_THHalfTensor_zero :: (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_maskedFill : tensor mask value -> void
foreign import ccall "THTensorMath.h THHalfTensor_maskedFill"
  c_THHalfTensor_maskedFill :: (Ptr CTHHalfTensor) -> Ptr CTHByteTensor -> THHalf -> IO ()

-- |c_THHalfTensor_maskedCopy : tensor mask src -> void
foreign import ccall "THTensorMath.h THHalfTensor_maskedCopy"
  c_THHalfTensor_maskedCopy :: (Ptr CTHHalfTensor) -> Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_maskedSelect : tensor src mask -> void
foreign import ccall "THTensorMath.h THHalfTensor_maskedSelect"
  c_THHalfTensor_maskedSelect :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THHalfTensor_nonzero : subscript tensor -> void
foreign import ccall "THTensorMath.h THHalfTensor_nonzero"
  c_THHalfTensor_nonzero :: Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_indexSelect : tensor src dim index -> void
foreign import ccall "THTensorMath.h THHalfTensor_indexSelect"
  c_THHalfTensor_indexSelect :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THHalfTensor_indexCopy : tensor dim index src -> void
foreign import ccall "THTensorMath.h THHalfTensor_indexCopy"
  c_THHalfTensor_indexCopy :: (Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_indexAdd : tensor dim index src -> void
foreign import ccall "THTensorMath.h THHalfTensor_indexAdd"
  c_THHalfTensor_indexAdd :: (Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_indexFill : tensor dim index val -> void
foreign import ccall "THTensorMath.h THHalfTensor_indexFill"
  c_THHalfTensor_indexFill :: (Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> THHalf -> IO ()

-- |c_THHalfTensor_take : tensor src index -> void
foreign import ccall "THTensorMath.h THHalfTensor_take"
  c_THHalfTensor_take :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> IO ()

-- |c_THHalfTensor_put : tensor index src accumulate -> void
foreign import ccall "THTensorMath.h THHalfTensor_put"
  c_THHalfTensor_put :: (Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CInt -> IO ()

-- |c_THHalfTensor_gather : tensor src dim index -> void
foreign import ccall "THTensorMath.h THHalfTensor_gather"
  c_THHalfTensor_gather :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THHalfTensor_scatter : tensor dim index src -> void
foreign import ccall "THTensorMath.h THHalfTensor_scatter"
  c_THHalfTensor_scatter :: (Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_scatterAdd : tensor dim index src -> void
foreign import ccall "THTensorMath.h THHalfTensor_scatterAdd"
  c_THHalfTensor_scatterAdd :: (Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_scatterFill : tensor dim index val -> void
foreign import ccall "THTensorMath.h THHalfTensor_scatterFill"
  c_THHalfTensor_scatterFill :: (Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> THHalf -> IO ()

-- |c_THHalfTensor_dot : t src -> accreal
foreign import ccall "THTensorMath.h THHalfTensor_dot"
  c_THHalfTensor_dot :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CFloat

-- |c_THHalfTensor_minall : t -> real
foreign import ccall "THTensorMath.h THHalfTensor_minall"
  c_THHalfTensor_minall :: (Ptr CTHHalfTensor) -> THHalf

-- |c_THHalfTensor_maxall : t -> real
foreign import ccall "THTensorMath.h THHalfTensor_maxall"
  c_THHalfTensor_maxall :: (Ptr CTHHalfTensor) -> THHalf

-- |c_THHalfTensor_medianall : t -> real
foreign import ccall "THTensorMath.h THHalfTensor_medianall"
  c_THHalfTensor_medianall :: (Ptr CTHHalfTensor) -> THHalf

-- |c_THHalfTensor_sumall : t -> accreal
foreign import ccall "THTensorMath.h THHalfTensor_sumall"
  c_THHalfTensor_sumall :: (Ptr CTHHalfTensor) -> CFloat

-- |c_THHalfTensor_prodall : t -> accreal
foreign import ccall "THTensorMath.h THHalfTensor_prodall"
  c_THHalfTensor_prodall :: (Ptr CTHHalfTensor) -> CFloat

-- |c_THHalfTensor_add : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_add"
  c_THHalfTensor_add :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_sub : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_sub"
  c_THHalfTensor_sub :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_add_scaled : r_ t value alpha -> void
foreign import ccall "THTensorMath.h THHalfTensor_add_scaled"
  c_THHalfTensor_add_scaled :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> THHalf -> IO ()

-- |c_THHalfTensor_sub_scaled : r_ t value alpha -> void
foreign import ccall "THTensorMath.h THHalfTensor_sub_scaled"
  c_THHalfTensor_sub_scaled :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> THHalf -> IO ()

-- |c_THHalfTensor_mul : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_mul"
  c_THHalfTensor_mul :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_div : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_div"
  c_THHalfTensor_div :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_lshift : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_lshift"
  c_THHalfTensor_lshift :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_rshift : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_rshift"
  c_THHalfTensor_rshift :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_fmod : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_fmod"
  c_THHalfTensor_fmod :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_remainder : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_remainder"
  c_THHalfTensor_remainder :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_clamp : r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h THHalfTensor_clamp"
  c_THHalfTensor_clamp :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> THHalf -> IO ()

-- |c_THHalfTensor_bitand : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_bitand"
  c_THHalfTensor_bitand :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_bitor : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_bitor"
  c_THHalfTensor_bitor :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_bitxor : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_bitxor"
  c_THHalfTensor_bitxor :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_cadd : r_ t value src -> void
foreign import ccall "THTensorMath.h THHalfTensor_cadd"
  c_THHalfTensor_cadd :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_csub : self src1 value src2 -> void
foreign import ccall "THTensorMath.h THHalfTensor_csub"
  c_THHalfTensor_csub :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_cmul : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensor_cmul"
  c_THHalfTensor_cmul :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_cpow : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensor_cpow"
  c_THHalfTensor_cpow :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_cdiv : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensor_cdiv"
  c_THHalfTensor_cdiv :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_clshift : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensor_clshift"
  c_THHalfTensor_clshift :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_crshift : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensor_crshift"
  c_THHalfTensor_crshift :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_cfmod : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensor_cfmod"
  c_THHalfTensor_cfmod :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_cremainder : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensor_cremainder"
  c_THHalfTensor_cremainder :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_cbitand : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensor_cbitand"
  c_THHalfTensor_cbitand :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_cbitor : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensor_cbitor"
  c_THHalfTensor_cbitor :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_cbitxor : r_ t src -> void
foreign import ccall "THTensorMath.h THHalfTensor_cbitxor"
  c_THHalfTensor_cbitxor :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_addcmul : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THHalfTensor_addcmul"
  c_THHalfTensor_addcmul :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_addcdiv : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THHalfTensor_addcdiv"
  c_THHalfTensor_addcdiv :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_addmv : r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h THHalfTensor_addmv"
  c_THHalfTensor_addmv :: (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_addmm : r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h THHalfTensor_addmm"
  c_THHalfTensor_addmm :: (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_addr : r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h THHalfTensor_addr"
  c_THHalfTensor_addr :: (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_addbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THHalfTensor_addbmm"
  c_THHalfTensor_addbmm :: (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_baddbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THHalfTensor_baddbmm"
  c_THHalfTensor_baddbmm :: (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_match : r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h THHalfTensor_match"
  c_THHalfTensor_match :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_numel : t -> ptrdiff_t
foreign import ccall "THTensorMath.h THHalfTensor_numel"
  c_THHalfTensor_numel :: (Ptr CTHHalfTensor) -> CPtrdiff

-- |c_THHalfTensor_max : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensor_max"
  c_THHalfTensor_max :: (Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_min : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensor_min"
  c_THHalfTensor_min :: (Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_kthvalue : values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensor_kthvalue"
  c_THHalfTensor_kthvalue :: (Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CLLong -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_mode : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensor_mode"
  c_THHalfTensor_mode :: (Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_median : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensor_median"
  c_THHalfTensor_median :: (Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_sum : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensor_sum"
  c_THHalfTensor_sum :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_prod : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THHalfTensor_prod"
  c_THHalfTensor_prod :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_cumsum : r_ t dimension -> void
foreign import ccall "THTensorMath.h THHalfTensor_cumsum"
  c_THHalfTensor_cumsum :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> IO ()

-- |c_THHalfTensor_cumprod : r_ t dimension -> void
foreign import ccall "THTensorMath.h THHalfTensor_cumprod"
  c_THHalfTensor_cumprod :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> IO ()

-- |c_THHalfTensor_sign : r_ t -> void
foreign import ccall "THTensorMath.h THHalfTensor_sign"
  c_THHalfTensor_sign :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_trace : t -> accreal
foreign import ccall "THTensorMath.h THHalfTensor_trace"
  c_THHalfTensor_trace :: (Ptr CTHHalfTensor) -> CFloat

-- |c_THHalfTensor_cross : r_ a b dimension -> void
foreign import ccall "THTensorMath.h THHalfTensor_cross"
  c_THHalfTensor_cross :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> IO ()

-- |c_THHalfTensor_cmax : r t src -> void
foreign import ccall "THTensorMath.h THHalfTensor_cmax"
  c_THHalfTensor_cmax :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_cmin : r t src -> void
foreign import ccall "THTensorMath.h THHalfTensor_cmin"
  c_THHalfTensor_cmin :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_cmaxValue : r t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_cmaxValue"
  c_THHalfTensor_cmaxValue :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_cminValue : r t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_cminValue"
  c_THHalfTensor_cminValue :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_zeros : r_ size -> void
foreign import ccall "THTensorMath.h THHalfTensor_zeros"
  c_THHalfTensor_zeros :: (Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THHalfTensor_zerosLike : r_ input -> void
foreign import ccall "THTensorMath.h THHalfTensor_zerosLike"
  c_THHalfTensor_zerosLike :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_ones : r_ size -> void
foreign import ccall "THTensorMath.h THHalfTensor_ones"
  c_THHalfTensor_ones :: (Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THHalfTensor_onesLike : r_ input -> void
foreign import ccall "THTensorMath.h THHalfTensor_onesLike"
  c_THHalfTensor_onesLike :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_diag : r_ t k -> void
foreign import ccall "THTensorMath.h THHalfTensor_diag"
  c_THHalfTensor_diag :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> IO ()

-- |c_THHalfTensor_eye : r_ n m -> void
foreign import ccall "THTensorMath.h THHalfTensor_eye"
  c_THHalfTensor_eye :: (Ptr CTHHalfTensor) -> CLLong -> CLLong -> IO ()

-- |c_THHalfTensor_arange : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THHalfTensor_arange"
  c_THHalfTensor_arange :: (Ptr CTHHalfTensor) -> CFloat -> CFloat -> CFloat -> IO ()

-- |c_THHalfTensor_range : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THHalfTensor_range"
  c_THHalfTensor_range :: (Ptr CTHHalfTensor) -> CFloat -> CFloat -> CFloat -> IO ()

-- |c_THHalfTensor_randperm : r_ _generator n -> void
foreign import ccall "THTensorMath.h THHalfTensor_randperm"
  c_THHalfTensor_randperm :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> CLLong -> IO ()

-- |c_THHalfTensor_reshape : r_ t size -> void
foreign import ccall "THTensorMath.h THHalfTensor_reshape"
  c_THHalfTensor_reshape :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THHalfTensor_sort : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h THHalfTensor_sort"
  c_THHalfTensor_sort :: (Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_topk : rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h THHalfTensor_topk"
  c_THHalfTensor_topk :: (Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CLLong -> CInt -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_tril : r_ t k -> void
foreign import ccall "THTensorMath.h THHalfTensor_tril"
  c_THHalfTensor_tril :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLLong -> IO ()

-- |c_THHalfTensor_triu : r_ t k -> void
foreign import ccall "THTensorMath.h THHalfTensor_triu"
  c_THHalfTensor_triu :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLLong -> IO ()

-- |c_THHalfTensor_cat : r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h THHalfTensor_cat"
  c_THHalfTensor_cat :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> IO ()

-- |c_THHalfTensor_catArray : result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h THHalfTensor_catArray"
  c_THHalfTensor_catArray :: (Ptr CTHHalfTensor) -> Ptr (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_equal : ta tb -> int
foreign import ccall "THTensorMath.h THHalfTensor_equal"
  c_THHalfTensor_equal :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt

-- |c_THHalfTensor_ltValue : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_ltValue"
  c_THHalfTensor_ltValue :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_leValue : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_leValue"
  c_THHalfTensor_leValue :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_gtValue : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_gtValue"
  c_THHalfTensor_gtValue :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_geValue : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_geValue"
  c_THHalfTensor_geValue :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_neValue : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_neValue"
  c_THHalfTensor_neValue :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_eqValue : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_eqValue"
  c_THHalfTensor_eqValue :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_ltValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_ltValueT"
  c_THHalfTensor_ltValueT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_leValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_leValueT"
  c_THHalfTensor_leValueT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_gtValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_gtValueT"
  c_THHalfTensor_gtValueT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_geValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_geValueT"
  c_THHalfTensor_geValueT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_neValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_neValueT"
  c_THHalfTensor_neValueT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_eqValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THHalfTensor_eqValueT"
  c_THHalfTensor_eqValueT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ()

-- |c_THHalfTensor_ltTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensor_ltTensor"
  c_THHalfTensor_ltTensor :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_leTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensor_leTensor"
  c_THHalfTensor_leTensor :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_gtTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensor_gtTensor"
  c_THHalfTensor_gtTensor :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_geTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensor_geTensor"
  c_THHalfTensor_geTensor :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_neTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensor_neTensor"
  c_THHalfTensor_neTensor :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_eqTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensor_eqTensor"
  c_THHalfTensor_eqTensor :: Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_ltTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensor_ltTensorT"
  c_THHalfTensor_ltTensorT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_leTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensor_leTensorT"
  c_THHalfTensor_leTensorT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_gtTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensor_gtTensorT"
  c_THHalfTensor_gtTensorT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_geTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensor_geTensorT"
  c_THHalfTensor_geTensorT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_neTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensor_neTensorT"
  c_THHalfTensor_neTensorT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_eqTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THHalfTensor_eqTensorT"
  c_THHalfTensor_eqTensorT :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |p_THHalfTensor_fill : Pointer to function : r_ value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_fill"
  p_THHalfTensor_fill :: FunPtr ((Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_zero : Pointer to function : r_ -> void
foreign import ccall "THTensorMath.h &THHalfTensor_zero"
  p_THHalfTensor_zero :: FunPtr ((Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_maskedFill : Pointer to function : tensor mask value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_maskedFill"
  p_THHalfTensor_maskedFill :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHByteTensor -> THHalf -> IO ())

-- |p_THHalfTensor_maskedCopy : Pointer to function : tensor mask src -> void
foreign import ccall "THTensorMath.h &THHalfTensor_maskedCopy"
  p_THHalfTensor_maskedCopy :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_maskedSelect : Pointer to function : tensor src mask -> void
foreign import ccall "THTensorMath.h &THHalfTensor_maskedSelect"
  p_THHalfTensor_maskedSelect :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CTHByteTensor -> IO ())

-- |p_THHalfTensor_nonzero : Pointer to function : subscript tensor -> void
foreign import ccall "THTensorMath.h &THHalfTensor_nonzero"
  p_THHalfTensor_nonzero :: FunPtr (Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_indexSelect : Pointer to function : tensor src dim index -> void
foreign import ccall "THTensorMath.h &THHalfTensor_indexSelect"
  p_THHalfTensor_indexSelect :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> IO ())

-- |p_THHalfTensor_indexCopy : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THHalfTensor_indexCopy"
  p_THHalfTensor_indexCopy :: FunPtr ((Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_indexAdd : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THHalfTensor_indexAdd"
  p_THHalfTensor_indexAdd :: FunPtr ((Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_indexFill : Pointer to function : tensor dim index val -> void
foreign import ccall "THTensorMath.h &THHalfTensor_indexFill"
  p_THHalfTensor_indexFill :: FunPtr ((Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> THHalf -> IO ())

-- |p_THHalfTensor_take : Pointer to function : tensor src index -> void
foreign import ccall "THTensorMath.h &THHalfTensor_take"
  p_THHalfTensor_take :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> IO ())

-- |p_THHalfTensor_put : Pointer to function : tensor index src accumulate -> void
foreign import ccall "THTensorMath.h &THHalfTensor_put"
  p_THHalfTensor_put :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CInt -> IO ())

-- |p_THHalfTensor_gather : Pointer to function : tensor src dim index -> void
foreign import ccall "THTensorMath.h &THHalfTensor_gather"
  p_THHalfTensor_gather :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> IO ())

-- |p_THHalfTensor_scatter : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THHalfTensor_scatter"
  p_THHalfTensor_scatter :: FunPtr ((Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_scatterAdd : Pointer to function : tensor dim index src -> void
foreign import ccall "THTensorMath.h &THHalfTensor_scatterAdd"
  p_THHalfTensor_scatterAdd :: FunPtr ((Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_scatterFill : Pointer to function : tensor dim index val -> void
foreign import ccall "THTensorMath.h &THHalfTensor_scatterFill"
  p_THHalfTensor_scatterFill :: FunPtr ((Ptr CTHHalfTensor) -> CInt -> Ptr CTHLongTensor -> THHalf -> IO ())

-- |p_THHalfTensor_dot : Pointer to function : t src -> accreal
foreign import ccall "THTensorMath.h &THHalfTensor_dot"
  p_THHalfTensor_dot :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CFloat)

-- |p_THHalfTensor_minall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THHalfTensor_minall"
  p_THHalfTensor_minall :: FunPtr ((Ptr CTHHalfTensor) -> THHalf)

-- |p_THHalfTensor_maxall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THHalfTensor_maxall"
  p_THHalfTensor_maxall :: FunPtr ((Ptr CTHHalfTensor) -> THHalf)

-- |p_THHalfTensor_medianall : Pointer to function : t -> real
foreign import ccall "THTensorMath.h &THHalfTensor_medianall"
  p_THHalfTensor_medianall :: FunPtr ((Ptr CTHHalfTensor) -> THHalf)

-- |p_THHalfTensor_sumall : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THHalfTensor_sumall"
  p_THHalfTensor_sumall :: FunPtr ((Ptr CTHHalfTensor) -> CFloat)

-- |p_THHalfTensor_prodall : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THHalfTensor_prodall"
  p_THHalfTensor_prodall :: FunPtr ((Ptr CTHHalfTensor) -> CFloat)

-- |p_THHalfTensor_add : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_add"
  p_THHalfTensor_add :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_sub : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_sub"
  p_THHalfTensor_sub :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_add_scaled : Pointer to function : r_ t value alpha -> void
foreign import ccall "THTensorMath.h &THHalfTensor_add_scaled"
  p_THHalfTensor_add_scaled :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> THHalf -> IO ())

-- |p_THHalfTensor_sub_scaled : Pointer to function : r_ t value alpha -> void
foreign import ccall "THTensorMath.h &THHalfTensor_sub_scaled"
  p_THHalfTensor_sub_scaled :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> THHalf -> IO ())

-- |p_THHalfTensor_mul : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_mul"
  p_THHalfTensor_mul :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_div : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_div"
  p_THHalfTensor_div :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_lshift : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_lshift"
  p_THHalfTensor_lshift :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_rshift : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_rshift"
  p_THHalfTensor_rshift :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_fmod : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_fmod"
  p_THHalfTensor_fmod :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_remainder : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_remainder"
  p_THHalfTensor_remainder :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_clamp : Pointer to function : r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_clamp"
  p_THHalfTensor_clamp :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> THHalf -> IO ())

-- |p_THHalfTensor_bitand : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_bitand"
  p_THHalfTensor_bitand :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_bitor : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_bitor"
  p_THHalfTensor_bitor :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_bitxor : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_bitxor"
  p_THHalfTensor_bitxor :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_cadd : Pointer to function : r_ t value src -> void
foreign import ccall "THTensorMath.h &THHalfTensor_cadd"
  p_THHalfTensor_cadd :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_csub : Pointer to function : self src1 value src2 -> void
foreign import ccall "THTensorMath.h &THHalfTensor_csub"
  p_THHalfTensor_csub :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_cmul : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THHalfTensor_cmul"
  p_THHalfTensor_cmul :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_cpow : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THHalfTensor_cpow"
  p_THHalfTensor_cpow :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_cdiv : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THHalfTensor_cdiv"
  p_THHalfTensor_cdiv :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_clshift : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THHalfTensor_clshift"
  p_THHalfTensor_clshift :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_crshift : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THHalfTensor_crshift"
  p_THHalfTensor_crshift :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_cfmod : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THHalfTensor_cfmod"
  p_THHalfTensor_cfmod :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_cremainder : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THHalfTensor_cremainder"
  p_THHalfTensor_cremainder :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_cbitand : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THHalfTensor_cbitand"
  p_THHalfTensor_cbitand :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_cbitor : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THHalfTensor_cbitor"
  p_THHalfTensor_cbitor :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_cbitxor : Pointer to function : r_ t src -> void
foreign import ccall "THTensorMath.h &THHalfTensor_cbitxor"
  p_THHalfTensor_cbitxor :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_addcmul : Pointer to function : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h &THHalfTensor_addcmul"
  p_THHalfTensor_addcmul :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_addcdiv : Pointer to function : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h &THHalfTensor_addcdiv"
  p_THHalfTensor_addcdiv :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_addmv : Pointer to function : r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h &THHalfTensor_addmv"
  p_THHalfTensor_addmv :: FunPtr ((Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_addmm : Pointer to function : r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h &THHalfTensor_addmm"
  p_THHalfTensor_addmm :: FunPtr ((Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_addr : Pointer to function : r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h &THHalfTensor_addr"
  p_THHalfTensor_addr :: FunPtr ((Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_addbmm : Pointer to function : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h &THHalfTensor_addbmm"
  p_THHalfTensor_addbmm :: FunPtr ((Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_baddbmm : Pointer to function : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h &THHalfTensor_baddbmm"
  p_THHalfTensor_baddbmm :: FunPtr ((Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_match : Pointer to function : r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h &THHalfTensor_match"
  p_THHalfTensor_match :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_numel : Pointer to function : t -> ptrdiff_t
foreign import ccall "THTensorMath.h &THHalfTensor_numel"
  p_THHalfTensor_numel :: FunPtr ((Ptr CTHHalfTensor) -> CPtrdiff)

-- |p_THHalfTensor_max : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THHalfTensor_max"
  p_THHalfTensor_max :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ())

-- |p_THHalfTensor_min : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THHalfTensor_min"
  p_THHalfTensor_min :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ())

-- |p_THHalfTensor_kthvalue : Pointer to function : values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h &THHalfTensor_kthvalue"
  p_THHalfTensor_kthvalue :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CLLong -> CInt -> CInt -> IO ())

-- |p_THHalfTensor_mode : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THHalfTensor_mode"
  p_THHalfTensor_mode :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ())

-- |p_THHalfTensor_median : Pointer to function : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THHalfTensor_median"
  p_THHalfTensor_median :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ())

-- |p_THHalfTensor_sum : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THHalfTensor_sum"
  p_THHalfTensor_sum :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ())

-- |p_THHalfTensor_prod : Pointer to function : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h &THHalfTensor_prod"
  p_THHalfTensor_prod :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ())

-- |p_THHalfTensor_cumsum : Pointer to function : r_ t dimension -> void
foreign import ccall "THTensorMath.h &THHalfTensor_cumsum"
  p_THHalfTensor_cumsum :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> IO ())

-- |p_THHalfTensor_cumprod : Pointer to function : r_ t dimension -> void
foreign import ccall "THTensorMath.h &THHalfTensor_cumprod"
  p_THHalfTensor_cumprod :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> IO ())

-- |p_THHalfTensor_sign : Pointer to function : r_ t -> void
foreign import ccall "THTensorMath.h &THHalfTensor_sign"
  p_THHalfTensor_sign :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_trace : Pointer to function : t -> accreal
foreign import ccall "THTensorMath.h &THHalfTensor_trace"
  p_THHalfTensor_trace :: FunPtr ((Ptr CTHHalfTensor) -> CFloat)

-- |p_THHalfTensor_cross : Pointer to function : r_ a b dimension -> void
foreign import ccall "THTensorMath.h &THHalfTensor_cross"
  p_THHalfTensor_cross :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> IO ())

-- |p_THHalfTensor_cmax : Pointer to function : r t src -> void
foreign import ccall "THTensorMath.h &THHalfTensor_cmax"
  p_THHalfTensor_cmax :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_cmin : Pointer to function : r t src -> void
foreign import ccall "THTensorMath.h &THHalfTensor_cmin"
  p_THHalfTensor_cmin :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_cmaxValue : Pointer to function : r t value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_cmaxValue"
  p_THHalfTensor_cmaxValue :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_cminValue : Pointer to function : r t value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_cminValue"
  p_THHalfTensor_cminValue :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_zeros : Pointer to function : r_ size -> void
foreign import ccall "THTensorMath.h &THHalfTensor_zeros"
  p_THHalfTensor_zeros :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> IO ())

-- |p_THHalfTensor_zerosLike : Pointer to function : r_ input -> void
foreign import ccall "THTensorMath.h &THHalfTensor_zerosLike"
  p_THHalfTensor_zerosLike :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_ones : Pointer to function : r_ size -> void
foreign import ccall "THTensorMath.h &THHalfTensor_ones"
  p_THHalfTensor_ones :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> IO ())

-- |p_THHalfTensor_onesLike : Pointer to function : r_ input -> void
foreign import ccall "THTensorMath.h &THHalfTensor_onesLike"
  p_THHalfTensor_onesLike :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_diag : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THHalfTensor_diag"
  p_THHalfTensor_diag :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> IO ())

-- |p_THHalfTensor_eye : Pointer to function : r_ n m -> void
foreign import ccall "THTensorMath.h &THHalfTensor_eye"
  p_THHalfTensor_eye :: FunPtr ((Ptr CTHHalfTensor) -> CLLong -> CLLong -> IO ())

-- |p_THHalfTensor_arange : Pointer to function : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h &THHalfTensor_arange"
  p_THHalfTensor_arange :: FunPtr ((Ptr CTHHalfTensor) -> CFloat -> CFloat -> CFloat -> IO ())

-- |p_THHalfTensor_range : Pointer to function : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h &THHalfTensor_range"
  p_THHalfTensor_range :: FunPtr ((Ptr CTHHalfTensor) -> CFloat -> CFloat -> CFloat -> IO ())

-- |p_THHalfTensor_randperm : Pointer to function : r_ _generator n -> void
foreign import ccall "THTensorMath.h &THHalfTensor_randperm"
  p_THHalfTensor_randperm :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHGenerator -> CLLong -> IO ())

-- |p_THHalfTensor_reshape : Pointer to function : r_ t size -> void
foreign import ccall "THTensorMath.h &THHalfTensor_reshape"
  p_THHalfTensor_reshape :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CTHLongStorage -> IO ())

-- |p_THHalfTensor_sort : Pointer to function : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h &THHalfTensor_sort"
  p_THHalfTensor_sort :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ())

-- |p_THHalfTensor_topk : Pointer to function : rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h &THHalfTensor_topk"
  p_THHalfTensor_topk :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> CLLong -> CInt -> CInt -> CInt -> IO ())

-- |p_THHalfTensor_tril : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THHalfTensor_tril"
  p_THHalfTensor_tril :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLLong -> IO ())

-- |p_THHalfTensor_triu : Pointer to function : r_ t k -> void
foreign import ccall "THTensorMath.h &THHalfTensor_triu"
  p_THHalfTensor_triu :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLLong -> IO ())

-- |p_THHalfTensor_cat : Pointer to function : r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h &THHalfTensor_cat"
  p_THHalfTensor_cat :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt -> IO ())

-- |p_THHalfTensor_catArray : Pointer to function : result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h &THHalfTensor_catArray"
  p_THHalfTensor_catArray :: FunPtr ((Ptr CTHHalfTensor) -> Ptr (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ())

-- |p_THHalfTensor_equal : Pointer to function : ta tb -> int
foreign import ccall "THTensorMath.h &THHalfTensor_equal"
  p_THHalfTensor_equal :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CInt)

-- |p_THHalfTensor_ltValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_ltValue"
  p_THHalfTensor_ltValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_leValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_leValue"
  p_THHalfTensor_leValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_gtValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_gtValue"
  p_THHalfTensor_gtValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_geValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_geValue"
  p_THHalfTensor_geValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_neValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_neValue"
  p_THHalfTensor_neValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_eqValue : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_eqValue"
  p_THHalfTensor_eqValue :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_ltValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_ltValueT"
  p_THHalfTensor_ltValueT :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_leValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_leValueT"
  p_THHalfTensor_leValueT :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_gtValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_gtValueT"
  p_THHalfTensor_gtValueT :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_geValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_geValueT"
  p_THHalfTensor_geValueT :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_neValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_neValueT"
  p_THHalfTensor_neValueT :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_eqValueT : Pointer to function : r_ t value -> void
foreign import ccall "THTensorMath.h &THHalfTensor_eqValueT"
  p_THHalfTensor_eqValueT :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> THHalf -> IO ())

-- |p_THHalfTensor_ltTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THHalfTensor_ltTensor"
  p_THHalfTensor_ltTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_leTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THHalfTensor_leTensor"
  p_THHalfTensor_leTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_gtTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THHalfTensor_gtTensor"
  p_THHalfTensor_gtTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_geTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THHalfTensor_geTensor"
  p_THHalfTensor_geTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_neTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THHalfTensor_neTensor"
  p_THHalfTensor_neTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_eqTensor : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THHalfTensor_eqTensor"
  p_THHalfTensor_eqTensor :: FunPtr (Ptr CTHByteTensor -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_ltTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THHalfTensor_ltTensorT"
  p_THHalfTensor_ltTensorT :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_leTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THHalfTensor_leTensorT"
  p_THHalfTensor_leTensorT :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_gtTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THHalfTensor_gtTensorT"
  p_THHalfTensor_gtTensorT :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_geTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THHalfTensor_geTensorT"
  p_THHalfTensor_geTensorT :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_neTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THHalfTensor_neTensorT"
  p_THHalfTensor_neTensorT :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_eqTensorT : Pointer to function : r_ ta tb -> void
foreign import ccall "THTensorMath.h &THHalfTensor_eqTensorT"
  p_THHalfTensor_eqTensorT :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())