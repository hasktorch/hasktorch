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
    c_THByteTensor_eqTensorT) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THByteTensor_fill : r_ value -> void
foreign import ccall "THTensorMath.h THByteTensor_fill"
  c_THByteTensor_fill :: (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_zero : r_ -> void
foreign import ccall "THTensorMath.h THByteTensor_zero"
  c_THByteTensor_zero :: (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_maskedFill : tensor mask value -> void
foreign import ccall "THTensorMath.h THByteTensor_maskedFill"
  c_THByteTensor_maskedFill :: (Ptr CTHByteTensor) -> Ptr CTHByteTensor -> CChar -> IO ()

-- |c_THByteTensor_maskedCopy : tensor mask src -> void
foreign import ccall "THTensorMath.h THByteTensor_maskedCopy"
  c_THByteTensor_maskedCopy :: (Ptr CTHByteTensor) -> Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_maskedSelect : tensor src mask -> void
foreign import ccall "THTensorMath.h THByteTensor_maskedSelect"
  c_THByteTensor_maskedSelect :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THByteTensor_nonzero : subscript tensor -> void
foreign import ccall "THTensorMath.h THByteTensor_nonzero"
  c_THByteTensor_nonzero :: Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_indexSelect : tensor src dim index -> void
foreign import ccall "THTensorMath.h THByteTensor_indexSelect"
  c_THByteTensor_indexSelect :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THByteTensor_indexCopy : tensor dim index src -> void
foreign import ccall "THTensorMath.h THByteTensor_indexCopy"
  c_THByteTensor_indexCopy :: (Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_indexAdd : tensor dim index src -> void
foreign import ccall "THTensorMath.h THByteTensor_indexAdd"
  c_THByteTensor_indexAdd :: (Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_indexFill : tensor dim index val -> void
foreign import ccall "THTensorMath.h THByteTensor_indexFill"
  c_THByteTensor_indexFill :: (Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> CChar -> IO ()

-- |c_THByteTensor_gather : tensor src dim index -> void
foreign import ccall "THTensorMath.h THByteTensor_gather"
  c_THByteTensor_gather :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THByteTensor_scatter : tensor dim index src -> void
foreign import ccall "THTensorMath.h THByteTensor_scatter"
  c_THByteTensor_scatter :: (Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_scatterAdd : tensor dim index src -> void
foreign import ccall "THTensorMath.h THByteTensor_scatterAdd"
  c_THByteTensor_scatterAdd :: (Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_scatterFill : tensor dim index val -> void
foreign import ccall "THTensorMath.h THByteTensor_scatterFill"
  c_THByteTensor_scatterFill :: (Ptr CTHByteTensor) -> CInt -> Ptr CTHLongTensor -> CChar -> IO ()

-- |c_THByteTensor_dot : t src -> accreal
foreign import ccall "THTensorMath.h THByteTensor_dot"
  c_THByteTensor_dot :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CLong

-- |c_THByteTensor_minall : t -> real
foreign import ccall "THTensorMath.h THByteTensor_minall"
  c_THByteTensor_minall :: (Ptr CTHByteTensor) -> CChar

-- |c_THByteTensor_maxall : t -> real
foreign import ccall "THTensorMath.h THByteTensor_maxall"
  c_THByteTensor_maxall :: (Ptr CTHByteTensor) -> CChar

-- |c_THByteTensor_medianall : t -> real
foreign import ccall "THTensorMath.h THByteTensor_medianall"
  c_THByteTensor_medianall :: (Ptr CTHByteTensor) -> CChar

-- |c_THByteTensor_sumall : t -> accreal
foreign import ccall "THTensorMath.h THByteTensor_sumall"
  c_THByteTensor_sumall :: (Ptr CTHByteTensor) -> CLong

-- |c_THByteTensor_prodall : t -> accreal
foreign import ccall "THTensorMath.h THByteTensor_prodall"
  c_THByteTensor_prodall :: (Ptr CTHByteTensor) -> CLong

-- |c_THByteTensor_add : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_add"
  c_THByteTensor_add :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_sub : self src value -> void
foreign import ccall "THTensorMath.h THByteTensor_sub"
  c_THByteTensor_sub :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_mul : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_mul"
  c_THByteTensor_mul :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_div : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_div"
  c_THByteTensor_div :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_lshift : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_lshift"
  c_THByteTensor_lshift :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_rshift : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_rshift"
  c_THByteTensor_rshift :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_fmod : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_fmod"
  c_THByteTensor_fmod :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_remainder : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_remainder"
  c_THByteTensor_remainder :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_clamp : r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h THByteTensor_clamp"
  c_THByteTensor_clamp :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> CChar -> IO ()

-- |c_THByteTensor_bitand : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_bitand"
  c_THByteTensor_bitand :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_bitor : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_bitor"
  c_THByteTensor_bitor :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_bitxor : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_bitxor"
  c_THByteTensor_bitxor :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_cadd : r_ t value src -> void
foreign import ccall "THTensorMath.h THByteTensor_cadd"
  c_THByteTensor_cadd :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_csub : self src1 value src2 -> void
foreign import ccall "THTensorMath.h THByteTensor_csub"
  c_THByteTensor_csub :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_cmul : r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensor_cmul"
  c_THByteTensor_cmul :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_cpow : r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensor_cpow"
  c_THByteTensor_cpow :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_cdiv : r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensor_cdiv"
  c_THByteTensor_cdiv :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_clshift : r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensor_clshift"
  c_THByteTensor_clshift :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_crshift : r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensor_crshift"
  c_THByteTensor_crshift :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_cfmod : r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensor_cfmod"
  c_THByteTensor_cfmod :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_cremainder : r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensor_cremainder"
  c_THByteTensor_cremainder :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_cbitand : r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensor_cbitand"
  c_THByteTensor_cbitand :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_cbitor : r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensor_cbitor"
  c_THByteTensor_cbitor :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_cbitxor : r_ t src -> void
foreign import ccall "THTensorMath.h THByteTensor_cbitxor"
  c_THByteTensor_cbitxor :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_addcmul : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THByteTensor_addcmul"
  c_THByteTensor_addcmul :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_addcdiv : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THByteTensor_addcdiv"
  c_THByteTensor_addcdiv :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_addmv : r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h THByteTensor_addmv"
  c_THByteTensor_addmv :: (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_addmm : r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h THByteTensor_addmm"
  c_THByteTensor_addmm :: (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_addr : r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h THByteTensor_addr"
  c_THByteTensor_addr :: (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_addbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THByteTensor_addbmm"
  c_THByteTensor_addbmm :: (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_baddbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THByteTensor_baddbmm"
  c_THByteTensor_baddbmm :: (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_match : r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h THByteTensor_match"
  c_THByteTensor_match :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_numel : t -> ptrdiff_t
foreign import ccall "THTensorMath.h THByteTensor_numel"
  c_THByteTensor_numel :: (Ptr CTHByteTensor) -> CPtrdiff

-- |c_THByteTensor_max : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THByteTensor_max"
  c_THByteTensor_max :: (Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensor_min : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THByteTensor_min"
  c_THByteTensor_min :: (Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensor_kthvalue : values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h THByteTensor_kthvalue"
  c_THByteTensor_kthvalue :: (Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CLong -> CInt -> CInt -> IO ()

-- |c_THByteTensor_mode : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THByteTensor_mode"
  c_THByteTensor_mode :: (Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensor_median : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THByteTensor_median"
  c_THByteTensor_median :: (Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensor_sum : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THByteTensor_sum"
  c_THByteTensor_sum :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensor_prod : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THByteTensor_prod"
  c_THByteTensor_prod :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensor_cumsum : r_ t dimension -> void
foreign import ccall "THTensorMath.h THByteTensor_cumsum"
  c_THByteTensor_cumsum :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> IO ()

-- |c_THByteTensor_cumprod : r_ t dimension -> void
foreign import ccall "THTensorMath.h THByteTensor_cumprod"
  c_THByteTensor_cumprod :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> IO ()

-- |c_THByteTensor_sign : r_ t -> void
foreign import ccall "THTensorMath.h THByteTensor_sign"
  c_THByteTensor_sign :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_trace : t -> accreal
foreign import ccall "THTensorMath.h THByteTensor_trace"
  c_THByteTensor_trace :: (Ptr CTHByteTensor) -> CLong

-- |c_THByteTensor_cross : r_ a b dimension -> void
foreign import ccall "THTensorMath.h THByteTensor_cross"
  c_THByteTensor_cross :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> IO ()

-- |c_THByteTensor_cmax : r t src -> void
foreign import ccall "THTensorMath.h THByteTensor_cmax"
  c_THByteTensor_cmax :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_cmin : r t src -> void
foreign import ccall "THTensorMath.h THByteTensor_cmin"
  c_THByteTensor_cmin :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_cmaxValue : r t value -> void
foreign import ccall "THTensorMath.h THByteTensor_cmaxValue"
  c_THByteTensor_cmaxValue :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_cminValue : r t value -> void
foreign import ccall "THTensorMath.h THByteTensor_cminValue"
  c_THByteTensor_cminValue :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_zeros : r_ size -> void
foreign import ccall "THTensorMath.h THByteTensor_zeros"
  c_THByteTensor_zeros :: (Ptr CTHByteTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THByteTensor_zerosLike : r_ input -> void
foreign import ccall "THTensorMath.h THByteTensor_zerosLike"
  c_THByteTensor_zerosLike :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_ones : r_ size -> void
foreign import ccall "THTensorMath.h THByteTensor_ones"
  c_THByteTensor_ones :: (Ptr CTHByteTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THByteTensor_onesLike : r_ input -> void
foreign import ccall "THTensorMath.h THByteTensor_onesLike"
  c_THByteTensor_onesLike :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_diag : r_ t k -> void
foreign import ccall "THTensorMath.h THByteTensor_diag"
  c_THByteTensor_diag :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> IO ()

-- |c_THByteTensor_eye : r_ n m -> void
foreign import ccall "THTensorMath.h THByteTensor_eye"
  c_THByteTensor_eye :: (Ptr CTHByteTensor) -> CLong -> CLong -> IO ()

-- |c_THByteTensor_arange : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THByteTensor_arange"
  c_THByteTensor_arange :: (Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_range : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THByteTensor_range"
  c_THByteTensor_range :: (Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_randperm : r_ _generator n -> void
foreign import ccall "THTensorMath.h THByteTensor_randperm"
  c_THByteTensor_randperm :: (Ptr CTHByteTensor) -> Ptr CTHGenerator -> CLong -> IO ()

-- |c_THByteTensor_reshape : r_ t size -> void
foreign import ccall "THTensorMath.h THByteTensor_reshape"
  c_THByteTensor_reshape :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THByteTensor_sort : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h THByteTensor_sort"
  c_THByteTensor_sort :: (Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensor_topk : rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h THByteTensor_topk"
  c_THByteTensor_topk :: (Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> CLong -> CInt -> CInt -> CInt -> IO ()

-- |c_THByteTensor_tril : r_ t k -> void
foreign import ccall "THTensorMath.h THByteTensor_tril"
  c_THByteTensor_tril :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CLong -> IO ()

-- |c_THByteTensor_triu : r_ t k -> void
foreign import ccall "THTensorMath.h THByteTensor_triu"
  c_THByteTensor_triu :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CLong -> IO ()

-- |c_THByteTensor_cat : r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h THByteTensor_cat"
  c_THByteTensor_cat :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt -> IO ()

-- |c_THByteTensor_catArray : result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h THByteTensor_catArray"
  c_THByteTensor_catArray :: (Ptr CTHByteTensor) -> Ptr (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensor_equal : ta tb -> int
foreign import ccall "THTensorMath.h THByteTensor_equal"
  c_THByteTensor_equal :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CInt

-- |c_THByteTensor_ltValue : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_ltValue"
  c_THByteTensor_ltValue :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_leValue : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_leValue"
  c_THByteTensor_leValue :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_gtValue : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_gtValue"
  c_THByteTensor_gtValue :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_geValue : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_geValue"
  c_THByteTensor_geValue :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_neValue : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_neValue"
  c_THByteTensor_neValue :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_eqValue : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_eqValue"
  c_THByteTensor_eqValue :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_ltValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_ltValueT"
  c_THByteTensor_ltValueT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_leValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_leValueT"
  c_THByteTensor_leValueT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_gtValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_gtValueT"
  c_THByteTensor_gtValueT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_geValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_geValueT"
  c_THByteTensor_geValueT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_neValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_neValueT"
  c_THByteTensor_neValueT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_eqValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THByteTensor_eqValueT"
  c_THByteTensor_eqValueT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CChar -> IO ()

-- |c_THByteTensor_ltTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensor_ltTensor"
  c_THByteTensor_ltTensor :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_leTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensor_leTensor"
  c_THByteTensor_leTensor :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_gtTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensor_gtTensor"
  c_THByteTensor_gtTensor :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_geTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensor_geTensor"
  c_THByteTensor_geTensor :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_neTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensor_neTensor"
  c_THByteTensor_neTensor :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_eqTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensor_eqTensor"
  c_THByteTensor_eqTensor :: Ptr CTHByteTensor -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_ltTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensor_ltTensorT"
  c_THByteTensor_ltTensorT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_leTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensor_leTensorT"
  c_THByteTensor_leTensorT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_gtTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensor_gtTensorT"
  c_THByteTensor_gtTensorT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_geTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensor_geTensorT"
  c_THByteTensor_geTensorT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_neTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensor_neTensorT"
  c_THByteTensor_neTensorT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_eqTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THByteTensor_eqTensorT"
  c_THByteTensor_eqTensorT :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()