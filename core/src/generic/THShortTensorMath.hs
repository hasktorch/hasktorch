{-# LANGUAGE ForeignFunctionInterface #-}

module THShortTensorMath (
    c_THShortTensor_fill,
    c_THShortTensor_zero,
    c_THShortTensor_maskedFill,
    c_THShortTensor_maskedCopy,
    c_THShortTensor_maskedSelect,
    c_THShortTensor_nonzero,
    c_THShortTensor_indexSelect,
    c_THShortTensor_indexCopy,
    c_THShortTensor_indexAdd,
    c_THShortTensor_indexFill,
    c_THShortTensor_gather,
    c_THShortTensor_scatter,
    c_THShortTensor_scatterAdd,
    c_THShortTensor_scatterFill,
    c_THShortTensor_dot,
    c_THShortTensor_minall,
    c_THShortTensor_maxall,
    c_THShortTensor_medianall,
    c_THShortTensor_sumall,
    c_THShortTensor_prodall,
    c_THShortTensor_neg,
    c_THShortTensor_add,
    c_THShortTensor_sub,
    c_THShortTensor_mul,
    c_THShortTensor_div,
    c_THShortTensor_lshift,
    c_THShortTensor_rshift,
    c_THShortTensor_fmod,
    c_THShortTensor_remainder,
    c_THShortTensor_clamp,
    c_THShortTensor_bitand,
    c_THShortTensor_bitor,
    c_THShortTensor_bitxor,
    c_THShortTensor_cadd,
    c_THShortTensor_csub,
    c_THShortTensor_cmul,
    c_THShortTensor_cpow,
    c_THShortTensor_cdiv,
    c_THShortTensor_clshift,
    c_THShortTensor_crshift,
    c_THShortTensor_cfmod,
    c_THShortTensor_cremainder,
    c_THShortTensor_cbitand,
    c_THShortTensor_cbitor,
    c_THShortTensor_cbitxor,
    c_THShortTensor_addcmul,
    c_THShortTensor_addcdiv,
    c_THShortTensor_addmv,
    c_THShortTensor_addmm,
    c_THShortTensor_addr,
    c_THShortTensor_addbmm,
    c_THShortTensor_baddbmm,
    c_THShortTensor_match,
    c_THShortTensor_numel,
    c_THShortTensor_max,
    c_THShortTensor_min,
    c_THShortTensor_kthvalue,
    c_THShortTensor_mode,
    c_THShortTensor_median,
    c_THShortTensor_sum,
    c_THShortTensor_prod,
    c_THShortTensor_cumsum,
    c_THShortTensor_cumprod,
    c_THShortTensor_sign,
    c_THShortTensor_trace,
    c_THShortTensor_cross,
    c_THShortTensor_cmax,
    c_THShortTensor_cmin,
    c_THShortTensor_cmaxValue,
    c_THShortTensor_cminValue,
    c_THShortTensor_zeros,
    c_THShortTensor_zerosLike,
    c_THShortTensor_ones,
    c_THShortTensor_onesLike,
    c_THShortTensor_diag,
    c_THShortTensor_eye,
    c_THShortTensor_arange,
    c_THShortTensor_range,
    c_THShortTensor_randperm,
    c_THShortTensor_reshape,
    c_THShortTensor_sort,
    c_THShortTensor_topk,
    c_THShortTensor_tril,
    c_THShortTensor_triu,
    c_THShortTensor_cat,
    c_THShortTensor_catArray,
    c_THShortTensor_equal,
    c_THShortTensor_ltValue,
    c_THShortTensor_leValue,
    c_THShortTensor_gtValue,
    c_THShortTensor_geValue,
    c_THShortTensor_neValue,
    c_THShortTensor_eqValue,
    c_THShortTensor_ltValueT,
    c_THShortTensor_leValueT,
    c_THShortTensor_gtValueT,
    c_THShortTensor_geValueT,
    c_THShortTensor_neValueT,
    c_THShortTensor_eqValueT,
    c_THShortTensor_ltTensor,
    c_THShortTensor_leTensor,
    c_THShortTensor_gtTensor,
    c_THShortTensor_geTensor,
    c_THShortTensor_neTensor,
    c_THShortTensor_eqTensor,
    c_THShortTensor_ltTensorT,
    c_THShortTensor_leTensorT,
    c_THShortTensor_gtTensorT,
    c_THShortTensor_geTensorT,
    c_THShortTensor_neTensorT,
    c_THShortTensor_eqTensorT,
    c_THShortTensor_abs) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THShortTensor_fill : r_ value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_fill"
  c_THShortTensor_fill :: (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_zero : r_ -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_zero"
  c_THShortTensor_zero :: (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_maskedFill : tensor mask value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_maskedFill"
  c_THShortTensor_maskedFill :: (Ptr CTHShortTensor) -> Ptr CTHByteTensor -> CShort -> IO ()

-- |c_THShortTensor_maskedCopy : tensor mask src -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_maskedCopy"
  c_THShortTensor_maskedCopy :: (Ptr CTHShortTensor) -> Ptr CTHByteTensor -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_maskedSelect : tensor src mask -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_maskedSelect"
  c_THShortTensor_maskedSelect :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THShortTensor_nonzero : subscript tensor -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_nonzero"
  c_THShortTensor_nonzero :: Ptr CTHLongTensor -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_indexSelect : tensor src dim index -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_indexSelect"
  c_THShortTensor_indexSelect :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THShortTensor_indexCopy : tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_indexCopy"
  c_THShortTensor_indexCopy :: (Ptr CTHShortTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_indexAdd : tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_indexAdd"
  c_THShortTensor_indexAdd :: (Ptr CTHShortTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_indexFill : tensor dim index val -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_indexFill"
  c_THShortTensor_indexFill :: (Ptr CTHShortTensor) -> CInt -> Ptr CTHLongTensor -> CShort -> IO ()

-- |c_THShortTensor_gather : tensor src dim index -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_gather"
  c_THShortTensor_gather :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THShortTensor_scatter : tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_scatter"
  c_THShortTensor_scatter :: (Ptr CTHShortTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_scatterAdd : tensor dim index src -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_scatterAdd"
  c_THShortTensor_scatterAdd :: (Ptr CTHShortTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_scatterFill : tensor dim index val -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_scatterFill"
  c_THShortTensor_scatterFill :: (Ptr CTHShortTensor) -> CInt -> Ptr CTHLongTensor -> CShort -> IO ()

-- |c_THShortTensor_dot : t src -> accreal
foreign import ccall unsafe "THTensorMath.h THShortTensor_dot"
  c_THShortTensor_dot :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CLong

-- |c_THShortTensor_minall : t -> real
foreign import ccall unsafe "THTensorMath.h THShortTensor_minall"
  c_THShortTensor_minall :: (Ptr CTHShortTensor) -> CShort

-- |c_THShortTensor_maxall : t -> real
foreign import ccall unsafe "THTensorMath.h THShortTensor_maxall"
  c_THShortTensor_maxall :: (Ptr CTHShortTensor) -> CShort

-- |c_THShortTensor_medianall : t -> real
foreign import ccall unsafe "THTensorMath.h THShortTensor_medianall"
  c_THShortTensor_medianall :: (Ptr CTHShortTensor) -> CShort

-- |c_THShortTensor_sumall : t -> accreal
foreign import ccall unsafe "THTensorMath.h THShortTensor_sumall"
  c_THShortTensor_sumall :: (Ptr CTHShortTensor) -> CLong

-- |c_THShortTensor_prodall : t -> accreal
foreign import ccall unsafe "THTensorMath.h THShortTensor_prodall"
  c_THShortTensor_prodall :: (Ptr CTHShortTensor) -> CLong

-- |c_THShortTensor_neg : self src -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_neg"
  c_THShortTensor_neg :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_add : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_add"
  c_THShortTensor_add :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_sub : self src value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_sub"
  c_THShortTensor_sub :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_mul : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_mul"
  c_THShortTensor_mul :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_div : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_div"
  c_THShortTensor_div :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_lshift : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_lshift"
  c_THShortTensor_lshift :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_rshift : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_rshift"
  c_THShortTensor_rshift :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_fmod : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_fmod"
  c_THShortTensor_fmod :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_remainder : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_remainder"
  c_THShortTensor_remainder :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_clamp : r_ t min_value max_value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_clamp"
  c_THShortTensor_clamp :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> CShort -> IO ()

-- |c_THShortTensor_bitand : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_bitand"
  c_THShortTensor_bitand :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_bitor : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_bitor"
  c_THShortTensor_bitor :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_bitxor : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_bitxor"
  c_THShortTensor_bitxor :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_cadd : r_ t value src -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_cadd"
  c_THShortTensor_cadd :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_csub : self src1 value src2 -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_csub"
  c_THShortTensor_csub :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_cmul : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_cmul"
  c_THShortTensor_cmul :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_cpow : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_cpow"
  c_THShortTensor_cpow :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_cdiv : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_cdiv"
  c_THShortTensor_cdiv :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_clshift : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_clshift"
  c_THShortTensor_clshift :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_crshift : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_crshift"
  c_THShortTensor_crshift :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_cfmod : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_cfmod"
  c_THShortTensor_cfmod :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_cremainder : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_cremainder"
  c_THShortTensor_cremainder :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_cbitand : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_cbitand"
  c_THShortTensor_cbitand :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_cbitor : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_cbitor"
  c_THShortTensor_cbitor :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_cbitxor : r_ t src -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_cbitxor"
  c_THShortTensor_cbitxor :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_addcmul : r_ t value src1 src2 -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_addcmul"
  c_THShortTensor_addcmul :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_addcdiv : r_ t value src1 src2 -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_addcdiv"
  c_THShortTensor_addcdiv :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_addmv : r_ beta t alpha mat vec -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_addmv"
  c_THShortTensor_addmv :: (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_addmm : r_ beta t alpha mat1 mat2 -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_addmm"
  c_THShortTensor_addmm :: (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_addr : r_ beta t alpha vec1 vec2 -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_addr"
  c_THShortTensor_addr :: (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_addbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_addbmm"
  c_THShortTensor_addbmm :: (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_baddbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_baddbmm"
  c_THShortTensor_baddbmm :: (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> CShort -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_match : r_ m1 m2 gain -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_match"
  c_THShortTensor_match :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_numel : t -> ptrdiff_t
foreign import ccall unsafe "THTensorMath.h THShortTensor_numel"
  c_THShortTensor_numel :: (Ptr CTHShortTensor) -> CPtrdiff

-- |c_THShortTensor_max : values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_max"
  c_THShortTensor_max :: (Ptr CTHShortTensor) -> Ptr CTHLongTensor -> (Ptr CTHShortTensor) -> CInt -> CInt -> IO ()

-- |c_THShortTensor_min : values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_min"
  c_THShortTensor_min :: (Ptr CTHShortTensor) -> Ptr CTHLongTensor -> (Ptr CTHShortTensor) -> CInt -> CInt -> IO ()

-- |c_THShortTensor_kthvalue : values_ indices_ t k dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_kthvalue"
  c_THShortTensor_kthvalue :: (Ptr CTHShortTensor) -> Ptr CTHLongTensor -> (Ptr CTHShortTensor) -> CLong -> CInt -> CInt -> IO ()

-- |c_THShortTensor_mode : values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_mode"
  c_THShortTensor_mode :: (Ptr CTHShortTensor) -> Ptr CTHLongTensor -> (Ptr CTHShortTensor) -> CInt -> CInt -> IO ()

-- |c_THShortTensor_median : values_ indices_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_median"
  c_THShortTensor_median :: (Ptr CTHShortTensor) -> Ptr CTHLongTensor -> (Ptr CTHShortTensor) -> CInt -> CInt -> IO ()

-- |c_THShortTensor_sum : r_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_sum"
  c_THShortTensor_sum :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> CInt -> IO ()

-- |c_THShortTensor_prod : r_ t dimension keepdim -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_prod"
  c_THShortTensor_prod :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> CInt -> IO ()

-- |c_THShortTensor_cumsum : r_ t dimension -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_cumsum"
  c_THShortTensor_cumsum :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> IO ()

-- |c_THShortTensor_cumprod : r_ t dimension -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_cumprod"
  c_THShortTensor_cumprod :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> IO ()

-- |c_THShortTensor_sign : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_sign"
  c_THShortTensor_sign :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_trace : t -> accreal
foreign import ccall unsafe "THTensorMath.h THShortTensor_trace"
  c_THShortTensor_trace :: (Ptr CTHShortTensor) -> CLong

-- |c_THShortTensor_cross : r_ a b dimension -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_cross"
  c_THShortTensor_cross :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> IO ()

-- |c_THShortTensor_cmax : r t src -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_cmax"
  c_THShortTensor_cmax :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_cmin : r t src -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_cmin"
  c_THShortTensor_cmin :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_cmaxValue : r t value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_cmaxValue"
  c_THShortTensor_cmaxValue :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_cminValue : r t value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_cminValue"
  c_THShortTensor_cminValue :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_zeros : r_ size -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_zeros"
  c_THShortTensor_zeros :: (Ptr CTHShortTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THShortTensor_zerosLike : r_ input -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_zerosLike"
  c_THShortTensor_zerosLike :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_ones : r_ size -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_ones"
  c_THShortTensor_ones :: (Ptr CTHShortTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THShortTensor_onesLike : r_ input -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_onesLike"
  c_THShortTensor_onesLike :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_diag : r_ t k -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_diag"
  c_THShortTensor_diag :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> IO ()

-- |c_THShortTensor_eye : r_ n m -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_eye"
  c_THShortTensor_eye :: (Ptr CTHShortTensor) -> CLong -> CLong -> IO ()

-- |c_THShortTensor_arange : r_ xmin xmax step -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_arange"
  c_THShortTensor_arange :: (Ptr CTHShortTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THShortTensor_range : r_ xmin xmax step -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_range"
  c_THShortTensor_range :: (Ptr CTHShortTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THShortTensor_randperm : r_ _generator n -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_randperm"
  c_THShortTensor_randperm :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> CLong -> IO ()

-- |c_THShortTensor_reshape : r_ t size -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_reshape"
  c_THShortTensor_reshape :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THShortTensor_sort : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_sort"
  c_THShortTensor_sort :: (Ptr CTHShortTensor) -> Ptr CTHLongTensor -> (Ptr CTHShortTensor) -> CInt -> CInt -> IO ()

-- |c_THShortTensor_topk : rt_ ri_ t k dim dir sorted -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_topk"
  c_THShortTensor_topk :: (Ptr CTHShortTensor) -> Ptr CTHLongTensor -> (Ptr CTHShortTensor) -> CLong -> CInt -> CInt -> CInt -> IO ()

-- |c_THShortTensor_tril : r_ t k -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_tril"
  c_THShortTensor_tril :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CLong -> IO ()

-- |c_THShortTensor_triu : r_ t k -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_triu"
  c_THShortTensor_triu :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CLong -> IO ()

-- |c_THShortTensor_cat : r_ ta tb dimension -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_cat"
  c_THShortTensor_cat :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt -> IO ()

-- |c_THShortTensor_catArray : result inputs numInputs dimension -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_catArray"
  c_THShortTensor_catArray :: (Ptr CTHShortTensor) -> Ptr (Ptr CTHShortTensor) -> CInt -> CInt -> IO ()

-- |c_THShortTensor_equal : ta tb -> int
foreign import ccall unsafe "THTensorMath.h THShortTensor_equal"
  c_THShortTensor_equal :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CInt

-- |c_THShortTensor_ltValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_ltValue"
  c_THShortTensor_ltValue :: Ptr CTHByteTensor -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_leValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_leValue"
  c_THShortTensor_leValue :: Ptr CTHByteTensor -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_gtValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_gtValue"
  c_THShortTensor_gtValue :: Ptr CTHByteTensor -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_geValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_geValue"
  c_THShortTensor_geValue :: Ptr CTHByteTensor -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_neValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_neValue"
  c_THShortTensor_neValue :: Ptr CTHByteTensor -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_eqValue : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_eqValue"
  c_THShortTensor_eqValue :: Ptr CTHByteTensor -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_ltValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_ltValueT"
  c_THShortTensor_ltValueT :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_leValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_leValueT"
  c_THShortTensor_leValueT :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_gtValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_gtValueT"
  c_THShortTensor_gtValueT :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_geValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_geValueT"
  c_THShortTensor_geValueT :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_neValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_neValueT"
  c_THShortTensor_neValueT :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_eqValueT : r_ t value -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_eqValueT"
  c_THShortTensor_eqValueT :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> CShort -> IO ()

-- |c_THShortTensor_ltTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_ltTensor"
  c_THShortTensor_ltTensor :: Ptr CTHByteTensor -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_leTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_leTensor"
  c_THShortTensor_leTensor :: Ptr CTHByteTensor -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_gtTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_gtTensor"
  c_THShortTensor_gtTensor :: Ptr CTHByteTensor -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_geTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_geTensor"
  c_THShortTensor_geTensor :: Ptr CTHByteTensor -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_neTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_neTensor"
  c_THShortTensor_neTensor :: Ptr CTHByteTensor -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_eqTensor : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_eqTensor"
  c_THShortTensor_eqTensor :: Ptr CTHByteTensor -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_ltTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_ltTensorT"
  c_THShortTensor_ltTensorT :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_leTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_leTensorT"
  c_THShortTensor_leTensorT :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_gtTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_gtTensorT"
  c_THShortTensor_gtTensorT :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_geTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_geTensorT"
  c_THShortTensor_geTensorT :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_neTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_neTensorT"
  c_THShortTensor_neTensorT :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_eqTensorT : r_ ta tb -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_eqTensorT"
  c_THShortTensor_eqTensorT :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_abs : r_ t -> void
foreign import ccall unsafe "THTensorMath.h THShortTensor_abs"
  c_THShortTensor_abs :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()