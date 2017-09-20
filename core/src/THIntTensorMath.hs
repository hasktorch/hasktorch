{-# LANGUAGE ForeignFunctionInterface#-}

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
    c_THIntTensor_cinv,
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
    c_THIntTensor_abs) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THIntTensor_fill : r_ value -> void
foreign import ccall "THTensorMath.h THIntTensor_fill"
  c_THIntTensor_fill :: (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_zero : r_ -> void
foreign import ccall "THTensorMath.h THIntTensor_zero"
  c_THIntTensor_zero :: (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_maskedFill : tensor mask value -> void
foreign import ccall "THTensorMath.h THIntTensor_maskedFill"
  c_THIntTensor_maskedFill :: (Ptr CTHIntTensor) -> Ptr CTHByteTensor -> CInt -> IO ()

-- |c_THIntTensor_maskedCopy : tensor mask src -> void
foreign import ccall "THTensorMath.h THIntTensor_maskedCopy"
  c_THIntTensor_maskedCopy :: (Ptr CTHIntTensor) -> Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_maskedSelect : tensor src mask -> void
foreign import ccall "THTensorMath.h THIntTensor_maskedSelect"
  c_THIntTensor_maskedSelect :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THIntTensor_nonzero : subscript tensor -> void
foreign import ccall "THTensorMath.h THIntTensor_nonzero"
  c_THIntTensor_nonzero :: Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_indexSelect : tensor src dim index -> void
foreign import ccall "THTensorMath.h THIntTensor_indexSelect"
  c_THIntTensor_indexSelect :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THIntTensor_indexCopy : tensor dim index src -> void
foreign import ccall "THTensorMath.h THIntTensor_indexCopy"
  c_THIntTensor_indexCopy :: (Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_indexAdd : tensor dim index src -> void
foreign import ccall "THTensorMath.h THIntTensor_indexAdd"
  c_THIntTensor_indexAdd :: (Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_indexFill : tensor dim index val -> void
foreign import ccall "THTensorMath.h THIntTensor_indexFill"
  c_THIntTensor_indexFill :: (Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> CInt -> IO ()

-- |c_THIntTensor_gather : tensor src dim index -> void
foreign import ccall "THTensorMath.h THIntTensor_gather"
  c_THIntTensor_gather :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> IO ()

-- |c_THIntTensor_scatter : tensor dim index src -> void
foreign import ccall "THTensorMath.h THIntTensor_scatter"
  c_THIntTensor_scatter :: (Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_scatterAdd : tensor dim index src -> void
foreign import ccall "THTensorMath.h THIntTensor_scatterAdd"
  c_THIntTensor_scatterAdd :: (Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_scatterFill : tensor dim index val -> void
foreign import ccall "THTensorMath.h THIntTensor_scatterFill"
  c_THIntTensor_scatterFill :: (Ptr CTHIntTensor) -> CInt -> Ptr CTHLongTensor -> CInt -> IO ()

-- |c_THIntTensor_dot : t src -> accreal
foreign import ccall "THTensorMath.h THIntTensor_dot"
  c_THIntTensor_dot :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CLong

-- |c_THIntTensor_minall : t -> real
foreign import ccall "THTensorMath.h THIntTensor_minall"
  c_THIntTensor_minall :: (Ptr CTHIntTensor) -> CInt

-- |c_THIntTensor_maxall : t -> real
foreign import ccall "THTensorMath.h THIntTensor_maxall"
  c_THIntTensor_maxall :: (Ptr CTHIntTensor) -> CInt

-- |c_THIntTensor_medianall : t -> real
foreign import ccall "THTensorMath.h THIntTensor_medianall"
  c_THIntTensor_medianall :: (Ptr CTHIntTensor) -> CInt

-- |c_THIntTensor_sumall : t -> accreal
foreign import ccall "THTensorMath.h THIntTensor_sumall"
  c_THIntTensor_sumall :: (Ptr CTHIntTensor) -> CLong

-- |c_THIntTensor_prodall : t -> accreal
foreign import ccall "THTensorMath.h THIntTensor_prodall"
  c_THIntTensor_prodall :: (Ptr CTHIntTensor) -> CLong

-- |c_THIntTensor_neg : self src -> void
foreign import ccall "THTensorMath.h THIntTensor_neg"
  c_THIntTensor_neg :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_cinv : self src -> void
foreign import ccall "THTensorMath.h THIntTensor_cinv"
  c_THIntTensor_cinv :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_add : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_add"
  c_THIntTensor_add :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_sub : self src value -> void
foreign import ccall "THTensorMath.h THIntTensor_sub"
  c_THIntTensor_sub :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_mul : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_mul"
  c_THIntTensor_mul :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_div : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_div"
  c_THIntTensor_div :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_lshift : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_lshift"
  c_THIntTensor_lshift :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_rshift : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_rshift"
  c_THIntTensor_rshift :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_fmod : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_fmod"
  c_THIntTensor_fmod :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_remainder : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_remainder"
  c_THIntTensor_remainder :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_clamp : r_ t min_value max_value -> void
foreign import ccall "THTensorMath.h THIntTensor_clamp"
  c_THIntTensor_clamp :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensor_bitand : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_bitand"
  c_THIntTensor_bitand :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_bitor : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_bitor"
  c_THIntTensor_bitor :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_bitxor : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_bitxor"
  c_THIntTensor_bitxor :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_cadd : r_ t value src -> void
foreign import ccall "THTensorMath.h THIntTensor_cadd"
  c_THIntTensor_cadd :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_csub : self src1 value src2 -> void
foreign import ccall "THTensorMath.h THIntTensor_csub"
  c_THIntTensor_csub :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_cmul : r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cmul"
  c_THIntTensor_cmul :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_cpow : r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cpow"
  c_THIntTensor_cpow :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_cdiv : r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cdiv"
  c_THIntTensor_cdiv :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_clshift : r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_clshift"
  c_THIntTensor_clshift :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_crshift : r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_crshift"
  c_THIntTensor_crshift :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_cfmod : r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cfmod"
  c_THIntTensor_cfmod :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_cremainder : r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cremainder"
  c_THIntTensor_cremainder :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_cbitand : r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cbitand"
  c_THIntTensor_cbitand :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_cbitor : r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cbitor"
  c_THIntTensor_cbitor :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_cbitxor : r_ t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cbitxor"
  c_THIntTensor_cbitxor :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_addcmul : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THIntTensor_addcmul"
  c_THIntTensor_addcmul :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_addcdiv : r_ t value src1 src2 -> void
foreign import ccall "THTensorMath.h THIntTensor_addcdiv"
  c_THIntTensor_addcdiv :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_addmv : r_ beta t alpha mat vec -> void
foreign import ccall "THTensorMath.h THIntTensor_addmv"
  c_THIntTensor_addmv :: (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_addmm : r_ beta t alpha mat1 mat2 -> void
foreign import ccall "THTensorMath.h THIntTensor_addmm"
  c_THIntTensor_addmm :: (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_addr : r_ beta t alpha vec1 vec2 -> void
foreign import ccall "THTensorMath.h THIntTensor_addr"
  c_THIntTensor_addr :: (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_addbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THIntTensor_addbmm"
  c_THIntTensor_addbmm :: (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_baddbmm : r_ beta t alpha batch1 batch2 -> void
foreign import ccall "THTensorMath.h THIntTensor_baddbmm"
  c_THIntTensor_baddbmm :: (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_match : r_ m1 m2 gain -> void
foreign import ccall "THTensorMath.h THIntTensor_match"
  c_THIntTensor_match :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_numel : t -> THStorage *
foreign import ccall "THTensorMath.h THIntTensor_numel"
  c_THIntTensor_numel :: (Ptr CTHIntTensor) -> IO (Ptr CTHIntStorage)

-- |c_THIntTensor_max : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensor_max"
  c_THIntTensor_max :: (Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensor_min : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensor_min"
  c_THIntTensor_min :: (Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensor_kthvalue : values_ indices_ t k dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensor_kthvalue"
  c_THIntTensor_kthvalue :: (Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CLong -> CInt -> CInt -> IO ()

-- |c_THIntTensor_mode : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensor_mode"
  c_THIntTensor_mode :: (Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensor_median : values_ indices_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensor_median"
  c_THIntTensor_median :: (Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensor_sum : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensor_sum"
  c_THIntTensor_sum :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensor_prod : r_ t dimension keepdim -> void
foreign import ccall "THTensorMath.h THIntTensor_prod"
  c_THIntTensor_prod :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensor_cumsum : r_ t dimension -> void
foreign import ccall "THTensorMath.h THIntTensor_cumsum"
  c_THIntTensor_cumsum :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_cumprod : r_ t dimension -> void
foreign import ccall "THTensorMath.h THIntTensor_cumprod"
  c_THIntTensor_cumprod :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_sign : r_ t -> void
foreign import ccall "THTensorMath.h THIntTensor_sign"
  c_THIntTensor_sign :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_trace : t -> accreal
foreign import ccall "THTensorMath.h THIntTensor_trace"
  c_THIntTensor_trace :: (Ptr CTHIntTensor) -> CLong

-- |c_THIntTensor_cross : r_ a b dimension -> void
foreign import ccall "THTensorMath.h THIntTensor_cross"
  c_THIntTensor_cross :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_cmax : r t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cmax"
  c_THIntTensor_cmax :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_cmin : r t src -> void
foreign import ccall "THTensorMath.h THIntTensor_cmin"
  c_THIntTensor_cmin :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_cmaxValue : r t value -> void
foreign import ccall "THTensorMath.h THIntTensor_cmaxValue"
  c_THIntTensor_cmaxValue :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_cminValue : r t value -> void
foreign import ccall "THTensorMath.h THIntTensor_cminValue"
  c_THIntTensor_cminValue :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_zeros : r_ size -> void
foreign import ccall "THTensorMath.h THIntTensor_zeros"
  c_THIntTensor_zeros :: (Ptr CTHIntTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THIntTensor_zerosLike : r_ input -> void
foreign import ccall "THTensorMath.h THIntTensor_zerosLike"
  c_THIntTensor_zerosLike :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_ones : r_ size -> void
foreign import ccall "THTensorMath.h THIntTensor_ones"
  c_THIntTensor_ones :: (Ptr CTHIntTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THIntTensor_onesLike : r_ input -> void
foreign import ccall "THTensorMath.h THIntTensor_onesLike"
  c_THIntTensor_onesLike :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_diag : r_ t k -> void
foreign import ccall "THTensorMath.h THIntTensor_diag"
  c_THIntTensor_diag :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_eye : r_ n m -> void
foreign import ccall "THTensorMath.h THIntTensor_eye"
  c_THIntTensor_eye :: (Ptr CTHIntTensor) -> CLong -> CLong -> IO ()

-- |c_THIntTensor_arange : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THIntTensor_arange"
  c_THIntTensor_arange :: (Ptr CTHIntTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THIntTensor_range : r_ xmin xmax step -> void
foreign import ccall "THTensorMath.h THIntTensor_range"
  c_THIntTensor_range :: (Ptr CTHIntTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THIntTensor_randperm : r_ _generator n -> void
foreign import ccall "THTensorMath.h THIntTensor_randperm"
  c_THIntTensor_randperm :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> CLong -> IO ()

-- |c_THIntTensor_reshape : r_ t size -> void
foreign import ccall "THTensorMath.h THIntTensor_reshape"
  c_THIntTensor_reshape :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CTHLongStorage -> IO ()

-- |c_THIntTensor_sort : rt_ ri_ t dimension descendingOrder -> void
foreign import ccall "THTensorMath.h THIntTensor_sort"
  c_THIntTensor_sort :: (Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensor_topk : rt_ ri_ t k dim dir sorted -> void
foreign import ccall "THTensorMath.h THIntTensor_topk"
  c_THIntTensor_topk :: (Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> CLong -> CInt -> CInt -> CInt -> IO ()

-- |c_THIntTensor_tril : r_ t k -> void
foreign import ccall "THTensorMath.h THIntTensor_tril"
  c_THIntTensor_tril :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CLong -> IO ()

-- |c_THIntTensor_triu : r_ t k -> void
foreign import ccall "THTensorMath.h THIntTensor_triu"
  c_THIntTensor_triu :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CLong -> IO ()

-- |c_THIntTensor_cat : r_ ta tb dimension -> void
foreign import ccall "THTensorMath.h THIntTensor_cat"
  c_THIntTensor_cat :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_catArray : result inputs numInputs dimension -> void
foreign import ccall "THTensorMath.h THIntTensor_catArray"
  c_THIntTensor_catArray :: (Ptr CTHIntTensor) -> Ptr (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensor_equal : ta tb -> int
foreign import ccall "THTensorMath.h THIntTensor_equal"
  c_THIntTensor_equal :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt

-- |c_THIntTensor_ltValue : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_ltValue"
  c_THIntTensor_ltValue :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_leValue : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_leValue"
  c_THIntTensor_leValue :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_gtValue : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_gtValue"
  c_THIntTensor_gtValue :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_geValue : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_geValue"
  c_THIntTensor_geValue :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_neValue : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_neValue"
  c_THIntTensor_neValue :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_eqValue : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_eqValue"
  c_THIntTensor_eqValue :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_ltValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_ltValueT"
  c_THIntTensor_ltValueT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_leValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_leValueT"
  c_THIntTensor_leValueT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_gtValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_gtValueT"
  c_THIntTensor_gtValueT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_geValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_geValueT"
  c_THIntTensor_geValueT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_neValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_neValueT"
  c_THIntTensor_neValueT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_eqValueT : r_ t value -> void
foreign import ccall "THTensorMath.h THIntTensor_eqValueT"
  c_THIntTensor_eqValueT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CInt -> IO ()

-- |c_THIntTensor_ltTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_ltTensor"
  c_THIntTensor_ltTensor :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_leTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_leTensor"
  c_THIntTensor_leTensor :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_gtTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_gtTensor"
  c_THIntTensor_gtTensor :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_geTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_geTensor"
  c_THIntTensor_geTensor :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_neTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_neTensor"
  c_THIntTensor_neTensor :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_eqTensor : r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_eqTensor"
  c_THIntTensor_eqTensor :: Ptr CTHByteTensor -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_ltTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_ltTensorT"
  c_THIntTensor_ltTensorT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_leTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_leTensorT"
  c_THIntTensor_leTensorT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_gtTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_gtTensorT"
  c_THIntTensor_gtTensorT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_geTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_geTensorT"
  c_THIntTensor_geTensorT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_neTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_neTensorT"
  c_THIntTensor_neTensorT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_eqTensorT : r_ ta tb -> void
foreign import ccall "THTensorMath.h THIntTensor_eqTensorT"
  c_THIntTensor_eqTensorT :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_abs : r_ t -> void
foreign import ccall "THTensorMath.h THIntTensor_abs"
  c_THIntTensor_abs :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()