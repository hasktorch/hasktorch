{-# OPTIONS_GHC -fno-warn-unused-imports #-}


module Torch.Types.CuRand.Structs where
import Foreign.Ptr
import Foreign.Ptr (Ptr,FunPtr,plusPtr)
import Foreign.Ptr (wordPtrToPtr,castPtrToFunPtr)
import Foreign.Storable
import Foreign.C.Types
import Foreign.C.String (CString,CStringLen,CWString,CWStringLen)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Marshal.Array (peekArray,pokeArray)
import Data.Int
import Data.Word


{- struct mtgp32_params_fast {
    int mexp;
    int pos;
    int sh1;
    int sh2;
    unsigned int tbl[16];
    unsigned int tmp_tbl[16];
    unsigned int flt_tmp_tbl[16];
    unsigned int mask;
    unsigned char poly_sha1[21];
}; -}










data C'mtgp32_params_fast = C'mtgp32_params_fast{
  c'mtgp32_params_fast'mexp :: CInt,
  c'mtgp32_params_fast'pos :: CInt,
  c'mtgp32_params_fast'sh1 :: CInt,
  c'mtgp32_params_fast'sh2 :: CInt,
  c'mtgp32_params_fast'tbl :: [CUInt],
  c'mtgp32_params_fast'tmp_tbl :: [CUInt],
  c'mtgp32_params_fast'flt_tmp_tbl :: [CUInt],
  c'mtgp32_params_fast'mask :: CUInt,
  c'mtgp32_params_fast'poly_sha1 :: [CUChar]
} deriving (Eq,Show)
p'mtgp32_params_fast'mexp p = plusPtr p 0
p'mtgp32_params_fast'mexp :: Ptr (C'mtgp32_params_fast) -> Ptr (CInt)
p'mtgp32_params_fast'pos p = plusPtr p 4
p'mtgp32_params_fast'pos :: Ptr (C'mtgp32_params_fast) -> Ptr (CInt)
p'mtgp32_params_fast'sh1 p = plusPtr p 8
p'mtgp32_params_fast'sh1 :: Ptr (C'mtgp32_params_fast) -> Ptr (CInt)
p'mtgp32_params_fast'sh2 p = plusPtr p 12
p'mtgp32_params_fast'sh2 :: Ptr (C'mtgp32_params_fast) -> Ptr (CInt)
p'mtgp32_params_fast'tbl p = plusPtr p 16
p'mtgp32_params_fast'tbl :: Ptr (C'mtgp32_params_fast) -> Ptr (CUInt)
p'mtgp32_params_fast'tmp_tbl p = plusPtr p 80
p'mtgp32_params_fast'tmp_tbl :: Ptr (C'mtgp32_params_fast) -> Ptr (CUInt)
p'mtgp32_params_fast'flt_tmp_tbl p = plusPtr p 144
p'mtgp32_params_fast'flt_tmp_tbl :: Ptr (C'mtgp32_params_fast) -> Ptr (CUInt)
p'mtgp32_params_fast'mask p = plusPtr p 208
p'mtgp32_params_fast'mask :: Ptr (C'mtgp32_params_fast) -> Ptr (CUInt)
p'mtgp32_params_fast'poly_sha1 p = plusPtr p 212
p'mtgp32_params_fast'poly_sha1 :: Ptr (C'mtgp32_params_fast) -> Ptr (CUChar)
instance Storable C'mtgp32_params_fast where
  sizeOf _ = 236
  alignment _ = 4
  peek _p = do
    v0 <- peekByteOff _p 0
    v1 <- peekByteOff _p 4
    v2 <- peekByteOff _p 8
    v3 <- peekByteOff _p 12
    v4 <- let s4 = div 64 $ sizeOf $ (undefined :: CUInt) in peekArray s4 (plusPtr _p 16)
    v5 <- let s5 = div 64 $ sizeOf $ (undefined :: CUInt) in peekArray s5 (plusPtr _p 80)
    v6 <- let s6 = div 64 $ sizeOf $ (undefined :: CUInt) in peekArray s6 (plusPtr _p 144)
    v7 <- peekByteOff _p 208
    v8 <- let s8 = div 21 $ sizeOf $ (undefined :: CUChar) in peekArray s8 (plusPtr _p 212)
    return $ C'mtgp32_params_fast v0 v1 v2 v3 v4 v5 v6 v7 v8
  poke _p (C'mtgp32_params_fast v0 v1 v2 v3 v4 v5 v6 v7 v8) = do
    pokeByteOff _p 0 v0
    pokeByteOff _p 4 v1
    pokeByteOff _p 8 v2
    pokeByteOff _p 12 v3
    let s4 = div 64 $ sizeOf $ (undefined :: CUInt)
    pokeArray (plusPtr _p 16) (take s4 v4)
    let s5 = div 64 $ sizeOf $ (undefined :: CUInt)
    pokeArray (plusPtr _p 80) (take s5 v5)
    let s6 = div 64 $ sizeOf $ (undefined :: CUInt)
    pokeArray (plusPtr _p 144) (take s6 v6)
    pokeByteOff _p 208 v7
    let s8 = div 21 $ sizeOf $ (undefined :: CUChar)
    pokeArray (plusPtr _p 212) (take s8 v8)
    return ()

{- struct mtgp32_kernel_params {
    unsigned int pos_tbl[200];
    unsigned int param_tbl[200][16];
    unsigned int temper_tbl[200][16];
    unsigned int single_temper_tbl[200][16];
    unsigned int sh1_tbl[200];
    unsigned int sh2_tbl[200];
    unsigned int mask[1];
}; -}








data C'mtgp32_kernel_params = C'mtgp32_kernel_params{
  c'mtgp32_kernel_params'pos_tbl :: [CUInt],
  c'mtgp32_kernel_params'param_tbl :: [CUInt],
  c'mtgp32_kernel_params'temper_tbl :: [CUInt],
  c'mtgp32_kernel_params'single_temper_tbl :: [CUInt],
  c'mtgp32_kernel_params'sh1_tbl :: [CUInt],
  c'mtgp32_kernel_params'sh2_tbl :: [CUInt],
  c'mtgp32_kernel_params'mask :: [CUInt]
} deriving (Eq,Show)
p'mtgp32_kernel_params'pos_tbl p = plusPtr p 0
p'mtgp32_kernel_params'pos_tbl :: Ptr (C'mtgp32_kernel_params) -> Ptr (CUInt)
p'mtgp32_kernel_params'param_tbl p = plusPtr p 800
p'mtgp32_kernel_params'param_tbl :: Ptr (C'mtgp32_kernel_params) -> Ptr (CUInt)
p'mtgp32_kernel_params'temper_tbl p = plusPtr p 13600
p'mtgp32_kernel_params'temper_tbl :: Ptr (C'mtgp32_kernel_params) -> Ptr (CUInt)
p'mtgp32_kernel_params'single_temper_tbl p = plusPtr p 26400
p'mtgp32_kernel_params'single_temper_tbl :: Ptr (C'mtgp32_kernel_params) -> Ptr (CUInt)
p'mtgp32_kernel_params'sh1_tbl p = plusPtr p 39200
p'mtgp32_kernel_params'sh1_tbl :: Ptr (C'mtgp32_kernel_params) -> Ptr (CUInt)
p'mtgp32_kernel_params'sh2_tbl p = plusPtr p 40000
p'mtgp32_kernel_params'sh2_tbl :: Ptr (C'mtgp32_kernel_params) -> Ptr (CUInt)
p'mtgp32_kernel_params'mask p = plusPtr p 40800
p'mtgp32_kernel_params'mask :: Ptr (C'mtgp32_kernel_params) -> Ptr (CUInt)
instance Storable C'mtgp32_kernel_params where
  sizeOf _ = 40804
  alignment _ = 4
  peek _p = do
    v0 <- let s0 = div 800 $ sizeOf $ (undefined :: CUInt) in peekArray s0 (plusPtr _p 0)
    v1 <- let s1 = div 12800 $ sizeOf $ (undefined :: CUInt) in peekArray s1 (plusPtr _p 800)
    v2 <- let s2 = div 12800 $ sizeOf $ (undefined :: CUInt) in peekArray s2 (plusPtr _p 13600)
    v3 <- let s3 = div 12800 $ sizeOf $ (undefined :: CUInt) in peekArray s3 (plusPtr _p 26400)
    v4 <- let s4 = div 800 $ sizeOf $ (undefined :: CUInt) in peekArray s4 (plusPtr _p 39200)
    v5 <- let s5 = div 800 $ sizeOf $ (undefined :: CUInt) in peekArray s5 (plusPtr _p 40000)
    v6 <- let s6 = div 4 $ sizeOf $ (undefined :: CUInt) in peekArray s6 (plusPtr _p 40800)
    return $ C'mtgp32_kernel_params v0 v1 v2 v3 v4 v5 v6
  poke _p (C'mtgp32_kernel_params v0 v1 v2 v3 v4 v5 v6) = do
    let s0 = div 800 $ sizeOf $ (undefined :: CUInt)
    pokeArray (plusPtr _p 0) (take s0 v0)
    let s1 = div 12800 $ sizeOf $ (undefined :: CUInt)
    pokeArray (plusPtr _p 800) (take s1 v1)
    let s2 = div 12800 $ sizeOf $ (undefined :: CUInt)
    pokeArray (plusPtr _p 13600) (take s2 v2)
    let s3 = div 12800 $ sizeOf $ (undefined :: CUInt)
    pokeArray (plusPtr _p 26400) (take s3 v3)
    let s4 = div 800 $ sizeOf $ (undefined :: CUInt)
    pokeArray (plusPtr _p 39200) (take s4 v4)
    let s5 = div 800 $ sizeOf $ (undefined :: CUInt)
    pokeArray (plusPtr _p 40000) (take s5 v5)
    let s6 = div 4 $ sizeOf $ (undefined :: CUInt)
    pokeArray (plusPtr _p 40800) (take s6 v6)
    return ()

{- struct curandStateMtgp32 {
    unsigned int s[1024];
    int offset;
    int pIdx;
    struct mtgp32_kernel_params * k;
    int precise_double_flag;
}; -}






data C'curandStateMtgp32 = C'curandStateMtgp32{
  c'curandStateMtgp32's :: [CUInt],
  c'curandStateMtgp32'offset :: CInt,
  c'curandStateMtgp32'pIdx :: CInt,
  c'curandStateMtgp32'k :: Ptr C'mtgp32_kernel_params,
  c'curandStateMtgp32'precise_double_flag :: CInt
} deriving (Eq,Show)
p'curandStateMtgp32's p = plusPtr p 0
p'curandStateMtgp32's :: Ptr (C'curandStateMtgp32) -> Ptr (CUInt)
p'curandStateMtgp32'offset p = plusPtr p 4096
p'curandStateMtgp32'offset :: Ptr (C'curandStateMtgp32) -> Ptr (CInt)
p'curandStateMtgp32'pIdx p = plusPtr p 4100
p'curandStateMtgp32'pIdx :: Ptr (C'curandStateMtgp32) -> Ptr (CInt)
p'curandStateMtgp32'k p = plusPtr p 4104
p'curandStateMtgp32'k :: Ptr (C'curandStateMtgp32) -> Ptr (Ptr C'mtgp32_kernel_params)
p'curandStateMtgp32'precise_double_flag p = plusPtr p 4112
p'curandStateMtgp32'precise_double_flag :: Ptr (C'curandStateMtgp32) -> Ptr (CInt)
instance Storable C'curandStateMtgp32 where
  sizeOf _ = 4120
  alignment _ = 8
  peek _p = do
    v0 <- let s0 = div 4096 $ sizeOf $ (undefined :: CUInt) in peekArray s0 (plusPtr _p 0)
    v1 <- peekByteOff _p 4096
    v2 <- peekByteOff _p 4100
    v3 <- peekByteOff _p 4104
    v4 <- peekByteOff _p 4112
    return $ C'curandStateMtgp32 v0 v1 v2 v3 v4
  poke _p (C'curandStateMtgp32 v0 v1 v2 v3 v4) = do
    let s0 = div 4096 $ sizeOf $ (undefined :: CUInt)
    pokeArray (plusPtr _p 0) (take s0 v0)
    pokeByteOff _p 4096 v1
    pokeByteOff _p 4100 v2
    pokeByteOff _p 4104 v3
    pokeByteOff _p 4112 v4
    return ()

