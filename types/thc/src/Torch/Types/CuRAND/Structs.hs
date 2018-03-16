-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Types.CuRAND.Structs
-- Copyright :  (c) Hasktorch devs 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- 
-- cuRAND structs. Only what THC relies on. look in your
-- /usr/local/cuda-9.0/include/curand_mtgp32.h
-------------------------------------------------------------------------------
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Types.CuRAND.Structs where

import Control.Monad
import Data.Int
import Data.Word
import Data.Sequence
import Data.Foldable (toList)

import Foreign
import Foreign.Ptr (wordPtrToPtr, castPtrToFunPtr)
import Foreign.Storable
import Foreign.C.Types
import Foreign.C.String (CString,CStringLen,CWString,CWStringLen)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Marshal.Array (peekArray,pokeArray)

-- #define MTGPDC_MEXP 11213
mtgpdc_mexp = 11213

-- #define MTGPDC_N 351
mtgpdc_n = 351

-- #define MTGPDC_FLOOR_2P 256
mtgpdc_floor_2p = 256

-- #define MTGPDC_CEIL_2P 512
mtgpdc_ceil_2p = 512

-- #define MTGPDC_PARAM_TABLE mtgp32dc_params_fast_11213
-- mtgpdc_param_table = mtgp32dc_params_fast_11213

-- #define MTGP32_STATE_SIZE 1024
mtgp32_state_size = 1024

-- #define MTGP32_STATE_MASK 1023
mtgp32_state_mask = 1023

-- #define CURAND_NUM_MTGP32_PARAMS 200
curand_num_mtgp32_params :: Int
curand_num_mtgp32_params = 200

-- #define MEXP 11213
mexp = 11213

-- #define THREAD_NUM MTGPDC_FLOOR_2P
thread_num = mtgpdc_floor_2p

-- #define LARGE_SIZE (THREAD_NUM * 3)
large_size = thread_num * 3

-- #define BLOCK_NUM_MAX CURAND_NUM_MTGP32_PARAMS
block_num_max = curand_num_mtgp32_params

-- #define TBL_SIZE 16
tbl_size = 16


-- /*
--  * Generator Parameters.
--  */
-- struct mtgp32_kernel_params;
-- struct mtgp32_kernel_params {
--     unsigned int pos_tbl[CURAND_NUM_MTGP32_PARAMS];
--     unsigned int param_tbl[CURAND_NUM_MTGP32_PARAMS][TBL_SIZE];
--     unsigned int temper_tbl[CURAND_NUM_MTGP32_PARAMS][TBL_SIZE];
--     unsigned int single_temper_tbl[CURAND_NUM_MTGP32_PARAMS][TBL_SIZE];
--     unsigned int sh1_tbl[CURAND_NUM_MTGP32_PARAMS];
--     unsigned int sh2_tbl[CURAND_NUM_MTGP32_PARAMS];
--     unsigned int mask[1];
-- };
--
-- /** \cond UNHIDE_TYPEDEFS */
-- typedef struct mtgp32_kernel_params mtgp32_kernel_params_t;
-- /** \endcond */
type CMtgp32_kernel_params = C'Mtgp32_kernel_params

data C'Mtgp32_kernel_params = C'Mtgp32_kernel_params
  { c'Mtgp32_kernel_params'pos_tbl :: [CUInt]
  , c'Mtgp32_kernel_params'param_tbl :: [[CUInt]]
  , c'Mtgp32_kernel_params'temper_tbl :: [[CUInt]]
  , c'Mtgp32_kernel_params'single_temper_tbl :: [[CUInt]]
  , c'Mtgp32_kernel_params'sh1_tbl :: [CUInt]
  , c'Mtgp32_kernel_params'sh2_tbl :: [CUInt]
  , c'Mtgp32_kernel_params'mask :: [CUInt]
  } deriving (Eq,Show)

sizeOfCUInt      = sizeOf (undefined :: CUInt)
cuintMtg32Params = sizeOfCUInt * curand_num_mtgp32_params
cuintMtg32Tbl    = cuintMtg32Params * tbl_size

size'Mtgp32_kernel_params'pos_tbl           = cuintMtg32Params
size'Mtgp32_kernel_params'param_tbl         = cuintMtg32Tbl
size'Mtgp32_kernel_params'temper_tbl        = cuintMtg32Tbl
size'Mtgp32_kernel_params'single_temper_tbl = cuintMtg32Tbl
size'Mtgp32_kernel_params'sh1_tbl           = cuintMtg32Params
size'Mtgp32_kernel_params'sh2_tbl           = cuintMtg32Params
size'Mtgp32_kernel_params'mask              = sizeOfCUInt
size'Mtgp32_kernel_params = sum
  [ size'Mtgp32_kernel_params'pos_tbl
  , size'Mtgp32_kernel_params'param_tbl
  , size'Mtgp32_kernel_params'temper_tbl
  , size'Mtgp32_kernel_params'single_temper_tbl
  , size'Mtgp32_kernel_params'sh1_tbl
  , size'Mtgp32_kernel_params'sh2_tbl
  , size'Mtgp32_kernel_params'mask
  ]

-- addr for "address of". Each is dependent on the initial position and size of the prior record-field
addr'Mtgp32_kernel_params'pos_tbl           = 0
addr'Mtgp32_kernel_params'param_tbl         = addr'Mtgp32_kernel_params'pos_tbl           + size'Mtgp32_kernel_params'pos_tbl
addr'Mtgp32_kernel_params'temper_tbl        = addr'Mtgp32_kernel_params'param_tbl         + size'Mtgp32_kernel_params'param_tbl
addr'Mtgp32_kernel_params'single_temper_tbl = addr'Mtgp32_kernel_params'temper_tbl        + size'Mtgp32_kernel_params'temper_tbl
addr'Mtgp32_kernel_params'sh1_tbl           = addr'Mtgp32_kernel_params'single_temper_tbl + size'Mtgp32_kernel_params'single_temper_tbl
addr'Mtgp32_kernel_params'sh2_tbl           = addr'Mtgp32_kernel_params'sh1_tbl           + size'Mtgp32_kernel_params'sh1_tbl
addr'Mtgp32_kernel_params'mask              = addr'Mtgp32_kernel_params'sh2_tbl           + size'Mtgp32_kernel_params'sh2_tbl

-- access pointers to each storable item
p'Mtgp32_kernel_params'pos_tbl           :: Ptr C'Mtgp32_kernel_params -> Ptr CUInt
p'Mtgp32_kernel_params'param_tbl         :: Ptr C'Mtgp32_kernel_params -> Ptr (Ptr CUInt)
p'Mtgp32_kernel_params'temper_tbl        :: Ptr C'Mtgp32_kernel_params -> Ptr (Ptr CUInt)
p'Mtgp32_kernel_params'single_temper_tbl :: Ptr C'Mtgp32_kernel_params -> Ptr (Ptr CUInt)
p'Mtgp32_kernel_params'sh1_tbl           :: Ptr C'Mtgp32_kernel_params -> Ptr CUInt
p'Mtgp32_kernel_params'sh2_tbl           :: Ptr C'Mtgp32_kernel_params -> Ptr CUInt
p'Mtgp32_kernel_params'mask              :: Ptr C'Mtgp32_kernel_params -> Ptr CUInt
p'Mtgp32_kernel_params'pos_tbl           p = plusPtr p addr'Mtgp32_kernel_params'pos_tbl
p'Mtgp32_kernel_params'param_tbl         p = plusPtr p addr'Mtgp32_kernel_params'param_tbl
p'Mtgp32_kernel_params'temper_tbl        p = plusPtr p addr'Mtgp32_kernel_params'temper_tbl
p'Mtgp32_kernel_params'single_temper_tbl p = plusPtr p addr'Mtgp32_kernel_params'single_temper_tbl
p'Mtgp32_kernel_params'sh1_tbl           p = plusPtr p addr'Mtgp32_kernel_params'sh1_tbl
p'Mtgp32_kernel_params'sh2_tbl           p = plusPtr p addr'Mtgp32_kernel_params'sh2_tbl
p'Mtgp32_kernel_params'mask              p = plusPtr p addr'Mtgp32_kernel_params'mask

instance Storable C'Mtgp32_kernel_params where
  sizeOf    _ = size'Mtgp32_kernel_params
  alignment _ = sizeOfCUInt -- see Storable haddocks: "An alignment constraint x is fulfilled by any address divisible by x."
  peek _p = do
    v0 :: [CUInt]   <- peekArray  numParams (p'Mtgp32_kernel_params'pos_tbl _p)
    v1 :: [[CUInt]] <- peekMatrix numParams tbl_size (p'Mtgp32_kernel_params'param_tbl         _p)
    v2 :: [[CUInt]] <- peekMatrix numParams tbl_size (p'Mtgp32_kernel_params'temper_tbl        _p)
    v3 :: [[CUInt]] <- peekMatrix numParams tbl_size (p'Mtgp32_kernel_params'single_temper_tbl _p)
    v4 :: [CUInt]   <- peekArray  numParams (p'Mtgp32_kernel_params'sh1_tbl _p)
    v5 :: [CUInt]   <- peekArray  numParams (p'Mtgp32_kernel_params'sh2_tbl _p)
    v6              <- peekArray         1  (p'Mtgp32_kernel_params'mask    _p)
    pure $ C'Mtgp32_kernel_params v0 v1 v2 v3 v4 v5 v6
   where
    numParams = curand_num_mtgp32_params
    tblSize = tbl_size

  poke _p (C'Mtgp32_kernel_params v0 v1 v2 v3 v4 v5 v6) = do
    pokeArray  (p'Mtgp32_kernel_params'pos_tbl           _p) v0
    pokeMatrix (p'Mtgp32_kernel_params'param_tbl         _p) v1
    pokeMatrix (p'Mtgp32_kernel_params'temper_tbl        _p) v2
    pokeMatrix (p'Mtgp32_kernel_params'single_temper_tbl _p) v3
    pokeArray  (p'Mtgp32_kernel_params'sh1_tbl           _p) v4
    pokeArray  (p'Mtgp32_kernel_params'sh2_tbl           _p) v5
    pokeArray  (p'Mtgp32_kernel_params'mask              _p) v6


peekMatrix :: forall a . Storable a => Int -> Int -> Ptr (Ptr a) -> IO [[a]]
peekMatrix nr nc pps = do
  cols :: [Ptr a] <- peekArray nr pps
  mapM (peekArray nc) cols

pokeMatrix :: forall a . Storable a => Ptr (Ptr a) -> [[a]] -> IO ()
pokeMatrix pps mat = zipPtrs mempty pps mat >>= mapM_ go
  where
    go :: (Ptr a, [a]) -> IO ()
    go (ptr, col) = pokeArray ptr col

    zipPtrs :: Seq (Ptr a, [a]) -> Ptr (Ptr a) -> [[a]] -> IO [(Ptr a, [a])]
    zipPtrs acc   _     [] = pure (toList acc)
    zipPtrs acc pIx (r:rs) = do
      rp <- peek pIx
      zipPtrs (acc |> (rp, r)) (advancePtr pIx 1) rs


-- /*
--  * kernel I/O
--  * This structure must be initialized before first use.
--  */
--
-- /* MTGP (Mersenne Twister) RNG */
-- /* This generator uses the Mersenne Twister algorithm of
--  * http://arxiv.org/abs/1005.4973v2
--  * Has period 2^11213.
-- */
--
-- /**
--  * CURAND MTGP32 state
--  */
-- struct curandStateMtgp32;
--
-- struct curandStateMtgp32 {
--     unsigned int s[MTGP32_STATE_SIZE];
--     int offset;
--     int pIdx;
--     mtgp32_kernel_params_t * k;
--     int precise_double_flag;
-- };
--
-- /*
--  * CURAND MTGP32 state
--  */
-- /** \cond UNHIDE_TYPEDEFS */
-- typedef struct curandStateMtgp32 curandStateMtgp32_t;
-- /** \endcond */
type CCurandStateMtgp32 = C'CurandStateMtgp32

data C'CurandStateMtgp32 = C'CurandStateMtgp32
  { c'CurandStateMtgp32's      :: [CUInt]
  , c'CurandStateMtgp32'offset :: CInt
  , c'CurandStateMtgp32'pIdx   :: CInt
  , c'CurandStateMtgp32'k      :: Ptr C'Mtgp32_kernel_params
  , c'CurandStateMtgp32'precise_double_flag :: CInt
  } deriving (Eq,Show)

sizeOfCInt = sizeOf (undefined :: CInt)

size'CurandStateMtgp32's                   = sizeOfCUInt * mtgp32_state_size
size'CurandStateMtgp32'offset              = sizeOfCInt
size'CurandStateMtgp32'pIdx                = sizeOfCInt
size'CurandStateMtgp32'k                   = sizeOf (undefined :: Ptr C'Mtgp32_kernel_params)
size'CurandStateMtgp32'precise_double_flag = sizeOfCInt
size'CurandStateMtgp32 = sum
  [ size'CurandStateMtgp32's
  , size'CurandStateMtgp32'offset
  , size'CurandStateMtgp32'pIdx
  , size'CurandStateMtgp32'k
  , size'CurandStateMtgp32'precise_double_flag
  ]

-- addr for "address of". Each is dependent on the initial position and size of the prior record-field
addr'CurandStateMtgp32's                   = 0
addr'CurandStateMtgp32'offset              = addr'CurandStateMtgp32's      + size'CurandStateMtgp32's
addr'CurandStateMtgp32'pIdx                = addr'CurandStateMtgp32'offset + size'CurandStateMtgp32'offset
addr'CurandStateMtgp32'k                   = addr'CurandStateMtgp32'pIdx   + size'CurandStateMtgp32'pIdx
addr'CurandStateMtgp32'precise_double_flag = addr'CurandStateMtgp32'k      + size'CurandStateMtgp32'k

-- access pointers to each storable item
p'CurandStateMtgp32's                   :: Ptr C'CurandStateMtgp32 -> Ptr CUInt
p'CurandStateMtgp32'offset              :: Ptr C'CurandStateMtgp32 -> Ptr CInt
p'CurandStateMtgp32'pIdx                :: Ptr C'CurandStateMtgp32 -> Ptr CInt
p'CurandStateMtgp32'k                   :: Ptr C'CurandStateMtgp32 -> Ptr C'Mtgp32_kernel_params
p'CurandStateMtgp32'precise_double_flag :: Ptr C'CurandStateMtgp32 -> Ptr CInt
p'CurandStateMtgp32's                   p = plusPtr p addr'CurandStateMtgp32's
p'CurandStateMtgp32'offset              p = plusPtr p addr'CurandStateMtgp32'offset
p'CurandStateMtgp32'pIdx                p = plusPtr p addr'CurandStateMtgp32'pIdx
p'CurandStateMtgp32'k                   p = plusPtr p addr'CurandStateMtgp32'k
p'CurandStateMtgp32'precise_double_flag p = plusPtr p addr'CurandStateMtgp32'precise_double_flag

instance Storable C'CurandStateMtgp32 where
  sizeOf    _ = size'Mtgp32_kernel_params
  alignment _ = sizeOfCUInt -- see Storable haddocks: "An alignment constraint x is fulfilled by any address divisible by x."
  peek _p = C'CurandStateMtgp32
    <$> peekArray mtgp32_state_size (p'CurandStateMtgp32's _p)
    <*> peekByteOff _p addr'CurandStateMtgp32'offset
    <*> peekByteOff _p addr'CurandStateMtgp32'pIdx
    <*> peekByteOff _p addr'CurandStateMtgp32'k
    <*> peekByteOff _p addr'CurandStateMtgp32'precise_double_flag

  poke _p (C'CurandStateMtgp32 v0 v1 v2 v3 v4) = do
    pokeArray (p'CurandStateMtgp32's _p) v0
    pokeByteOff _p addr'CurandStateMtgp32'offset v1
    pokeByteOff _p addr'CurandStateMtgp32'pIdx   v2
    pokeByteOff _p addr'CurandStateMtgp32'k      v3
    pokeByteOff _p addr'CurandStateMtgp32'precise_double_flag v4

