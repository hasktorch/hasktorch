{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeApplications #-}

module Torch.Tensor where

import Foreign.ForeignPtr
import Foreign.Ptr
import Foreign.Storable
import System.IO.Unsafe
import Data.Int (Int64)
import Data.List (intercalate)
import Numeric

import ATen.Cast
import ATen.Class (Castable(..))
import qualified ATen.Managed.Type.Tensor as ATen
import qualified ATen.Managed.Type.TensorOptions as ATen
import qualified ATen.Managed.Native as ATen
import qualified ATen.Managed.Cast as ATen
import qualified ATen.Type as ATen
import qualified ATen.Const as ATen
import qualified Torch.Managed.Native as LibTorch

import Torch.DType
import Torch.TensorOptions


type ATenTensor = ForeignPtr ATen.Tensor

data Tensor = Tensor ATenTensor

instance Castable Tensor ATenTensor where
  cast (Tensor aten_tensor) f = f aten_tensor
  uncast aten_tensor f = f $ Tensor aten_tensor

--------------------------------------------------------------------------------
-- Basic tensor properties
--------------------------------------------------------------------------------

numel :: Tensor -> Int
numel t = unsafePerformIO $ cast1 ATen.tensor_numel $ t

size :: Tensor -> Int -> Int
size t dim = unsafePerformIO $ (cast2 ATen.tensor_size_l) t dim

shape :: Tensor -> [Int]
shape t = unsafePerformIO $ (cast1 ATen.tensor_sizes) t

dim :: Tensor -> Int
dim t = unsafePerformIO $ (cast1 ATen.tensor_dim) t

dtype :: Tensor -> DType
dtype t = unsafePerformIO $ (cast1 ATen.tensor_scalar_type) t

select :: Tensor -> Int -> Int -> Tensor
select t dim idx = unsafePerformIO $ (cast3 ATen.tensor_select_ll) t dim idx

toDouble :: Tensor -> Double
toDouble t = unsafePerformIO $ cast1 ATen.tensor_item_double $ t

toInt :: Tensor -> Int
toInt t = unsafePerformIO $ cast1 ATen.tensor_item_int64_t $ t

select :: Tensor -> Int -> Int -> Tensor
select t dim idx = unsafePerformIO $ (cast3 ATen.tensor_select_ll) t dim idx

reshape :: Tensor -> [Int] -> Tensor
reshape t shape = unsafePerformIO $ (cast2 ATen.reshape_tl) t shape
--------------------------------------------------------------------------------
-- Indexing support
--------------------------------------------------------------------------------

(@@) :: TensorIndex a => Tensor -> a -> Tensor
t @@ idx = fst $ indexWith (t, 0) idx

class TensorIndex a where
  indexWith :: (Tensor, Int) -> a -> (Tensor, Int)

instance TensorIndex Int where
  indexWith (t, offset) idx = (select t offset idx, offset)

instance TensorIndex Integer where
  indexWith (t, offset) idx = (select t offset (fromIntegral idx), offset)

instance TensorIndex () where
  indexWith (t, offset) idx = (t, offset + 1)

instance (TensorIndex a, TensorIndex b) => TensorIndex (a,b) where
  indexWith toff (a, b) = indexWith (indexWith toff a) b

instance (TensorIndex a, TensorIndex b, TensorIndex c) => TensorIndex (a,b,c) where
  indexWith toff (a, b, c) = indexWith (indexWith (indexWith toff a) b) c

instance (TensorIndex a, TensorIndex b, TensorIndex c, TensorIndex d) => TensorIndex (a,b,c,d) where
  indexWith toff (a, b, c, d) = indexWith (indexWith (indexWith (indexWith toff a) b) c) d

--------------------------------------------------------------------------------
-- Scalar -> Tensor promotion
--------------------------------------------------------------------------------

class TensorLike a where
  asTensor :: a -> Tensor

mkScalarTensor :: Storable a => a -> TensorOptions -> Tensor
mkScalarTensor v opts = unsafePerformIO $ do
  t <- ((cast2 LibTorch.empty_lo) :: [Int] -> TensorOptions -> IO Tensor) [] opts
  ptr <- ((cast1 ATen.tensor_data_ptr) :: Tensor -> IO (Ptr ())) t
  poke (castPtr ptr) v
  return t

int64_opts = withDType Int64 defaultOpts
float_opts = withDType Float defaultOpts

instance TensorLike Integer where
  asTensor v = mkScalarTensor @Int64 (fromIntegral v) int64_opts

instance TensorLike Double where
  -- XXX: This implicit cast to float is very meh, but I don't have any better ideas
  --      This is our default dtype, so it's the most convenient thing we can do
  asTensor v = mkScalarTensor @Float (realToFrac v) float_opts

--------------------------------------------------------------------------------
-- Show
--------------------------------------------------------------------------------

instance Show Tensor where
  show t = case (dim t) of
      0 -> details ++ show0d t
      1 -> details ++ show1d t
      2 -> details ++ show2d t
      _ -> details
    where
      -- TODO: this is obviously not the right way to do it,
      -- and will be terribly slow, so please fix it.
      showElems elemShow sep t = "[" ++ (intercalate sep $ map elemShow [t @@ i | i <- [0..((size t 0) - 1)]]) ++ "]"
      padPositive x s = if x >= 0 then " " ++ s else s
      -- TODO: this assumes that scientific notation only uses one-digit exponents, which is not
      --       true in general
      padLarge x s = if (abs x) >= 0.1 then s ++ "   " else s
      show0d x = if isIntegral (dtype t)
                  then padPositive (toInt x) $ show $ toInt x
                  else padLarge (toDouble x) $ padPositive (toDouble x) $ showGFloat (Just 4) (toDouble x) ""
      show1d = showElems show0d ", "
      show2d = showElems show1d (",\n " ++ padding)
      details = "Tensor " ++ (show $ dtype t) ++ " " ++ (show $ shape t) ++ " "
      padding = map (const ' ') details
