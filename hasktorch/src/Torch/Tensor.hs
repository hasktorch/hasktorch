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

debugPrint :: Tensor -> IO ()
debugPrint = cast1 ATen.tensor_print

numel :: Tensor -> Int
numel t = unsafePerformIO $ cast1 ATen.tensor_numel $ t

select :: Tensor -> Int -> Int -> Tensor
select t dim idx = unsafePerformIO $ (cast3 ATen.tensor_select_ll) t dim idx

size :: Tensor -> Int -> Int
size t dim = unsafePerformIO $ (cast2 ATen.tensor_size_l) t dim

shape :: Tensor -> [Int]
shape t = unsafePerformIO $ (cast1 ATen.tensor_sizes) t

dim :: Tensor -> Int
dim t = unsafePerformIO $ (cast1 ATen.tensor_dim) t

asDouble :: Tensor -> Double
asDouble t = unsafePerformIO $ cast1 ATen.tensor_item_double $ t

asInt :: Tensor -> Int
asInt t = unsafePerformIO $ cast1 ATen.tensor_item_int64_t $ t



class TensorIndex a where
  (@@) :: Tensor -> a -> Tensor

instance TensorIndex Int where
  t @@ idx = select t 0 idx

instance TensorIndex Integer where
  t @@ idx = select t 0 $ fromIntegral idx

-- TODO: advanced indexing

instance (TensorIndex a, TensorIndex b) => TensorIndex (a,b) where
  t @@ (a, b) = (t @@ a) @@ b

instance (TensorIndex a, TensorIndex b, TensorIndex c) => TensorIndex (a,b,c) where
  t @@ (a, b, c) = (t @@ (a, b)) @@ c

instance (TensorIndex a, TensorIndex b, TensorIndex c, TensorIndex d) => TensorIndex (a,b,c,d) where
  t @@ (a, b, c, d) = (t @@ (a, b, c)) @@ c


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
  asTensor v = mkScalarTensor @Float (realToFrac v) float_opts


instance Show Tensor where
  show t = case (dim t) of
      0 -> show0d t
      1 -> show1d t
      2 -> show2d t
      _ -> "Tensor"
    where
      -- TODO: this is obviously not the right way to do it,
      -- and will be terribly slow, so please fix it.
      showElems elemShow sep t = "[" ++ (intercalate sep $ map elemShow [t @@ i | i <- [0..((size t 0) - 1)]]) ++ "]"
      show0d = show . asDouble
      show1d = showElems show0d ", "
      show2d = showElems show1d ",\n "
