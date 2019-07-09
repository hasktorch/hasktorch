{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Tensor where

import Control.Monad (forM_, forM)
import Foreign.ForeignPtr
import Foreign.Ptr
import Foreign.Storable
import Foreign.C.Types (CDouble, CFloat, CInt)
import System.IO.Unsafe
import Data.Int (Int64)
import Data.List (intercalate)
import Numeric

import ATen.Cast
import ATen.Class (Castable(..), CppTuple2(..), CppTuple3(..), CppTuple4(..), CppTuple5(..))
import qualified ATen.Managed.Type.Tensor as ATen
import qualified ATen.Managed.Type.TensorOptions as ATen
import qualified ATen.Managed.Type.Tuple as ATen
import qualified ATen.Managed.Native as ATen
import qualified ATen.Managed.Cast as ATen
import qualified ATen.Type as ATen
import qualified ATen.Const as ATen
import qualified Torch.Managed.Native as LibTorch

import Torch.DType
import Torch.TensorOptions


type ATenTensor = ForeignPtr ATen.Tensor

-- do not use the constructor
newtype Tensor = Unsafe ATenTensor

instance Castable Tensor ATenTensor where
  cast (Unsafe aten_tensor) f = f aten_tensor
  uncast aten_tensor f = f $ Unsafe aten_tensor

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

toDouble :: Tensor -> Double
toDouble t = unsafePerformIO $ cast1 ATen.tensor_item_double $ t

toInt :: Tensor -> Int
toInt t = unsafePerformIO $ cast1 ATen.tensor_item_int64_t $ t

select :: Tensor -> Int -> Int -> Tensor
select t dim idx = unsafePerformIO $ (cast3 ATen.tensor_select_ll) t dim idx

reshape :: Tensor -> [Int] -> Tensor
reshape t shape = unsafePerformIO $ (cast2 ATen.reshape_tl) t shape

--------------------------------------------------------------------------------
-- Move backend
--------------------------------------------------------------------------------

toSparse :: Tensor -> Tensor
toSparse t = unsafePerformIO $ (cast1 ATen.tensor_to_sparse) t

toDense :: Tensor -> Tensor
toDense t = unsafePerformIO $ (cast1 ATen.tensor_to_dense) t

toCPU :: Tensor -> Tensor
toCPU t = unsafePerformIO $ (cast1 ATen.tensor_cpu) t

toCUDA :: Tensor -> Tensor
toCUDA t = unsafePerformIO $ (cast1 ATen.tensor_cuda) t

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
-- Scalar <-> Tensor promotion
--------------------------------------------------------------------------------

class TensorLike a where
  asTensor :: a -> Tensor
  asValue :: Tensor -> a

mkScalarTensor :: Storable a => a -> TensorOptions -> Tensor
mkScalarTensor v opts = unsafePerformIO $ do
  t <- ((cast2 LibTorch.empty_lo) :: [Int] -> TensorOptions -> IO Tensor) [] opts
  ptr <- ((cast1 ATen.tensor_data_ptr) :: Tensor -> IO (Ptr ())) t
  poke (castPtr ptr) v
  return t

mkScalarValue :: Storable a => Tensor -> a
mkScalarValue t = unsafePerformIO $ do
  ptr <- ((cast1 ATen.tensor_data_ptr) :: Tensor -> IO (Ptr ())) t
  peek (castPtr ptr)

int64_opts = withDType Int64 defaultOpts
float_opts = withDType Float defaultOpts

instance TensorLike Integer where
  asTensor v = mkScalarTensor @Int64 (fromIntegral v) int64_opts
  asValue t = fromIntegral $ mkScalarValue @Int64 t

instance TensorLike Double where
  -- XXX: This implicit cast to float is very meh, but I don't have any better ideas
  --      This is our default dtype, so it's the most convenient thing we can do
  asTensor v = mkScalarTensor @Float (realToFrac v) float_opts
  asValue t = realToFrac $ mkScalarValue @Float t

--------------------------------------------------------------------------------
-- List <-> Tensor promotion
--------------------------------------------------------------------------------

mkTensor :: (Storable a) => [a] -> [Int] -> TensorOptions -> Tensor
mkTensor vs dims opts = unsafePerformIO $ do
  t <- ((cast2 LibTorch.empty_lo) :: [Int] -> TensorOptions -> IO Tensor) dims opts
  ptr <- ((cast1 ATen.tensor_data_ptr) :: Tensor -> IO (Ptr ())) t
  forM_ (zip [0..] vs) $ \(i,v) ->
    pokeElemOff (castPtr ptr) i v
  return t

mkValue :: (Storable a) => Tensor -> [a]
mkValue t = unsafePerformIO $ do
  let dims = shape t
  ptr <- ((cast1 ATen.tensor_data_ptr) :: Tensor -> IO (Ptr ())) t
  forM [0..((product dims)-1)] $ \i ->
    peekElemOff (castPtr ptr) i

split1d :: [Int] -> [a] -> [[a]]
split1d _ [] = []
split1d ns xs =
  let n = head $ reverse ns
      (x,y) = splitAt n xs
  in x:(split1d ns y)

split2d :: [Int] -> [a] -> [[[a]]]
split2d _ [] = []
split2d ns xs =
  let (n0:n1:_) = reverse ns
  in flip map (split1d [n0] xs) $ split1d [n1]

instance TensorLike [Integer] where
  asTensor v = mkTensor @Int64 (map fromIntegral v) [length v] int64_opts
  asValue t = map fromIntegral $ mkValue @Int64 t

instance TensorLike [Double] where
  asTensor v = mkTensor @Float (map realToFrac v) [length v] float_opts
  asValue t = map realToFrac $ mkValue @Float t

instance TensorLike [[Integer]] where
  asTensor v = mkTensor @Int64 (map fromIntegral (concat v)) [length v, length (head v)] int64_opts
  asValue t = split1d (shape t) $ map fromIntegral $ mkValue @Int64 t

instance TensorLike [[Double]] where
  asTensor v = mkTensor @Float (map realToFrac (concat v)) [length v, length (head v)] float_opts
  asValue t = split1d (shape t) $ map realToFrac $ mkValue @Float t

instance TensorLike [[[Integer]]] where
  asTensor v = mkTensor @Int64 (map fromIntegral (concat (concat v))) [length v, length (head v), length (head (head v))] int64_opts
  asValue t = split2d (shape t) $ map fromIntegral $ mkValue @Int64 t

instance TensorLike [[[Double]]] where
  asTensor v = mkTensor @Float (map realToFrac (concat (concat v))) [length v, length (head v), length (head (head v))] float_opts
  asValue t = split2d (shape t) $ map realToFrac $ mkValue @Float t

--------------------------------------------------------------------------------
-- Tuple -> ForeignPtr (ATen's Tuple)
--------------------------------------------------------------------------------

instance Castable (Tensor,Tensor) (ForeignPtr (ATen.Tensor,ATen.Tensor)) where
  cast _ _ = undefined
  uncast t f = do
    t0 <- get0 t
    t1 <- get1 t
    f (Unsafe t0, Unsafe t1)

instance Castable (Tensor,Tensor,Tensor) (ForeignPtr (ATen.Tensor,ATen.Tensor,ATen.Tensor)) where
  cast _ _ = undefined
  uncast t f = do
    t0 <- get0 t
    t1 <- get1 t
    t2 <- get2 t
    f (Unsafe t0, Unsafe t1, Unsafe t2)

instance Castable (Tensor,Tensor,Tensor,Tensor) (ForeignPtr (ATen.Tensor,ATen.Tensor,ATen.Tensor,ATen.Tensor)) where
  cast _ _ = undefined
  uncast t f = do
    t0 <- get0 t
    t1 <- get1 t
    t2 <- get2 t
    t3 <- get3 t
    f (Unsafe t0, Unsafe t1, Unsafe t2, Unsafe t3)

instance Castable (Tensor,Tensor,Tensor,[Tensor]) (ForeignPtr (ATen.Tensor,ATen.Tensor,ATen.Tensor,ATen.TensorList)) where
  cast _ _ = undefined
  uncast t f = do
    t0 <- get0 t
    t1 <- get1 t
    t2 <- get2 t
    xs <- get3 t
    uncast xs $ \ptr_list -> do
      tensor_list <- mapM (\(x :: ForeignPtr ATen.Tensor) -> uncast x return) ptr_list
      f (Unsafe t0, Unsafe t1, Unsafe t2, map Unsafe tensor_list)

instance Castable (Tensor,Tensor,Tensor,Tensor,Tensor) (ForeignPtr (ATen.Tensor,ATen.Tensor,ATen.Tensor,ATen.Tensor,ATen.Tensor)) where
  cast _ _ = undefined
  uncast t f = do
    t0 <- get0 t
    t1 <- get1 t
    t2 <- get2 t
    t3 <- get3 t
    t4 <- get4 t
    f (Unsafe t0, Unsafe t1, Unsafe t2, Unsafe t3, Unsafe t4)

instance Castable (Tensor,Tensor,Double,Int64) (ForeignPtr (ATen.Tensor,ATen.Tensor,CDouble,Int64)) where
  cast _ _ = undefined
  uncast t f = do
    t0 <- get0 t
    t1 <- get1 t
    t2 <- get2 t
    t3 <- get3 t
    f (Unsafe t0, Unsafe t1, realToFrac t2, t3)

instance Castable (Tensor,Tensor,Float,Int) (ForeignPtr (ATen.Tensor,ATen.Tensor,CFloat,CInt)) where
  cast _ _ = undefined
  uncast t f = do
    t0 <- get0 t
    t1 <- get1 t
    t2 <- get2 t
    t3 <- get3 t
    f (Unsafe t0, Unsafe t1, realToFrac t2, fromIntegral t3)

instance Castable (Tensor,Tensor,Tensor,Int64) (ForeignPtr (ATen.Tensor,ATen.Tensor,ATen.Tensor,Int64)) where
  cast _ _ = undefined
  uncast t f = do
    t0 <- get0 t
    t1 <- get1 t
    t2 <- get2 t
    t3 <- get3 t
    f (Unsafe t0, Unsafe t1, Unsafe t2, t3)

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
