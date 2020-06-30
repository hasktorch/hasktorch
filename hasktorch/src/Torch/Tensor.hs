{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RecordWildCards #-}

module Torch.Tensor where

import Control.Monad (forM_, forM)
import Control.Exception.Safe (throwIO)
import Foreign.ForeignPtr
import Foreign.Ptr
import Foreign.Storable
import Foreign.C.Types
import System.IO.Unsafe
import Data.Int (Int16, Int64)
import Data.Word (Word8)
import Data.List (intercalate)
import Data.Proxy
import Data.Reflection
import Numeric

import Torch.Internal.Cast
import Torch.Internal.Class (Castable(..), CppTuple2(..), CppTuple3(..), CppTuple4(..))
import qualified Torch.Internal.Unmanaged.Type.Tensor as Unmanaged (tensor_data_ptr)
import qualified Torch.Internal.Managed.Type.Context as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Managed.Type.TensorOptions as ATen
import qualified Torch.Internal.Managed.Type.TensorIndex as ATen
import qualified Torch.Internal.Managed.Type.StdArray as ATen
import qualified Torch.Internal.Managed.Type.StdString as ATen
import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Managed.Cast as ATen
import qualified Torch.Internal.Type as ATen
import qualified Torch.Internal.Const as ATen
import qualified Torch.Internal.Managed.TensorFactories as LibTorch

import Torch.Device
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

-- | Returns the total number of elements in the input tensor.
numel 
 :: Tensor -- ^ input 
 -> Int -- ^ number of elements in tensor
numel t = unsafePerformIO $ cast1 ATen.tensor_numel $ t

size :: Tensor -> Int -> Int
size t dim = unsafePerformIO $ (cast2 ATen.tensor_size_l) t dim

-- | Returns the shape of the tensor
shape 
 :: Tensor -- ^ input
 -> [Int] -- ^ list of integers representing the shape of the tensor
shape t = unsafePerformIO $ (cast1 ATen.tensor_sizes) t

-- | Returns the dimensions of the input tensor
dim 
 :: Tensor -- ^ input 
 -> Int -- ^ output
dim t = unsafePerformIO $ (cast1 ATen.tensor_dim) t

-- | Returns the device on which the tensor is currently allocated
device 
 :: Tensor -- ^ input
 -> Device -- ^ object representing the device
device t = unsafePerformIO $ do
  hasCUDA <- cast0 ATen.hasCUDA :: IO Bool
  if hasCUDA
    then do
      isCUDA <- cast1 ATen.tensor_is_cuda t :: IO Bool
      if isCUDA then cuda <$> cast1 ATen.tensor_get_device t else pure cpu
    else pure cpu
 where
  cpu = Device { deviceType = CPU, deviceIndex = 0 }
  cuda :: Int -> Device
  cuda di = Device { deviceType = CUDA, deviceIndex = fromIntegral di }

-- | Returns the data type of the input tensor
dtype 
 :: Tensor -- ^ input
 -> DType -- ^ data type of the input tensor
dtype t = unsafePerformIO $ cast1 ATen.tensor_scalar_type t


toDouble :: Tensor -> Double  
toDouble t = unsafePerformIO $ cast1 ATen.tensor_item_double t

toInt :: Tensor -> Int
toInt t = unsafePerformIO $ cast1 ATen.tensor_item_int64_t t

-- | Casts the input tensor to the given data type
toType 
 :: DType -- ^ data type to cast input to 
 -> Tensor -- ^ input 
 -> Tensor -- ^ output
toType dtype t = unsafePerformIO $ cast2 ATen.tensor_toType_s t dtype

-- | Casts the input tensor to given device
toDevice 
 :: Device -- ^ device to cast input to
 -> Tensor -- ^ input
 -> Tensor -- ^ output
toDevice device' t = unsafePerformIO $ do
  hasCUDA <- cast0 ATen.hasCUDA :: IO Bool
  let device = Torch.Tensor.device t
  t' <- toDevice' (deviceType device)
                  (deviceType device')
                  (deviceIndex device)
                  (deviceIndex device')
                  hasCUDA
  check (deviceType device')
        (deviceType $ Torch.Tensor.device t')
        (deviceIndex device')
        (deviceIndex $ Torch.Tensor.device t')
  pure t'
 where
  toDevice' dt   dt'  di di' _    | dt == dt' && di == di' = pure t -- do nothing
  toDevice' CUDA CUDA di di' True | di /= di'              = getOpts t >>= withDeviceIndex di' >>= to t -- copy from di to di'
  toDevice' CPU  CUDA 0  di' True | di' >= 0               = getOpts t >>= withDeviceIndex di' >>= to t -- copy from cpu:0 to cuda:di'
  toDevice' CUDA CPU  di 0   True | di >= 0                = getOpts t >>= withDeviceType CPU  >>= to t -- copy from cuda:di to cpu:0
  toDevice' dt   dt'  di di' _                             =
    error
      $  "cannot move tensor from \""
      <> show dt
      <> ":"
      <> show di
      <> "\" to \""
      <> show dt'
      <> ":"
      <> show di'
      <> "\""
  getOpts :: Tensor -> IO TensorOptions
  getOpts = cast1 ATen.tensor_options
  withDeviceType :: DeviceType -> TensorOptions -> IO TensorOptions
  withDeviceType dt opts = cast2 ATen.tensorOptions_device_D opts dt
  withDeviceIndex :: Int16 -> TensorOptions -> IO TensorOptions
  withDeviceIndex di opts = cast2 ATen.tensorOptions_device_index_s opts di -- careful, setting the device index implies setting the device type to CUDA!
  to :: Tensor -> TensorOptions -> IO Tensor
  to t opts = cast4 ATen.tensor_to_obb t opts nonBlocking copy
   where
    nonBlocking = False
    copy = False
  check dt dt' di di' | dt == dt' && di == di' = pure ()
  check dt dt' di di' =
    error
      $  "moving of tensor failed: device should have been \""
      <> show dt
      <> ":"
      <> show di
      <> "\" but is \""
      <> show dt'
      <> ":"
      <> show di'
      <> "\""

-- | Slices the input tensor along the selected dimension at the given index. 
select 
 :: Tensor -- ^ input
 -> Int -- ^ dimension to slice along
 -> Int -- ^ index in the given dimension 
 -> Tensor -- ^ output
select t dim idx = unsafePerformIO $ cast3 ATen.tensor_select_ll t dim idx

-- | Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor.
indexSelect 
 :: Tensor 
 -> Int 
 -> Tensor 
 -> Tensor
indexSelect t dim indexTensor = unsafePerformIO $ (cast3 ATen.index_select_tlt) t dim indexTensor

-- | Slices the input tensor along the selected dimension at the given range. 
slice
  :: Tensor -- ^ input
  -> Int -- ^ dim
  -> Int -- ^ start
  -> Int -- ^ end
  -> Int -- ^ step
  -> Tensor
slice _self _dim _start _end _step = unsafePerformIO $ (cast5 ATen.slice_tllll) _self _dim _start _end _step

isContiguous
  :: Tensor
  -> Bool
isContiguous t = unsafePerformIO $ (cast1 ATen.tensor_is_contiguous) t

contiguous
  :: Tensor
  -> Tensor
contiguous t = unsafePerformIO $ (cast1 ATen.tensor_contiguous) t

-- | Returns a tensor with the same data and number of elements as input, but with the specified shape.
reshape 
 :: [Int] 
 -> Tensor 
 -> Tensor
reshape shape t = unsafePerformIO $ cast2 ATen.reshape_tl t shape

--------------------------------------------------------------------------------
-- Move backend
--------------------------------------------------------------------------------

toSparse :: Tensor -> Tensor
toSparse t = unsafePerformIO $ (cast1 ATen.tensor_to_sparse) t

toDense :: Tensor -> Tensor
toDense t = unsafePerformIO $ (cast1 ATen.tensor_to_dense) t

toMKLDNN :: Tensor -> Tensor
toMKLDNN t = unsafePerformIO $ (cast1 ATen.tensor_to_mkldnn) t

toCPU :: Tensor -> Tensor
toCPU t = unsafePerformIO $ (cast1 ATen.tensor_cpu) t

toCUDA :: Tensor -> Tensor
toCUDA t = unsafePerformIO $ (cast1 ATen.tensor_cuda) t

--------------------------------------------------------------------------------
-- Indexing support
--------------------------------------------------------------------------------

-- TensorIndex is the same as slice of pytorch.
--
-- There is one-to-one correspondence between Pytorch and Hasktorch tensor index types:
-- Pytorch                 | Hasktorch
-- -----------------------------------------------------
-- `None`                  | `None`
-- `Ellipsis`              | `Ellipsis`
-- `...`                   | `Ellipsis`
-- `123`                   | `123`
-- `True` / `False`        | `True` / `False`
-- `:`                     | `Slice ()`
-- `::`                    | `Slice ()`
-- `1:`                    | `Slice (1, None)`
-- `1::`                   | `Slice (1, None)`
-- `:3`                    | `Slice (None, 3)`
-- `:3:`                   | `Slice (None, 3)`
-- `::2`                   | `Slice (None, None, 2)`
-- `1:3`                   | `Slice (1, 3)`
-- `1::2`                  | `Slice (1, None, 2)`
-- `:3:2`                  | `Slice (None, 3, 2)`
-- `1:3:2`                 | `Slice (1, 3, 2)`
-- `torch.tensor([1, 2])`) | `asTensor([1, 2 ::Int])`

newtype RawTensorIndexList = RawTensorIndexList (ForeignPtr (ATen.StdVector ATen.TensorIndex))
newtype RawTensorIndex = RawTensorIndex (ForeignPtr ATen.TensorIndex)

(@@) :: TensorIndex a => Tensor -> a -> Tensor
(Unsafe t) @@ idx = unsafePerformIO $ do
  let idxs = pushIndex [] idx
  vec <- ATen.newTensorIndexList
  forM_ idxs $ \(RawTensorIndex i) -> do
    ATen.tensorIndexList_push_back vec i
  ATen.index t vec >>= (return . Unsafe)

data None = None
  deriving (Show, Eq)

data Ellipsis = Ellipsis
  deriving (Show, Eq)

data Slice a = Slice a
  deriving (Show, Eq)

instance Castable RawTensorIndex (ForeignPtr ATen.TensorIndex) where
  cast (RawTensorIndex obj) f = f obj
  uncast obj f = f $ RawTensorIndex obj

class TensorIndex a where
  pushIndex :: [RawTensorIndex] -> a -> [RawTensorIndex]

instance {-# OVERLAPS #-} TensorIndex None where
  pushIndex vec _ = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithNone
    return ((RawTensorIndex idx):vec)
    
instance {-# OVERLAPS #-} TensorIndex Ellipsis where
  pushIndex vec _ = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithEllipsis
    return ((RawTensorIndex idx):vec)
    
instance {-# OVERLAPS #-} TensorIndex Bool where
  pushIndex vec b = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithBool (if b then 1 else 0)
    return ((RawTensorIndex idx):vec)
    
instance {-# OVERLAPS #-} (Integral a) => TensorIndex (Slice (a,a)) where
  pushIndex vec (Slice (start,end)) = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithSlice (fromIntegral start :: CInt) (fromIntegral end ::CInt) 1
    return ((RawTensorIndex idx):vec)
    
instance {-# OVERLAPS #-} (Integral a) => TensorIndex (Slice (a,a,a)) where
  pushIndex vec (Slice (start,end,step)) = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithSlice (fromIntegral start :: CInt) (fromIntegral end ::CInt) (fromIntegral step ::CInt)
    return ((RawTensorIndex idx):vec)

instance {-# OVERLAPS #-} (Integral a) => TensorIndex (Slice (None,None,a)) where
  pushIndex vec (Slice (_,_,step)) = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithSlice 0 (maxBound ::CInt) (fromIntegral step ::CInt)
    return ((RawTensorIndex idx):vec)

instance {-# OVERLAPS #-} (Integral a) => TensorIndex (Slice a) where
  pushIndex vec (Slice start) = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithSlice (fromIntegral start :: CInt) (maxBound ::CInt) 1
    return ((RawTensorIndex idx):vec)

instance {-# OVERLAPS #-} (Integral a) => TensorIndex (Slice (a,None)) where
  pushIndex vec (Slice (start,_)) = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithSlice (fromIntegral start :: CInt) (maxBound ::CInt) 1
    return ((RawTensorIndex idx):vec)

instance {-# OVERLAPS #-} (Integral a) => TensorIndex (Slice (a,None,a)) where
  pushIndex vec (Slice (start,_,step)) = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithSlice (fromIntegral start :: CInt) (maxBound ::CInt) (fromIntegral step :: CInt)
    return ((RawTensorIndex idx):vec)

instance {-# OVERLAPS #-} (Integral a) => TensorIndex (Slice (None,a,a)) where
  pushIndex vec (Slice (_,end,step)) = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithSlice 0 (fromIntegral end :: CInt) (fromIntegral step :: CInt)
    return ((RawTensorIndex idx):vec)

instance {-# OVERLAPS #-} (Integral a) => TensorIndex (Slice (None,a)) where
  pushIndex vec (Slice (_,end)) = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithSlice 0 (fromIntegral end :: CInt) 1
    return ((RawTensorIndex idx):vec)

instance {-# OVERLAPS #-} TensorIndex (Slice ()) where
  pushIndex vec (Slice ()) = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithSlice 0 (maxBound :: CInt) 1
    return ((RawTensorIndex idx):vec)

instance TensorIndex Int where
  pushIndex vec v = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithInt (fromIntegral v::CInt)
    return ((RawTensorIndex idx):vec)

instance TensorIndex Integer where
  pushIndex vec v = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithInt (fromIntegral v::CInt)
    return ((RawTensorIndex idx):vec)

instance TensorIndex Tensor where
  pushIndex vec v = unsafePerformIO $ do
    idx <- cast1 ATen.newTensorIndexWithTensor v
    return (idx:vec)

instance TensorIndex () where
  pushIndex vec _ = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithSlice 0 (maxBound :: CInt) 1
    return ((RawTensorIndex idx):vec)

instance (TensorIndex a, TensorIndex b) => TensorIndex (a,b) where
  pushIndex vec (a,b) = (flip pushIndex a) . (flip pushIndex b) $ vec
  
instance (TensorIndex a, TensorIndex b, TensorIndex c) => TensorIndex (a,b,c) where
  pushIndex vec (a,b,c) = (flip pushIndex a) . (flip pushIndex b) . (flip pushIndex c) $ vec

instance (TensorIndex a, TensorIndex b, TensorIndex c, TensorIndex d) => TensorIndex (a,b,c,d) where
  pushIndex vec (a,b,c,d) = (flip pushIndex a) . (flip pushIndex b) . (flip pushIndex c) . (flip pushIndex d) $ vec

instance (TensorIndex a, TensorIndex b, TensorIndex c, TensorIndex d, TensorIndex e) => TensorIndex (a,b,c,d,e) where
  pushIndex vec (a,b,c,d,e) = (flip pushIndex a) . (flip pushIndex b) . (flip pushIndex c) . (flip pushIndex d) . (flip pushIndex e) $ vec

--------------------------------------------------------------------------------
-- Scalar <-> Tensor promotion
--------------------------------------------------------------------------------

class TensorLike a where
  asTensor' :: a -> TensorOptions -> Tensor
  asTensor :: a -> Tensor
  asValue :: Tensor -> a
  -- Internal functions(like "_xxx") are below. Do not use them directly.
  _dtype :: DType
  _dims :: a -> [Int]
  _deepDims :: a -> Maybe [Int]
  _peekElemOff :: Ptr () -> Int -> [Int] -> IO a
  _pokeElemOff :: Ptr () -> Int -> a -> IO ()

int64_opts = withDType Int64 defaultOpts
float_opts = withDType Float defaultOpts

withTensor :: Tensor -> (Ptr () -> IO a) -> IO a
withTensor t fn = cast t $ \t' -> withForeignPtr t' $ \tensor_ptr -> Unmanaged.tensor_data_ptr tensor_ptr >>= fn

instance (Reifies a DType, Storable a) => TensorLike a where
  asTensor' v opts = unsafePerformIO $ do
    t <- ((cast2 LibTorch.empty_lo) :: [Int] -> TensorOptions -> IO Tensor) [] $ withDType (_dtype @a) opts
    withTensor t $ \ptr -> do
      _pokeElemOff ptr 0 v
    return t

  asTensor v = asTensor' v defaultOpts

  asValue t = unsafePerformIO $ do
    if _dtype @a == dtype t
    then do
      withTensor t $ \ptr -> do
        _peekElemOff ptr 0 []
    else
      throwIO $ userError $ "The infered DType of asValue is " ++ show (_dtype @a)  ++ ", but the DType of tensor on memory is " ++ show (dtype t) ++ "."

  _dtype = reflect (Proxy :: Proxy a)
  _dims _ = []
  _deepDims _ = Just []
  _peekElemOff ptr offset _ = peekElemOff (castPtr ptr) offset
  _pokeElemOff ptr offset v = pokeElemOff (castPtr ptr) offset v


instance {-# OVERLAPPING #-}TensorLike Bool where
  asTensor' v opts = unsafePerformIO $ do
    t <- ((cast2 LibTorch.empty_lo) :: [Int] -> TensorOptions -> IO Tensor) [] $ withDType (_dtype @Bool) opts
    withTensor t $ \ptr -> do
      _pokeElemOff ptr 0 v
    return t

  asTensor v = asTensor' v defaultOpts

  asValue t = unsafePerformIO $ do
    if _dtype @Bool == dtype t
    then do
      withTensor t $ \ptr -> do
        _peekElemOff ptr 0 []
    else
      throwIO $ userError $ "The infered DType of asValue is " ++ show (_dtype @Bool)  ++ ", but the DType of tensor on memory is " ++ show (dtype t) ++ "."

  _dtype = reflect (Proxy :: Proxy Bool)
  _dims _ = []
  _deepDims _ = Just []
  _peekElemOff ptr offset _ = (/= 0) <$> (peekElemOff (castPtr ptr) offset :: IO Word8)
  _pokeElemOff ptr offset v = pokeElemOff (castPtr ptr) offset ((if v then 1 else 0) :: Word8)


instance {-# OVERLAPPING #-}TensorLike a => TensorLike [a] where
  asTensor' v opts = unsafePerformIO $ do
    t <- ((cast2 LibTorch.empty_lo) :: [Int] -> TensorOptions -> IO Tensor) (_dims v) $ withDType (_dtype @a) opts
    withTensor t $ \ptr -> do
      _pokeElemOff ptr 0 v
    return t

  asTensor v = asTensor' v defaultOpts

  asValue t' = unsafePerformIO $ do
    let t = if isContiguous t' then t' else contiguous t'
    if _dtype @a == dtype t
    then do
      withTensor t $ \ptr -> do
        _peekElemOff ptr 0 (shape t)
    else
      throwIO $ userError $ "The infered DType of asValue is " ++ show (_dtype @a)  ++ ", but the DType of tensor on memory is " ++ show (dtype t) ++ "."

  _dtype = _dtype @a

  _dims [] = []
  _dims v@(x:_) = (length v):(_dims x)

  _deepDims [] = Just []
  _deepDims v@(x:xs) = do
    deepDimsX <- _deepDims x
    deepDimsXs <- traverse _deepDims xs
    if and $ fmap (deepDimsX ==) deepDimsXs
    then return $ length v : deepDimsX
    else Nothing

  _peekElemOff ptr offset [] = return []
  _peekElemOff ptr offset (d:dims) =
    let width = product dims
    in forM [0..(d-1)] $ \i ->
         _peekElemOff ptr (offset+i*width) dims

  _pokeElemOff ptr offset [] = return ()
  _pokeElemOff ptr offset v@(x:_) =
    let width = product (_dims x)
    in forM_ (zip [0..] v) $ \(i,d) ->
         if product (_dims d) == width -- This validation may be slow.
         then (_pokeElemOff @a) ptr (offset+i*width) d
         else throwIO $ userError $ "There are lists having different length."

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


--------------------------------------------------------------------------------

-- Castable instances
--------------------------------------------------------------------------------

-- NB: ATen only defines Castable [ForeignPtr ATen.Tensor] (ForeignPtr ATen.TensorList)
instance Castable [Tensor] (ForeignPtr ATen.TensorList) where
  cast xs f = do
    ptr_list <- mapM (\x -> (cast x return :: IO (ForeignPtr ATen.Tensor))) xs
    cast ptr_list f
  uncast xs f = uncast xs $ \ptr_list -> do
    tensor_list <- mapM (\(x :: ForeignPtr ATen.Tensor) -> uncast x return) ptr_list
    f tensor_list

instance Castable [Tensor] (ForeignPtr (ATen.C10List ATen.Tensor)) where
  cast xs f = do
    ptr_list <- mapM (\x -> (cast x return :: IO (ForeignPtr ATen.Tensor))) xs
    cast ptr_list f
  uncast xs f = uncast xs $ \ptr_list -> do
    tensor_list <- mapM (\(x :: ForeignPtr ATen.Tensor) -> uncast x return) ptr_list
    f tensor_list
