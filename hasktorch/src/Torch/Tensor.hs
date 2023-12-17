{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Tensor where

import Control.Exception.Safe (throwIO)
import Control.Monad (forM, forM_)
import Numeric.Half
import Data.Complex
import Data.Int (Int16, Int64)
import Data.List (intercalate)
import Data.Proxy
import Data.Reflection
import qualified Data.Vector as V
import Data.Word (Word8)
import Foreign.C.Types
import Foreign.ForeignPtr
import Foreign.Ptr
import Foreign.Storable
import GHC.Generics
import Numeric
import System.IO.Unsafe
import Torch.DType
import Torch.Device
import Torch.Internal.Cast
import Torch.Internal.Class (Castable (..), CppTuple2 (..), CppTuple3 (..), CppTuple4 (..))
import qualified Torch.Internal.Const as ATen
import qualified Torch.Internal.Managed.Cast as ATen
import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Managed.TensorFactories as LibTorch
import qualified Torch.Internal.Managed.Type.Context as ATen
import qualified Torch.Internal.Managed.Type.StdArray as ATen
import qualified Torch.Internal.Managed.Type.StdString as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Managed.Type.TensorIndex as ATen
import qualified Torch.Internal.Managed.Type.TensorOptions as ATen
import qualified Torch.Internal.Managed.Type.Extra as ATen
import qualified Torch.Internal.Type as ATen
import qualified Torch.Internal.Unmanaged.Type.Tensor as Unmanaged (tensor_data_ptr)
import Torch.Lens
import Torch.TensorOptions

type ATenTensor = ForeignPtr ATen.Tensor

-- do not use the constructor
newtype Tensor = Unsafe ATenTensor

instance Castable Tensor ATenTensor where
  cast (Unsafe aten_tensor) f = f aten_tensor
  uncast aten_tensor f = f $ Unsafe aten_tensor

newtype MutableTensor = MutableTensor Tensor deriving Show

newMutableTensor :: Tensor -> IO MutableTensor
newMutableTensor tensor = MutableTensor <$> cast1 ATen.detach_t tensor

toImmutable :: MutableTensor -> IO Tensor
toImmutable (MutableTensor tensor) = cast1 ATen.detach_t tensor

--------------------------------------------------------------------------------
-- Basic tensor properties
--------------------------------------------------------------------------------

-- | Returns the total number of elements in the input tensor.
numel ::
  -- | input
  Tensor ->
  -- | number of elements in tensor
  Int
numel t = unsafePerformIO $ cast1 ATen.tensor_numel $ t

-- | Returns the size of a given dimension of the input tensor.
size ::
  -- | dimension
  Int ->
  -- | input
  Tensor ->
  Int
size dim t = unsafePerformIO $ (cast2 ATen.tensor_size_l) t dim

-- | Returns the shape of the tensor
shape ::
  -- | input
  Tensor ->
  -- | list of integers representing the shape of the tensor
  [Int]
shape t = unsafePerformIO $ (cast1 ATen.tensor_sizes) t

-- | Returns the dimensions of the input tensor
dim ::
  -- | input
  Tensor ->
  -- | output
  Int
dim t = unsafePerformIO $ (cast1 ATen.tensor_dim) t

-- | Returns the dimensions of the input tensor
dimUnsafe ::
  -- | input
  Tensor ->
  -- | output
  Int
dimUnsafe t = unsafePerformIO $ (cast1 ATen.tensor_dim_unsafe) t

-- | Returns the dimensions of the input tensor
dimCUnsafe ::
  -- | input
  Tensor ->
  -- | output
  Int
dimCUnsafe t = unsafePerformIO $ (cast1 ATen.tensor_dim_c_unsafe) t

-- | Returns the device on which the tensor is currently allocated
device ::
  -- | input
  Tensor ->
  -- | object representing the device
  Device
device t = unsafePerformIO $ do
  hasCUDA <- cast0 ATen.hasCUDA :: IO Bool
  if hasCUDA
    then do
      isCUDA <- cast1 ATen.tensor_is_cuda t :: IO Bool
      if isCUDA then cuda <$> cast1 ATen.tensor_get_device t else pure cpu
    else pure cpu
  where
    cpu = Device {deviceType = CPU, deviceIndex = 0}
    cuda :: Int -> Device
    cuda di = Device {deviceType = CUDA, deviceIndex = fromIntegral di}

-- | Returns the data type of the input tensor
dtype ::
  -- | input
  Tensor ->
  -- | data type of the input tensor
  DType
dtype t = unsafePerformIO $ cast1 ATen.tensor_scalar_type t

toComplex :: Tensor -> Complex Double
toComplex t = unsafePerformIO $
    case dtype t of
      ComplexHalf -> do
        r :+ i  <- withTensor t $ \ptr -> peekElemOff (castPtr ptr) 0 :: IO (Complex Half)
        return (realToFrac r :+ realToFrac i)
      ComplexFloat -> do
        r :+ i  <- withTensor t $ \ptr -> peekElemOff (castPtr ptr) 0 :: IO (Complex Float)
        return (realToFrac r :+ realToFrac i)
      ComplexDouble -> withTensor t $ \ptr -> peekElemOff (castPtr ptr) 0 :: IO (Complex Double)
      _ -> (:+ 0) <$> cast1 ATen.tensor_item_double t

toDouble :: Tensor -> Double
toDouble t = unsafePerformIO $ cast1 ATen.tensor_item_double t

toInt :: Tensor -> Int
toInt t = unsafePerformIO $ cast1 ATen.tensor_item_int64_t t

-- | Casts the input tensor to the given data type
_toType ::
  -- | data type to cast input to
  DType ->
  -- | input
  Tensor ->
  -- | output
  Tensor
_toType dtype t = unsafePerformIO $ cast2 ATen.tensor_toType_s t dtype

instance HasTypes Tensor Tensor where
  types_ = id

instance HasTypes (a -> a) Tensor where
  types_ _ = pure

instance HasTypes Int Tensor where
  types_ _ = pure

instance HasTypes Double Tensor where
  types_ _ = pure

instance HasTypes Float Tensor where
  types_ _ = pure

instance HasTypes Bool Tensor where
  types_ _ = pure

instance HasTypes Int Int where
  types_ = id

instance HasTypes Float Float where
  types_ = id

instance HasTypes Double Double where
  types_ = id

instance HasTypes Bool Bool where
  types_ = id

toType :: forall a. HasTypes a Tensor => DType -> a -> a
toType dtype t = over (types @Tensor @a) (_toType dtype) t

toDevice :: forall a. HasTypes a Tensor => Device -> a -> a
toDevice device' t = over (types @Tensor @a) (_toDevice device') t

-- | Casts the input tensor to given device
_toDevice ::
  -- | device to cast input to
  Device ->
  -- | input
  Tensor ->
  -- | output
  Tensor
_toDevice device' t = unsafePerformIO $ do
  hasCUDA <- cast0 ATen.hasCUDA :: IO Bool
  let device = Torch.Tensor.device t
  t' <-
    toDevice'
      (deviceType device)
      (deviceType device')
      (deviceIndex device)
      (deviceIndex device')
      hasCUDA
  check
    (deviceType device')
    (deviceType $ Torch.Tensor.device t')
    (deviceIndex device')
    (deviceIndex $ Torch.Tensor.device t')
  pure t'
  where
    toDevice' dt dt' di di' _ | dt == dt' && di == di' = pure t -- do nothing
    toDevice' CUDA CUDA di di' True | di /= di' = getOpts t >>= withDeviceIndex di' >>= to t -- copy from di to di'
    toDevice' CPU CUDA 0 di' True | di' >= 0 = getOpts t >>= withDeviceIndex di' >>= to t -- copy from cpu:0 to cuda:di'
    toDevice' CUDA CPU di 0 True | di >= 0 = getOpts t >>= withDeviceType CPU >>= to t -- copy from cuda:di to cpu:0
    toDevice' dt dt' di di' _ =
      error $
        "cannot move tensor from \""
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
      error $
        "moving of tensor failed: device should have been \""
          <> show dt
          <> ":"
          <> show di
          <> "\" but is \""
          <> show dt'
          <> ":"
          <> show di'
          <> "\""

toDeviceWithTensor :: Tensor -> Tensor -> Tensor
toDeviceWithTensor reference input = unsafePerformIO $ cast2 ATen.tensor_to_device reference input

-- | Slices the input tensor along the selected dimension at the given index.
select ::
  -- | dimension to slice along
  Int ->
  -- | index in the given dimension
  Int ->
  -- | input
  Tensor ->
  -- | output
  Tensor
select dim idx t = unsafePerformIO $ cast3 ATen.tensor_select_ll t dim idx

-- | Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor.
indexSelect ::
  -- | dim
  Int ->
  -- | indexTensor
  Tensor ->
  -- | input
  Tensor ->
  -- | output
  Tensor
indexSelect dim indexTensor t = unsafePerformIO $ (cast3 ATen.index_select_tlt) t dim indexTensor

indexSelect' ::
  -- | dim
  Int ->
  -- | indexList
  [Int] ->
  -- | input
  Tensor ->
  -- | output
  Tensor
indexSelect' dim indexList t = unsafePerformIO $ (cast3 ATen.index_select_tlt) t dim (_toDevice (device t) (asTensor indexList))

-- | Slices the input tensor along the selected dimension at the given range.
sliceDim ::
  -- | dim
  Int ->
  -- | start
  Int ->
  -- | end
  Int ->
  -- | step
  Int ->
  -- | input
  Tensor ->
  Tensor
sliceDim _dim _start _end _step _self = unsafePerformIO $ (cast5 ATen.slice_tllll) _self _dim _start _end _step

isContiguous ::
  Tensor ->
  Bool
isContiguous t = unsafePerformIO $ (cast1 ATen.tensor_is_contiguous) t

contiguous ::
  Tensor ->
  Tensor
contiguous t = unsafePerformIO $ (cast1 ATen.tensor_contiguous) t

-- | Returns a tensor with the same data and number of elements as input, but with the specified shape.
reshape ::
  [Int] ->
  Tensor ->
  Tensor
reshape shape t = unsafePerformIO $ cast2 ATen.reshape_tl t shape

--------------------------------------------------------------------------------
-- Move backend
--------------------------------------------------------------------------------

toSparse :: Tensor -> Tensor
toSparse t = unsafePerformIO $ (cast2 ATen.tensor_to_sparse_l) t (dimCUnsafe t)

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

(!) :: TensorIndex a => Tensor -> a -> Tensor
(Unsafe t) ! idx = unsafePerformIO $ do
  let idxs = pushIndex [] idx
  vec <- ATen.newTensorIndexList
  forM_ idxs $ \(RawTensorIndex i) -> do
    ATen.tensorIndexList_push_back vec i
  ATen.index t vec >>= (return . Unsafe)

maskedFill :: (TensorIndex a, TensorLike t) => Tensor -> a -> t -> Tensor
maskedFill (Unsafe t') idx v' = unsafePerformIO $ do
  let idxs = pushIndex [] idx
      (Unsafe v) = asTensor v'
  t <- ATen.clone_t t'
  vec <- ATen.newTensorIndexList
  forM_ idxs $ \(RawTensorIndex i) -> do
    ATen.tensorIndexList_push_back vec i
  ATen.index_put_ t vec v
  return $ Unsafe t

data None = None
  deriving (Show, Eq)

data Ellipsis = Ellipsis
  deriving (Show, Eq)

newtype Slice a = Slice a
  deriving (Show, Eq)

instance Castable RawTensorIndex (ForeignPtr ATen.TensorIndex) where
  cast (RawTensorIndex obj) f = f obj
  uncast obj f = f $ RawTensorIndex obj

class TensorIndex a where
  pushIndex :: [RawTensorIndex] -> a -> [RawTensorIndex]
  toLens :: TensorLike b => a -> Lens' Tensor b
  default toLens :: TensorLike b => a -> Lens' Tensor b
  toLens idx func s = maskedFill s idx <$> (asTensor <$> func (asValue (s ! idx)))

instance {-# OVERLAPS #-} TensorIndex None where
  pushIndex vec _ = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithNone
    return ((RawTensorIndex idx) : vec)

instance {-# OVERLAPS #-} TensorIndex Ellipsis where
  pushIndex vec _ = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithEllipsis
    return ((RawTensorIndex idx) : vec)

instance {-# OVERLAPS #-} TensorIndex Bool where
  pushIndex vec b = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithBool (if b then 1 else 0)
    return ((RawTensorIndex idx) : vec)

instance {-# OVERLAPS #-} (Integral a) => TensorIndex (Slice (a, a)) where
  pushIndex vec (Slice (start, end)) = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithSlice (fromIntegral start :: CInt) (fromIntegral end :: CInt) 1
    return ((RawTensorIndex idx) : vec)

instance {-# OVERLAPS #-} (Integral a) => TensorIndex (Slice (a, a, a)) where
  pushIndex vec (Slice (start, end, step)) = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithSlice (fromIntegral start :: CInt) (fromIntegral end :: CInt) (fromIntegral step :: CInt)
    return ((RawTensorIndex idx) : vec)

instance {-# OVERLAPS #-} (Integral a) => TensorIndex (Slice (None, None, a)) where
  pushIndex vec (Slice (_, _, step)) = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithSlice 0 (maxBound :: CInt) (fromIntegral step :: CInt)
    return ((RawTensorIndex idx) : vec)

instance {-# OVERLAPS #-} (Integral a) => TensorIndex (Slice a) where
  pushIndex vec (Slice start) = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithSlice (fromIntegral start :: CInt) (maxBound :: CInt) 1
    return ((RawTensorIndex idx) : vec)

instance {-# OVERLAPS #-} (Integral a) => TensorIndex (Slice (a, None)) where
  pushIndex vec (Slice (start, _)) = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithSlice (fromIntegral start :: CInt) (maxBound :: CInt) 1
    return ((RawTensorIndex idx) : vec)

instance {-# OVERLAPS #-} (Integral a) => TensorIndex (Slice (a, None, a)) where
  pushIndex vec (Slice (start, _, step)) = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithSlice (fromIntegral start :: CInt) (maxBound :: CInt) (fromIntegral step :: CInt)
    return ((RawTensorIndex idx) : vec)

instance {-# OVERLAPS #-} (Integral a) => TensorIndex (Slice (None, a, a)) where
  pushIndex vec (Slice (_, end, step)) = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithSlice 0 (fromIntegral end :: CInt) (fromIntegral step :: CInt)
    return ((RawTensorIndex idx) : vec)

instance {-# OVERLAPS #-} (Integral a) => TensorIndex (Slice (None, a)) where
  pushIndex vec (Slice (_, end)) = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithSlice 0 (fromIntegral end :: CInt) 1
    return ((RawTensorIndex idx) : vec)

instance {-# OVERLAPS #-} TensorIndex (Slice ()) where
  pushIndex vec (Slice ()) = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithSlice 0 (maxBound :: CInt) 1
    return ((RawTensorIndex idx) : vec)

instance TensorIndex Int where
  pushIndex vec v = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithInt (fromIntegral v :: CInt)
    return ((RawTensorIndex idx) : vec)

instance TensorIndex Integer where
  pushIndex vec v = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithInt (fromIntegral v :: CInt)
    return ((RawTensorIndex idx) : vec)

instance TensorIndex Tensor where
  pushIndex vec v = unsafePerformIO $ do
    idx <- cast1 ATen.newTensorIndexWithTensor v
    return (idx : vec)

instance TensorIndex () where
  pushIndex vec _ = unsafePerformIO $ do
    idx <- ATen.newTensorIndexWithSlice 0 (maxBound :: CInt) 1
    return ((RawTensorIndex idx) : vec)

instance (TensorIndex a, TensorIndex b) => TensorIndex (a, b) where
  pushIndex vec (a, b) = (flip pushIndex a) . (flip pushIndex b) $ vec

instance (TensorIndex a, TensorIndex b, TensorIndex c) => TensorIndex (a, b, c) where
  pushIndex vec (a, b, c) = (flip pushIndex a) . (flip pushIndex b) . (flip pushIndex c) $ vec

instance (TensorIndex a, TensorIndex b, TensorIndex c, TensorIndex d) => TensorIndex (a, b, c, d) where
  pushIndex vec (a, b, c, d) = (flip pushIndex a) . (flip pushIndex b) . (flip pushIndex c) . (flip pushIndex d) $ vec

instance (TensorIndex a, TensorIndex b, TensorIndex c, TensorIndex d, TensorIndex e) => TensorIndex (a, b, c, d, e) where
  pushIndex vec (a, b, c, d, e) = (flip pushIndex a) . (flip pushIndex b) . (flip pushIndex c) . (flip pushIndex d) . (flip pushIndex e) $ vec

--------------------------------------------------------------------------------
-- Scalar <-> Tensor promotion
--------------------------------------------------------------------------------

asValue :: TensorLike a => Tensor -> a
asValue t =
  let cpuTensor = if device t == Device CPU 0 then t else toCPU t
      contTensor = if isContiguous cpuTensor then cpuTensor else contiguous cpuTensor
   in _asValue contTensor

class TensorOptionLike a where
  withTensorOptions :: Tensor -> a -> Tensor

instance  TensorOptionLike TensorOptions where
  withTensorOptions t opts = unsafePerformIO $ cast4 ATen.tensor_to_obb t opts nonBlocking copy
    where
      nonBlocking = False
      copy = False

instance  TensorOptionLike Tensor where
  withTensorOptions t opts = unsafePerformIO $ cast4 ATen.tensor_to_tbb t opts nonBlocking copy
    where
      nonBlocking = False
      copy = False

class TensorLike a where
  asTensor' :: TensorOptionLike opt => a -> opt -> Tensor
  asTensor' v opts = withTensorOptions (asTensor v) opts
  asTensor :: a -> Tensor
  _asValue :: Tensor -> a

  -- Internal functions(like "_xxx") are below. Do not use them directly.
  _dtype :: DType
  _dims :: a -> [Int]
  _deepDims :: a -> Maybe [Int]
  _peekElemOff :: Ptr () -> Int -> [Int] -> IO a
  _pokeElemOff :: Ptr () -> Int -> a -> IO ()

bool_opts = withDType Bool defaultOpts

uint8_opts = withDType UInt8 defaultOpts

int64_opts = withDType Int64 defaultOpts

float_opts = withDType Float defaultOpts

double_opts = withDType Double defaultOpts

withTensor :: Tensor -> (Ptr () -> IO a) -> IO a
withTensor t fn =
  let tensor = if isContiguous t then t else contiguous t
   in cast tensor $ \t' -> withForeignPtr t' $ \tensor_ptr -> Unmanaged.tensor_data_ptr tensor_ptr >>= fn

-- | The internal function of withTensor. It does not check contiguous memory-layout.
_withTensor :: Tensor -> (Ptr () -> IO a) -> IO a
_withTensor t fn =
  cast t $ \t' -> withForeignPtr t' $ \tensor_ptr -> Unmanaged.tensor_data_ptr tensor_ptr >>= fn

instance {-# OVERLAPPING #-} (Reifies a DType, Storable a) => TensorLike a where
  asTensor v = unsafePerformIO $ do
    t <- ((cast2 ATen.new_empty_tensor) :: [Int] -> TensorOptions -> IO Tensor) [] $ withDType (_dtype @a) defaultOpts
    _withTensor t $ \ptr -> do
      _pokeElemOff ptr 0 v
    return t

  _asValue t = unsafePerformIO $ do
    if _dtype @a == dtype t
      then do
        withTensor t $ \ptr -> do
          _peekElemOff ptr 0 []
      else throwIO $ userError $ "The infered DType of asValue is " ++ show (_dtype @a) ++ ", but the DType of tensor on memory is " ++ show (dtype t) ++ "."

  _dtype = reflect (Proxy :: Proxy a)
  _dims _ = []
  _deepDims _ = Just []
  _peekElemOff ptr offset _ = peekElemOff (castPtr ptr) offset
  _pokeElemOff ptr offset v = pokeElemOff (castPtr ptr) offset v

instance {-# OVERLAPPING #-} TensorLike Bool where
  asTensor v = unsafePerformIO $ do
    t <- ((cast2 ATen.new_empty_tensor) :: [Int] -> TensorOptions -> IO Tensor) [] $ withDType (_dtype @Bool) defaultOpts
    _withTensor t $ \ptr -> do
      _pokeElemOff ptr 0 v
    return t

  _asValue t = unsafePerformIO $ do
    if _dtype @Bool == dtype t
      then do
        withTensor t $ \ptr -> do
          _peekElemOff ptr 0 []
      else throwIO $ userError $ "The infered DType of asValue is " ++ show (_dtype @Bool) ++ ", but the DType of tensor on memory is " ++ show (dtype t) ++ "."

  _dtype = reflect (Proxy :: Proxy Bool)
  _dims _ = []
  _deepDims _ = Just []
  _peekElemOff ptr offset _ = (/= 0) <$> (peekElemOff (castPtr ptr) offset :: IO Word8)
  _pokeElemOff ptr offset v = pokeElemOff (castPtr ptr) offset ((if v then 1 else 0) :: Word8)

instance {-# OVERLAPPING #-} TensorLike Tensor where
  asTensor' v opts = withTensorOptions v opts
  asTensor = id
  _asValue = id
  _dtype = error "Not implemented for Tensor-type"
  _dims v = error "Not implemented for Tensor-type"
  _deepDims v = error "Not implemented for Tensor-type"
  _peekElemOff = error "Not implemented for Tensor-type"
  _pokeElemOff = error "Not implemented for Tensor-type"

instance {-# OVERLAPPING #-} TensorLike a => TensorLike (a, a) where
  asTensor (a, b) = asTensor [a, b]
  _asValue v =
    let [a, b] = _asValue v
     in (a, b)
  _dtype = error "Not implemented for tuple-type"
  _dims v = error "Not implemented for tuple-type"
  _deepDims v = error "Not implemented for tuple-type"
  _peekElemOff = error "Not implemented for tuple-type"
  _pokeElemOff = error "Not implemented for tuple-type"

instance {-# OVERLAPPING #-} TensorLike a => TensorLike [a] where
  asTensor v = unsafePerformIO $ do
    t <- ((cast2 ATen.new_empty_tensor) :: [Int] -> TensorOptions -> IO Tensor) (_dims v) $ withDType (_dtype @a) defaultOpts
    _withTensor t $ \ptr -> do
      _pokeElemOff ptr 0 v
    return t

  _asValue t = unsafePerformIO $ do
    if _dtype @a == dtype t
      then do
        withTensor t $ \ptr -> do
          _peekElemOff ptr 0 (shape t)
      else throwIO $ userError $ "The infered DType of asValue is " ++ show (_dtype @a) ++ ", but the DType of tensor on memory is " ++ show (dtype t) ++ "."

  _dtype = _dtype @a

  _dims [] = [0]
  _dims v@(x : _) = (length v) : (_dims x)

  _deepDims [] = Just [0]
  _deepDims v@(x : xs) = do
    deepDimsX <- _deepDims x
    deepDimsXs <- traverse _deepDims xs
    if and $ fmap (deepDimsX ==) deepDimsXs
      then return $ length v : deepDimsX
      else Nothing

  _peekElemOff ptr offset [] = return []
  _peekElemOff ptr offset (d : dims) =
    let width = product dims
     in forM [0 .. (d -1)] $ \i ->
          _peekElemOff ptr (offset + i * width) dims

  _pokeElemOff ptr offset [] = return ()
  _pokeElemOff ptr offset v@(x : _) =
    let width = product (_dims x)
     in forM_ (zip [0 ..] v) $ \(i, d) ->
          if product (_dims d) == width -- This validation may be slow.
            then (_pokeElemOff @a) ptr (offset + i * width) d
            else throwIO $ userError $ "There are lists having different length."

class AsTensors as where
  toTensors :: as -> V.Vector Tensor
  default toTensors :: (Generic as, GAsTensors (Rep as)) => as -> V.Vector Tensor
  toTensors a = gToTensors $ from a

instance TensorLike a => AsTensors a where
  toTensors = pure . asTensor

class GAsTensors record where
  gToTensors :: record as -> V.Vector Tensor

instance (GAsTensors ls, GAsTensors rs) => GAsTensors (ls :*: rs) where
  gToTensors (g :*: d) = gToTensors g V.++ gToTensors d

instance (GAsTensors ls, GAsTensors rs) => GAsTensors (ls :+: rs) where
  gToTensors (L1 g) = gToTensors g
  gToTensors (R1 g) = gToTensors g

instance (GAsTensors ls) => GAsTensors (M1 i c ls) where
  gToTensors (M1 g) = gToTensors g

instance (TensorLike ls) => GAsTensors (K1 i ls) where
  gToTensors (K1 g) = pure $ asTensor g

--------------------------------------------------------------------------------
-- Show
--------------------------------------------------------------------------------

instance Show Tensor where
  show t' =
    case (dim t) of
      0 -> details ++ show0d t
      1 -> details ++ show1d t
      n -> details ++ shownd n 0 t
    where
      t = if device t' == Device CPU 0 then t' else toCPU t'
      -- TODO: this is obviously not the right way to do it,
      -- and will be terribly slow, so please fix it.
      showElems elemShow sep t = "[" ++ (intercalate sep $ map elemShow [t ! i | i <- [0 .. ((size 0 t) - 1)]]) ++ "]"
      padPositive x s = if x >= 0 then " " ++ s else s
      -- TODO: this assumes that scientific notation only uses one-digit exponents, which is not
      --       true in general
      padLarge x s = if (abs x) >= 0.1 then s ++ "   " else s
      show0d x =
        if isIntegral (dtype t)
          then padPositive (toInt x) $ show $ toInt x
          else
            if isComplex (dtype t)
               then
                 let r :+ i = toComplex x
                 in (padLarge r $ padPositive r $ showGFloat (Just 4) r "") ++ " + i" ++
                    (padLarge i $ padPositive i $ showGFloat (Just 4) i "")
               else padLarge (toDouble x) $ padPositive (toDouble x) $ showGFloat (Just 4) (toDouble x) ""
      show1d = showElems show0d ", "
      shownd n offset =
        case n of
          2 -> showElems show1d (",\n " ++ padding ++ replicate offset ' ')
          _ -> showElems (shownd (n -1) (offset + 1)) (",\n " ++ padding ++ replicate offset ' ')
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

instance Castable [Tensor] (ForeignPtr (ATen.C10List (ATen.C10Optional ATen.Tensor))) where
  cast xs f = do
    ptr_list <- mapM (\x -> (cast x return :: IO (ForeignPtr ATen.Tensor))) xs
    cast ptr_list f
  uncast xs f = uncast xs $ \ptr_list -> do
    tensor_list <- mapM (\(x :: ForeignPtr ATen.Tensor) -> uncast x return) ptr_list
    f tensor_list
