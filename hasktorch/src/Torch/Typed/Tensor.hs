{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

module Torch.Typed.Tensor where

import           Prelude                 hiding ( (.), id )
import           Control.Arrow
import           Control.Category
import           Data.Finite
import           Torch.HList
import           Data.Kind                      ( Constraint
                                                , Type
                                                )
import           Data.Maybe
import           Data.Proxy
import           Data.Reflection
import           Foreign.ForeignPtr
import           Foreign.Storable
import           GHC.TypeLits
import           GHC.Exts

import           Torch.Internal.Cast
import           Torch.Internal.Class                     ( Castable(..)
                                                , CppTuple2(..)
                                                , CppTuple3(..)
                                                , CppTuple4(..)
                                                )
import qualified Torch.Internal.Type                     as ATen
import qualified Torch.Tensor                  as D
import qualified Torch.TensorFactories         as D
import qualified Torch.Functional               as D hiding (select)
import qualified Torch.DType                   as D
import qualified Torch.Device                  as D
import           Torch.Typed.Aux

class KnownShape (shape :: [Nat]) where
    shapeVal :: [Int]

instance KnownShape '[] where
    shapeVal = []

instance (KnownNat h, KnownShape t) => KnownShape (h ': t) where
    shapeVal = natValI @h : shapeVal @t

getFiniteI :: Finite n -> Int
getFiniteI = fromIntegral . getFinite

class KnownDType (dtype :: D.DType) where
  dtypeVal :: D.DType

instance KnownDType 'D.Bool where
  dtypeVal = D.Bool
instance KnownDType 'D.UInt8 where
  dtypeVal = D.UInt8
instance KnownDType 'D.Int8 where
  dtypeVal = D.Int8
instance KnownDType 'D.Int16 where
  dtypeVal = D.Int16
instance KnownDType 'D.Int32 where
  dtypeVal = D.Int32
instance KnownDType 'D.Int64 where
  dtypeVal = D.Int64
instance KnownDType 'D.Half where
  dtypeVal = D.Half
instance KnownDType 'D.Float where
  dtypeVal = D.Float
instance KnownDType 'D.Double where
  dtypeVal = D.Double

type family ComputeDType (dtype' :: dtype) :: D.DType where
  ComputeDType Bool = D.Bool
  ComputeDType D.Bool = D.Bool
  ComputeDType D.UInt8 = D.UInt8
  ComputeDType D.Int8 = D.Int8
  ComputeDType D.Int16 = D.Int16
  ComputeDType D.Int32 = D.Int32
  ComputeDType Int = D.Int64
  ComputeDType D.Int64 = D.Int64
  ComputeDType Float = D.Float
  ComputeDType D.Float = D.Float
  ComputeDType Double = D.Double
  ComputeDType D.Double = D.Double
  ComputeDType dtype' = TypeError (Text "Unsupported tensor type " :<>: ShowType dtype')

class KnownDevice (device :: (D.DeviceType, Nat)) where
  deviceVal :: D.Device

instance (KnownNat n) => KnownDevice '( 'D.CPU, n) where
  deviceVal = D.Device D.CPU (natValInt16 @n)

instance (KnownNat n) => KnownDevice '( 'D.CUDA, n) where
  deviceVal = D.Device D.CUDA (natValInt16 @n)

data Tensor (device :: (D.DeviceType, Nat)) (dtype :: D.DType) (shape :: [Nat]) where
  UnsafeMkTensor :: forall device dtype shape . { toDynamic :: D.Tensor } -> Tensor device dtype shape

type CPUTensor = Tensor '( 'D.CPU, 0)
type CUDATensor deviceIndex = Tensor '( 'D.CUDA, deviceIndex)

type family ComputeHaskellType (dtype :: D.DType) :: Type where
  ComputeHaskellType D.Bool = Bool
  ComputeHaskellType D.Int64 = Int
  ComputeHaskellType D.Float = Float
  ComputeHaskellType D.Double = Double
  ComputeHaskellType dtype = TypeError (Text "Unsupported tensor type " :<>: ShowType dtype)

type family ComputeItemType (ty :: Type) (shape :: [Nat]) :: Type where
  ComputeItemType _ '[] = TypeError (Text "Scalars are not supported")
  ComputeItemType ty (_ ': '[]) = ty
  ComputeItemType ty (_ ': h ': t) = [ComputeItemType ty (h ': t)]

instance ( D.TensorLike [ComputeItemType (ComputeHaskellType dtype) shape]
         , KnownShape shape)
  => IsList (Maybe (Tensor '( 'D.CPU, 0) dtype shape))
 where
  type Item (Maybe (Tensor '( 'D.CPU, 0) dtype shape)) = ComputeItemType (ComputeHaskellType dtype) shape
  fromList xs = do
    shapeXs <- D._deepDims xs
    if shapeVal @shape == shapeXs
    then return $ UnsafeMkTensor . D.asTensor $ xs
    else Nothing
  toList Nothing = []
  toList (Just t) = D.asValue . toDynamic $ t

instance Num (Tensor device dtype shape) where
  (+) a b = UnsafeMkTensor $ toDynamic a + toDynamic b
  (-) a b = UnsafeMkTensor $ toDynamic a - toDynamic b
  (*) a b = UnsafeMkTensor $ toDynamic a * toDynamic b
  negate t = UnsafeMkTensor $ negate $ toDynamic t
  abs t = UnsafeMkTensor $ abs $ toDynamic t
  signum t = UnsafeMkTensor $ signum $ toDynamic t
  fromInteger i = UnsafeMkTensor $ D.asTensor @Int $ fromInteger @Int i

instance Fractional (Tensor device dtype shape) where
  a / b = UnsafeMkTensor $ toDynamic a / toDynamic b
  recip t = UnsafeMkTensor $ recip $ toDynamic t
  fromRational i = UnsafeMkTensor $ D.asTensor @Float $ fromRational @Float i

instance Show (Tensor device dtype shape) where
    show (UnsafeMkTensor dynamic) = show dynamic

class TensorOptions (shape :: [Nat]) (dtype :: D.DType) (device :: (D.DeviceType, Nat)) where
  optionsRuntimeShape :: [Int]
  optionsRuntimeDType :: D.DType
  optionsRuntimeDevice :: D.Device

instance (KnownDType dtype, KnownDevice device) => TensorOptions '[] dtype device where
  optionsRuntimeShape = []
  optionsRuntimeDType = dtypeVal @dtype
  optionsRuntimeDevice = deviceVal @device

instance (KnownNat h, TensorOptions t dtype device) => TensorOptions (h ': t) dtype device where
  optionsRuntimeShape = natValI @h : optionsRuntimeShape @t @dtype @device
  optionsRuntimeDType = optionsRuntimeDType @t @dtype @device
  optionsRuntimeDevice = optionsRuntimeDevice @t @dtype @device

--------------------------------------------------------------------------------
-- Untyped -> Typed typecasts
--------------------------------------------------------------------------------

type family All (pred :: a -> Constraint) (l :: [a]) :: Constraint where
  All _    '[] = ()
  All pred (h ': t) = (pred h, All pred t)

data SomeShape where
  SomeShape :: forall (shape :: [Nat]) . KnownShape shape => Proxy shape -> SomeShape

someShape :: [Int] -> SomeShape
someShape []      = SomeShape $ Proxy @'[]
someShape (h : t) = case someNatVal (fromIntegral h) of
  Nothing -> error "Negative dimension in someShape!"
  (Just (SomeNat (Proxy :: Proxy ht))) -> case someShape t of
    (SomeShape (Proxy :: Proxy tt)) -> SomeShape $ Proxy @(ht ': tt)

data SomeDType where
  SomeDType :: forall (dtype :: D.DType) . KnownDType dtype => Proxy dtype -> SomeDType

someDType :: D.DType -> SomeDType
someDType D.Bool   = SomeDType $ Proxy @D.Bool
someDType D.UInt8  = SomeDType $ Proxy @D.UInt8
someDType D.Int8   = SomeDType $ Proxy @D.Int8
someDType D.Int16  = SomeDType $ Proxy @D.Int16
someDType D.Int32  = SomeDType $ Proxy @D.Int32
someDType D.Int64  = SomeDType $ Proxy @D.Int64
someDType D.Half   = SomeDType $ Proxy @D.Half
someDType D.Float  = SomeDType $ Proxy @D.Float
someDType D.Double = SomeDType $ Proxy @D.Double

data SomeDevice where
  SomeDevice :: forall (device :: (D.DeviceType, Nat)) . KnownDevice device => Proxy device -> SomeDevice

someDevice :: D.Device -> SomeDevice
someDevice D.Device {..} = case someNatVal (fromIntegral deviceIndex) of
  Nothing -> error "Negative device index in someDevice!"
  Just (SomeNat (Proxy :: Proxy n)) -> case deviceType of
    D.CPU  -> SomeDevice $ Proxy @'( 'D.CPU, n)
    D.CUDA -> SomeDevice $ Proxy @'( 'D.CUDA, n)

withTensor
  :: D.Tensor
  -> (  forall shape dtype device
      . KnownShape shape
     => Tensor device dtype shape
     -> r
     )
  -> r
withTensor untypedTensor f = case someShape (D.shape untypedTensor) of
    (SomeShape (Proxy :: Proxy shape)) -> case someDType (D.dtype untypedTensor) of
        (SomeDType (Proxy :: Proxy dtype)) -> case someDevice (D.device untypedTensor) of
          (SomeDevice (Proxy :: Proxy device)) -> f $ UnsafeMkTensor @device @dtype @shape untypedTensor

--------------------------------------------------------------------------------
-- Broadcast type-level function
--------------------------------------------------------------------------------

type family ComputeBroadcast (reversedShape :: [Nat]) (reversedShape' :: [Nat]) :: Maybe [Nat] where
    ComputeBroadcast '[]           reversedShape = Just reversedShape
    ComputeBroadcast reversedShape '[]           = Just reversedShape
    ComputeBroadcast (h ': t)      (h ': t2)     = AppendToMaybe h (ComputeBroadcast t t2)
    ComputeBroadcast (h ': t)      (1 ': t2)     = AppendToMaybe h (ComputeBroadcast t t2)
    ComputeBroadcast (1 ': t)      (h ': t2)     = AppendToMaybe h (ComputeBroadcast t t2)
    ComputeBroadcast _             _             = Nothing

type family CheckBroadcast (shape :: [Nat]) (shape' :: [Nat]) (result :: Maybe [Nat]) :: [Nat] where
    CheckBroadcast shape shape' Nothing       = TypeError (Text "The shapes " :<>:
                                                           ShowType shape :<>:
                                                           Text " and " :<>:
                                                           ShowType shape' :<>:
                                                           Text " cannot be broadcast")
    CheckBroadcast _     _      (Just result) = (Reverse result)

type Broadcast shape shape' = CheckBroadcast shape shape' (ComputeBroadcast (Reverse shape)
                                                                            (Reverse shape'))

type family BasicArithmeticDTypeIsValid (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  BasicArithmeticDTypeIsValid '( 'D.CPU, 0)    dtype = ( DTypeIsNotBool '( 'D.CPU, 0) dtype
                                                       , DTypeIsNotHalf '( 'D.CPU, 0) dtype
                                                       )
  BasicArithmeticDTypeIsValid '( 'D.CUDA, _)   dtype = ()
  BasicArithmeticDTypeIsValid '(deviceType, _) dtype = UnsupportedDTypeForDevice deviceType dtype

add, sub, mul
  :: forall shape'' shape shape' dtype dtype' dtype'' device
   . ( dtype'' ~ DTypePromotion dtype dtype'
     , shape'' ~ Broadcast shape shape'
     , BasicArithmeticDTypeIsValid device dtype
     , BasicArithmeticDTypeIsValid device dtype'
     , BasicArithmeticDTypeIsValid device dtype''
     )
  => Tensor device dtype   shape
  -> Tensor device dtype'  shape'
  -> Tensor device dtype'' shape''
add a b = UnsafeMkTensor $ D.add (toDynamic a) (toDynamic b)
sub a b = UnsafeMkTensor $ D.sub (toDynamic a) (toDynamic b)
mul a b = UnsafeMkTensor $ D.mul (toDynamic a) (toDynamic b)

type family ComparisonDTypeIsValid (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  ComparisonDTypeIsValid '( 'D.CPU, 0)    dtype = ( DTypeIsNotBool '( 'D.CPU, 0) dtype
                                                  , DTypeIsNotHalf '( 'D.CPU, 0) dtype
                                                  )
  ComparisonDTypeIsValid '( 'D.CUDA, _)   dtype = ()
  ComparisonDTypeIsValid '(deviceType, _) dtype = UnsupportedDTypeForDevice deviceType dtype

gt, lt, ge, le, eq, ne, (>.), (<.), (>=.), (<=.), (==.), (/=.)
  :: forall shape'' shape shape' dtype dtype' device
   . ( shape'' ~ Broadcast shape shape'
     , ComparisonDTypeIsValid device dtype
     , ComparisonDTypeIsValid device dtype'
     )
  => Tensor device dtype   shape
  -> Tensor device dtype'  shape'
  -> Tensor device 'D.Bool shape''
gt a b = UnsafeMkTensor $ D.gt (toDynamic a) (toDynamic b)
lt a b = UnsafeMkTensor $ D.lt (toDynamic a) (toDynamic b)
ge a b = UnsafeMkTensor $ D.ge (toDynamic a) (toDynamic b)
le a b = UnsafeMkTensor $ D.le (toDynamic a) (toDynamic b)
eq a b = UnsafeMkTensor $ D.eq (toDynamic a) (toDynamic b)
ne a b = UnsafeMkTensor $ D.ne (toDynamic a) (toDynamic b)
(>.) = gt
(<.) = lt
(>=.) = ge
(<=.) = le
(==.) = eq
(/=.) = ne

type family ComputeMatMul (reversedShape :: [Nat]) (reversedShape' :: [Nat]) :: Maybe [Nat] where
  ComputeMatMul (k ': '[])                         (k ': '[])                          = Just '[]
  ComputeMatMul (k ': '[])                         (m ': k ': reversedBroadcastShape') = AppendToMaybe m (ComputeBroadcast '[] reversedBroadcastShape')
  ComputeMatMul (k ': n ': reversedBroadcastShape) (k ': '[])                          = AppendToMaybe n (ComputeBroadcast '[] reversedBroadcastShape)
  ComputeMatMul (k ': n ': reversedBroadcastShape) (m ': k ': reversedBroadcastShape') = AppendToMaybe m (AppendToMaybe n (ComputeBroadcast reversedBroadcastShape reversedBroadcastShape'))

type family CheckMatMul (shape :: [Nat]) (shape' :: [Nat]) (result :: Maybe [Nat]) :: [Nat] where
  CheckMatMul shape shape' Nothing       = TypeError (Text "The shapes " :<>:
                                                      ShowType shape :<>:
                                                      Text " and " :<>:
                                                      ShowType shape' :<>:
                                                      Text " are not compatible with matrix multiplication")
  CheckMatMul _     _      (Just result) = (Reverse result)

type MatMul shape shape' = CheckMatMul shape shape' (ComputeMatMul (Reverse shape) (Reverse shape'))

type family MatMulDTypeIsValid (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  MatMulDTypeIsValid '( 'D.CPU, 0)            dtype = ( DTypeIsNotBool '( 'D.CPU, 0) dtype
                                                      , DTypeIsNotHalf '( 'D.CPU, 0) dtype
                                                      )
  MatMulDTypeIsValid '( 'D.CUDA, deviceIndex) dtype = DTypeIsFloatingPoint '( 'D.CUDA, deviceIndex) dtype
  MatMulDTypeIsValid '(deviceType, _)         dtype = UnsupportedDTypeForDevice deviceType dtype

-- | matrix multiplication
-- See https://pytorch.org/docs/stable/torch.html#torch.matmul.
matmul
  :: forall shape'' shape shape' dtype device
   . ( shape'' ~ MatMul shape shape'
     , MatMulDTypeIsValid device dtype
     )
  => Tensor device dtype shape
  -> Tensor device dtype shape'
  -> Tensor device dtype shape''
matmul a b = UnsafeMkTensor $ D.matmul (toDynamic a) (toDynamic b)

select
  :: forall dim idx shape' shape dtype device
   . ( KnownNat dim
     , KnownNat idx
     , InRange shape dim idx
     , shape' ~ Remove shape dim
     )
  => Tensor device dtype shape
  -> Tensor device dtype shape'
select t = UnsafeMkTensor $ D.select (toDynamic t) (natValI @dim) (natValI @idx)

selectIdx
  :: forall dim n shape' shape dtype device
   . ( KnownNat dim
     , n ~ Index shape dim
     , shape' ~ Remove shape dim
     )
  => Tensor device dtype shape
  -> Finite n
  -> Tensor device dtype shape'
selectIdx t idx = UnsafeMkTensor $ D.select (toDynamic t) (natValI @dim) (getFiniteI idx)

type family Numel (shape :: [Nat]) :: Nat where
    Numel '[] = 1
    Numel (h ': t) = h * (Numel t)

-- | reshape
-- >>> t :: CPUTensor 'D.Int64 '[2,3,4] = fromJust [[[111,112,113,114],[121,122,123,124],[131,132,133,134]],[[211,212,213,214],[221,222,223,224],[231,232,233,234]]]
-- >>> t' = reshape @'[24] t
-- >>> toList . Just $ t'
-- [111,112,113,114,121,122,123,124,131,132,133,134,211,212,213,214,221,222,223,224,231,232,233,234]
-- >>> toList . Just $ reshape @'[2,3,4] t'
-- [[[111,112,113,114],[121,122,123,124],[131,132,133,134]],[[211,212,213,214],[221,222,223,224],[231,232,233,234]]]
reshape
  :: forall shape' shape dtype device
   . ( KnownShape shape'
     , Numel shape ~ Numel shape'
     )
  => Tensor device dtype shape
  -> Tensor device dtype shape'
reshape t = UnsafeMkTensor $ D.reshape (shapeVal @shape') (toDynamic t)

instance Castable (Tensor device dtype shape) D.ATenTensor where
  cast (UnsafeMkTensor (D.Unsafe aten_tensor)) f = f aten_tensor
  uncast aten_tensor f = f $ UnsafeMkTensor (D.Unsafe aten_tensor)

instance Castable [Tensor device dtype shape] (ForeignPtr ATen.TensorList) where
  cast xs f = do
    ptr_list <- mapM (\x -> (cast x return :: IO (ForeignPtr ATen.Tensor))) xs
    cast ptr_list f
  uncast xs f = uncast xs $ \ptr_list -> do
    tensor_list <- mapM (\(x :: ForeignPtr ATen.Tensor) -> uncast x return) ptr_list
    f tensor_list

data TensorListFold = TensorListFold

instance (Castable x D.ATenTensor) => Apply TensorListFold x ([D.ATenTensor] -> IO [D.ATenTensor]) where
  apply _ x = \xs -> do
    x' <- cast x return
    return (x' : xs)

data TensorListUnfold = TensorListUnfold

instance Apply TensorListUnfold [D.ATenTensor] (IO HNothing) where
  apply _ [] = pure HNothing

instance (Castable x D.ATenTensor) => Apply TensorListUnfold [D.ATenTensor] (IO (HJust (x, [D.ATenTensor]))) where
  apply _ (x : xs) = do
    x' <- uncast x return
    return $ HJust (x', xs)

instance ( HFoldrM IO TensorListFold [D.ATenTensor] l
         , Apply TensorListUnfold [D.ATenTensor] res
         , HUnfoldM IO TensorListUnfold res l
         , res ~ (HUnfoldMRes IO [D.ATenTensor] l)
         )
  => Castable (HList l) [D.ATenTensor]
 where
  cast xs f = f =<< go xs
   where
    go :: HList l -> IO [D.ATenTensor]
    go xs = hfoldrM TensorListFold [] xs
  uncast xs f = f =<< go xs
   where
    go :: [D.ATenTensor] -> IO (HList l)
    go xs = hunfoldrM TensorListUnfold xs

instance Castable (HList l) [D.ATenTensor] => Castable (HList l) (ForeignPtr ATen.TensorList) where
  cast xs f = do
    ts <- cast xs return :: IO [ForeignPtr ATen.Tensor]
    cast ts f
  uncast xs f = uncast xs $ \(ptrList :: [ForeignPtr ATen.Tensor]) -> do
    ts <- uncast ptrList return :: IO (HList l)
    f ts

--------------------------------------------------------------------------------
-- Move tensors
--------------------------------------------------------------------------------

-- TODO: track sparsity in tensor type
toSparse :: Tensor device dtype shape -> Tensor device dtype shape
toSparse t = UnsafeMkTensor $ D.toSparse (toDynamic t)

-- TODO: track sparsity in tensor type
toDense :: Tensor device dtype shape -> Tensor device dtype shape
toDense t = UnsafeMkTensor $ D.toDense (toDynamic t)

-- -- TODO: is this a device?
-- toMKLDNN
--   :: forall device' device shape dtype
--    . Tensor device  dtype shape
--   -> Tensor device' dtype shape
-- toMKLDNN t = UnsafeMkTensor $ D.toMKLDNN (toDynamic t)

-- | move tensor to CPU
-- TODO: can this fail?
toCPU
  :: forall device shape dtype
   . Tensor device  dtype shape
  -> CPUTensor dtype shape
toCPU input = UnsafeMkTensor $ D.toCPU (toDynamic input)

-- | move tensor to the first CUDA device
-- TODO: what if this fails?
toCUDA
  :: forall device' device shape dtype
   . Tensor device  dtype shape
  -> CUDATensor 0 dtype shape
toCUDA t = UnsafeMkTensor $ D.toCUDA (toDynamic t)

-- | move tensor to device
-- TODO: what if this fails?
toDevice
  :: forall device' device dtype shape
   . KnownDevice device'
  => Tensor device  dtype shape
  -> Tensor device' dtype shape
toDevice = UnsafeMkTensor . D.toDevice (deviceVal @device') . toDynamic

-- | change tensor data type
toDType
  :: forall dtype' dtype device shape
   . KnownDType dtype'
  => Tensor device dtype  shape
  -> Tensor device dtype' shape
toDType = UnsafeMkTensor . D.toType (dtypeVal @dtype') . toDynamic

--------------------------------------------------------------------------------
-- Auxiliary functions for accessing tensor options as values
--------------------------------------------------------------------------------

-- | returns tensor dimension
--   uses compile-time information only
dim
  :: forall device dtype shape
   . TensorOptions shape dtype device
  => Tensor device dtype shape
  -> Int
dim t = length $ optionsRuntimeShape @shape @dtype @device

-- | returns tensor shape as list
--   uses compile-time information only
shape
  :: forall device dtype shape
   . TensorOptions shape dtype device
  => Tensor device dtype shape
  -> [Int]
shape _ = optionsRuntimeShape @shape @dtype @device

-- | returns tensor data type
--   uses compile-time information only
dtype
  :: forall device dtype shape
   . TensorOptions shape dtype device
  => Tensor device dtype shape
  -> D.DType
dtype _ = optionsRuntimeDType @shape @dtype @device

-- | returns tensor device
--   uses compile-time information only
device
  :: forall device dtype shape
   . TensorOptions shape dtype device
  => Tensor device dtype shape
  -> D.Device
device _ = optionsRuntimeDevice @shape @dtype @device

--------------------------------------------------------------------------------
-- Auxiliary functions for accessing tensors as values
--------------------------------------------------------------------------------

-- TODO: figure out what device, dtype, and shape we need for this
toInt
  :: Tensor device dtype shape
  -> Int
toInt t = D.toInt $ toDynamic t

toFloat :: forall device . Tensor device 'D.Float '[] -> Float
toFloat t = D.asValue . toDynamic . toCPU $ t

toDouble :: forall device . Tensor device 'D.Double '[] -> Double
toDouble t = D.asValue . toDynamic . toCPU $ t
