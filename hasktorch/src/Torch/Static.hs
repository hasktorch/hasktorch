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

module Torch.Static where

import Data.Proxy
import Data.Finite
import Data.Kind (Constraint, Type)
import Data.Reflection
import Foreign.Storable
import GHC.TypeLits
import GHC.Exts

import ATen.Cast
import ATen.Class (Castable(..), CppTuple2(..), CppTuple3(..), CppTuple4(..))
import qualified ATen.Type as ATen
import Foreign.ForeignPtr
import qualified Torch.Tensor as D
import qualified Torch.TensorFactories as D
import qualified Torch.Functions as D
import qualified Torch.DType as D

natValI :: forall n. KnownNat n => Int
natValI = fromIntegral $ natVal $ Proxy @n

class KnownShape shape where
    shapeVal :: [Int]

instance KnownShape '[] where
    shapeVal = []

instance (KnownNat h, KnownShape t) => KnownShape (h ': t) where
    shapeVal = natValI @h : shapeVal @t

getFiniteI :: Finite n -> Int
getFiniteI = fromIntegral . getFinite

class KnownDType dtype where
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

data Tensor (dtype :: D.DType) (shape :: [Nat]) where
  UnsafeMkTensor :: forall dtype shape . { toDynamic :: D.Tensor } -> Tensor dtype shape

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

instance (D.TensorLike [ComputeItemType (ComputeHaskellType dtype) shape], KnownShape shape) => IsList (Maybe (Tensor dtype shape)) where
  type Item (Maybe (Tensor dtype shape)) = ComputeItemType (ComputeHaskellType dtype) shape
  fromList xs = do
    shapeXs <- D._deepDims xs
    if shapeVal @shape == shapeXs
    then return $ UnsafeMkTensor . D.asTensor $ xs
    else Nothing
  toList Nothing = []
  toList (Just t) = D.asValue . toDynamic $ t
 
instance Num (Tensor dtype shape) where
  (+) a b = UnsafeMkTensor $ toDynamic a + toDynamic b
  (-) a b = UnsafeMkTensor $ toDynamic a - toDynamic b
  (*) a b = UnsafeMkTensor $ toDynamic a * toDynamic b
  negate t = UnsafeMkTensor $ negate $ toDynamic t
  abs t = UnsafeMkTensor $ abs $ toDynamic t
  signum t = UnsafeMkTensor $ signum $ toDynamic t
  fromInteger i = UnsafeMkTensor $ D.asTensor @Int $ fromInteger @Int i

instance Fractional (Tensor dtype shape) where
  a / b = UnsafeMkTensor $ toDynamic a / toDynamic b
  recip t = UnsafeMkTensor $ recip $ toDynamic t
  fromRational i = UnsafeMkTensor $ D.asTensor @Float $ fromRational @Float i

instance Show (Tensor dtype shape) where
    show (UnsafeMkTensor dynamic) = show dynamic

class TensorOptions (dtype :: D.DType) (shape :: [Nat]) where
  optionsRuntimeDType :: D.DType
  optionsRuntimeShape :: [Int]

instance (KnownDType dtype) => TensorOptions dtype '[] where
  optionsRuntimeDType = dtypeVal @dtype
  optionsRuntimeShape = []

instance (KnownDType dtype, KnownNat h, TensorOptions dtype t) => TensorOptions dtype (h ': t) where
  optionsRuntimeDType = dtypeVal @dtype
  optionsRuntimeShape = (natValI @h : optionsRuntimeShape @dtype @t)

--------------------------------------------------------------------------------
-- Dynamic -> Static typecasts
--------------------------------------------------------------------------------

type family All (pred :: a -> Constraint) (l :: [a]) :: Constraint where
    All _    '[] = ()
    All pred (h ': t) = (pred h, All pred t)

data SomeShape where
    SomeShape :: forall (shape :: [Nat]). KnownShape shape => Proxy shape -> SomeShape

someShape :: [Int] -> SomeShape
someShape [] = SomeShape $ Proxy @'[]
someShape (h : t) = case someNatVal (fromIntegral h) of
    Nothing -> error "Negative dimension in someShape!"
    (Just (SomeNat (Proxy :: Proxy ht))) -> case someShape t of
        (SomeShape (Proxy :: Proxy tt)) -> SomeShape $ Proxy @(ht ': tt)

data SomeDType where
    SomeDType :: forall (dtype :: D.DType). Proxy dtype -> SomeDType

someDType :: D.DType -> SomeDType
someDType D.Float = SomeDType $ Proxy @D.Float

withTensor :: D.Tensor ->
              (forall (dtype :: D.DType) (shape :: [Nat]).
                    KnownShape shape => Tensor dtype shape -> r) ->
              r

withTensor d f = case someShape (D.shape d) of
    (SomeShape (Proxy :: Proxy shape)) -> case someDType (D.dtype d) of
        (SomeDType (Proxy :: Proxy dtype)) -> f $ UnsafeMkTensor @dtype @shape d

--------------------------------------------------------------------------------
-- Broadcast type-level function
--------------------------------------------------------------------------------

type family AppendToMaybe (n :: Nat) (l :: Maybe [Nat]) where
    AppendToMaybe n Nothing = Nothing
    AppendToMaybe n (Just l) = Just (n : l)

type family ComputeBroadcast (shape :: [Nat]) (shape' :: [Nat]) :: Maybe [Nat] where
    ComputeBroadcast '[] shape = Just shape
    ComputeBroadcast shape '[] = Just shape
    ComputeBroadcast (h ': t) (h ': t2) = AppendToMaybe h (ComputeBroadcast t t2)
    ComputeBroadcast (h ': t) (1 ': t2) = AppendToMaybe h (ComputeBroadcast t t2)
    ComputeBroadcast (1 ': t) (h ': t2) = AppendToMaybe h (ComputeBroadcast t t2)
    ComputeBroadcast _ _ = Nothing

type family CheckBroadcast (shape :: [Nat]) (shape' :: [Nat]) (result :: Maybe [Nat]) :: [Nat] where
    CheckBroadcast shape shape' Nothing = TypeError (Text "The shapes " :<>:
                                                       ShowType shape :<>:
                                                       Text " and " :<>:
                                                       ShowType shape' :<>:
                                                       Text " cannot be broadcast")
    CheckBroadcast _ _ (Just result) = (Reverse result)

type Broadcast shape shape' = CheckBroadcast shape shape' (ComputeBroadcast (Reverse shape)
                                                                            (Reverse shape'))

--------------------------------------------------------------------------------
-- Nice error messages for type checking failures
--------------------------------------------------------------------------------

type family DimOutOfBound (shape :: [a]) (dim :: Nat) where
    DimOutOfBound shape dim = TypeError (Text "Out of bound dimension: " :<>:
                                         ShowType dim :<>:
                                         Text " (the tensor is only " :<>:
                                         ShowType (ListLength shape) :<>:
                                         Text "D)")

type family IndexOutOfBound (shape :: [a]) (dim :: Nat) (idx :: Nat) where
    IndexOutOfBound shape dim idx = TypeError (Text "Out of bound index " :<>:
                                               ShowType idx :<>:
                                               Text " for dimension " :<>:
                                               ShowType dim :<>:
                                               Text " (the tensor shape is " :<>:
                                               ShowType shape :<>:
                                               Text ")")

--------------------------------------------------------------------------------
-- Type-level helpers for working with dimension lists
--------------------------------------------------------------------------------

type family ListLength (l :: [a]) :: Nat where
    ListLength '[] = 0
    ListLength (_ ': t) = 1 + ListLength t

                    ----------------------------------------

type family RemoveImpl (l :: [a]) (n :: Nat) :: Maybe [a] where
    RemoveImpl (h ': t) 0 = Just t
    RemoveImpl (h ': t) n = AppendToMaybe h (RemoveImpl t (n - 1))
    RemoveImpl _        _ = Nothing

type family CheckRemove (l :: [a]) (n :: Nat) (result :: Maybe [a]) :: [a] where
    CheckRemove l n Nothing       = DimOutOfBound l n
    CheckRemove _ _ (Just result) = result

type Remove l n = CheckRemove l n (RemoveImpl l n)

                    ----------------------------------------

type family IndexImpl (l :: [a]) (n :: Nat) :: Maybe a where
    IndexImpl (h ': t) 0 = Just h
    IndexImpl (h ': t) n = IndexImpl t (n - 1)
    IndexImpl _        _ = Nothing

type family CheckIndex (l :: [a]) (n :: Nat) (result :: Maybe a) :: a where
    CheckIndex l n Nothing       = DimOutOfBound l n
    CheckIndex _ _ (Just result) = result

type Index l n = CheckIndex l n (IndexImpl l n)

                    ----------------------------------------

type family InRangeCheck (shape :: [Nat]) (dim :: Nat) (idx :: Nat) (ok :: Ordering) :: Constraint where
    InRangeCheck _     _   _   'LT    = ()
    InRangeCheck shape dim idx _      = IndexOutOfBound shape dim idx

type InRange shape dim idx = InRangeCheck shape dim idx (CmpNat idx (Index shape dim))

                    ----------------------------------------

type family ReverseImpl (l :: [a]) (acc :: [a]) :: [a] where
    ReverseImpl '[]      acc = acc
    ReverseImpl (h ': t) acc = ReverseImpl t (h ': acc)

type Reverse l = ReverseImpl l '[]

--------------------------------------------------------------------------------
-- Operations
--------------------------------------------------------------------------------

type family IsAtLeast (n :: Nat) (m :: Nat) (cmp :: Ordering) :: Constraint where
    IsAtLeast n m LT = TypeError (Text "Expected a dimension of size at least " :<>:
                                  ShowType n :<>:
                                  Text " but got " :<>:
                                  ShowType m :<>:
                                  Text "!")
    IsAtLeast _ _ _  = ()

-- IsAtLeast goes first, because while it doesn't help with inferring any
-- inequality constraints, it will give a _significantly_ nicer error message
-- than KnownNat.
-- TODO: This is designed for inequalities of the form <expression> >= constant, but
--       we have variables on both sides in ConvSideCheck which sometimes leads to
--       funny error messages like "expected at least 5, got 29!".
type (>=) (n :: Nat) (m :: Nat) = (IsAtLeast n m (CmpNat n m), KnownNat (n - m))

add :: (shape'' ~ Broadcast shape shape') =>
       Tensor dtype shape -> Tensor dtype shape' -> Tensor dtype shape''
add a b = UnsafeMkTensor $ D.add (toDynamic a) (toDynamic b)

sub :: (shape'' ~ Broadcast shape shape') =>
        Tensor dtype shape -> Tensor dtype shape' -> Tensor dtype shape''
sub a b = UnsafeMkTensor $ D.sub (toDynamic a) (toDynamic b)

mul :: (shape'' ~ Broadcast shape shape') =>
        Tensor dtype shape -> Tensor dtype shape' -> Tensor dtype shape''
mul a b = UnsafeMkTensor $ D.mul (toDynamic a) (toDynamic b)

gt :: (shape'' ~ Broadcast shape shape') =>
      Tensor dtype shape -> Tensor dtype shape' -> Tensor 'D.Bool shape''
gt a b = UnsafeMkTensor $ D.gt (toDynamic a) (toDynamic b)

(>.) = gt

lt :: (shape'' ~ Broadcast shape shape') =>
      Tensor dtype shape -> Tensor dtype shape' -> Tensor 'D.Bool shape''
lt a b = UnsafeMkTensor $ D.lt (toDynamic a) (toDynamic b)

(<.) = lt

ge :: (shape'' ~ Broadcast shape shape') =>
      Tensor dtype shape -> Tensor dtype shape' -> Tensor 'D.Bool shape''
ge a b = UnsafeMkTensor $ D.ge (toDynamic a) (toDynamic b)

(>=.) = ge

le :: (shape'' ~ Broadcast shape shape') =>
      Tensor dtype shape -> Tensor dtype shape' -> Tensor 'D.Bool shape''
le a b = UnsafeMkTensor $ D.le (toDynamic a) (toDynamic b)

(<=.) = le

eq :: (shape'' ~ Broadcast shape shape') =>
      Tensor dtype shape -> Tensor dtype shape' -> Tensor 'D.Bool shape''
eq a b = UnsafeMkTensor $ D.eq (toDynamic a) (toDynamic b)

(==.) = eq

ne :: (shape'' ~ Broadcast shape shape') =>
      Tensor dtype shape -> Tensor dtype shape' -> Tensor 'D.Bool shape''
ne a b = UnsafeMkTensor $ D.ne (toDynamic a) (toDynamic b)

(/=.) = ne

relu :: Tensor dtype shape -> Tensor dtype shape
relu t = UnsafeMkTensor $ D.relu (toDynamic t)

mm :: Tensor dtype [n, k] -> Tensor dtype [k, m] -> Tensor dtype [n, m]
mm a b = UnsafeMkTensor $ D.matmul (toDynamic a) (toDynamic b)

select :: forall dim idx shape dtype shape'.
          (KnownNat dim, KnownNat idx,
           InRange shape dim idx,
           shape' ~ Remove shape dim) => Tensor dtype shape -> Tensor dtype shape'
select t = UnsafeMkTensor $ D.select (toDynamic t) (natValI @dim) (natValI @idx)

selectIdx :: forall dim n shape dtype shape'.
             (KnownNat dim,
              n ~ Index shape dim,
              shape' ~ Remove shape dim) => Tensor dtype shape -> Finite n -> Tensor dtype shape'
selectIdx t idx = UnsafeMkTensor $ D.select (toDynamic t) (natValI @dim) (getFiniteI idx)

type ConvSideCheck h k d (p :: Nat) o =
  (
  -- kernel and step size must be > 0
    k >= 1, d >= 1
  -- kernel size can't be greater than actual input size
  , ((h + (2 * p)) + 1) >= k
  -- output size must be greater than 0
  , o >= 1
  -- output forumlation:
  , o ~ ((Div ((h + (2 * p)) - k) d) + 1)
  )

type family Fst (t :: (a, b)) :: a where
    Fst '(x,_) = x

type family Snd (t :: (a, b)) :: b where
    Snd '(_,x) = x

type family Fst3 (t :: (a, b, c)) :: a where
    Fst3 '(x,_,_) = x

type family Snd3 (t :: (a, b, c)) :: b where
    Snd3 '(_,x,_) = x

type family Trd3 (t :: (a, b, c)) :: c where
    Trd3 '(_,_,x) = x

-- TODO: Perhaps use distinct types for stride and padding so that people
-- don't confuse them later?
conv2dBias
    :: forall stride padding dtype n ic oc ih iw oh ow kh kw.
       ( All KnownNat [Fst stride, Snd stride, Fst padding, Snd padding, n, ic, oc, ih, iw, oh, ow, kh, kw]
       , ConvSideCheck ih kh (Fst stride) (Fst padding) oh
       , ConvSideCheck iw kw (Snd stride) (Snd padding) ow ) =>
         Tensor dtype '[n, ic, ih, iw] ->
         Tensor dtype '[oc, ic, kh, kw] ->
         Tensor dtype '[oc] ->
         Tensor dtype '[n, oc, oh, ow]
conv2dBias input weight bias = UnsafeMkTensor $
    D.conv2d (toDynamic input) (toDynamic weight) (toDynamic bias)
             (natValI @(Fst stride), natValI @(Snd stride))
             (natValI @(Fst padding), natValI @(Snd padding))

maxPool2d :: forall kernel_size stride padding dtype n c ih iw oh ow.
             ( All KnownNat [ Fst kernel_size, Snd kernel_size
                            , Fst stride, Snd stride
                            , Fst padding, Snd padding ]
             , ConvSideCheck ih (Fst kernel_size) (Fst stride) (Fst padding) oh
             , ConvSideCheck iw (Snd kernel_size) (Snd stride) (Snd padding) ow ) =>
               Tensor dtype '[n, c, ih, iw] ->
               Tensor dtype '[n, c, oh, ow]
maxPool2d input = UnsafeMkTensor $
    D.maxPool2d (toDynamic input)
                (natValI @(Fst kernel_size), natValI @(Snd kernel_size))
                (natValI @(Fst stride), natValI @(Snd stride))
                (natValI @(Fst padding), natValI @(Snd padding))


type family Numel (shape :: [Nat]) :: Nat where
    Numel '[] = 1
    Numel (h ': t) = h * (Numel t)

reshape :: forall shape' dtype shape. (KnownShape shape', Numel shape ~ Numel shape') => Tensor dtype shape -> Tensor dtype shape'
reshape t = UnsafeMkTensor $ D.reshape (toDynamic t) (shapeVal @shape')

logSoftmax :: KnownShape shape => Tensor dtype shape -> Int -> Tensor dtype shape
logSoftmax input dim = UnsafeMkTensor $ D.logSoftmax (toDynamic input) dim

--instance Castable (Tensor dtype shape) D.Tensor where
--  cast (UnsafeMkTensor dtensor) f = f dtensor
--  uncast dtensor f = f $ UnsafeMkTensor dtensor

instance Castable (Tensor dtype shape) D.ATenTensor where
  cast (UnsafeMkTensor (D.Unsafe aten_tensor)) f = f aten_tensor
  uncast aten_tensor f = f $ (UnsafeMkTensor (D.Unsafe aten_tensor))

instance Castable [Tensor dtype shape] (ForeignPtr ATen.TensorList) where
  cast xs f = do
    ptr_list <- mapM (\x -> (cast x return :: IO (ForeignPtr ATen.Tensor))) xs
    cast ptr_list f
  uncast xs f = uncast xs $ \ptr_list -> do
    tensor_list <- mapM (\(x :: ForeignPtr ATen.Tensor) -> uncast x return) ptr_list
    f tensor_list
