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
import Data.Kind (Constraint)
import GHC.TypeLits

import qualified Torch.Tensor as D
import qualified Torch.TensorFactories as D
import qualified Torch.Functions as D
import qualified Torch.DType as DType

natValI :: forall n. KnownNat n => Int
natValI = fromIntegral $ natVal $ Proxy @n

class (All KnownNat shape) => KnownShape shape where
    shapeVal :: [Int]

instance KnownShape '[] where
    shapeVal = []

instance (KnownNat h, KnownShape t) => KnownShape (h ': t) where
    shapeVal = natValI @h : shapeVal @t

getFiniteI :: Finite n -> Int
getFiniteI = fromIntegral . getFinite

data Tensor (dtype :: DType.DType) (shape :: [Nat]) = UnsafeMkTensor { toDynamic :: D.Tensor }

instance Show (Tensor dtype shape) where
    show (UnsafeMkTensor dynamic) = show dynamic

class TensorOptions (dtype :: DType.DType) (shape :: [Nat]) where
    optionsRuntimeShape :: [Int]
    ones :: Tensor dtype shape

instance TensorOptions DType.Float '[] where
    optionsRuntimeShape = []
    ones = UnsafeMkTensor $ D.ones' []

instance (KnownNat h, TensorOptions DType.Float t) => TensorOptions DType.Float (h ': t) where
    optionsRuntimeShape = (natValI @h : optionsRuntimeShape @DType.Float @t)
    ones = UnsafeMkTensor $ D.ones' (optionsRuntimeShape @DType.Float @(h ': t))

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
    SomeDType :: forall (dtype :: DType.DType). Proxy dtype -> SomeDType

someDType :: DType.DType -> SomeDType
someDType DType.Float = SomeDType $ Proxy @DType.Float

withTensor :: D.Tensor ->
              (forall (dtype :: DType.DType) (shape :: [Nat]).
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

-- TODO: broadcast with a one!
type family ComputeBroadcast (shape :: [Nat]) (shape' :: [Nat]) :: Maybe [Nat] where
    ComputeBroadcast '[] shape = Just shape
    ComputeBroadcast shape '[] = Just shape
    ComputeBroadcast (h ': t) (h ': t2) = AppendToMaybe h (ComputeBroadcast t t2)
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
