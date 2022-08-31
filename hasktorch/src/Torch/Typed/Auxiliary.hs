{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE RankNTypes #-}

module Torch.Typed.Auxiliary where

import qualified Data.Int as I
import Data.Kind
import Data.Proxy
import Data.Type.Equality
import GHC.TypeLits
import qualified Torch.DType as D
import qualified Torch.Device as D
import Torch.HList
import Data.Constraint
import Unsafe.Coerce (unsafeCoerce)

natValI :: forall n. KnownNat n => Int
natValI = fromIntegral $ natVal $ Proxy @n

natValInt16 :: forall n. KnownNat n => I.Int16
natValInt16 = fromIntegral $ natVal $ Proxy @n

type family Fst (t :: (a, b)) :: a where
  Torch.Typed.Auxiliary.Fst '(x, _) = x

type family Snd (t :: (a, b)) :: b where
  Torch.Typed.Auxiliary.Snd '(_, x) = x

type family Fst3 (t :: (a, b, c)) :: a where
  Fst3 '(x, _, _) = x

type family Snd3 (t :: (a, b, c)) :: b where
  Snd3 '(_, x, _) = x

type family Trd3 (t :: (a, b, c)) :: c where
  Trd3 '(_, _, x) = x

--------------------------------------------------------------------------------
-- Nice error messages for type checking failures
--------------------------------------------------------------------------------

type family DimOutOfBoundCheckImpl (shape :: [a]) (dim :: Nat) (xs :: [a]) (n :: Nat) :: Constraint where
  DimOutOfBoundCheckImpl shape dim '[] _ = DimOutOfBound shape dim
  DimOutOfBoundCheckImpl _ _ _ 0 = ()
  DimOutOfBoundCheckImpl shape dim (_ ': xs) n = DimOutOfBoundCheckImpl shape dim xs (n - 1)

type DimOutOfBoundCheck shape dim = DimOutOfBoundCheckImpl shape dim shape dim

type family DimOutOfBound (shape :: [a]) (dim :: Nat) where
  DimOutOfBound shape dim =
    TypeError
      ( Text "Out of bound dimension: "
          :<>: ShowType dim
          :<>: Text " (the tensor is only "
          :<>: ShowType (ListLength shape)
          :<>: Text "D)"
      )

type family IndexOutOfBound (shape :: [a]) (dim :: Nat) (idx :: Nat) where
  IndexOutOfBound shape dim idx =
    TypeError
      ( Text "Out of bound index "
          :<>: ShowType idx
          :<>: Text " for dimension "
          :<>: ShowType dim
          :<>: Text " (the tensor shape is "
          :<>: ShowType shape
          :<>: Text ")"
      )

--------------------------------------------------------------------------------
-- Type-level helpers for working with dimension lists
--------------------------------------------------------------------------------

type family AppendToMaybe (h :: a) (mt :: Maybe [a]) :: Maybe [a] where
  AppendToMaybe h Nothing = Nothing
  AppendToMaybe h (Just t) = Just (h : t)

type family AppendToMaybe' (h :: Maybe a) (mt :: Maybe [a]) :: Maybe [a] where
  AppendToMaybe' Nothing _ = Nothing
  AppendToMaybe' _ Nothing = Nothing
  AppendToMaybe' (Just h) (Just t) = Just (h : t)

type family MaybePrepend (mh :: Maybe a) (t :: [a]) :: [a] where
  MaybePrepend Nothing t = t
  MaybePrepend (Just h) t = h : t

----------------------------------------

type family LastDim (l :: [a]) :: Nat where
  LastDim (_ ': '[]) = 0
  LastDim (_ ': t) = 1 + LastDim t

type family Product (xs :: [Nat]) :: Nat where
  Product '[] = 1
  Product (x ': xs) = x GHC.TypeLits.* Product xs

type family BackwardsImpl (last :: Nat) (n :: Nat) :: Nat where
  BackwardsImpl last n = last - n

type Backwards l n = BackwardsImpl (LastDim l) n

-- | Evaluate a type-level constraint for whether or not the former shape is a suffix of the latter shape
--
-- >>> :kind! IsSuffixOf '[1] '[1]
-- IsSuffixOf '[1] '[1] :: Constraint
-- = () :: Constraint
-- >>> :kind! IsSuffixOf '[1] '[2, 1]
-- IsSuffixOf '[1] '[2, 1] :: Constraint
-- = () :: Constraint
-- >>> :kind! IsSuffixOf '[2] '[2, 1]
-- IsSuffixOf '[2] '[2, 1] :: Constraint
-- = (TypeError ...)
-- >>> :kind! IsSuffixOf '[1, 1] '[2, 1]
-- IsSuffixOf '[1, 1] '[2, 1] :: Constraint
-- = (TypeError ...)
-- >>> :kind! IsSuffixOf '[2, 1] '[2, 1]
-- IsSuffixOf '[2, 1] '[2, 1] :: Constraint
-- = () :: Constraint
type IsSuffixOf xs ys = CheckIsSuffixOf xs ys (IsSuffixOfImpl xs ys (DropLengthMaybe xs ys))

type family CheckIsSuffixOf (xs :: [a]) (ys :: [a]) (result :: Bool) :: Constraint where
  CheckIsSuffixOf _ _ 'True = ()
  CheckIsSuffixOf xs ys 'False = TypeError (ShowType xs :<>: Text " is not a suffix of " :<>: ShowType ys)

type family IsSuffixOfImpl (xs :: [a]) (ys :: [a]) (mDelta :: Maybe [b]) :: Bool where
  IsSuffixOfImpl xs ys ('Just delta) = xs == DropLength delta ys
  IsSuffixOfImpl _ _ 'Nothing = 'False

type family DropLengthMaybe (xs :: [a]) (ys :: [b]) :: Maybe [b] where
  DropLengthMaybe '[] ys = 'Just ys
  DropLengthMaybe _ '[] = 'Nothing
  DropLengthMaybe (_ : xs) (_ : ys) = DropLengthMaybe xs ys

type family DropLength (xs :: [a]) (ys :: [b]) :: [b] where
  DropLength '[] ys = ys
  DropLength _ '[] = '[]
  DropLength (_ : xs) (_ : ys) = DropLength xs ys

----------------------------------------

type family Init (xs :: [a]) :: [a] where
  Init '[] = TypeError (Text "Init of empty list.")
  Init (x ': '[]) = '[]
  Init (x ': xs) = x ': Init xs

type family Last (xs :: [a]) :: a where
  Last '[] = TypeError (Text "Last of empty list.")
  Last (x ': '[]) = x
  Last (x ': xs) = Last xs

type family InsertImpl (n :: Nat) (x :: a) (l :: [a]) :: Maybe [a] where
  InsertImpl 0 x l = Just (x ': l)
  InsertImpl n x '[] = Nothing
  InsertImpl n x (h ': t) = AppendToMaybe h (InsertImpl (n - 1) x t)

type family CheckInsert (n :: Nat) (x :: a) (l :: [a]) (result :: Maybe [a]) :: [a] where
  CheckInsert _ _ _ (Just xs) = xs
  CheckInsert n x l Nothing = DimOutOfBound l n

type family Insert (n :: Nat) (x :: a) (l :: [a]) :: [a] where
  Insert n x l = CheckInsert n x l (InsertImpl n x l)

type family RemoveImpl (l :: [a]) (n :: Nat) :: Maybe [a] where
  RemoveImpl (h ': t) 0 = Just t
  RemoveImpl (h ': t) n = AppendToMaybe h (RemoveImpl t (n - 1))
  RemoveImpl _ _ = Nothing

type family CheckRemove (l :: [a]) (n :: Nat) (result :: Maybe [a]) :: [a] where
  CheckRemove l n Nothing = DimOutOfBound l n
  CheckRemove _ _ (Just result) = result

type Remove l n = CheckRemove l n (RemoveImpl l n)

----------------------------------------

type family IndexImpl (l :: [a]) (n :: Nat) :: Maybe a where
  IndexImpl (h ': t) 0 = Just h
  IndexImpl (h ': t) n = IndexImpl t (n - 1)
  IndexImpl _ _ = Nothing

type family CheckIndex (l :: [a]) (n :: Nat) (result :: Maybe a) :: a where
  CheckIndex l n Nothing = DimOutOfBound l n
  CheckIndex _ _ (Just result) = result

type Index l n = CheckIndex l n (IndexImpl l n)

----------------------------------------

type family InRangeCheck (shape :: [Nat]) (dim :: Nat) (idx :: Nat) (ok :: Ordering) :: Constraint where
  InRangeCheck _ _ _ 'LT = ()
  InRangeCheck shape dim idx _ = IndexOutOfBound shape dim idx

type InRange shape dim idx = InRangeCheck shape dim idx (CmpNat idx (Index shape dim))

----------------------------------------

type family ReverseImpl (l :: [a]) (acc :: [a]) :: [a] where
  ReverseImpl '[] acc = acc
  ReverseImpl (h ': t) acc = ReverseImpl t (h ': acc)

type Reverse l = ReverseImpl l '[]

type family ExtractDim (dim :: Nat) (shape :: [Nat]) :: Maybe Nat where
  ExtractDim 0 (h ': _) = Just h
  ExtractDim dim (_ ': t) = ExtractDim (dim - 1) t
  ExtractDim _ _ = Nothing

type family ReplaceDim (dim :: Nat) (shape :: [Nat]) (n :: Nat) :: Maybe [Nat] where
  ReplaceDim 0 (_ ': t) n = Just (n ': t)
  ReplaceDim dim (h ': t) n = AppendToMaybe h (ReplaceDim (dim - 1) t n)
  ReplaceDim _ _ _ = Nothing

type family If c t e where
  If 'True t e = t
  If 'False t e = e

type family AllDimsPositive (shape :: [Nat]) :: Constraint where
  AllDimsPositive '[] = ()
  AllDimsPositive (x ': xs) = If (1 <=? x) (AllDimsPositive xs) (TypeError (Text "Expected positive dimension but got " :<>: ShowType x :<>: Text "!"))

--------------------------------------------------------------------------------
-- Operations
--------------------------------------------------------------------------------

type family IsAtLeast (n :: Nat) (m :: Nat) (cmp :: Ordering) :: Constraint where
  IsAtLeast n m LT =
    TypeError
      ( Text "Expected a dimension of size at least "
          :<>: ShowType n
          :<>: Text " but got "
          :<>: ShowType m
          :<>: Text "!"
      )
  IsAtLeast _ _ _ = ()

-- IsAtLeast goes first, because while it doesn't help with inferring any
-- inequality constraints, it will give a _significantly_ nicer error message
-- than KnownNat.
-- TODO: This is designed for inequalities of the form <expression> >= constant, but
--       we have variables on both sides in ConvSideCheck which sometimes leads to
--       funny error messages like "expected at least 5, got 29!".
type (>=) (n :: Nat) (m :: Nat) = (IsAtLeast n m (CmpNat n m), KnownNat (n - m))

--------------------------------------------------------------------------------
-- DType Promotion
--------------------------------------------------------------------------------

type family CmpDType (dtype :: D.DType) (dtype' :: D.DType) :: Ordering where
  CmpDType dtype dtype = 'EQ
  CmpDType D.Bool D.UInt8 = 'LT
  CmpDType D.Bool D.Int8 = 'LT
  CmpDType D.Bool D.Int16 = 'LT
  CmpDType D.Bool D.Int32 = 'LT
  CmpDType D.Bool D.Int64 = 'LT
  CmpDType D.Bool D.Half = 'LT
  CmpDType D.Bool D.Float = 'LT
  CmpDType D.Bool D.Double = 'LT
  CmpDType D.UInt8 D.Int8 = 'LT
  CmpDType D.UInt8 D.Int16 = 'LT
  CmpDType D.UInt8 D.Int32 = 'LT
  CmpDType D.UInt8 D.Int64 = 'LT
  CmpDType D.UInt8 D.Half = 'LT
  CmpDType D.UInt8 D.Float = 'LT
  CmpDType D.UInt8 D.Double = 'LT
  CmpDType D.Int8 D.Int16 = 'LT
  CmpDType D.Int8 D.Int32 = 'LT
  CmpDType D.Int8 D.Int64 = 'LT
  CmpDType D.Int8 D.Half = 'LT
  CmpDType D.Int8 D.Float = 'LT
  CmpDType D.Int8 D.Double = 'LT
  CmpDType D.Int16 D.Int32 = 'LT
  CmpDType D.Int16 D.Int64 = 'LT
  CmpDType D.Int16 D.Half = 'LT
  CmpDType D.Int16 D.Float = 'LT
  CmpDType D.Int16 D.Double = 'LT
  CmpDType D.Int32 D.Int64 = 'LT
  CmpDType D.Int32 D.Half = 'LT
  CmpDType D.Int32 D.Float = 'LT
  CmpDType D.Int32 D.Double = 'LT
  CmpDType D.Int64 D.Half = 'LT
  CmpDType D.Int64 D.Float = 'LT
  CmpDType D.Int64 D.Double = 'LT
  CmpDType D.Half D.Float = 'LT
  CmpDType D.Half D.Double = 'LT
  CmpDType D.Float D.Double = 'LT
  CmpDType _ _ = 'GT

type family DTypePromotionImpl (dtype :: D.DType) (dtype' :: D.DType) (ord :: Ordering) :: D.DType where
  DTypePromotionImpl D.UInt8 D.Int8 _ = D.Int16
  DTypePromotionImpl D.Int8 D.UInt8 _ = D.Int16
  DTypePromotionImpl dtype _ EQ = dtype
  DTypePromotionImpl _ dtype LT = dtype
  DTypePromotionImpl dtype _ GT = dtype

type DTypePromotion dtype dtype' = DTypePromotionImpl dtype dtype' (CmpDType dtype dtype')

--------------------------------------------------------------------------------
-- DType Validation
--------------------------------------------------------------------------------

type family DTypeIsFloatingPoint (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  DTypeIsFloatingPoint _ 'D.Half = ()
  DTypeIsFloatingPoint _ 'D.Float = ()
  DTypeIsFloatingPoint _ 'D.Double = ()
  DTypeIsFloatingPoint '(deviceType, _) dtype = UnsupportedDTypeForDevice deviceType dtype

type family DTypeIsIntegral (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  DTypeIsIntegral _ 'D.Bool = ()
  DTypeIsIntegral _ 'D.UInt8 = ()
  DTypeIsIntegral _ 'D.Int8 = ()
  DTypeIsIntegral _ 'D.Int16 = ()
  DTypeIsIntegral _ 'D.Int32 = ()
  DTypeIsIntegral _ 'D.Int64 = ()
  DTypeIsIntegral '(deviceType, _) dtype = UnsupportedDTypeForDevice deviceType dtype

type family DTypeIsNotHalf (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  DTypeIsNotHalf '(deviceType, _) D.Half = UnsupportedDTypeForDevice deviceType D.Half
  DTypeIsNotHalf _ _ = ()

type family DTypeIsNotBool (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  DTypeIsNotBool '(deviceType, _) D.Bool = UnsupportedDTypeForDevice deviceType D.Bool
  DTypeIsNotBool _ _ = ()

type family UnsupportedDTypeForDevice (deviceType :: D.DeviceType) (dtype :: D.DType) :: Constraint where
  UnsupportedDTypeForDevice deviceType dtype =
    TypeError
      ( Text "This operation does not support "
          :<>: ShowType dtype
          :<>: Text " tensors on devices of type "
          :<>: ShowType deviceType
          :<>: Text "."
      )

type family StandardFloatingPointDTypeValidation (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  StandardFloatingPointDTypeValidation '( 'D.CPU, 0) dtype =
    ( DTypeIsFloatingPoint '( 'D.CPU, 0) dtype,
      DTypeIsNotHalf '( 'D.CPU, 0) dtype
    )
  StandardFloatingPointDTypeValidation '( 'D.CUDA, deviceIndex) dtype = DTypeIsFloatingPoint '( 'D.CUDA, deviceIndex) dtype
  StandardFloatingPointDTypeValidation '(deviceType, _) dtype = UnsupportedDTypeForDevice deviceType dtype

type family StandardDTypeValidation (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  StandardDTypeValidation '( 'D.CPU, 0) dtype =
    ( DTypeIsNotBool '( 'D.CPU, 0) dtype,
      DTypeIsNotHalf '( 'D.CPU, 0) dtype
    )
  StandardDTypeValidation '( 'D.CUDA, deviceIndex) dtype = DTypeIsNotBool '( 'D.CUDA, deviceIndex) dtype
  StandardDTypeValidation '(deviceType, _) dtype = UnsupportedDTypeForDevice deviceType dtype


--------------------------------------------------------------------------------
-- An unsafe function to enforce a constraint.
--------------------------------------------------------------------------------

unsafeConstraint :: forall c a. (c => a) -> a
unsafeConstraint = withDict (dummyDict @c)
  where
    dummyDict :: forall b. Dict b
    dummyDict = unsafeCoerce (Dict :: Dict ())

--------------------------------------------------------------------------------
-- Helper functions to handle nat at runtime.
--------------------------------------------------------------------------------

withNat ::
  Int ->
  ( forall n.
    KnownNat n =>
    Proxy n ->
    r
  ) ->
  r
withNat i f = case someNatVal (fromIntegral i) of
  Nothing -> error "Negative Number in withNat!"
  (Just (SomeNat p)) -> f p

forEachNat :: forall n a. KnownNat n => (forall i. KnownNat i => Proxy i -> a) -> [a]
forEachNat func = map (\i -> withNat i func) [0 .. (natValI @n -1)]
