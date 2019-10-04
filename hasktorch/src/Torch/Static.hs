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

-- type family Flip (constraint :: a -> b -> Constraint) (fst :: b) (snd :: a) :: Constraint where
--   Flip constraint b a = constraint a b

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

type family AppendToMaybe (h :: a) (mt :: Maybe [a]) :: Maybe [a]  where
    AppendToMaybe h Nothing  = Nothing
    AppendToMaybe h (Just t) = Just (h : t)

type family AppendToMaybe' (h :: Maybe a) (mt :: Maybe [a]) :: Maybe [a] where
  AppendToMaybe' Nothing  _        = Nothing
  AppendToMaybe' _        Nothing  = Nothing
  AppendToMaybe' (Just h) (Just t) = Just (h : t)

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

--------------------------------------------------------------------------------
-- Nice error messages for type checking failures
--------------------------------------------------------------------------------

type family DimOutOfBoundCheckImpl (shape :: [a]) (dim :: Nat) (xs :: [a]) (n :: Nat) :: Constraint where
  DimOutOfBoundCheckImpl shape dim '[]       _ = DimOutOfBound shape dim
  DimOutOfBoundCheckImpl _     _   _         0 = ()
  DimOutOfBoundCheckImpl shape dim (_ ': xs) n = DimOutOfBoundCheckImpl shape dim xs (n - 1)

type DimOutOfBoundCheck shape dim = DimOutOfBoundCheckImpl shape dim shape dim

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
    ListLength '[]      = 0
    ListLength (_ ': t) = 1 + ListLength t

type family LastDim (l :: [a]) :: Nat where
  LastDim (_ ': '[]) = 0
  LastDim (_ ': t)   = 1 + LastDim t

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

type family ExtractDim (dim :: Nat) (shape :: [Nat]) :: Maybe Nat where
  ExtractDim 0   (h ': _) = Just h
  ExtractDim dim (_ ': t) = ExtractDim (dim - 1) t
  ExtractDim _   _        = Nothing

type family ReplaceDim (dim :: Nat) (shape :: [Nat]) (n :: Nat) :: Maybe [Nat] where
  ReplaceDim 0   (_ ': t) n = Just (n ': t)
  ReplaceDim dim (h ': t) n = AppendToMaybe h (ReplaceDim (dim - 1) t n)
  ReplaceDim _   _        _ = Nothing

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

-- | matmul, see https://pytorch.org/docs/stable/torch.html#torch.matmul
matmul
  :: forall dtype shape shape' shape''
   . (shape'' ~ MatMul shape shape')
  => Tensor dtype shape
  -> Tensor dtype shape'
  -> Tensor dtype shape''
matmul a b = UnsafeMkTensor $ D.matmul (toDynamic a) (toDynamic b)

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

type family Numel (shape :: [Nat]) :: Nat where
    Numel '[] = 1
    Numel (h ': t) = h * (Numel t)

reshape :: forall shape' dtype shape. (KnownShape shape', Numel shape ~ Numel shape') => Tensor dtype shape -> Tensor dtype shape'
reshape t = UnsafeMkTensor $ D.reshape (toDynamic t) (shapeVal @shape')

instance Castable (Tensor dtype shape) D.ATenTensor where
  cast (UnsafeMkTensor (D.Unsafe aten_tensor)) f = f aten_tensor
  uncast aten_tensor f = f $ UnsafeMkTensor (D.Unsafe aten_tensor)

instance Castable [Tensor dtype shape] (ForeignPtr ATen.TensorList) where
  cast xs f = do
    ptr_list <- mapM (\x -> (cast x return :: IO (ForeignPtr ATen.Tensor))) xs
    cast ptr_list f
  uncast xs f = uncast xs $ \ptr_list -> do
    tensor_list <- mapM (\(x :: ForeignPtr ATen.Tensor) -> uncast x return) ptr_list
    f tensor_list


data family HList (l :: [Type])
data instance HList '[] = HNil
newtype instance HList (x ': xs) = HCons (x, HList xs)
pattern (:.) x xs = HCons (x, xs)

infixr 2 :.

instance Eq (HList '[]) where
  HNil == HNil = True

instance (Eq x, Eq (HList xs)) => Eq (HList (x ': xs)) where
  (x :. xs) == (y :. ys) = x == y && xs == ys

class Apply f a b where
  apply :: f -> a -> b

class HMap f xs ys where
  hmap :: f -> HList xs -> HList ys

instance HMap f '[] '[] where
  hmap _ _ = HNil

instance (Apply f x y, HMap f xs ys) => HMap f (x ': xs) (y ': ys) where
  hmap f (x :. xs) = (apply f x) :. (hmap f xs)

class HFoldr f acc xs where
  hfoldr :: f -> acc -> HList xs -> acc

instance HFoldr f acc '[] where
  hfoldr _ acc _ = acc

instance (Apply f x (acc -> acc), HFoldr f acc xs) => HFoldr f acc (x ': xs) where
  hfoldr f acc (x :. xs) = apply f x $ hfoldr f acc xs

class HFoldrM m f acc xs where
  hfoldrM :: f -> acc -> HList xs -> m acc

instance (Monad m) => HFoldrM m f acc '[] where
  hfoldrM _ acc _ = pure acc

instance (Monad m, Apply f x (acc -> m acc), HFoldrM m f acc xs) => HFoldrM m f acc (x ': xs) where
  hfoldrM f acc (x :. xs) = apply f x =<< hfoldrM f acc xs

data HNothing  = HNothing
data HJust x   = HJust x

class HUnfold f res xs where
  hunfoldr' :: f -> res -> HList xs

type family HUnfoldRes s xs where
  HUnfoldRes _ '[] = HNothing
  HUnfoldRes s (x ': _) = HJust (x, s)

instance HUnfold f HNothing '[] where
  hunfoldr' _ _ = HNil

instance (Apply f s res, HUnfold f res xs, res ~ HUnfoldRes s xs) => HUnfold f (HJust (x, s)) (x ': xs) where
  hunfoldr' f (HJust (x, s)) = x :. (hunfoldr' f (apply f s :: res))

hunfoldr
  :: forall f res (xs :: [Type]) a
   . (Apply f a res, HUnfold f res xs)
  => f
  -> a
  -> HList xs
hunfoldr f s = hunfoldr' f (apply f s :: res)

class HUnfoldM m f res xs where
  hunfoldrM' :: f -> res -> m (HList xs)

type family HUnfoldMRes m s xs where
  HUnfoldMRes m _ '[] = m HNothing
  HUnfoldMRes m s (x ': _) = m (HJust (x, s))

instance (Monad m) => HUnfoldM m f (m HNothing) '[] where
  hunfoldrM' _ _ = pure HNil

instance (Monad m, HUnfoldM m f res xs, Apply f s res, res ~ HUnfoldMRes m s xs) => HUnfoldM m f (m (HJust (x, s))) (x ': xs) where
  hunfoldrM' f just = do
    HJust (x, s) <- just
    xs <- hunfoldrM' f (apply f s :: res)
    return (x :. xs)

hunfoldrM
  :: forall (m :: Type -> Type) f res (xs :: [Type]) a
   . (HUnfoldM m f res xs, Apply f a res, res ~ HUnfoldMRes m a xs)
  => f
  -> a
  -> m (HList xs)
hunfoldrM f s = hunfoldrM' f (apply f s :: res)

data TensorListFolds = TensorListFolds

instance (Castable x D.ATenTensor) => Apply TensorListFolds x ([D.ATenTensor] -> IO [D.ATenTensor]) where
  apply _ x = \xs -> do
    x' <- cast x return
    return (x' : xs)

instance Apply TensorListFolds [D.ATenTensor] (IO HNothing) where
  apply _ [] = pure HNothing

instance (Castable x D.ATenTensor) => Apply TensorListFolds [D.ATenTensor] (IO (HJust (x, [D.ATenTensor]))) where
  apply _ (x : xs) = do
    x' <- uncast x return
    return $ HJust (x', xs)

instance (HFoldrM IO TensorListFolds [D.ATenTensor] l, Apply TensorListFolds [D.ATenTensor] res, HUnfoldM IO TensorListFolds res l, res ~ (HUnfoldMRes IO [D.ATenTensor] l)) => Castable (HList l) [D.ATenTensor] where
  cast xs f = f =<< go xs
   where
    go :: HList l -> IO [D.ATenTensor]
    go xs = hfoldrM TensorListFolds [] xs
  uncast xs f = f =<< go xs
   where
    go :: [D.ATenTensor] -> IO (HList l)
    go xs = hunfoldrM TensorListFolds xs

instance Castable (HList l) [D.ATenTensor] => Castable (HList l) (ForeignPtr ATen.TensorList) where
  cast xs f = do
    ts <- cast xs return :: IO [ForeignPtr ATen.Tensor]
    cast ts f
  uncast xs f = uncast xs $ \(ptrList :: [ForeignPtr ATen.Tensor]) -> do
    ts <- uncast ptrList return :: IO (HList l)
    f ts

test :: forall dtype shape . Tensor dtype shape -> IO [D.ATenTensor]
test t = hfoldrM TensorListFolds [] (t :. HNil)

test' :: forall dtype shape dtype' shape' . [D.ATenTensor] -> IO (HList '[Tensor dtype shape, Tensor dtype' shape'])
test' xs = hunfoldrM TensorListFolds xs

test'' :: HList '[Tensor dtype shape] -> IO [D.ATenTensor]
test'' xs = cast xs return

test''' :: [D.ATenTensor] -> IO (HList '[Tensor dtype shape])
test''' xs = uncast xs return

class (ListLength es ~ n) => HReplicate' (n :: Nat) e es where
    hReplicate :: Proxy n -> e -> HList es

instance HReplicate' 0 e '[] where
    hReplicate _ _ = HNil

instance (HReplicate' (n - 1) e es, e ~ e', 1 <= n) => HReplicate' n e (e' ': es) where
    hReplicate n e = e :. hReplicate (Proxy @(n - 1)) e

type HReplicate n e = HReplicate' n e (HReplicateR n e)

type family HReplicateR (n :: Nat) (e :: a) :: [a] where
  HReplicateR 0 e = '[]
  HReplicateR n e = e ': HReplicateR (n - 1) e

testReplicate :: forall dtype shape . Tensor dtype shape -> HList (HReplicateR 3 (Tensor dtype shape))
testReplicate t = hReplicate Proxy t

--------------------------------------------------------------------------------
-- Move backend
--------------------------------------------------------------------------------

toSparse :: Tensor dtype shape -> Tensor dtype shape
toSparse t = UnsafeMkTensor $ D.toSparse (toDynamic t)

toDense :: Tensor dtype shape -> Tensor dtype shape
toDense t = UnsafeMkTensor $ D.toDense (toDynamic t)

toMKLDNN :: Tensor dtype shape -> Tensor dtype shape
toMKLDNN t = UnsafeMkTensor $ D.toMKLDNN (toDynamic t)

toCPU :: Tensor dtype shape -> Tensor dtype shape
toCPU t = UnsafeMkTensor $ D.toCPU (toDynamic t)

toCUDA :: Tensor dtype shape -> Tensor dtype shape
toCUDA t = UnsafeMkTensor $ D.toCUDA (toDynamic t)

--------------------------------------------------------------------------------
-- Auxiliary functions for accessing tensor properties as values
--------------------------------------------------------------------------------

dim :: Tensor dtype shape -> Int
dim t = D.dim $ toDynamic t

shape :: Tensor dtype shape -> [Int]
shape t = D.shape $ toDynamic t

dtype :: Tensor dtype shape -> D.DType
dtype t = D.dtype $ toDynamic t

toInt :: Tensor dtype shape -> Int
toInt t = D.toInt $ toDynamic t
