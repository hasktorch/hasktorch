{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Typed.Lens where

import Control.Applicative (liftA2)
import Control.Monad.State.Strict
import Control.Monad (forM)
import Data.Kind
import Data.Maybe (fromJust)
import Data.Proxy
import Data.Reflection hiding (D)
import Data.Type.Bool
import Data.Vector.Sized (Vector)
import GHC.Generics
import GHC.TypeLits
import System.IO.Unsafe
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Functional as D hiding (select)
import qualified Torch.Functional.Internal as I
import qualified Torch.Internal.Managed.Type.TensorIndex as ATen
import Torch.Lens (Lens, Lens', Traversal, Traversal')
import qualified Torch.Tensor as T
import Torch.Typed.Auxiliary hiding (If)
import Torch.Typed.Tensor

class HasName (name :: Type -> Type) shape where
  name :: Traversal' (NamedTensor device dtype shape) (NamedTensor device dtype (DropName name shape))
  default name :: (KnownNat (NamedIdx name shape)) => Traversal' (NamedTensor device dtype shape) (NamedTensor device dtype (DropName name shape))
  name func s = func'
    where
      dimension :: Int
      dimension = natValI @(NamedIdx name shape)
      func' = (\v -> (fromUnnamed . UnsafeMkTensor $ D.stack (D.Dim dimension) (map toDynamic v))) <$> swapA (map func a') (pure [])
      s' = toDynamic s
      swapA [] v = v
      swapA (x : xs) v = swapA xs (liftA2 (\a b -> b ++ [a]) x v)
      a' :: [NamedTensor device dtype (DropName name shape)]
      a' = map (fromUnnamed . UnsafeMkTensor) $ I.unbind s' dimension

instance (KnownNat (NamedIdx name shape)) => HasName name shape

class HasField (field :: Symbol) shape where
  field :: Lens' (NamedTensor device dtype shape) (NamedTensor device dtype (DropField field shape))
  default field :: (FieldIdx field shape) => Lens' (NamedTensor device dtype shape) (NamedTensor device dtype (DropField field shape))
  field func s = fmap func' (func a')
    where
      index = fieldIdx @field @shape Proxy
      func' :: NamedTensor device dtype (DropField field shape) -> NamedTensor device dtype shape
      func' v = fromUnnamed . UnsafeMkTensor $ T.maskedFill s' index (toDynamic v)
      s' = toDynamic s
      a' :: NamedTensor device dtype (DropField field shape)
      a' = fromUnnamed . UnsafeMkTensor $ (s' T.! index)

instance {-# OVERLAPS #-} FieldIdx field shape => HasField field shape

type family GHasField (field :: Symbol) f :: Bool where
  GHasField field (S1 ('MetaSel ('Just field) _ _ _) _) = 'True
  GHasField field (S1 ('MetaSel _ _ _ _) _) = 'False
  GHasField field (D1 _ f) = GHasField field f
  GHasField field (C1 _ f) = GHasField field f
  GHasField field (l :*: r) = GHasField field l || GHasField field r
  GHasField field (l :+: r) = GHasField field l || GHasField field r
  GHasField field (K1 _ _) = 'False
  GHasField field U1 = 'False
  GHasField field (Vector n) = 'False
  GHasField field a = GHasField field (Rep (a ()))

type family DropField (field :: Symbol) (a :: [Type -> Type]) :: [Type -> Type] where
  DropField field '[] = '[]
  DropField field (x ': xs) = If (GHasField field x) xs (x ': DropField field xs)

type family DropName (name :: Type -> Type) (a :: [Type -> Type]) :: [Type -> Type] where
  DropName name '[] = '[]
  DropName name (name ': xs) = xs
  DropName name (x ': xs) = x ': DropName name xs

instance {-# OVERLAPS #-} T.TensorIndex [Maybe Int] where
  pushIndex vec list_of_maybe_int = unsafePerformIO $ do
    idx <- forM list_of_maybe_int $ \i -> do
      case i of
        Nothing -> T.RawTensorIndex <$> ATen.newTensorIndexWithSlice 0 maxBound 1
        Just v -> T.RawTensorIndex <$> ATen.newTensorIndexWithInt (fromIntegral v)
    return $ idx ++ vec

type family NamedIdx (name :: Type -> Type) (shape :: [Type -> Type]) :: Nat where
  NamedIdx name '[] = TypeError (Text "There is not the name in the shape.")
  NamedIdx name (name ': xs) = 0
  NamedIdx name (x ': xs) = NamedIdx name xs + 1

class FieldIdx (field :: Symbol) (a :: [Type -> Type]) where
  -- | Return field-id
  fieldIdx :: Proxy a -> [Maybe Int]

instance FieldIdx field '[] where
  fieldIdx _ = []

instance (FieldId field (x ()), FieldIdx field xs) => FieldIdx field (x ': xs) where
  fieldIdx _ = fieldId @field @(x ()) Proxy : fieldIdx @field @xs Proxy

class FieldId (field :: Symbol) a where
  -- | Return field-id
  fieldId :: Proxy a -> Maybe Int
  default fieldId :: (Generic a, GFieldId field (Rep a)) => Proxy a -> Maybe Int
  fieldId _ = gfieldId @field (Proxy :: Proxy (Rep a))

instance FieldId field (Vector n v) where
  fieldId _ = Nothing

instance {-# OVERLAPS #-} (Generic s, GFieldId field (Rep s)) => FieldId field s

class GFieldId (field :: Symbol) (a :: Type -> Type) where
  gfieldId :: Proxy a -> Maybe Int
  gfieldId p = fst $ gfieldId' @field @a p
  gfieldId' :: Proxy a -> (Maybe Int, Int)

instance (GFieldId field f) => GFieldId field (M1 D t f) where
  gfieldId' _ = gfieldId' @field (Proxy :: Proxy f)

instance (GFieldId field f) => GFieldId field (M1 C t f) where
  gfieldId' _ = gfieldId' @field (Proxy :: Proxy f)

instance (KnownSymbol field, KnownSymbol field_) => GFieldId field (S1 ('MetaSel ('Just field_) p f b) (Rec0 a)) where
  gfieldId' _ =
    if symbolVal (Proxy :: Proxy field) == symbolVal (Proxy :: Proxy field_)
      then (Just 0, 1)
      else (Nothing, 1)

instance GFieldId field (K1 c f) where
  gfieldId' _ = (Nothing, 1)

instance GFieldId field U1 where
  gfieldId' _ = (Nothing, 1)

instance (GFieldId field f, GFieldId field g) => GFieldId field (f :*: g) where
  gfieldId' _ =
    case (gfieldId' @field (Proxy :: Proxy f), gfieldId' @field (Proxy :: Proxy g)) of
      ((Nothing, t0), (Nothing, t1)) -> (Nothing, t0 + t1)
      ((Nothing, t0), (Just v1, t1)) -> (Just (v1 + t0), t1 + t0)
      ((Just v0, t0), (_, t1)) -> (Just v0, t0 + t1)

instance (GFieldId field f, GFieldId field g) => GFieldId field (f :+: g) where
  gfieldId' _ =
    case (gfieldId' @field (Proxy :: Proxy f), gfieldId' @field (Proxy :: Proxy g)) of
      ((Nothing, _), a1) -> a1
      (a0@(Just _, _), _) -> a0
