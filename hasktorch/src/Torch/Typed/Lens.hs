{-#LANGUAGE RankNTypes#-}
{-#LANGUAGE TypeFamilies #-}
{-#LANGUAGE TypeApplications#-}
{-#LANGUAGE TypeOperators#-}
{-#LANGUAGE DefaultSignatures#-}
{-#LANGUAGE KindSignatures#-}
{-#LANGUAGE FlexibleContexts#-}
{-#LANGUAGE DeriveGeneric#-}
{-#LANGUAGE DataKinds#-}
{-#LANGUAGE MultiParamTypeClasses#-}
{-#LANGUAGE AllowAmbiguousTypes#-}
{-#LANGUAGE FlexibleInstances#-}
{-#LANGUAGE ScopedTypeVariables#-}
{-#LANGUAGE UndecidableInstances#-}
{-#LANGUAGE FunctionalDependencies#-}

module Torch.Typed.Lens where

import Data.Kind
import GHC.Generics
import GHC.TypeLits
import Control.Monad.State.Strict
import Data.Maybe (fromJust)
import Data.Proxy
import Data.Type.Bool
import qualified Torch.Tensor as T
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Functional as D hiding (select)
import Torch.Typed.Tensor
import Data.Reflection hiding (D)
import qualified Torch.Internal.Managed.Type.TensorIndex as ATen
import System.IO.Unsafe

-- | Type alias for lens
type Lens' s a
  = Lens s s a a

type Lens s t a b
  = forall f. Functor f => (a -> f b) -> s -> f t

class FieldIdx field shape => HasField (field :: Symbol) shape where
  field :: Lens' (NamedTensor device dtype shape) (NamedTensor device dtype (DropField field shape))
  field func s = fmap func' (func a')
    where
      index = fieldIdx @field @shape Proxy
      func' :: NamedTensor device dtype (DropField field shape) -> NamedTensor device dtype shape
      func' v = fromUnnamed . UnsafeMkTensor $ T.maskedFill s' index (toDynamic v)
      s' = toDynamic s
      a' :: NamedTensor device dtype (DropField field shape)
      a' = fromUnnamed . UnsafeMkTensor $ (s' T.! index)

type family GHasField (field :: Symbol) f :: Bool where
  GHasField field (S1 ( 'MetaSel ( 'Just field) _ _ _) _) = 'True
  GHasField field (S1 ( 'MetaSel _ _ _ _) _) = 'False
  GHasField field (D1 _ f ) = GHasField field f
  GHasField field (C1 _ f ) = GHasField field f
  GHasField field (l :*: r) = GHasField field l || GHasField field r
  GHasField field (l :+: r) = GHasField field l || GHasField field r
  GHasField field (K1 _ _) = 'False
  GHasField field U1 = 'False

type family DropField (field :: Symbol) (a :: [Type->Type]) :: [Type->Type] where
  DropField field '[] = '[]
  DropField field (x ': xs) = If (GHasField field (Rep (x ()))) xs (DropField field xs)

instance {-# OVERLAPS #-} T.TensorIndex [Maybe Int] where
  pushIndex vec list_of_maybe_int = unsafePerformIO $ do
    idx <- forM list_of_maybe_int $ \i -> do
      case i of
        Nothing -> T.RawTensorIndex <$> ATen.newTensorIndexWithSlice 0 maxBound 1
        Just v -> T.RawTensorIndex <$> ATen.newTensorIndexWithInt (fromIntegral v)
    return $ idx ++ vec

class FieldIdx (field::Symbol) (a :: [Type->Type]) where
  -- | Return field-id
  fieldIdx :: Proxy a -> [Maybe Int]

instance FieldIdx field '[] where
  fieldIdx _ = []

instance (FieldId field (x ()), FieldIdx field xs) => FieldIdx field (x ': xs) where
  fieldIdx _ = fieldId @field @(x ()) Proxy : fieldIdx @field @xs Proxy

class FieldId (field::Symbol) a where
  -- | Return field-id
  fieldId :: Proxy a -> Maybe Int
  default fieldId :: (Generic a, GFieldId field (Rep a)) => Proxy a -> Maybe Int
  fieldId _ = fst $ runState (gfieldId @field @(Rep a) (from (undefined :: a))) 0

class GFieldId (field::Symbol) a where
  gfieldId :: a b -> State Int (Maybe Int)

instance GFieldId field V1 where
  gfieldId _ = return Nothing

instance GFieldId field U1 where
  gfieldId _ = return Nothing

instance (KnownSymbol field, KnownSymbol field_) => GFieldId field (S1 ('MetaSel ('Just field_) p f b) (Rec0 a)) where
  gfieldId _ = do
    i <- get
    put (i+1)
    if (symbolVal (Proxy :: Proxy field) == symbolVal (Proxy :: Proxy field_)) 
      then return (Just i)
      else return Nothing

instance GFieldId field f => GFieldId field (M1 D c f) where
  gfieldId (M1 x) = gfieldId @field x

instance GFieldId field f => GFieldId field (M1 C c f) where
  gfieldId (M1 x) = gfieldId @field x

instance (GFieldId field a, GFieldId field b) => GFieldId field (a :+: b) where
  gfieldId (L1 x) = gfieldId @field x
  gfieldId (R1 x) = gfieldId @field x

instance (GFieldId field a, GFieldId field b) => GFieldId field (a :*: b) where
  gfieldId (a :*: b) = do
    v <- gfieldId @field a
    case v of
      Just v' -> return v
      Nothing -> gfieldId @field b

