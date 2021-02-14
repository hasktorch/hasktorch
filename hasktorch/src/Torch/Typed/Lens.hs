{-#LANGUAGE RankNTypes#-}
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

import GHC.Generics
import GHC.TypeLits
import Control.Monad.State.Strict
import Data.Maybe (fromJust)
import Data.Proxy
import qualified Torch.Tensor as T
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Functional as D hiding (select)
import Torch.Typed.Tensor
import Data.Reflection hiding (D)

-- | Type alias for lens
type Lens' s a
  = Lens s s a a

type Lens s t a b
  = forall f. Functor f => (a -> f b) -> s -> f t

newtype TensorData (device :: (D.DeviceType, Nat)) (dtype :: D.DType) (shape :: [Nat]) a = MkTensorData {toTypedTensor :: Tensor device dtype shape} deriving (Generic)

{-
class (FieldId field s,Monoid s, Generic s, GFieldId field (Rep s)) => THasField (field :: Symbol) s a | field s -> a where
  field :: Lens' (TensorData device dtype shape s) (TensorData device dtype shape a)
  field func s = fmap func' (func a')
    where
      idx = fieldId @field @s (mempty :: s) :: Int
      func' :: TensorData device dtype shape a -> TensorData device dtype shape s
      func' v = toTensorData $ T.maskedFill s' index (toTensor v)
      index = (T.Ellipsis, idx)
      s' = toTensor s
      a' :: TensorData device dtype shape a
      a' = toTensorData (s' T.! index)
      toTensor = toDynamic . toTypedTensor
      toTensorData = (MkTensorData) . (UnsafeMkTensor)
-}

class (FieldId field s,Monoid s, Generic s, GFieldId field (Rep s)) => HasField (field :: Symbol) s where
  field :: Lens' (TensorData device dtype shape s) (Tensor device dtype shape)
  field func s = fmap func' (func a')
    where
      idx = fieldId @field @s (mempty :: s) :: Int
      func' :: Tensor device dtype shape -> TensorData device dtype shape s
      func' v = toTensorData $ T.maskedFill s' index (toDynamic v)
      index = (T.Ellipsis, idx)
      s' = toTensor s
      a' :: Tensor device dtype shape
      a' = UnsafeMkTensor (s' T.! index)
      toTensor = toDynamic . toTypedTensor
      toTensorData = (MkTensorData) . (UnsafeMkTensor)

class CountFields a where
  -- | Return number of constuctor fields for a value.
  countFields :: a -> Int
  default countFields :: (Generic a, GCountFields (Rep a)) => a -> Int
  countFields = gcountFields . from

instance (Generic a, GCountFields (Rep a)) => CountFields a

class GCountFields a where
  gcountFields :: forall b. a b -> Int

instance GCountFields V1 where
  gcountFields _ = 0

instance GCountFields U1 where
  gcountFields _ = 0

instance GCountFields (K1 i c) where
  gcountFields _ = 1

instance (GCountFields f) => GCountFields (M1 i c f) where
  gcountFields (M1 x) = gcountFields x

instance (GCountFields a, GCountFields b) => GCountFields (a :+: b) where
  gcountFields (L1 x) = gcountFields x
  gcountFields (R1 x) = gcountFields x

instance (GCountFields a, GCountFields b) => GCountFields (a :*: b) where
  gcountFields (a :*: b) = gcountFields a + gcountFields b

class FieldId (field::Symbol) a where
  -- | Return field-id
  fieldId :: a -> Int
  default fieldId :: (Generic a, GFieldId field (Rep a)) => a -> Int
  fieldId v = fromJust.fst $ runState (gfieldId @field @(Rep a) (from v)) 0

instance (Generic a, GFieldId field (Rep a)) => FieldId field a

instance FieldId field T.Tensor where
  fieldId _ = error "Tensor does not have any fields."

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

-- instance GFieldId field (K1 i c p) where

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

data RGB = RGB {
  r :: Float,
  g :: Float,
  b :: Float
} deriving (Generic,Show,Eq)

data RGB2 = RGBF {
  rf :: Float,
  gf :: Float,
  bf :: Float
} | RGBD {
  rd :: Double,
  gd :: Double,
  bd :: Double
} deriving (Generic,Show,Eq)
