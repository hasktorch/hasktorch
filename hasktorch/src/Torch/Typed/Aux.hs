{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.Typed.Aux where

import qualified Data.Int                      as I
import           Data.Proxy
import           GHC.TypeLits

type family Fst (t :: (a, b)) :: a where
  Fst '(x, _) = x

type family Snd (t :: (a, b)) :: b where
  Snd '(_, x) = x

type family Fst3 (t :: (a, b, c)) :: a where
  Fst3 '(x, _, _) = x

type family Snd3 (t :: (a, b, c)) :: b where
  Snd3 '(_, x, _) = x

type family Trd3 (t :: (a, b, c)) :: c where
  Trd3 '(_, _, x) = x

natValI :: forall n . KnownNat n => Int
natValI = fromIntegral $ natVal $ Proxy @n

natValInt16 :: forall n . KnownNat n => I.Int16
natValInt16 = fromIntegral $ natVal $ Proxy @n
