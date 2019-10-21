{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Typed.Aux where

import qualified Data.Int                      as I
import           Data.Proxy
import           GHC.TypeLits

natValI :: forall n . KnownNat n => Int
natValI = fromIntegral $ natVal $ Proxy @n

natValInt16 :: forall n . KnownNat n => I.Int16
natValInt16 = fromIntegral $ natVal $ Proxy @n
