{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Typed.Aux where

import           Data.Proxy
import           GHC.TypeLits

natValI :: forall n . KnownNat n => Int
natValI = fromIntegral $ natVal $ Proxy @n
