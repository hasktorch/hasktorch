{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE DeriveAnyClass #-}

module Serialise where

import GHC.Generics (Generic)
import Codec.Serialise
import Torch

instance Serialise Linear where
  encode p = encode p' 
    where p' :: ([[Float]], [Float]) = (asValue . toDependent . weight $ p, asValue . toDependent . bias $ p)
  decode = undefined 
  {- decode s'
    where s' = Linear { 
           weight = IndependentTensor . asTensor . fst $ decode s,
           bias = IndependentTensor . asTensor . snd $ decode s
           }
           -}
  -- decode = IndependentTensor . (asTensor :: [Float] -> Tensor) <$> decode
  --
-- deriving instance Serialise Linear
-- deriving instance Generic LinearSpec
-- deriving instance Serialise LinearSpec
