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

module Torch.Static.Factories where

import           Data.Proxy
import           Data.Finite
import           Data.Kind                      ( Constraint )
import           GHC.TypeLits

import           Prelude                 hiding ( sin )
import qualified Torch.Tensor                  as D
import qualified Torch.TensorFactories         as D
import qualified Torch.Functions               as D
import           Torch.DType
import           Torch.Static
import qualified Torch.TensorOptions           as D
import           Data.Reflection

zeros :: forall dtype shape . (TensorOptions dtype shape) => Tensor dtype shape
zeros = UnsafeMkTensor $ D.zeros
  (optionsRuntimeShape @dtype @shape)
  (D.withDType (optionsRuntimeDType @dtype @shape) D.defaultOpts)

ones :: forall dtype shape . (TensorOptions dtype shape) => Tensor dtype shape
ones = UnsafeMkTensor $ D.ones
  (optionsRuntimeShape @dtype @shape)
  (D.withDType (optionsRuntimeDType @dtype @shape) D.defaultOpts)

rand
  :: forall dtype shape . (TensorOptions dtype shape) => IO (Tensor dtype shape)
rand = UnsafeMkTensor <$> D.rand
  (optionsRuntimeShape @dtype @shape)
  (D.withDType (optionsRuntimeDType @dtype @shape) D.defaultOpts)

randn
  :: forall dtype shape . (TensorOptions dtype shape) => IO (Tensor dtype shape)
randn = UnsafeMkTensor <$> D.randn
  (optionsRuntimeShape @dtype @shape)
  (D.withDType (optionsRuntimeDType @dtype @shape) D.defaultOpts)

type family SquareCheck (shape :: [Nat]) :: Constraint where
  SquareCheck '[] = ()
  SquareCheck (h ': '[]) = ()
  SquareCheck (h ': h ': t) = SquareCheck (h ': t)
  SquareCheck (h ': h2 ': _) = TypeError (Text "Shape is not square " :<>:
                                          Text "(dimensions " :<>:
                                          ShowType h :<>:
                                          Text " and " :<>:
                                          ShowType h2 :<>:
                                          Text " differ)")

eyeSquare
  :: forall dtype shape
   . ( SquareCheck shape
     , KnownNat (ListLength shape)
     , TensorOptions dtype shape
     )
  => Tensor dtype shape
eyeSquare = UnsafeMkTensor $ D.eyeSquare
  (natValI @(ListLength shape))
  (D.withDType (optionsRuntimeDType @dtype @shape) D.defaultOpts)
