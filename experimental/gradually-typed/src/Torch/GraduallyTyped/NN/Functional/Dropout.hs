{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE TypeOperators #-}
{-# OPTIONS_GHC -Wall #-}

module Torch.GraduallyTyped.NN.Functional.Dropout where

import Foreign.ForeignPtr (ForeignPtr)
import Torch.GraduallyTyped.Random (Generator, withGenerator)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))
import Torch.Internal.Cast (cast3)
import qualified Torch.Internal.Managed.Native as ATen (_fused_dropout_tdG)
import qualified Torch.Internal.Managed.Type.Tuple as ATen ()
import qualified Torch.Internal.Type as ATen (Tensor)

-- $setup
-- >>> import Data.Singletons.Prelude.List (SList (..))
-- >>> import Torch.GraduallyTyped

-- | Dropout randomly zeroes some of the elements of
-- the input tensor with probability 'p' using samples from a Bernoulli distribution.
dropout ::
  forall gradient layout device dataType shape generatorDevice.
  -- | probability of an element to be zeroed
  Double ->
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | generator
  Generator generatorDevice ->
  -- | output
  (Tensor gradient layout (device <+> generatorDevice) dataType shape, Generator (device <+> generatorDevice))
dropout p tensor = withGenerator @device $ \gptr -> do
  (t :: ForeignPtr ATen.Tensor, _ :: ForeignPtr ATen.Tensor) <- cast3 ATen._fused_dropout_tdG tensor (1 - p) gptr
  pure t
