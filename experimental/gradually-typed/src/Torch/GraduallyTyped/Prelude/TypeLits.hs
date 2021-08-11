{-# LANGUAGE CPP #-}

module Torch.GraduallyTyped.Prelude.TypeLits (
#if MIN_VERSION_singletons(3,0,0)
    module GHC.TypeLits.Singletons
#else
    module Torch.GraduallyTyped.Prelude.TypeLits
#endif
) where

#if MIN_VERSION_singletons(3,0,0)
import GHC.TypeLits.Singletons
#else
import Torch.GraduallyTyped.Prelude.TypeLits
#endif
