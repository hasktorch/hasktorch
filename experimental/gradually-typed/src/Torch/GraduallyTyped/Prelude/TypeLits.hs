{-# LANGUAGE CPP #-}

#if MIN_VERSION_singletons(3,0,0)
module Torch.GraduallyTyped.Prelude.TypeLits (module GHC.TypeLits.Singletons) where
import GHC.TypeLits.Singletons
#else
module Torch.GraduallyTyped.Prelude.TypeLits (module Data.Singletons.TypeLits) where
import Data.Singletons.TypeLits
#endif
