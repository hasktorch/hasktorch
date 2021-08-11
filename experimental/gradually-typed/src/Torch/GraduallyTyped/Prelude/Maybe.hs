{-# LANGUAGE CPP #-}

module Torch.GraduallyTyped.Prelude.Maybe (
#if MIN_VERSION_singletons(3,0,0)
    module Data.Maybe.Singletons
#else
    module Torch.GraduallyTyped.Prelude.Maybe
#endif
) where

#if MIN_VERSION_singletons(3,0,0)
import Data.Maybe.Singletons
#else
import Torch.GraduallyTyped.Prelude.Maybe
#endif
