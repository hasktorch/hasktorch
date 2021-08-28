{-# LANGUAGE CPP #-}

module Torch.GraduallyTyped.Prelude.Maybe
  (
  )
where

#if MIN_VERSION_singletons(3,0,0)
    module Data.Maybe.Singletons
#else
    module Data.Singletons.Prelude.Maybe
#endif

#if MIN_VERSION_singletons(3,0,0)
import Data.Maybe.Singletons
#else
import Data.Singletons.Prelude.Maybe
#endif
