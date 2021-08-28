{-# LANGUAGE CPP #-}

module Torch.GraduallyTyped.Prelude.Bool
  (
  )
where

#if MIN_VERSION_singletons(3,0,0)
    module Data.Bool.Singletons
#else
    module Data.Singletons.Prelude.Bool
#endif

#if MIN_VERSION_singletons(3,0,0)
import Data.Bool.Singletons
#else
import Data.Singletons.Prelude.Bool
#endif
