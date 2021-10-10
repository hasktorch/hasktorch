{-# LANGUAGE CPP #-}

#if MIN_VERSION_singletons(3,0,0)
module Torch.GraduallyTyped.Prelude.Bool (module Data.Bool.Singletons) where
import Data.Bool.Singletons
#else
module Torch.GraduallyTyped.Prelude.Bool (module Data.Singletons.Prelude.Bool) where
import Data.Singletons.Prelude.Bool
#endif
