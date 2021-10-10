{-# LANGUAGE CPP #-}

#if MIN_VERSION_singletons(3,0,0)
module Torch.GraduallyTyped.Prelude.List (module Data.List.Singletons) where
import Data.List.Singletons
#else
module Torch.GraduallyTyped.Prelude.List (module Data.Singletons.Prelude.List) where
import Data.Singletons.Prelude.List
#endif
