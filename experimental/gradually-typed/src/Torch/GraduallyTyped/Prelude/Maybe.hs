{-# LANGUAGE CPP #-}

#if MIN_VERSION_singletons(3,0,0)
module Torch.GraduallyTyped.Prelude.Maybe (module Data.Maybe.Singletons) where
import Data.Maybe.Singletons
#else
module Torch.GraduallyTyped.Prelude.Maybe (module Data.Singletons.Prelude.Maybe) where
import Data.Singletons.Prelude.Maybe
#endif
