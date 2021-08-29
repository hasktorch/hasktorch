{-# LANGUAGE CPP #-}

module Torch.GraduallyTyped.Internal.Vector where

import qualified Data.Vector as V

#if MIN_VERSION_vector(0,12,2)
uncons :: V.Vector a -> Maybe (a, V.Vector a)
uncons = V.uncons
#else
uncons :: V.Vector a -> Maybe (a, V.Vector a)
uncons xs = flip (,) (V.unsafeTail xs) `fmap` (xs V.!? 0)
#endif
