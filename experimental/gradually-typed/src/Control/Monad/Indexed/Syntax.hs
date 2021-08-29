-- |
-- For https://ghc.gitlab.haskell.org/ghc/doc/users_guide/exts/qualified_do.html.
module Control.Monad.Indexed.Syntax where

import Control.Monad.Indexed (IxApplicative (..), IxFunctor (..), IxMonad (..))
import Prelude hiding ((=<<), (>>), (>>=))

(<$>) :: IxFunctor f => (a -> b) -> f j k a -> f j k b
(<$>) = imap

(<*>) :: IxApplicative f => f i j (a -> b) -> f j k a -> f i k b
(<*>) = iap

(=<<) :: IxMonad m => (a -> m j k b) -> m i j a -> m i k b
(=<<) = ibind

(>>=) :: IxMonad m => m i j a -> (a -> m j k b) -> m i k b
(>>=) = flip (=<<)

(>>) :: IxMonad m => m i j a -> m j k b -> m i k b
a >> b = a >>= const b
