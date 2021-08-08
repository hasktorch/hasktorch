{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Control.Monad.Fresh where

import Control.Monad.Cont (ContT, MonadCont (..))
import Control.Monad.Except (ExceptT, MonadError (..))
import Control.Monad.Fix (MonadFix (..))
import Control.Monad.IO.Class (MonadIO (..))
import Control.Monad.Identity (Identity (..), IdentityT)
import Control.Monad.Morph (MFunctor (..))
import Control.Monad.RWS (RWST)
import Control.Monad.Reader (MonadReader (..), ReaderT, asks, runReaderT)
import Control.Monad.State (MonadState (..), StateT, evalStateT, modify)
import Control.Monad.Trans (MonadTrans (..))
import Control.Monad.Trans.Maybe (MaybeT)
import Control.Monad.Writer (MonadWriter (..), WriterT)
import GHC.Base (Alternative (..), MonadPlus (..))
import Hedgehog (MonadGen (..), distributeT)

newtype Successor a = Successor {suc :: a -> a}

-- | The monad transformer for generating fresh values.
newtype FreshT e m a = FreshT {unFreshT :: ReaderT (Successor e) (StateT e m) a}
  deriving (Functor)

instance Monad m => MonadFresh e (FreshT e m) where
  fresh = FreshT $ do
    e <- get
    s <- asks suc
    modify s
    pure e

instance Monad m => Monad (FreshT e m) where
  return = FreshT . return
  (FreshT m) >>= f = FreshT $ m >>= unFreshT . f

instance MonadPlus m => MonadPlus (FreshT e m) where
  mzero = FreshT mzero
  mplus (FreshT m) (FreshT m') = FreshT $ mplus m m'

instance (Functor f, Monad f) => Applicative (FreshT e f) where
  pure = FreshT . pure
  (FreshT f) <*> (FreshT a) = FreshT $ f <*> a

instance (Monad m, Functor m, MonadPlus m) => Alternative (FreshT e m) where
  empty = mzero
  (<|>) = mplus

type Fresh e = FreshT e Identity

instance MonadTrans (FreshT e) where
  lift = FreshT . lift . lift

instance MonadReader r m => MonadReader r (FreshT e m) where
  local f m = FreshT $ ask >>= lift . local f . runReaderT (unFreshT m)
  ask = FreshT (lift ask)

instance MonadState s m => MonadState s (FreshT e m) where
  get = FreshT $ (lift . lift) get
  put = FreshT . lift . lift . put

instance (MonadWriter w m) => MonadWriter w (FreshT e m) where
  tell m = lift $ tell m
  listen = FreshT . listen . unFreshT
  pass = FreshT . pass . unFreshT

instance MonadFix m => MonadFix (FreshT e m) where
  mfix = FreshT . mfix . (unFreshT .)

instance MonadIO m => MonadIO (FreshT e m) where
  liftIO = FreshT . liftIO

instance MonadCont m => MonadCont (FreshT e m) where
  callCC f = FreshT $ callCC (unFreshT . f . (FreshT .))

instance MonadError e m => MonadError e (FreshT e' m) where
  throwError = FreshT . throwError
  catchError m h = FreshT $ catchError (unFreshT m) (unFreshT . h)

instance MFunctor (FreshT e) where
  hoist nat m = FreshT $ hoist (hoist nat) (unFreshT m)

instance MonadGen m => MonadGen (FreshT e m) where
  type GenBase (FreshT e m) = FreshT e (GenBase m)
  toGenT = hoist FreshT . distributeT . hoist distributeT . unFreshT . hoist toGenT
  fromGenT = hoist fromGenT . distributeT

successor :: forall e. (e -> e) -> Successor e
successor = Successor

enumSucc :: forall e. Enum e => Successor e
enumSucc = Successor succ

-- | Run a @FreshT@ computation starting from the value
-- @toEnum 0@
runFreshT :: forall e m a. (Enum e, Monad m) => FreshT e m a -> m a
runFreshT = runFreshTFrom (toEnum 0)

-- | Run a @Fresh@ computation starting from the value
-- @toEnum 0@
runFresh :: forall e a. Enum e => Fresh e a -> a
runFresh = runFreshFrom (toEnum 0)

-- | Run a @FreshT@ computation starting from a specific value @e@.
runFreshTFrom :: forall e m a. (Monad m, Enum e) => e -> FreshT e m a -> m a
runFreshTFrom = runFreshTWith enumSucc

-- | Run a @Fresh@ computation starting from a specific value @e@.
runFreshFrom :: forall e a. Enum e => e -> Fresh e a -> a
runFreshFrom = runFreshWith enumSucc

-- | Run a @FreshT@ computation starting from a specific value @e@ with
-- a the next fresh value determined by @Successor e@.
runFreshTWith :: forall e m a. Monad m => Successor e -> e -> FreshT e m a -> m a
runFreshTWith s e =
  flip evalStateT e
    . flip runReaderT s
    . unFreshT

-- | Run a @FreshT@ computation starting from a specific value @e@ with
-- a the next fresh value determined by @Successor e@.
runFreshWith :: forall e a. Successor e -> e -> Fresh e a -> a
runFreshWith s e = runIdentity . runFreshTWith s e

-- | The MTL style class for generating fresh values
class Monad m => MonadFresh e m | m -> e where
  -- | Generate a fresh value @e@, @fresh@ should never produce the
  -- same value within a monadic computation.
  fresh :: m e

instance MonadFresh e m => MonadFresh e (IdentityT m) where
  fresh = lift fresh

instance MonadFresh e m => MonadFresh e (StateT s m) where
  fresh = lift fresh

instance MonadFresh e m => MonadFresh e (ReaderT s m) where
  fresh = lift fresh

instance (MonadFresh e m, Monoid s) => MonadFresh e (WriterT s m) where
  fresh = lift fresh

instance MonadFresh e m => MonadFresh e (MaybeT m) where
  fresh = lift fresh

instance MonadFresh e m => MonadFresh e (ContT r m) where
  fresh = lift fresh

instance (Monoid w, MonadFresh e m) => MonadFresh e (RWST r w s m) where
  fresh = lift fresh

instance (MonadFresh e m) => MonadFresh e (ExceptT e' m) where
  fresh = lift fresh
