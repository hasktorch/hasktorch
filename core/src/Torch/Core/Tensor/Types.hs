{-# LANGUAGE TypeFamilies #-}
{- LANGUAGE ConstraintKinds #-}
{- LANGUAGE TypeInType #-}
{- LANGUAGE GADTs #-}
{- LANGUAGE Rank2Types #-}
{- LANGUAGE UndecidableInstances #-}
module Torch.Core.Tensor.Types
  ( -- Tensor(..)
  ) where


-- import Control.Monad.IO.Class (liftIO)
-- import Control.Monad.Managed (managed, runManaged)
import Foreign (ForeignPtr) -- , withForeignPtr)
import Torch.Class.Internal
import qualified Tensor as Sig

import Torch.Core.Storage (Storage)

-- ========================================================================= --
-- helper functions:
--
-- withManaged4
--   :: (THForeignRef t)
--   => (Ptr (THForeignType t) -> Ptr (THForeignType t) -> Ptr (THForeignType t) -> Ptr (THForeignType t) -> IO ())
--   -> t -> t -> t -> t -> IO ()
-- withManaged4 fn resA resB a b = runManaged $ do
--   resBRaw <- managed (withForeignPtr (getForeign resB))
--   resARaw <- managed (withForeignPtr (getForeign resA))
--   bRaw <- managed (withForeignPtr (getForeign b))
--   aRaw <- managed (withForeignPtr (getForeign a))
--   liftIO (fn resBRaw resARaw bRaw aRaw)
--
-- withManaged3
--   :: (THForeignRef t)
--   => (Ptr (THForeignType t) -> Ptr (THForeignType t) -> Ptr (THForeignType t) -> IO ())
--   -> t -> t -> t -> IO ()
-- withManaged3 fn a b c = runManaged $ do
--   a' <- managed (withForeignPtr (getForeign a))
--   b' <- managed (withForeignPtr (getForeign b))
--   c' <- managed (withForeignPtr (getForeign c))
--   liftIO (fn a' b' c')
--
-- withManaged2
--   :: (THForeignRef t)
--   => (Ptr (THForeignType t) -> Ptr (THForeignType t) -> IO ())
--   -> t -> t -> IO ()
-- withManaged2 fn resA a = runManaged $ do
--   resARaw <- managed (withForeignPtr (getForeign resA))
--   aRaw <- managed (withForeignPtr (getForeign a))
--   liftIO (fn resARaw aRaw)
--
-- withManaged1
--   :: (THForeignRef t)
--   => (Ptr (THForeignType t) -> IO ())
--   -> t -> IO ()
-- withManaged1 fn a = runManaged $ do
--   a' <- managed (withForeignPtr (getForeign a))
--   liftIO (fn a')


