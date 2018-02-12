-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Class.C.Internal
-- Copyright :  (c) Sam Stites 2017
-- License   :  MIT
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Should be "Torch.Class.C.Types"
-------------------------------------------------------------------------------
{-# LANGUAGE TypeFamilies #-}
module Torch.Class.C.Internal where

type family HsReal t
type family HsAccReal t
type family HsStorage t
type family AsDynamic t


