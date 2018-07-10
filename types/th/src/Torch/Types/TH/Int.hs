-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Types.TH.Int
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- A package of type aliases to satisfy a backpack Torch.Sig.Types signature.
-------------------------------------------------------------------------------
{-# LANGUAGE DataKinds #-}
module Torch.Types.TH.Int where

import Foreign.C.Types
import Foreign
import GHC.TypeLits (Nat)
import GHC.Int
import Torch.Types.TH

-- | type alias of 'CIntStorage'
type CStorage = CIntStorage
-- | type alias of 'CIntTensor'
type CTensor = CIntTensor

-- type CInt' = CInt

-- | C-level representation of a torch "real" -- the inhabitant of a 'CTensor'. Alias to 'CInt'.
type CReal = CInt
-- | C-level representation of a torch "accreal" -- the accumulating type of a 'CTensor'. Alias to 'CLong'.
type CAccReal = CLong
-- | Hask-level representation of a torch "real" -- the accumulating type of a 'Tensor'. Alias to 'Int32'.
type HsReal = Int32
-- | Hask-level representation of a torch "accreal" -- the accumulating type of a 'Tensor'. Alias to 'Int64'.
type HsAccReal = Int64

-- | convert a real to its accumulating representation
real2acc :: HsReal -> HsAccReal
real2acc = fromIntegral

-- | convert an accumulating value to its base rperesentation
acc2real :: HsAccReal -> HsReal
acc2real = fromIntegral

-- | convert an 'HsReal' to its C-level representation
hs2cReal :: HsReal -> CReal
hs2cReal = fromIntegral

-- | convert an 'HsAccReal' to its C-level representation
hs2cAccReal :: HsAccReal -> CAccReal
hs2cAccReal = fromIntegral

-- | convert a 'CReal' to its Hask-level representation
c2hsReal :: CReal -> HsReal
c2hsReal = fromIntegral

-- | convert a 'CAccReal' to its Hask-level representation
c2hsAccReal :: CAccReal -> HsAccReal
c2hsAccReal = fromIntegral

-- | convert an integral type to its hask-level representation
i2hsReal :: Integral i => i -> HsReal
i2hsReal = fromIntegral

-- | type alias to 'IntStorage'
type Storage = IntStorage
-- | get the C-level representation of a 'IntStorage' out of 'intStorageState'
cstorage        = snd . intStorageState
-- | alias to 'intStorage'
storage         = intStorage
-- | alias to 'intStorageState'
storageState    = intStorageState
-- | get the C-level representation of the 'CState' from a 'IntStorage' out of 'intStorageState'
storageStateRef = fst . intStorageState

-- | type alias to 'IntDynamic'
type Dynamic = IntDynamic
-- | get the C-level representation of a 'IntDynamic' out of 'intDynamicState'
ctensor = snd . intDynamicState
-- | alias to 'intDynamic'
dynamic = intDynamic
-- | alias to 'intDynamicState'
dynamicState = intDynamicState
-- | get the C-level representation of the 'CState' from a 'IntDynamic' out of 'intDynamicState'
dynamicStateRef = fst . intDynamicState

-- | type alias to 'IntTensor'
type Tensor = IntTensor
-- | type alias to 'intAsDynamic'
asDynamic = intAsDynamic
-- | type alias to 'intAsStatic'
asStatic = intAsStatic

-- this is to make indef work with NN code and shouldn't be exported.
-- FIXME: Remove this
-- instance Fractional Int32

