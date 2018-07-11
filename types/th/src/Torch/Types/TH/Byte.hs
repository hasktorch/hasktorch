-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Types.TH.Byte
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- A package of type aliases to satisfy a backpack Torch.Sig.Types signature.
-------------------------------------------------------------------------------
{-# LANGUAGE DataKinds #-}
module Torch.Types.TH.Byte where

import Foreign
import Foreign.C.Types
import GHC.TypeLits (Nat)
import GHC.Word
import Torch.Types.TH

-- | type alias of 'CByteStorage'
type CStorage = CByteStorage
-- | type alias of 'CByteTensor'
type CTensor = CByteTensor

-- | C-level representation of a torch "real" -- the inhabitant of a 'CTensor'. Alias to 'CUChar'.
type CReal = CUChar
-- | C-level representation of a torch "accreal" -- the accumulating type of a 'CTensor'. Alias to 'CLong'.
type CAccReal = CLong
-- | Hask-level representation of a torch "real" -- the accumulating type of a 'Tensor'. Alias to 'Word32'.
type HsReal = Word8
-- | Hask-level representation of a torch "accreal" -- the accumulating type of a 'Tensor'. Alias to 'Word64'.
type HsAccReal = Word64

real2acc :: HsReal -> HsAccReal
real2acc = fromIntegral

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

-- | type alias to 'ByteStorage'
type Storage = ByteStorage
-- | get the C-level representation of a 'ByteStorage' out of 'byteStorageState'
cstorage        = snd . byteStorageState
-- | alias to 'byteStorage'
storage         = byteStorage
-- | alias to 'byteStorageState'
storageState    = byteStorageState
-- | get the C-level representation of the 'CState' from a 'ByteStorage' out of 'byteStorageState'
storageStateRef = fst . byteStorageState

-- | type alias to 'ByteDynamic'
type Dynamic = ByteDynamic
-- | get the C-level representation of a 'ByteDynamic' out of 'byteDynamicState'
ctensor = snd . byteDynamicState
-- | alias to 'byteDynamic'
dynamic = byteDynamic
-- | alias to 'byteDynamicState'
dynamicState = byteDynamicState
-- | get the C-level representation of the 'CState' from a 'ByteDynamic' out of 'byteDynamicState'
dynamicStateRef = fst . byteDynamicState

-- | type alias to 'ByteTensor'
type Tensor = ByteTensor
-- | type alias to 'byteAsDynamic'
asDynamic = byteAsDynamic
-- | type alias to 'byteAsStatic'
asStatic = byteAsStatic

-- instance Fractional Word8
