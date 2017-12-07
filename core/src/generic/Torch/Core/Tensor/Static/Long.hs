{-# OPTIONS_GHC -fno-cse -fno-full-laziness #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances   #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies        #-}

module Torch.Core.Tensor.Static.Long (
  ) where

import Data.Singletons
import Data.Singletons.TypeLits
import Data.Singletons.Prelude.List
import Data.Singletons.Prelude.Num
import Foreign (Ptr)
import Foreign.C.Types (CLong)
import Foreign.ForeignPtr (ForeignPtr, withForeignPtr, newForeignPtr)
import GHC.Exts -- (IsList(..))
import System.IO.Unsafe (unsafePerformIO)

import Torch.Core.Internal (w2cl)
import Torch.Core.Tensor.Raw
import Torch.Core.Tensor.Types

import THTypes
import THLongTensor

newtype TensorLongStatic (d :: [Nat]) = TLS {
  tlsTensor :: ForeignPtr CTHLongTensor
  } deriving (Show)

type TLS = TensorLongStatic

class StaticLongTensor t where
  tls_new :: t
  tls_cloneDim :: t -> t
  tls_init :: Int -> t
  tls_p ::  t -> IO ()

list2dim :: (Num a2, Integral a1) => [a1] -> TensorDim a2
list2dim lst  = case (length lst) of
  0 -> D0
  1 -> D1 (d !! 0)
  2 -> D2 ((d !! 0), (d !! 1))
  3 -> D3 ((d !! 0), (d !! 1), (d !! 2))
  4 -> D4 ((d !! 0), (d !! 1), (d !! 2), (d !! 3))
  _ -> error "Tensor type signature has invalid dimensions"
  where
    d = fromIntegral <$> lst -- cast as needed for tensordim

tls_dim :: (Num a2, SingI d) => TensorLongStatic d -> TensorDim a2
tls_dim (x :: TensorLongStatic d) = list2dim $ fromSing (sing :: Sing d)

-- |Make an initialized raw pointer with requested dimensions
mkPtr :: TensorDim Word -> Int -> IO TensorLongRaw
mkPtr dim value = tensorLongRaw dim value

-- |Make a foreign pointer from requested dimensions
mkTHelper :: TensorDim Word -> (ForeignPtr CTHLongTensor -> TLS d) -> Int -> TLS d
mkTHelper dims makeStatic value = unsafePerformIO $ do
  newPtr <- mkPtr dims value
  fPtr <- newForeignPtr p_THLongTensor_free newPtr
  pure $ makeStatic fPtr
{-# NOINLINE mkTHelper #-}

instance SingI d => StaticLongTensor (TensorLongStatic d)  where
  tls_init initVal = mkTHelper dims makeStatic initVal
    where
      dims = list2dim $ fromSing (sing :: Sing d)
      makeStatic fptr = (TLS fptr) :: TLS d
  tls_new = tls_init 0
  tls_cloneDim _ = tls_new :: TLS d
  tls_p tensor = undefined -- (withForeignPtr (tlsTensor tensor) dispRaw)

instance KnownNat l => IsList (TLS '[l]) where
  type Item (TLS '[l]) = Int
  fromList l = if (fromIntegral $ natVal (Proxy :: Proxy l)) /= length l
               then error "List length does not match tensor dimensions"
               else unsafePerformIO $ go result
               -- TODO: try to force strict evaluation
               -- to avoid potential FFI + IO + mutation bugs.
               -- however `go` never executes with deepseq:
               -- else unsafePerformIO $ pure (deepseq go result)
    where
      result = tls_new
      go t = do
        mapM_ mutTensor (zip [0..(length l) - 1] l)
        pure t
        where
          mutTensor (idx, value) =
            let (idxC, valueC) = (fromIntegral idx, fromIntegral value) in
              withForeignPtr (tlsTensor t)
                (\tp -> do
                    -- print idx -- check to see when mutation happens
                    c_THLongTensor_set1d tp idxC valueC
                )
  {-# NOINLINE fromList #-}
