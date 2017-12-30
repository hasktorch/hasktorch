{-# OPTIONS_GHC -fno-cse -fno-full-laziness #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE FlexibleInstances   #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE RecordWildCards     #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies        #-}
{-# LANGUAGE TypeOperators #-}

module Torch.Core.Tensor.Static.Float (
  tfs_dim,
  tfs_expand,
  tfs_new,
  tfs_new_,
  tfs_fromList,
  tfs_init,
  tfs_cloneDim,
  tfs_newClone,
  tfs_p,
  tfs_resize,
  tfs_trans,
  TensorFloatStatic(..),
  TFS(..),
  Nat
  ) where

import Control.Monad.Managed
import Data.Singletons
import Data.Singletons.TypeLits
import Data.Singletons.Prelude.List
import Data.Singletons.Prelude.Num
import Foreign (Ptr)
import Foreign.C.Types (CLong)
import Foreign.ForeignPtr (ForeignPtr, withForeignPtr, newForeignPtr)
import GHC.Exts
import System.IO.Unsafe (unsafePerformIO)

import Torch.Core.Internal (w2cl)
import Torch.Core.StorageLong
import Torch.Core.StorageTypes
import Torch.Core.Tensor.Raw
import Torch.Core.Tensor.Types

import THTypes
import THFloatTensor
import THFloatTensorMath

-- TODO: get rid of this double-specific typeclass and just extend functionality
-- as independent functions using singletons
class TFClass t where
  -- |tensor dimensions
  -- |create tensor
  tfs_new_ :: IO t
  tfs_new :: t
  -- |create tensor of the same dimensions
  tfs_cloneDim :: t -> t -- takes unused argument, gets dimensions by matching types
  -- |create and initialize tensor
  tfs_init_ :: Float -> IO t
  tfs_init :: Float -> t
  -- |Display tensor
  tfs_p ::  t -> IO ()

instance KnownNat l => IsList (TFS '[l]) where
  type Item (TFS '[l]) = Float
  fromList l = if (fromIntegral $ natVal (Proxy :: Proxy l)) /= length l
               then error "List length does not match tensor dimensions"
               else unsafePerformIO $ go result
               -- TODO: try to force strict evaluation
               -- to avoid potential FFI + IO + mutation bugs.
               -- however `go` never executes with deepseq:
               -- else unsafePerformIO $ pure (deepseq go result)
    where
      result = tfs_new
      go t = do
        mapM_ mutTensor (zip [0..(length l) - 1] l)
        pure t
        where
          mutTensor (idx, value) =
            let (idxC, valueC) = (fromIntegral idx, realToFrac value) in
              withForeignPtr (tdsTensor t)
                (\tp -> do
                    -- print idx -- check to see when mutation happens
                    c_THFloatTensor_set1d tp idxC valueC
                )
  {-# NOINLINE fromList #-}

  toList t = undefined -- TODO
  -- check when fromList evaluates
  -- let foo = (tfs_fromList [1..3] :: TFS '[3])
  -- tfs_p foo -- prints indexes

-- |Initialize a 1D tensor from a list
tfs_fromList1D :: KnownNat n => [Float] -> TFS '[n]
tfs_fromList1D l = fromList l

-- |Initialize a tensor of arbitrary dimension from a list
tfs_fromList
  :: forall d2 . (SingI '[Product d2], SingI d2, KnownNat (Product d2))
  => [Float] -> TFS d2
tfs_fromList l = tfs_resize (tfs_fromList1D l :: TFS '[Product d2])

-- |Make a resized tensor
tfs_resize :: forall d1 d2. (Product d1 ~ Product d2, SingI d1, SingI d2) =>
  TFS d1 -> TFS d2
tfs_resize t = unsafePerformIO $ do
  let resDummy = tfs_new :: TFS d2
  newPtr <- withForeignPtr (tdsTensor t) (
    \tPtr ->
      c_THFloatTensor_newClone tPtr
    )
  newFPtr <- newForeignPtr p_THFloatTensor_free newPtr
  withForeignPtr (newFPtr)
    (\selfp ->
        withForeignPtr (tdsTensor resDummy)
          (\srcp ->
             c_THFloatTensor_resizeAs selfp srcp
          )
    )
  pure $ TFS newFPtr
{-# NOINLINE tfs_resize #-}

-- |Runtime type-level check of # dimensions
dimCheck :: Monad m => TensorDim Word -> Integer -> m ()
dimCheck dims n = case dims of
  D0 -> if n == 0 then pure () else fail "Incorrect Dimensions"
  D1 _ -> if n == 1 then pure () else fail "Incorrect Dimensions"
  D2 _ -> if n == 2 then pure () else fail "Incorrect Dimensions"
  D3 _ -> if n == 3 then pure () else fail "Incorrect Dimensions"
  D4 _ -> if n == 4 then pure () else fail "Incorrect Dimensions"

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

newtype TensorFloatStatic (d :: [Nat]) = TFS {
  tdsTensor :: ForeignPtr CTHFloatTensor
  } deriving (Show)

type TFS = TensorFloatStatic

instance Eq (TensorFloatStatic d) where
  (==) t1 t2 = unsafePerformIO $ withForeignPtr (tdsTensor t1)
    (\t1c -> withForeignPtr (tdsTensor t2)
             (\t2c -> pure $ (c_THFloatTensor_equal t1c t2c) == 1)
    )
  {-# NOINLINE (==) #-}

tfs_newClone :: TensorFloatStatic d -> TensorFloatStatic d
tfs_newClone t = unsafePerformIO $ do
  newPtr <- withForeignPtr (tdsTensor t) (
    \tPtr -> c_THFloatTensor_newClone tPtr
    )
  newFPtr <- newForeignPtr p_THFloatTensor_free newPtr
  pure $ TFS newFPtr
{-# NOINLINE tfs_newClone #-}

-- -- |generalized transpose - needs type level determination of perturbed dimensions
-- tfs_transpose :: Word -> Word -> TensorFloatStatic d1 -> TensorFloatStatic d2
-- tfs_transpose = undefined

-- |matrix specialization of transpose transpose
tfs_trans :: TensorFloatStatic '[r, c] -> TensorFloatStatic '[c, r]
tfs_trans t = unsafePerformIO $ do
  newPtr <- withForeignPtr (tdsTensor t) (
    \tPtr -> c_THFloatTensor_newTranspose tPtr 1 0
    )
  newFPtr <- newForeignPtr p_THFloatTensor_free newPtr
  pure $ TFS newFPtr
{-# NOINLINE tfs_trans #-}

tfs_dim :: (Num a2, SingI d) => TensorFloatStatic d -> TensorDim a2
tfs_dim (x :: TensorFloatStatic d) = list2dim $ fromSing (sing :: Sing d)

-- |Expand a vector by copying into a matrix by set dimensions, TODO -
-- generalize this beyond the matrix case
tfs_expand :: forall d1 d2 . (KnownNat d1, KnownNat d2) => TFS '[d1] -> TFS '[d2, d1]
tfs_expand t = unsafePerformIO $ do
  let r_ = tfs_new
  runManaged $ do
    rPtr <- managed (withForeignPtr (tdsTensor r_))
    tPtr <- managed (withForeignPtr (tdsTensor t))
    sPtr <- managed (withForeignPtr (slStorage s))
    liftIO $ c_THFloatTensor_expand rPtr tPtr sPtr
  pure r_
  where
    s1 = fromIntegral $ natVal (Proxy :: Proxy d1)
    s2 = fromIntegral $ natVal (Proxy :: Proxy d2)
    s = newStorageLong (S2 (s2, s1))
{-# NOINLINE tfs_expand #-}

-- |Make an initialized raw pointer with requested dimensions
mkPtr :: TensorDim Word -> Float -> IO TensorFloatRaw
mkPtr dim value = tensorFloatRaw dim value

-- |Make a foreign pointer from requested dimensions
mkTHelper :: TensorDim Word -> (ForeignPtr CTHFloatTensor -> TFS d) -> Float -> TFS d
mkTHelper dims makeStatic value = unsafePerformIO $ do
  newPtr <- mkPtr dims value
  fPtr <- newForeignPtr p_THFloatTensor_free newPtr
  pure $ makeStatic fPtr
{-# NOINLINE mkTHelper #-}

-- |Make a foreign pointer from requested dimensions
mkTHelper_ :: TensorDim Word -> (ForeignPtr CTHFloatTensor -> TFS d) -> Float -> IO (TFS d)
mkTHelper_ dims makeStatic value = do
  newPtr <- mkPtr dims value
  fPtr <- newForeignPtr p_THFloatTensor_free newPtr
  pure $ makeStatic fPtr

instance SingI d => TFClass (TensorFloatStatic d)  where
  tfs_init initVal = mkTHelper dims makeStatic initVal
    where
      dims = list2dim $ fromSing (sing :: Sing d)
      makeStatic fptr = (TFS fptr) :: TFS d
  tfs_init_ initVal = mkTHelper_ dims makeStatic initVal
    where
      dims = list2dim $ fromSing (sing :: Sing d)
      makeStatic fptr = (TFS fptr) :: TFS d
  tfs_new = tfs_init 0.0
  tfs_new_ = tfs_init_ 0.0
  tfs_cloneDim _ = tfs_new :: TFS d
  tfs_p tensor = (withForeignPtr (tdsTensor tensor) dispRawFloat)

