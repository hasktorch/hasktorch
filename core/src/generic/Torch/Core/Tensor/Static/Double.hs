{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE BangPatterns #-}

module Torch.Core.Tensor.Static.Double (
  tds_dim,
  tds_new,
  tds_fromList,
  tds_init,
  tds_cloneDim,
  tds_newClone,
  tds_p,
  tds_resize,
  tds_toDynamic,
  tds_fromDynamic,
  tds_trans, -- matrix specialization of transpose
  TensorDoubleStatic(..),
  TDS(..),
  Nat -- re-export for kind signature readability
  ) where

import Control.DeepSeq
import Data.Singletons
import Data.Singletons.TypeLits
import Data.Singletons.Prelude.List
import Foreign (Ptr)
import Foreign.C.Types (CLong)
import Foreign.ForeignPtr ( ForeignPtr, withForeignPtr, newForeignPtr )
import GHC.Exts
import System.IO.Unsafe (unsafePerformIO)

import Torch.Core.Internal (w2cl)
import Torch.Core.Tensor.Dynamic.Double
import Torch.Core.Tensor.Raw
import Torch.Core.Tensor.Types
import THTypes
import THDoubleTensor
import THDoubleTensorMath

class StaticTensor t where
  -- |tensor dimensions
  -- |create tensor
  tds_new :: t
  -- |create tensor of the same dimensions
  tds_cloneDim :: t -> t -- takes unused argument, gets dimensions by matching types
  -- |create and initialize tensor
  tds_init :: Double -> t
  -- |Display tensor
  tds_p ::  t -> IO ()

instance KnownNat l => IsList (TDS '[l]) where
  type Item (TDS '[l]) = Double
  fromList l = if (fromIntegral $ natVal (Proxy :: Proxy l)) /= length l
               then error "List length does not match tensor dimensions"
               else unsafePerformIO $ go result
               -- TODO: try to force strict evaluation
               -- to avoid potential FFI + IO + mutation bugs.
               -- however `go` never executes with deepseq:
               -- else unsafePerformIO $ pure (deepseq go result)
    where
      result = tds_new
      go t = do
        mapM_ mutTensor (zip [0..(length l) - 1] l)
        pure t
        where
          mutTensor (idx, value) =
            let (idxC, valueC) = (fromIntegral idx, realToFrac value) in
              withForeignPtr (tdsTensor t)
                (\tp -> do
                    -- print idx -- check to see when mutation happens
                    c_THDoubleTensor_set1d tp idxC valueC
                )
  toList t = undefined -- TODO
  -- check when fromList evaluates
  -- let foo = (tds_fromList [1..3] :: TDS '[3])
  -- tds_p foo -- prints indexes

-- |Initialize a 1D tensor from a list
tds_fromList1D :: KnownNat n => [Double] -> TDS '[n]
tds_fromList1D l = fromList l

-- |Initialize a tensor of arbitrary dimension from a list
tds_fromList
  :: forall d2 . (SingI '[Product d2], SingI d2, KnownNat (Product d2)) =>
     [Double] -> TDS d2
tds_fromList l = tds_resize (tds_fromList1D l :: TDS '[Product d2])

tds_resize :: forall d1 d2. (Product d1 ~ Product d2, SingI d1, SingI d2) =>
  TDS d1 -> TDS d2
tds_resize t = unsafePerformIO $ do
  let resDummy = tds_new :: TDS d2
  newPtr <- withForeignPtr (tdsTensor t) (
    \tPtr ->
      c_THDoubleTensor_newClone tPtr
    )
  newFPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  withForeignPtr (newFPtr)
    (\selfp ->
        withForeignPtr (tdsTensor resDummy)
          (\srcp ->
             c_THDoubleTensor_resizeAs selfp srcp
          )
    )
  pure $ TDS newFPtr

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

data TensorDoubleStatic (d :: [Nat]) = TDS {
  tdsTensor :: !(ForeignPtr CTHDoubleTensor)
  } deriving (Show)

type TDS = TensorDoubleStatic

instance Eq (TensorDoubleStatic d) where
  (==) t1 t2 = unsafePerformIO $ withForeignPtr (tdsTensor t1)
    (\t1c -> withForeignPtr (tdsTensor t2)
             (\t2c -> pure $ (c_THDoubleTensor_equal t1c t2c) == 1)
    )

tds_toDynamic :: TensorDoubleStatic d -> TensorDouble
tds_toDynamic t = unsafePerformIO $ do
  newPtr <- withForeignPtr (tdsTensor t) (
    \tPtr -> c_THDoubleTensor_newClone tPtr
    )
  newFPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  let dim = dimFromRaw newPtr
  pure $ TensorDouble newFPtr dim


-- |TODO: add dimension check
tds_fromDynamic :: SingI d => TensorDouble -> TensorDoubleStatic d
tds_fromDynamic t = unsafePerformIO $ do
  newPtr <- withForeignPtr (tdTensor t) (
    \tPtr -> c_THDoubleTensor_newClone tPtr
    )
  newFPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  pure $ TDS newFPtr

tds_newClone :: TensorDoubleStatic d -> TensorDoubleStatic d
tds_newClone t = unsafePerformIO $ do
  newPtr <- withForeignPtr (tdsTensor t) (
    \tPtr -> c_THDoubleTensor_newClone tPtr
    )
  newFPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  pure $ TDS newFPtr

-- -- |generalized transpose - needs type level determination of perturbed dimensions
-- tds_transpose :: Word -> Word -> TensorDoubleStatic d1 -> TensorDoubleStatic d2
-- tds_transpose = undefined

-- |matrix specialization of transpose transpose
tds_trans :: TensorDoubleStatic '[r, c] -> TensorDoubleStatic '[c, r]
tds_trans t = unsafePerformIO $ do
  newPtr <- withForeignPtr (tdsTensor t) (
    \tPtr -> c_THDoubleTensor_newTranspose tPtr 1 0
    )
  newFPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  pure $ TDS newFPtr

tds_dim :: (Num a2, SingI d) => TensorDoubleStatic d -> TensorDim a2
tds_dim (x :: TensorDoubleStatic d) = list2dim $ fromSing (sing :: Sing d)

-- |Make an initialized raw pointer with requested dimensions
mkPtr :: TensorDim Word -> Double -> IO TensorDoubleRaw
mkPtr dim value = tensorRaw dim value

-- |Make a foreign pointer from requested dimensions
mkTHelper :: TensorDim Word -> (ForeignPtr CTHDoubleTensor -> TDS d) -> Double -> TDS d
mkTHelper dims makeStatic value = unsafePerformIO $ do
  newPtr <- mkPtr dims value
  fPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  pure $ makeStatic fPtr

instance SingI d => StaticTensor (TensorDoubleStatic d)  where
  tds_init initVal = mkTHelper dims makeStatic initVal
    where
      dims = list2dim $ fromSing (sing :: Sing d)
      makeStatic fptr = (TDS fptr) :: TDS d
  tds_new = tds_init 0.0
  tds_cloneDim _ = tds_new :: TDS d
  tds_p tensor = (withForeignPtr (tdsTensor tensor) dispRaw)

{- Sanity checks -}

testCreate = do
  print("1")
  let t1 = tds_new :: TDS '[2, 2]
  tds_p t1
  print("2")
  let t2 = tds_new :: TDS '[2, 4]
  tds_p t2
  print("3")
  let t3 = tds_new :: TDS '[2, 2, 2]
  tds_p t3
  print("4")
  let t4 = tds_new :: TDS '[8, 4]
  tds_p t4
  pure ()

-- -- | from flat list
-- instance (SingI r, Num a) => IsList (TDS (r :: [Nat])) where
--     fromList l = Tensor $ V.fromList $ take n $ l ++ repeat 0
--       where
--         n = case (sing :: Sing r) of
--           SNil -> 1
--           (SCons x xs') -> product $ fromIntegral <$> (fromSing x: fromSing xs')
--     toList (Tensor v) = V.toList v

testEq = do
  print "Should be True:"
  print $ (tds_init 4.0 :: TDS '[2,3]) ==  (tds_init 4.0 :: TDS '[2,3])
  print "Should be False:"
  print $ (tds_init 3.0 :: TDS '[2,3]) ==  (tds_init 1.0 :: TDS '[2,3])

testTranspose = do
  tds_p $ tds_trans . tds_trans . tds_trans $ (tds_init 3.0 :: TDS '[3,2])
  print $ (tds_trans . tds_trans $ (tds_init 3.0 :: TDS '[3,2])) == (tds_init 3.0 :: TDS '[3,2])

testResize = do
  let vec = tds_fromList [1.0, 5.0, 2.0, 4.0, 3.0, 3.5] :: TDS '[6]
  let mtx = tds_resize vec :: TDS '[3,2]
  tds_p vec
  tds_p mtx

test = do
  testCreate
  testEq
  testTranspose
  disp $ tds_toDynamic (tds_init 2.0 :: TDS '[3, 4])
  tds_p $ tds_newClone (tds_init 2.0 :: TDS '[2, 3])
  testResize

