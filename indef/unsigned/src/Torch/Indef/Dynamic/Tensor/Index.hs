module Torch.Indef.Dynamic.Tensor.Index where

import Torch.Sig.Types
import qualified Torch.Sig.Types          as Sig
import qualified Torch.Sig.Types.Global   as Sig
import qualified Torch.Sig.Tensor.Index   as Sig
import qualified Torch.Class.Tensor.Index as Class

import Torch.Indef.Types

forIxFn
  :: Dynamic
  -> Dynamic
  -> IndexDynamic
  -> (Ptr CState -> Ptr CTensor -> Ptr CTensor  -> Ptr CIndexTensor -> IO ())
  -> IO ()
forIxFn ret t1 ix fn =
  with2DynamicState ret t1 $ \s' ret' t1' ->
    withForeignPtr (snd $ Sig.longDynamicState ix) $ \ix' ->
      fn s' ret' t1' ix'

instance Class.TensorIndex Dynamic where
  indexCopy_ :: Dynamic -> Int -> IndexDynamic -> Dynamic -> IO ()
  indexCopy_ ret i ix t = forIxFn ret t ix $ \s' ret' t' ix' ->
    Sig.c_indexCopy s' ret' (fromIntegral i) ix' t'

  indexAdd_ :: Dynamic -> Int -> IndexDynamic -> Dynamic -> IO ()
  indexAdd_ r i ix t = forIxFn r t ix $ \s' r' t' ix' ->
    Sig.c_indexAdd s' r' (fromIntegral i) ix' t'

  indexFill_ :: Dynamic -> Int -> IndexDynamic -> HsReal -> IO ()
  indexFill_ ret i ix v =
    withDynamicState ret $ \s' ret' ->
      withForeignPtr (snd $ Sig.longDynamicState ix) $ \ix' ->
        Sig.c_indexFill s' ret' (fromIntegral i) ix' (Sig.hs2cReal v)

  indexSelect_ :: Dynamic -> Dynamic -> Int -> IndexDynamic -> IO ()
  indexSelect_ ret t i ix = forIxFn ret t ix $ \s' ret' t' ix' ->
    Sig.c_indexSelect s' ret' t' (fromIntegral i) ix'

  take_ :: Dynamic -> Dynamic -> IndexDynamic -> IO ()
  take_ ret t ix = forIxFn ret t ix $ \s' ret' t' ix' ->
    Sig.c_take s' ret' t' ix'

  put_ :: Dynamic -> IndexDynamic -> Dynamic -> Int -> IO ()
  put_ ret ix t i = forIxFn ret t ix $ \s' ret' t' ix' ->
    Sig.c_put s' ret' ix' t' (fromIntegral i)

-- class GPUTensorIndex Dynamic where
--   indexCopy_long_ :: t -> Int -> IndexDynamic t -> t -> IO ()
--   indexAdd_long_ :: t -> Int -> IndexDynamic t -> t -> IO ()
--   indexFill_long_ :: t -> Int -> IndexDynamic t -> Word -> IO ()
--   indexSelect_long_ :: t -> t -> Int -> IndexDynamic t -> IO ()
--   calculateAdvancedIndexingOffsets_ :: IndexDynamic t -> t -> Integer -> [IndexTensor t] -> IO ()
