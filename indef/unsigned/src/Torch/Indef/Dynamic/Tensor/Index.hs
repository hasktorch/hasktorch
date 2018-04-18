module Torch.Indef.Dynamic.Tensor.Index () where

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
  _indexCopy :: Dynamic -> Int -> IndexDynamic -> Dynamic -> IO ()
  _indexCopy ret i ix t = forIxFn ret t ix $ \s' ret' t' ix' ->
    Sig.c_indexCopy s' ret' (fromIntegral i) ix' t'

  _indexAdd :: Dynamic -> Int -> IndexDynamic -> Dynamic -> IO ()
  _indexAdd r i ix t = forIxFn r t ix $ \s' r' t' ix' ->
    Sig.c_indexAdd s' r' (fromIntegral i) ix' t'

  _indexFill :: Dynamic -> Int -> IndexDynamic -> HsReal -> IO ()
  _indexFill ret i ix v =
    withDynamicState ret $ \s' ret' ->
      withForeignPtr (snd $ Sig.longDynamicState ix) $ \ix' ->
        Sig.c_indexFill s' ret' (fromIntegral i) ix' (Sig.hs2cReal v)

  _indexSelect :: Dynamic -> Dynamic -> Int -> IndexDynamic -> IO ()
  _indexSelect ret t i ix = forIxFn ret t ix $ \s' ret' t' ix' ->
    Sig.c_indexSelect s' ret' t' (fromIntegral i) ix'

  _take :: Dynamic -> Dynamic -> IndexDynamic -> IO ()
  _take ret t ix = forIxFn ret t ix $ \s' ret' t' ix' ->
    Sig.c_take s' ret' t' ix'

  _put :: Dynamic -> IndexDynamic -> Dynamic -> Int -> IO ()
  _put ret ix t i = forIxFn ret t ix $ \s' ret' t' ix' ->
    Sig.c_put s' ret' ix' t' (fromIntegral i)

-- class GPUTensorIndex Dynamic where
--   _indexCopy_long :: t -> Int -> IndexDynamic t -> t -> IO ()
--   _indexAdd_long :: t -> Int -> IndexDynamic t -> t -> IO ()
--   _indexFill_long :: t -> Int -> IndexDynamic t -> Word -> IO ()
--   _indexSelect_long :: t -> t -> Int -> IndexDynamic t -> IO ()
--   _calculateAdvancedIndexingOffsets :: IndexDynamic t -> t -> Integer -> [IndexTensor t] -> IO ()
