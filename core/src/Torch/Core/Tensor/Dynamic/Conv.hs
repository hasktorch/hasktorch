{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Core.Tensor.Dynamic.Conv
  ( TensorConv(..)
  ) where

import Control.Monad ((>=>))
import Torch.Class.Internal
import qualified Torch.Class.Tensor.Conv as CCall
import GHC.Int

import qualified Torch.Core.LongTensor.Dynamic   as L
import qualified Torch.Core.FloatTensor.Dynamic  as F
import qualified Torch.Core.ByteTensor.Dynamic   as B
-- import qualified Torch.Core.CharTensor.Dynamic   as C
import qualified Torch.Core.ShortTensor.Dynamic  as S
import qualified Torch.Core.IntTensor.Dynamic    as I
import qualified Torch.Core.DoubleTensor.Dynamic as D
-- import qualified Torch.Core.HalfTensor.Dynamic   as H

class CCall.TensorConv t => TensorConv t where
  conv2DRevger  :: t -> HsReal t -> HsReal t -> t -> t -> Int64 -> Int64 -> IO ()
  conv2DRevger  = CCall.conv2DRevger
  conv2DRevgerm :: t -> HsReal t -> HsReal t -> t -> t -> Int64 -> Int64 -> IO ()
  conv2DRevgerm = CCall.conv2DRevgerm
  conv2Dger     :: t -> HsReal t -> HsReal t -> t -> t -> Int64 -> Int64 -> [Int8] -> [Int8] -> IO ()
  conv2Dger     = CCall.conv2Dger
  conv2Dmv      :: t -> HsReal t -> HsReal t -> t -> t -> Int64 -> Int64 -> [Int8] -> [Int8] -> IO ()
  conv2Dmv      = CCall.conv2Dmv
  conv2Dmm      :: t -> HsReal t -> HsReal t -> t -> t -> Int64 -> Int64 -> [Int8] -> [Int8] -> IO ()
  conv2Dmm      = CCall.conv2Dmm
  conv2Dmul     :: t -> HsReal t -> HsReal t -> t -> t -> Int64 -> Int64 -> [Int8] -> [Int8] -> IO ()
  conv2Dmul     = CCall.conv2Dmul
  conv2Dcmul    :: t -> HsReal t -> HsReal t -> t -> t -> Int64 -> Int64 -> [Int8] -> [Int8] -> IO ()
  conv2Dcmul    = CCall.conv2Dcmul
  conv3DRevger  :: t -> HsReal t -> HsReal t -> t -> t -> Int64 -> Int64 -> Int64 -> IO ()
  conv3DRevger  = CCall.conv3DRevger
  conv3Dger     :: t -> HsReal t -> HsReal t -> t -> t -> Int64 -> Int64 -> Int64 -> [Int8] -> [Int8] -> IO ()
  conv3Dger     = CCall.conv3Dger
  conv3Dmv      :: t -> HsReal t -> HsReal t -> t -> t -> Int64 -> Int64 -> Int64 -> [Int8] -> [Int8] -> IO ()
  conv3Dmv      = CCall.conv3Dmv
  conv3Dmul     :: t -> HsReal t -> HsReal t -> t -> t -> Int64 -> Int64 -> Int64 -> [Int8] -> [Int8] -> IO ()
  conv3Dmul     = CCall.conv3Dmul
  conv3Dcmul    :: t -> HsReal t -> HsReal t -> t -> t -> Int64 -> Int64 -> Int64 -> [Int8] -> [Int8] -> IO ()
  conv3Dcmul    = CCall.conv3Dcmul

instance TensorConv B.Tensor where
instance TensorConv S.Tensor where
instance TensorConv I.Tensor where
instance TensorConv L.Tensor where
instance TensorConv F.Tensor where
instance TensorConv D.Tensor where

