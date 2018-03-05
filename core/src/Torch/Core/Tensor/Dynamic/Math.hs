{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Core.Tensor.Dynamic.Math
  ( module Class
  ) where

import Torch.Class.Internal
import GHC.Int
import Torch.Class.Tensor.Math as Class

import Torch.Types.TH
import Foreign

-- import any type-specific orphans
import Torch.Core.ByteTensor.Dynamic   ()
-- import Torch.Core.CharTensor.Dynamic   ()
import Torch.Core.ShortTensor.Dynamic  ()
import Torch.Core.IntTensor.Dynamic    ()
import Torch.Core.LongTensor.Dynamic    ()
-- import Torch.Core.HalfTensor.Dynamic   ()
import Torch.Core.FloatTensor.Dynamic  ()
import Torch.Core.DoubleTensor.Dynamic ()


-------------------------------------------------------------------------------
-- | Import any basic math orphans

import Torch.Core.ByteTensor.Dynamic.Math      ()
-- import Torch.Core.CharTensor.Dynamic.Math   ()
import Torch.Core.ShortTensor.Dynamic.Math     ()
import Torch.Core.IntTensor.Dynamic.Math       ()
import Torch.Core.LongTensor.Dynamic.Math      ()
-- import Torch.Core.HalfTensor.Dynamic.Math   ()
import Torch.Core.FloatTensor.Dynamic.Math     ()
import Torch.Core.DoubleTensor.Dynamic.Math    ()

-------------------------------------------------------------------------------
-- | Import any signed math orphans

import Torch.Core.ShortTensor.Dynamic.Math.Signed     ()
import Torch.Core.IntTensor.Dynamic.Math.Signed       ()
import Torch.Core.LongTensor.Dynamic.Math.Signed      ()
-- import Torch.Core.HalfTensor.Dynamic.Math.Signed   ()
import Torch.Core.FloatTensor.Dynamic.Math.Signed     ()
import Torch.Core.DoubleTensor.Dynamic.Math.Signed    ()

-------------------------------------------------------------------------------
-- | Import any floating math orphans

-- import Torch.Core.HalfTensor.Dynamic.Math.Floating   ()
import Torch.Core.FloatTensor.Dynamic.Math.Floating     ()
import Torch.Core.DoubleTensor.Dynamic.Math.Floating    ()


