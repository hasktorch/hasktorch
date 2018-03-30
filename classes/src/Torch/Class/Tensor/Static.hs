{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ConstraintKinds #-}
module Torch.Class.Tensor.Static
  ( module X
  , Static, Static2
  ) where

import Torch.Class.Tensor as X
import Torch.Class.Types

type Static t =
  ( IsStatic t
  , Tensor t
  , Tensor (AsDynamic t)
  , Tensor (IndexTensor t)
  , IndexTensor t ~ IndexTensor (AsDynamic t)
  , HsStorage t ~ HsStorage (AsDynamic t)
  , HsReal t ~ HsReal (AsDynamic t)
  , HsAccReal t ~ HsAccReal (AsDynamic t)
  , IndexStorage t ~ IndexStorage (AsDynamic t)
  , MaskTensor t ~ MaskTensor (AsDynamic t)
  )

type Static2 t0 t1 = (Static t0, Static t1, AsDynamic t0 ~ AsDynamic t1)


