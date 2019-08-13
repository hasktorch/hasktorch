{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.Static.Native where

import Data.Proxy
import Data.Finite
import Data.Kind (Constraint)
import GHC.TypeLits

import Prelude hiding (sin)
import qualified Torch.Tensor as D
import qualified Torch.TensorFactories as D
import Torch.Functions as D
import Torch.DType
import Torch.Static
import qualified Torch.TensorOptions as D
import Data.Reflection

sin :: Tensor dtype shape -> Tensor dtype shape
sin t = UnsafeMkTensor $ D.sin (toDynamic t)

sigmoid :: Tensor dtype shape -> Tensor dtype shape
sigmoid t = UnsafeMkTensor $ D.sigmoid (toDynamic t)


