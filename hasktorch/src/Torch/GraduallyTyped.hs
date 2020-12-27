{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
-- {-# OPTIONS_GHC -fplugin TypeLevel.Rewrite
--                 -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Prelude.TestLaw
--                 -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Prelude.TestRightAssociative #-}

module Torch.GraduallyTyped
  ( module Torch.Data,
    module Torch.GraduallyTyped,
    module Torch.GraduallyTyped.Prelude,
    module Torch.GraduallyTyped.Autograd,
    module Torch.GraduallyTyped.NN,
    module Torch.GraduallyTyped.Random,
    module Torch.GraduallyTyped.Tensor,
    module Torch.GraduallyTyped.Device,
    module Torch.GraduallyTyped.Shape,
    module Torch.GraduallyTyped.DType,
    module Torch.GraduallyTyped.Layout,
    module Torch.GraduallyTyped.RequiresGradient,
    module Torch.GraduallyTyped.Scalar,
    module Torch.HList,
    module Torch.DType,
  )
where

import Torch.DType
import Torch.Data
import Torch.GraduallyTyped.Autograd
import Torch.GraduallyTyped.DType
import Torch.GraduallyTyped.Device
import Torch.GraduallyTyped.Layout
import Torch.GraduallyTyped.NN
import Torch.GraduallyTyped.Prelude
import Torch.GraduallyTyped.Random
import Torch.GraduallyTyped.RequiresGradient
import Torch.GraduallyTyped.Scalar
import Torch.GraduallyTyped.Shape
import Torch.GraduallyTyped.Tensor
import Torch.HList

-- import Torch.GraduallyTyped.Prelude (TestF)

-- class Foo a where
--   foo :: ()

-- q :: forall a b . (Foo (TestF a b)) => ()
-- q = foo @(TestF a (TestF a b))

-- q' :: forall a b . (Foo (TestF a (TestF a (TestF a (TestF a b))))) => ()
-- q' = foo @(TestF a (TestF a b))

-- q'' :: forall a b c . (Foo (TestF a (TestF b c))) => ()
-- q'' = foo @(TestF (TestF a b) c)

-- q''' :: forall a b c . (Foo (TestF (TestF a b) c)) => ()
-- q''' = foo @(TestF a (TestF b c))

-- f' :: forall a b . (Foo (TestF a (TestF a b))) => ()
-- f' = foo @(TestF a b)

-- f'' :: forall a b c. (Foo (TestF a (TestF b c))) => ()
-- f'' = foo @(TestF (TestF a b) c)

-- f''' :: forall a b c. (Foo (TestF (TestF a b) c)) => ()
-- f''' = foo @(TestF a (TestF b c))