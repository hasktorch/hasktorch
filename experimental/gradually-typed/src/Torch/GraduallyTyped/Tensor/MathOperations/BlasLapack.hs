{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.Tensor.MathOperations.BlasLapack where

import Control.Monad.Catch (MonadThrow)
import Data.Type.Bool (If)
import GHC.TypeLits (Nat, Symbol, TypeError)
import Torch.GraduallyTyped.Prelude (Catch, PrependMaybe, Reverse)
import Torch.GraduallyTyped.Shape (BroadcastDimsImplF, Dim (..), Name (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (UnifyCheck, type (<+>), type (<|>))
import qualified Torch.Internal.Cast as ATen (cast2)
import Torch.Internal.GC (unsafeThrowableIO)
import qualified Torch.Internal.Managed.Native as ATen
import Type.Errors.Pretty (type (%), type (<>))

-- $setup
-- >>> import Torch.GraduallyTyped.Prelude.List (SList (..))
-- >>> import Torch.GraduallyTyped

type MatmulDimsImplF :: [Dim (Name Symbol) (Size Nat)] -> [Dim (Name Symbol) (Size Nat)] -> Maybe [Dim (Name Symbol) (Size Nat)]
type family MatmulDimsImplF reversedDims reversedDims' where
  MatmulDimsImplF (k ': '[]) (k' ': '[]) =
    If (UnifyCheck (Dim (Name Symbol) (Size Nat)) k k') ('Just '[]) 'Nothing
  MatmulDimsImplF (k ': '[]) (m ': k' ': reversedBroadcastDims') =
    If (UnifyCheck (Dim (Name Symbol) (Size Nat)) k k') (PrependMaybe ('Just m) (BroadcastDimsImplF '[] reversedBroadcastDims')) 'Nothing
  MatmulDimsImplF (k ': n ': reversedBroadcastDims) (k' ': '[]) =
    If (UnifyCheck (Dim (Name Symbol) (Size Nat)) k k') (PrependMaybe ('Just n) (BroadcastDimsImplF '[] reversedBroadcastDims)) 'Nothing
  MatmulDimsImplF (k ': n ': reversedBroadcastDims) (m ': k' ': reversedBroadcastDims') =
    If (UnifyCheck (Dim (Name Symbol) (Size Nat)) k k') (PrependMaybe ('Just m) (PrependMaybe ('Just n) (BroadcastDimsImplF reversedBroadcastDims reversedBroadcastDims'))) 'Nothing
  MatmulDimsImplF _ _ = 'Nothing

type MatmulDimsCheckF :: [Dim (Name Symbol) (Size Nat)] -> [Dim (Name Symbol) (Size Nat)] -> Maybe [Dim (Name Symbol) (Size Nat)] -> [Dim (Name Symbol) (Size Nat)]
type family MatmulDimsCheckF dims dims' result where
  MatmulDimsCheckF dims dims' 'Nothing =
    TypeError
      ( "Cannot multiply the tensors since the dimensions"
          % ""
          % "    '" <> dims <> "' and '" <> dims' <> "'"
          % ""
          % "are not compatible for matrix multiplation."
          % "You may need to reshape the tensor(s) first."
      )
  MatmulDimsCheckF _ _ ('Just result) = (Reverse result)

type MatmulDimsF :: [Dim (Name Symbol) (Size Nat)] -> [Dim (Name Symbol) (Size Nat)] -> [Dim (Name Symbol) (Size Nat)]

type MatmulDimsF dims dims' = MatmulDimsCheckF dims dims' (MatmulDimsImplF (Reverse dims) (Reverse dims'))

type MatmulF :: Shape [Dim (Name Symbol) (Size Nat)] -> Shape [Dim (Name Symbol) (Size Nat)] -> Shape [Dim (Name Symbol) (Size Nat)]
type family MatmulF shape shape' where
  MatmulF 'UncheckedShape _ = 'UncheckedShape
  MatmulF _ 'UncheckedShape = 'UncheckedShape
  MatmulF ('Shape dims) ('Shape dims') = 'Shape (MatmulDimsF dims dims')

-- | Matrix product of two tensors.
--
-- The following code serves the examples of @matmul@ below:
--
-- >>> g <- sMkGenerator (SDevice SCPU) 0
-- >>> sRandn' = sRandn . TensorSpec (SGradient SWithGradient) (SLayout SDense) (SDevice SCPU) (SDataType SFloat)
--
-- In order to understand the behavior of @matmul@, consider the following cases:
--
--     (1) If both tensors are 1-dimensional, the dot product (scalar) is returned:
--
--         >>> (tensor1, g') <- sRandn' (SShape $ SName @"*" :&: SSize @3 :|: SNil) g
--         >>> (tensor2, g'') <- sRandn' (SShape $ SName @"*" :&: SSize @3 :|: SNil) g'
--         >>> result = tensor1 `matmul` tensor2
--         >>> :type result
--         result
--           :: MonadThrow m =>
--              m (Tensor
--                   ('Gradient WithGradient)
--                   ('Layout Dense)
--                   ('Device CPU)
--                   ('DataType 'Float)
--                   ('Shape '[]))
--
--
--     (2) If both arguments are 2-dimensional, the matrix-matrix product is returned:
--
--         >>> (tensor1, g') <- sRandn' (SShape $ SName @"*" :&: SSize @3 :|: SName @"*" :&: SSize @4 :|: SNil) g
--         >>> (tensor2, g'') <- sRandn' (SShape $ SName @"*" :&: SSize @4 :|: SName @"*" :&: SSize @7 :|: SNil) g'
--         >>> result = tensor1 `matmul` tensor2
--         >>> :type result
--         result
--           :: MonadThrow m =>
--              m (Tensor
--                   ('Gradient WithGradient)
--                   ('Layout Dense)
--                   ('Device CPU)
--                   ('DataType 'Float)
--                   ('Shape ['Dim ('Name "*") ('Size 3), 'Dim ('Name "*") ('Size 7)]))
--
--
--     (3) If the first argument is 1-dimensional and the second argument is 2-dimensional,
--     a 1 is prepended to its dimension for the purpose of the matrix multiply.
--     After the matrix multiply, the prepended dimension is removed:
--
--         >>> (tensor1, g') <- sRandn' (SShape $ SName @"*" :&: SSize @4 :|: SNil) g
--         >>> (tensor2, g'') <- sRandn' (SShape $ SName @"*" :&: SSize @4 :|: SName @"*" :&: SSize @7 :|: SNil) g'
--         >>> result = tensor1 `matmul` tensor2
--         >>> :type result
--         result
--           :: MonadThrow m =>
--              m (Tensor
--                   ('Gradient WithGradient)
--                   ('Layout Dense)
--                   ('Device CPU)
--                   ('DataType 'Float)
--                   ('Shape '[ 'Dim ('Name "*") ('Size 7)]))
--
--
--     (4) If the first argument is 2-dimensional and the second argument is 1-dimensional,
--     the matrix-vector product is returned:
--
--         >>> (tensor1, g') <- sRandn' (SShape $ SName @"*" :&: SSize @3 :|: SName @"*" :&: SSize @4 :|: SNil) g
--         >>> (tensor2, g'') <- sRandn' (SShape $ SName @"*" :&: SSize @4 :|: SNil) g'
--         >>> result = tensor1 `matmul` tensor2
--         >>> :type result
--         result
--           :: MonadThrow m =>
--              m (Tensor
--                   ('Gradient WithGradient)
--                   ('Layout Dense)
--                   ('Device CPU)
--                   ('DataType 'Float)
--                   ('Shape '[ 'Dim ('Name "*") ('Size 3)]))
--
--
--     (5) If both arguments are at least 1-dimensional and at least one argument is \(n\)-dimensional (where \(n > 2\)),
--     then a batched matrix multiply is returned.
--
--     The following is an example of a batched matrix multiplication:
--
--         >>> (tensor1, g') <- sRandn' (SShape $ SName @"batch" :&: SSize @10 :|: SName @"*" :&: SSize @3 :|: SName @"*" :&: SSize @4 :|: SNil) g
--         >>> (tensor2, g'') <- sRandn' (SShape $ SName @"batch" :&: SSize @10 :|: SName @"*" :&: SSize @4 :|: SName @"*" :&: SSize @7 :|: SNil) g'
--         >>> result = tensor1 `matmul` tensor2
--         >>> :type result
--         result
--           :: MonadThrow m =>
--              m (Tensor
--                   ('Gradient WithGradient)
--                   ('Layout Dense)
--                   ('Device CPU)
--                   ('DataType 'Float)
--                   ('Shape
--                      ['Dim ('Name "batch") ('Size 10), 'Dim ('Name "*") ('Size 3),
--                       'Dim ('Name "*") ('Size 7)]))
--
--
--     If the first argument is 1-dimensional,
--     a 1 is prepended to its dimension for the purpose of the batched matrix multiply and removed after:
--
--         >>> (tensor1, g') <- sRandn' (SShape $ SName @"*" :&: SSize @4 :|: SNil) g
--         >>> (tensor2, g'') <- sRandn' (SShape $ SName @"batch" :&: SSize @10 :|: SName @"*" :&: SSize @4 :|: SName @"*" :&: SSize @7 :|: SNil) g'
--         >>> result = tensor1 `matmul` tensor2
--         >>> :type result
--         result
--           :: MonadThrow m =>
--              m (Tensor
--                   ('Gradient WithGradient)
--                   ('Layout Dense)
--                   ('Device CPU)
--                   ('DataType 'Float)
--                   ('Shape
--                      ['Dim ('Name "batch") ('Size 10), 'Dim ('Name "*") ('Size 7)]))
--
--
--     If the second argument is 1-dimensional,
--     a 1 is appended to its dimension for the purpose of the batched matrix multiply and removed after:
--
--         >>> (tensor1, g') <- sRandn' (SShape $ SName @"batch" :&: SSize @10 :|: SName @"*" :&: SSize @3 :|: SName @"*" :&: SSize @4 :|: SNil) g
--         >>> (tensor2, g'') <- sRandn' (SShape $ SName @"*" :&: SSize @4 :|: SNil) g'
--         >>> result = tensor1 `matmul` tensor2
--         >>> :type result
--         result
--           :: MonadThrow m =>
--              m (Tensor
--                   ('Gradient WithGradient)
--                   ('Layout Dense)
--                   ('Device CPU)
--                   ('DataType 'Float)
--                   ('Shape
--                      ['Dim ('Name "batch") ('Size 10), 'Dim ('Name "*") ('Size 3)]))
--
--
--     The non-matrix (i.e. batch) dimensions are broadcasted (and thus must be broadcastable).
--     For example, if 'input' is a \(j \times 1 \times n \times m\) tensor and
--     'other' is a \(k \times m \times p\) tensor, 'output' will be a \(j \times k \times n \times p\) tensor:
--
--         >>> (tensor1, g') <- sRandn' (SShape $ SName @"batch" :&: SSize @10 :|: SName @"*" :&: SSize @1 :|: SName @"*" :&: SSize @3 :|: SName @"*" :&: SSize @4 :|: SNil) g
--         >>> (tensor2, g'') <- sRandn' (SShape $ SName @"*" :&: SSize @5 :|: SName @"*" :&: SSize @4 :|: SName @"*" :&: SSize @7 :|: SNil) g'
--         >>> result = tensor1 `matmul` tensor2
--         >>> :type result
--         result
--           :: MonadThrow m =>
--              m (Tensor
--                   ('Gradient WithGradient)
--                   ('Layout Dense)
--                   ('Device CPU)
--                   ('DataType 'Float)
--                   ('Shape
--                      ['Dim ('Name "batch") ('Size 10), 'Dim ('Name "*") ('Size 5),
--                       'Dim ('Name "*") ('Size 3), 'Dim ('Name "*") ('Size 7)]))
matmul ::
  forall m gradient gradient' layout layout' device device' dataType dataType' shape shape' shape''.
  (MonadThrow m, shape'' ~ MatmulF shape shape', Catch shape'') =>
  -- input
  Tensor gradient layout device dataType shape ->
  -- other
  Tensor gradient' layout' device' dataType' shape' ->
  -- output
  m
    ( Tensor
        (gradient <|> gradient')
        (layout <+> layout')
        (device <+> device')
        (dataType <+> dataType')
        shape''
    )
input `matmul` other = unsafeThrowableIO $ ATen.cast2 ATen.matmul_tt input other
