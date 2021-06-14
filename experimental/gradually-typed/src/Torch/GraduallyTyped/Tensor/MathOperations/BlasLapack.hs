{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.Tensor.MathOperations.BlasLapack where

import GHC.TypeLits (Nat, Symbol, TypeError)
import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.Prelude (PrependMaybe, Reverse)
import Torch.GraduallyTyped.Shape (BroadcastDimsImplF, Dim, Name, Shape (..), Size)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))
import Torch.Internal.Cast (cast2)
import qualified Torch.Internal.Managed.Native as ATen
import Type.Errors.Pretty (type (%), type (<>))

type family MatmulDimsImplF (reversedDims :: [Dim (Name Symbol) (Size Nat)]) (reversedDims' :: [Dim (Name Symbol) (Size Nat)]) :: Maybe [Dim (Name Symbol) (Size Nat)] where
  MatmulDimsImplF (k ': '[]) (k ': '[]) = Just '[]
  MatmulDimsImplF (k ': '[]) (m ': k ': reversedBroadcastDims') =
    PrependMaybe (Just m) (BroadcastDimsImplF '[] reversedBroadcastDims')
  MatmulDimsImplF (k ': n ': reversedBroadcastDims) (k ': '[]) =
    PrependMaybe (Just n) (BroadcastDimsImplF '[] reversedBroadcastDims)
  MatmulDimsImplF (k ': n ': reversedBroadcastDims) (m ': k ': reversedBroadcastDims') =
    PrependMaybe (Just m) (PrependMaybe (Just n) (BroadcastDimsImplF reversedBroadcastDims reversedBroadcastDims'))
  MatmulDimsImplF _ _ = 'Nothing

type family MatmulDimsCheckF (dims :: [Dim (Name Symbol) (Size Nat)]) (dims' :: [Dim (Name Symbol) (Size Nat)]) (result :: Maybe [Dim (Name Symbol) (Size Nat)]) :: [Dim (Name Symbol) (Size Nat)] where
  MatmulDimsCheckF dims dims' Nothing =
    TypeError
      ( "Cannot multiply the tensors since the dimensions"
          % ""
          % "    '" <> dims <> "' and '" <> dims' <> "'"
          % ""
          % "are not compatible for matrix multiplation."
          % "You may need to reshape the tensor(s) first."
      )
  MatmulDimsCheckF _ _ (Just result) = (Reverse result)

type MatmulDimsF dims dims' = MatmulDimsCheckF dims dims' (MatmulDimsImplF (Reverse dims) (Reverse dims'))

type family MatmulF (shape :: Shape [Dim (Name Symbol) (Size Nat)]) (shape' :: Shape [Dim (Name Symbol) (Size Nat)]) :: Shape [Dim (Name Symbol) (Size Nat)] where
  MatmulF 'UncheckedShape _ = 'UncheckedShape
  MatmulF _ 'UncheckedShape = 'UncheckedShape
  MatmulF ('Shape dims) ('Shape dims') = 'Shape (MatmulDimsF dims dims')

-- | Matrix product of two tensors.
--
-- The behavior depends on the dimensionality of the tensors as follows:
--
--     (1) If both tensors are 1-dimensional, the dot product (scalar) is returned:
--
--         >>> g <- generator @('Device 'CPU) 0
--         >>> (tensor1, g') = randn @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Sized 3)]) g
--         >>> (tensor2, g'') = randn @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Sized 3)]) g'
--         >>> result = tensor1 `matmul` tensor2
--         >>> :type result
--         result
--           :: Tensor
--            'WithGradient
--            ('Layout 'Dense)
--            ('Device 'CPU)
--            ('DataType 'Float)
--            ('Shape '[])
--
--
--     (2) If both arguments are 2-dimensional, the matrix-matrix product is returned:
--
--         >>> g <- generator @('Device 'CPU) 0
--         >>> (tensor1, g') = randn @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Sized 3), 'Dim ('Sized 4)]) g
--         >>> (tensor2, g'') = randn @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Sized 4), 'Dim ('Sized 7)]) g'
--         >>> result = tensor1 `matmul` tensor2
--         >>> :type result
--         result
--           :: Tensor
--                'WithGradient
--                ('Layout 'Dense)
--                ('Device 'CPU)
--                ('DataType 'Float)
--                ('Shape '[ 'Dim ('Sized 3), 'Dim ('Sized 7)])
--
--
--     (3) If the first argument is 1-dimensional and the second argument is 2-dimensional,
--     a 1 is prepended to its dimension for the purpose of the matrix multiply.
--     After the matrix multiply, the prepended dimension is removed:
--
--         >>> g <- generator @('Device 'CPU) 0
--         >>> (tensor1, g') = randn @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Sized 4)]) g
--         >>> (tensor2, g'') = randn @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Sized 4), 'Dim ('Sized 7)]) g'
--         >>> result = tensor1 `matmul` tensor2
--         >>> :type result
--         result
--           :: Tensor
--                'WithGradient
--                ('Layout 'Dense)
--                ('Device 'CPU)
--                ('DataType 'Float)
--                ('Shape '[ 'Dim ('Sized 7)])
--
--
--     (4) If the first argument is 2-dimensional and the second argument is 1-dimensional,
--     the matrix-vector product is returned:
--
--         >>> g <- generator @('Device 'CPU) 0
--         >>> (tensor1, g') = randn @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Sized 3), 'Dim ('Sized 4)]) g
--         >>> (tensor2, g'') = randn @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Sized 4)]) g'
--         >>> result = tensor1 `matmul` tensor2
--         >>> :type result
--         result
--           :: Tensor
--                'WithGradient
--                ('Layout 'Dense)
--                ('Device 'CPU)
--                ('DataType 'Float)
--                ('Shape '[ 'Dim ('Sized 3)])
--
--
--     (5) If both arguments are at least 1-dimensional and at least one argument is \(n\)-dimensional (where \(n > 2\)),
--     then a batched matrix multiply is returned.
--
--     The following is an example of a batched matrix multiplication:
--
--         >>> g <- generator @('Device 'CPU) 0
--         >>> (tensor1, g') = randn @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('NamedSized "batch" 10), 'Dim ('Sized 3), 'Dim ('Sized 4)]) g
--         >>> (tensor2, g'') = randn @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('NamedSized "batch" 10), 'Dim ('Sized 4), 'Dim ('Sized 7)]) g'
--         >>> result = tensor1 `matmul` tensor2
--         >>> :type result
--         result
--           :: Tensor
--                'WithGradient
--                ('Layout 'Dense)
--                ('Device 'CPU)
--                ('DataType 'Float)
--                ('Shape '[ 'Dim ('NamedSized "batch" 10), 'Dim ('Sized 3), 'Dim ('Sized 7)])
--
--
--     If the first argument is 1-dimensional,
--     a 1 is prepended to its dimension for the purpose of the batched matrix multiply and removed after:
--
--         >>> g <- generator @('Device 'CPU) 0
--         >>> (tensor1, g') = randn @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Sized 4)]) g
--         >>> (tensor2, g'') = randn @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('NamedSized "batch" 10), 'Dim ('Sized 4), 'Dim ('Sized 7)]) g'
--         >>> result = tensor1 `matmul` tensor2
--         >>> :type result
--         result
--           :: Tensor
--                'WithGradient
--                ('Layout 'Dense)
--                ('Device 'CPU)
--                ('DataType 'Float)
--                ('Shape '[ 'Dim ('NamedSized "batch" 10), 'Dim ('Sized 7)])
--
--
--     If the second argument is 1-dimensional,
--     a 1 is appended to its dimension for the purpose of the batched matrix multiply and removed after:
--
--         >>> g <- generator @('Device 'CPU) 0
--         >>> (tensor1, g') = randn @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('NamedSized "batch" 10), 'Dim ('Sized 3), 'Dim ('Sized 4)]) g
--         >>> (tensor2, g'') = randn @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Sized 4)]) g'
--         >>> result = tensor1 `matmul` tensor2
--         >>> :type result
--         result
--           :: Tensor
--                'WithGradient
--                ('Layout 'Dense)
--                ('Device 'CPU)
--                ('DataType 'Float)
--                ('Shape '[ 'Dim ('NamedSized "batch" 10), 'Dim ('Sized 3)])
--
--
--     The non-matrix (i.e. batch) dimensions are broadcasted (and thus must be broadcastable).
--     For example, if 'input' is a \(j \times 1 \times n \times m\) tensor and
--     'other' is a \(k \times m \times p\) tensor, 'output' will be a \(j \times k \times n \times p\) tensor:
--
--         >>> g <- generator @('Device 'CPU) 0
--         >>> (tensor1, g') = randn @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('NamedSized "batch" 10), 'Dim ('Sized 1), 'Dim ('Sized 3), 'Dim ('Sized 4)]) g
--         >>> (tensor2, g'') = randn @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Sized 5), 'Dim ('Sized 4), 'Dim ('Sized 7)]) g'
--         >>> result = tensor1 `matmul` tensor2
--         >>> :type result
--         result
--           :: Tensor
--                'WithGradient
--                ('Layout 'Dense)
--                ('Device 'CPU)
--                ('DataType 'Float)
--                ('Shape '[ 'Dim ('NamedSized "batch" 10), 'Dim ('Sized 5), 'Dim ('Sized 3), 'Dim ('Sized 7)])
matmul ::
  forall requiresGradient layout layout' device device' dataType dataType' shape shape'.
  -- input
  Tensor requiresGradient layout device dataType shape ->
  -- other
  Tensor requiresGradient layout' device' dataType' shape' ->
  -- output
  Tensor
    requiresGradient
    (layout <+> layout')
    (device <+> device')
    (dataType <+> dataType')
    (MatmulF shape shape')
input `matmul` other = unsafePerformIO $ cast2 ATen.matmul_tt input other
