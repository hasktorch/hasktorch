-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Types.THC.Internal
-- Copyright :  (c) Hasktorch dev team 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Any manual work which has to be ammended to Torch.Types.THC.Structs after
-- running c2hsc.
-------------------------------------------------------------------------------
{-# LANGUAGE ConstraintKinds #-}
module Torch.Types.THC.Internal
  ( module X
  , C'THAllocator(..)
  , C'cudaStream_t, c'cudaStream_t
  , C'cusparseHandle_t, c'cusparseHandle_t
  , C'cublasHandle_t, c'cublasHandle_t
  , C'cusparseContext, c'cusparseContext
  ) where

import Foreign.Ptr

import Torch.Types.Cuda.Structs as X
import Torch.Types.CuRand.Structs as X
import Torch.Types.TH.Structs (C'THAllocator(..))

type C'cudaStream_t = Ptr ()
c'cudaStream_t = nullPtr

type C'cusparseHandle_t = Ptr ()
c'cusparseHandle_t = nullPtr

type C'cublasHandle_t = Ptr ()
c'cublasHandle_t = nullPtr

type C'cusparseContext = Ptr ()
c'cusparseContext = nullPtr


