{-# LANGUAGE PolyKinds #-}

module Torch.GraduallyTyped.Internal.Void where

{-
  Note [Uncluttering type signatures]
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Because various instances in this library are wrappers of constraint synonyms,
  GHC replaces them with their constraints, which results in large, unreadable types.
  Writing an (overlapping instance) for the 'Void' type means that the original
  instance might not be the one selected, thus GHC leaves the constraints in
  place until further information is provided, at which point the type
  machinery has sufficient information to reduce to sensible types.
  This solution has been adapted from Csongor Kiss and his generic-lens library, see
  https://kcsongor.github.io/opaque-constraint-synonyms/.
-}
data Void
