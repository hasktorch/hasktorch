{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}

module LapackTest where

import Prelude as P
import Torch.Double as T

-- TODO - move this under tests

{- Lapack sanity checks -}

testGesv = do
    putStrLn "\n\ngesv test\n\n"
    Just (t :: Tensor '[3, 3]) <- fromList [2, 4, 6, 0, -1, -8, 0, 0, 96]
    let trg = eye :: Tensor '[3, 3]
    let (invT, invTLU) = gesv (eye :: Tensor '[3, 3]) t
    print t
    print trg
    print invT
    print invTLU
    print (t !*! invT)

testGetri = do
    putStrLn "\n\ngetri test\n\n"
    Just (t :: Tensor '[3, 3]) <- fromList [2, 4, 6, 0, -1, -8, 0, 0, 96]
    let invT = (getri t) :: Tensor '[3, 3]
    print t
    print invT
    print (t !*! invT)