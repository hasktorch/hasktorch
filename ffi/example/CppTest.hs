{-

select compiler using:
`stack build --ghc-options='-pgmc [gcc/clang path]'`

run in ghci using:
`stack ghci --ghc-options='-fobject-code' --main-is ffi-experimental:exe:cpp-test`

-}

{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C
import qualified Data.Map as Map
import Control.Exception.Safe

import Foreign.C.String
import Foreign.C.Types
import Foreign

data Tensor

C.context $ C.cppCtx <> mempty {
    C.ctxTypesTable = Map.fromList [
      (C.TypeName "PtrTensor", [t|Ptr Tensor|])
    ]
}

C.include "<iostream>"
C.include "<torch/torch.h>"
C.include "<torch/csrc/autograd/variable.h>"
C.include "<torch/csrc/autograd/function.h>"

-- inline-c can not parse c++'s symbol like torch::Tensor which includes "::".
-- "helper.h" defines "typedef torch::Tensor* PtrTensor;" to remove "::".
C.include "helper.h"

testInit :: IO ()
testInit = do
    [C.block| void {
        std::cout << "Hello torch!" << std::endl;
        torch::Tensor tensor = torch::rand({2, 3});
        std::cout << tensor << std::endl;
    } |]

testException :: IO ()
testException = do
    v <- try [C.throwBlock| void {
        std::cout << "Hello Exception!" << std::endl;
        torch::Tensor tensor = torch::rand({2, 3});
        std::cout << tensor[3.14] << std::endl;
    } |]
    print (v :: Either SomeException ())

testAutograd :: IO ()
testAutograd = do
    [C.block| void {
        torch::Tensor a = torch::ones({2, 2}, torch::requires_grad());
        torch::Tensor b = torch::randn({2, 2});
        auto c = a + b;
        c.backward();
        std::cout << a << std::endl << b << std::endl << c << std::endl;
    } |]
    [C.block| void {
        std::cout << "Hello torch!" << std::endl;
    } |]
    a2 <- [C.block| PtrTensor {
      return new torch::Tensor(torch::ones({2, 2}, torch::requires_grad()));
    } |]
    b2 <- [C.block| PtrTensor {
      return new torch::Tensor(torch::randn({2, 2}));
    } |]
    [C.block| void {
        auto c2 = *$(PtrTensor a2) + *$(PtrTensor b2);
        c2.backward();
        std::cout << *$(PtrTensor a2) << std::endl << *$(PtrTensor b2) << std::endl << c2 << std::endl;
    } |]

testResource :: IO ()
testResource = do
    -- TODO: pass tensor between c++/haskell
    let x = [C.block| void {
        torch::Tensor x = torch::randn({2, 2});
    } |]
    pure ()

main :: IO ()
main = do
    testInit
    testException
    testAutograd
