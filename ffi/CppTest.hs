{-

select compiler using:
`stack build --ghc-options='-pgmc [gcc/clang path]'`

run in ghci using:
`stack ghci --ghc-options='-fobject-code' --main-is ffi-experimental:exe:cpp-test`

-}

{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C

C.context C.cppCtx

C.include "<iostream>"
C.include "<torch/torch.h>"
C.include "<torch/csrc/autograd/variable.h>"
C.include "<torch/csrc/autograd/function.h>"

testInit :: IO ()
testInit = do
    [C.block| void {
        std::cout << "Hello torch!" << std::endl;
        torch::Tensor tensor = torch::rand({2, 3});
        std::cout << tensor << std::endl;
    } |]

testAutograd :: IO ()
testAutograd = do
    [C.block| void {
        torch::Tensor a = torch::ones({2, 2}, torch::requires_grad());
        torch::Tensor b = torch::randn({2, 2});
        auto c = a + b;
        c.backward();
        std::cout << a << std::endl << b << std::endl << c << std::endl;
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
    testAutograd