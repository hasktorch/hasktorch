{-

select compiler using:
`stack build --ghc-options='-pgmc [gcc/clang path]'`

run in ghci using:
`stack ghci --ghc-options='-fobject-code' --main-is ffi-experimental:exe:cpp-test`

-}

{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}

import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C

C.context C.cppCtx

C.include "<iostream>"
C.include "<torch/torch.h>"

main :: IO ()
main = do
    [C.block| void {
        std::cout << "Hello torch!" << std::endl;
        torch::Tensor tensor = torch::rand({2, 3});
        std::cout << tensor << std::endl;
    } |]