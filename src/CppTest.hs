{-

select compiler using:
`stack build --ghc-options='-pgmc [gcc/clang path]'`

run in ghci using:
`stack ghci --ghc-options='-fobject-code' --main-is ffi-experimental:exe:cpp-test`

currently ghci run works, but `stack build` runs into an issue
similar to
https://github.com/fpco/inline-c/issues/75

-}

{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}

import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C

C.context C.cppCtx

C.include "<iostream>"

main :: IO ()
main = do
    let x = 3
    [C.block| void {
        std::cout << "Hello, world: " << $(int x) << std::endl;
    } |]