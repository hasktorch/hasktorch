   let prelude = https://raw.githubusercontent.com/dhall-lang/dhall-to-cabal/master/dhall/prelude.dhall
in let types = https://raw.githubusercontent.com/dhall-lang/dhall-to-cabal/master/dhall/types.dhall
in

{ anyver =
   λ(pkg : Text) →
   { bounds = prelude.anyVersion, package = pkg } : types.Dependency

, renameNoop =
   λ(pkg : Text) →
   { name = pkg, original = { name = pkg, package = [] : Optional Text } }

, mixinRequirements  =
   λ(renames : List { rename : Text, to : Text }) →
   < renaming = renames | default : {} | hiding : List Text >

, renameSig =
   λ(to : Text) →
   λ(specific : Text) →
   { rename = ("Torch.Sig."++specific) : Text
   , to = ("Torch."++ to ++ "."++specific) : Text
   }
}
