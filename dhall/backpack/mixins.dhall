   let prelude = https://raw.githubusercontent.com/dhall-lang/dhall-to-cabal/master/dhall/prelude.dhall
in let types = https://raw.githubusercontent.com/dhall-lang/dhall-to-cabal/master/dhall/types.dhall
in let common = ../common.dhall
in let packages = common.packages
in let cabalvars = common.cabalvars
in let fn = ../common/functions.dhall

in let unsigned =
    λ(isth : Bool) →
    λ(ttype : Text) →
        let lib = fn.showlib isth
     in let tensorMathOr = λ(pkg : Text) → if isth then "TensorMath" else pkg
     in let ffi = "Torch.FFI.${lib}.${ttype}"
     in [ { rename = "Torch.Sig.State"                 , to = if isth then "Torch.Types.TH" else "Torch.FFI.THC.State" }
        , { rename = "Torch.Sig.Types.Global"          , to = "Torch.Types.${lib}" }
        , { rename = "Torch.Sig.Types"                 , to = "Torch.Types.${lib}.${ttype}" }
        , { rename = "Torch.Sig.Storage"               , to = "Torch.FFI.${lib}.${ttype}.Storage" }
        , { rename = "Torch.Sig.Storage.Copy"          , to = "Torch.FFI.${lib}.${ttype}.StorageCopy" }
        , { rename = "Torch.Sig.Storage.Memory"        , to = "Torch.FFI.${lib}.${ttype}." ++ (if isth then "FreeStorage" else "Storage" ) }
        , { rename = "Torch.Sig.Tensor"                , to = "Torch.FFI.${lib}.${ttype}." ++ (if isth then "Tensor" else "Tensor" ) }
        , { rename = "Torch.Sig.Tensor.Copy"           , to = "Torch.FFI.${lib}.${ttype}.TensorCopy" }
        , { rename = "Torch.Sig.Tensor.Memory"         , to = "Torch.FFI.${lib}.${ttype}." ++ (if isth then "FreeTensor" else "Tensor" ) }
        , { rename = "Torch.Sig.Tensor.Index"          , to = "Torch.FFI.${lib}.${ttype}." ++ (tensorMathOr "TensorIndex" ) }
        , { rename = "Torch.Sig.Tensor.Masked"         , to = "Torch.FFI.${lib}.${ttype}." ++ (tensorMathOr "TensorMasked" ) }
        , { rename = "Torch.Sig.Tensor.Math"           , to = "Torch.FFI.${lib}.${ttype}." ++ (tensorMathOr "TensorMath" ) }
        , { rename = "Torch.Sig.Tensor.Math.Compare"   , to = "Torch.FFI.${lib}.${ttype}." ++ (tensorMathOr "TensorMathCompare" ) }
        , { rename = "Torch.Sig.Tensor.Math.CompareT"  , to = "Torch.FFI.${lib}.${ttype}." ++ (tensorMathOr "TensorMathCompareT" ) }
        , { rename = "Torch.Sig.Tensor.Math.Pairwise"  , to = "Torch.FFI.${lib}.${ttype}." ++ (tensorMathOr "TensorMathPairwise" ) }
        , { rename = "Torch.Sig.Tensor.Math.Pointwise" , to = "Torch.FFI.${lib}.${ttype}." ++ (tensorMathOr "TensorMathPointwise") }
        , { rename = "Torch.Sig.Tensor.Math.Reduce"    , to = "Torch.FFI.${lib}.${ttype}." ++ (tensorMathOr "TensorMathReduce")  }
        , { rename = "Torch.Sig.Tensor.Math.Scan"      , to = "Torch.FFI.${lib}.${ttype}." ++ (tensorMathOr "TensorMathScan" ) }
        , { rename = "Torch.Sig.Tensor.Mode"           , to = "Torch.FFI.${lib}.${ttype}." ++ (tensorMathOr  "TensorMode" ) }
        , { rename = "Torch.Sig.Tensor.ScatterGather"  , to = "Torch.FFI.${lib}.${ttype}." ++ (tensorMathOr "TensorScatterGather" ) }
        , { rename = "Torch.Sig.Tensor.Sort"           , to = "Torch.FFI.${lib}.${ttype}." ++ (tensorMathOr "TensorSort" ) }
        , { rename = "Torch.Sig.Tensor.TopK"           , to = "Torch.FFI.${lib}.${ttype}." ++ (tensorMathOr "TensorTopK" ) }
        ]

in let signed =
    λ(isth : Bool) →
    λ(ttype : Text) →
        let lib = fn.showlib isth
     in let tensorMathOr = λ(pkg : Text) → if isth then "TensorMath" else pkg
     in let ffi = "Torch.FFI.${lib}.${ttype}"
     in (unsigned isth ttype) #
        [ { rename = "Torch.Sig.Tensor.Math.Pointwise.Signed", to = "Torch.FFI.${lib}.${ttype}." ++ (tensorMathOr "TensorMathPointwise" ) }
        ]

in let floating =
    λ(isth : Bool) →
    λ(ttype : Text) →
        let lib = fn.showlib isth
     in let tensorMathOr = λ(pkg : Text) → if isth then "TensorMath" else pkg
     in let ffi = "Torch.FFI.${lib}.${ttype}"
     in (signed isth ttype) #
        [ { rename = "Torch.Sig.Tensor.Math.Pointwise.Floating" , to = "Torch.FFI.${lib}.${ttype}." ++ (tensorMathOr "TensorMathPointwise" ) }
        , { rename = "Torch.Sig.Tensor.Math.Reduce.Floating"    , to = "Torch.FFI.${lib}.${ttype}." ++ (tensorMathOr "TensorMathReduce" ) }
        , { rename = "Torch.Sig.Tensor.Math.Floating"           , to = "Torch.FFI.${lib}.${ttype}." ++ (tensorMathOr "TensorMath" ) }
        , { rename = "Torch.Sig.Tensor.Math.Blas"               , to = "Torch.FFI.${lib}.${ttype}." ++ (tensorMathOr "TensorMathBlas" ) }
        , { rename = "Torch.Sig.Tensor.Math.Lapack"             , to = "Torch.FFI.${lib}.${ttype}." ++ (if isth then "TensorLapack" else "TensorMathMagma" ) }
        , { rename = "Torch.Sig.NN"                             , to = "Torch.FFI.${lib}.NN.${ttype}" }
        , { rename = "Torch.Sig.Types.NN"                       , to = "Torch.Types.${lib}" }

        , { rename = "Torch.Sig.Tensor.Math.Random.TH", to =
            (if isth then "Torch.FFI.TH.${ttype}.TensorMath" else "Torch.Undefined.${ttype}.Tensor.Math.Random.TH") }

        , { rename = "Torch.Sig.Tensor.Random.TH", to =
            (if isth then "Torch.FFI.TH.${ttype}.TensorRandom" else "Torch.Undefined.${ttype}.Tensor.Random.TH") }

        , { rename = "Torch.Sig.Tensor.Random.THC", to =
            (if isth then "Torch.Undefined.${ttype}.Tensor.Random.THC" else "Torch.FFI.THC.${ttype}.TensorRandom") }
        ]
in
{ unsigned = unsigned
, signed   = signed
, floating = floating
}
