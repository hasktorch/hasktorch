   let prelude = https://raw.githubusercontent.com/dhall-lang/dhall-to-cabal/master/dhall/prelude.dhall
in let types = https://raw.githubusercontent.com/dhall-lang/dhall-to-cabal/master/dhall/types.dhall
in let fn = ../common/functions.dhall
in let unsigned =
    λ(isth : Bool) →
    λ(ttype : Text) →
      let namespace = if isth then "${ttype}" else "Cuda.${ttype}"
    in
      [ { rename = "Torch.Indef.Storage"                              , to = "Torch.Indef.${namespace}.Storage" }
      , { rename = "Torch.Indef.Storage.Copy"                         , to = "Torch.Indef.${namespace}.Storage.Copy" }
      , { rename = "Torch.Indef.Static.Tensor"                        , to = "Torch.Indef.${namespace}.Tensor" }
      , { rename = "Torch.Indef.Static.Tensor.Copy"                   , to = "Torch.Indef.${namespace}.Tensor.Copy" }
      , { rename = "Torch.Indef.Static.Tensor.Index"                  , to = "Torch.Indef.${namespace}.Tensor.Index" }
      , { rename = "Torch.Indef.Static.Tensor.Masked"                 , to = "Torch.Indef.${namespace}.Tensor.Masked" }
      , { rename = "Torch.Indef.Static.Tensor.Math"                   , to = "Torch.Indef.${namespace}.Tensor.Math" }
      , { rename = "Torch.Indef.Static.Tensor.Math.Compare"           , to = "Torch.Indef.${namespace}.Tensor.Math.Compare" }
      , { rename = "Torch.Indef.Static.Tensor.Math.CompareT"          , to = "Torch.Indef.${namespace}.Tensor.Math.CompareT" }
      , { rename = "Torch.Indef.Static.Tensor.Math.Pairwise"          , to = "Torch.Indef.${namespace}.Tensor.Math.Pairwise" }
      , { rename = "Torch.Indef.Static.Tensor.Math.Pointwise"         , to = "Torch.Indef.${namespace}.Tensor.Math.Pointwise" }
      , { rename = "Torch.Indef.Static.Tensor.Math.Reduce"            , to = "Torch.Indef.${namespace}.Tensor.Math.Reduce" }
      , { rename = "Torch.Indef.Static.Tensor.Math.Scan"              , to = "Torch.Indef.${namespace}.Tensor.Math.Scan" }
      , { rename = "Torch.Indef.Static.Tensor.Mode"                   , to = "Torch.Indef.${namespace}.Tensor.Mode" }
      , { rename = "Torch.Indef.Static.Tensor.ScatterGather"          , to = "Torch.Indef.${namespace}.Tensor.ScatterGather" }
      , { rename = "Torch.Indef.Static.Tensor.Sort"                   , to = "Torch.Indef.${namespace}.Tensor.Sort" }
      , { rename = "Torch.Indef.Static.Tensor.TopK"                   , to = "Torch.Indef.${namespace}.Tensor.TopK" }
      , { rename = "Torch.Indef.Dynamic.Tensor"                       , to = "Torch.Indef.${namespace}.Dynamic.Tensor" }
      , { rename = "Torch.Indef.Dynamic.Tensor.Copy"                  , to = "Torch.Indef.${namespace}.Dynamic.Tensor.Copy" }
      , { rename = "Torch.Indef.Dynamic.Tensor.Index"                 , to = "Torch.Indef.${namespace}.Dynamic.Tensor.Index" }
      , { rename = "Torch.Indef.Dynamic.Tensor.Masked"                , to = "Torch.Indef.${namespace}.Dynamic.Tensor.Masked" }
      , { rename = "Torch.Indef.Dynamic.Tensor.Math"                  , to = "Torch.Indef.${namespace}.Dynamic.Tensor.Math" }
      , { rename = "Torch.Indef.Dynamic.Tensor.Math.Compare"          , to = "Torch.Indef.${namespace}.Dynamic.Tensor.Math.Compare" }
      , { rename = "Torch.Indef.Dynamic.Tensor.Math.CompareT"         , to = "Torch.Indef.${namespace}.Dynamic.Tensor.Math.CompareT" }
      , { rename = "Torch.Indef.Dynamic.Tensor.Math.Pairwise"         , to = "Torch.Indef.${namespace}.Dynamic.Tensor.Math.Pairwise" }
      , { rename = "Torch.Indef.Dynamic.Tensor.Math.Pointwise"        , to = "Torch.Indef.${namespace}.Dynamic.Tensor.Math.Pointwise" }
      , { rename = "Torch.Indef.Dynamic.Tensor.Math.Reduce"           , to = "Torch.Indef.${namespace}.Dynamic.Tensor.Math.Reduce" }
      , { rename = "Torch.Indef.Dynamic.Tensor.Math.Scan"             , to = "Torch.Indef.${namespace}.Dynamic.Tensor.Math.Scan" }
      , { rename = "Torch.Indef.Dynamic.Tensor.Mode"                  , to = "Torch.Indef.${namespace}.Dynamic.Tensor.Mode" }
      , { rename = "Torch.Indef.Dynamic.Tensor.ScatterGather"         , to = "Torch.Indef.${namespace}.Dynamic.Tensor.ScatterGather" }
      , { rename = "Torch.Indef.Dynamic.Tensor.Sort"                  , to = "Torch.Indef.${namespace}.Dynamic.Tensor.Sort" }
      , { rename = "Torch.Indef.Dynamic.Tensor.TopK"                  , to = "Torch.Indef.${namespace}.Dynamic.Tensor.TopK" }

      , { rename = "Torch.Indef.Types"                                , to = "Torch.${namespace}.Types" }
      , { rename = "Torch.Indef.Index"                                , to = "Torch.${namespace}.Index" }
      , { rename = "Torch.Indef.Mask"                                 , to = "Torch.${namespace}.Mask" }
      ]
in let signed =
    λ(isth : Bool) →
    λ(ttype : Text) →
      let namespace = if isth then "${ttype}" else "Cuda.${ttype}"
    in (unsigned isth namespace) #
      [ { rename = "Torch.Indef.Static.Tensor.Math.Pointwise.Signed"  , to = "Torch.Indef.${namespace}.Tensor.Math.Pointwise.Signed" }
      , { rename = "Torch.Indef.Dynamic.Tensor.Math.Pointwise.Signed" , to = "Torch.Indef.${namespace}.Dynamic.Tensor.Math.Pointwise.Signed" }
      ]
in let floating =
    λ(isth : Bool) →
    λ(ttype : Text) →
      let namespace = if isth then "${ttype}" else "Cuda.${ttype}"
    in (signed isth namespace) #
      [ { rename = "Torch.Indef.Dynamic.Tensor.Math.Blas"               , to = "Torch.Indef.${namespace}.Dynamic.Tensor.Math.Blas" }
      , { rename = "Torch.Indef.Dynamic.Tensor.Math.Lapack"             , to = "Torch.Indef.${namespace}.Dynamic.Tensor.Math.Lapack" }

      , { rename = "Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating" , to = "Torch.Indef.${namespace}.Dynamic.Tensor.Math.Pointwise.Floating" }
      , { rename = "Torch.Indef.Dynamic.Tensor.Math.Reduce.Floating"    , to = "Torch.Indef.${namespace}.Dynamic.Tensor.Math.Reduce.Floating" }
      , { rename = "Torch.Indef.Dynamic.Tensor.Math.Floating"           , to = "Torch.Indef.${namespace}.Dynamic.Tensor.Math.Floating" }

      , { rename = "Torch.Indef.Static.Tensor.Math.Blas"                , to = "Torch.Indef.${namespace}.Tensor.Math.Blas" }
      , { rename = "Torch.Indef.Static.Tensor.Math.Lapack"              , to = "Torch.Indef.${namespace}.Tensor.Math.Lapack" }

      , { rename = "Torch.Indef.Static.Tensor.Math.Pointwise.Floating"  , to = "Torch.Indef.${namespace}.Tensor.Math.Pointwise.Floating" }
      , { rename = "Torch.Indef.Static.Tensor.Math.Reduce.Floating"     , to = "Torch.Indef.${namespace}.Tensor.Math.Reduce.Floating" }
      , { rename = "Torch.Indef.Static.Tensor.Math.Floating"            , to = "Torch.Indef.${namespace}.Tensor.Math.Floating" }

      -- NN modules
      , { rename = "Torch.Indef.Dynamic.NN"                             , to = "Torch.Indef.${namespace}.Dynamic.NN" }

      , { rename = "Torch.Indef.Static.NN"                              , to = "Torch.Indef.${namespace}.NN" }
      , { rename = "Torch.Indef.Static.NN"                              , to = "Torch.${namespace}.NN" }
      , { rename = "Torch.Indef.Static.NN.Activation"                   , to = "Torch.${namespace}.NN.Activation" }
      , { rename = "Torch.Indef.Static.NN.Backprop"                     , to = "Torch.${namespace}.NN.Backprop" }
      , { rename = "Torch.Indef.Static.NN.Conv1d"                       , to = "Torch.${namespace}.NN.Conv1d" }
      , { rename = "Torch.Indef.Static.NN.Conv2d"                       , to = "Torch.${namespace}.NN.Conv2d" }
      , { rename = "Torch.Indef.Static.NN.Criterion"                    , to = "Torch.${namespace}.NN.Criterion" }
      , { rename = "Torch.Indef.Static.NN.Layers"                       , to = "Torch.${namespace}.NN.Layers" }
      , { rename = "Torch.Indef.Static.NN.Linear"                       , to = "Torch.${namespace}.NN.Linear" }
      , { rename = "Torch.Indef.Static.NN.Math"                         , to = "Torch.${namespace}.NN.Math" }
      , { rename = "Torch.Indef.Static.NN.Padding"                      , to = "Torch.${namespace}.NN.Padding" }
      , { rename = "Torch.Indef.Static.NN.Pooling"                      , to = "Torch.${namespace}.NN.Pooling" }
      , { rename = "Torch.Indef.Static.NN.Sampling"                     , to = "Torch.${namespace}.NN.Sampling" }
      ]
      -- Random modules
      # if isth
        then
          [ { rename = "Torch.Indef.Static.Tensor.Random.TH"            , to = "Torch.Indef.${namespace}.Tensor.Random.TH" }
          , { rename = "Torch.Indef.Static.Tensor.Math.Random.TH"       , to = "Torch.Indef.${namespace}.Tensor.Math.Random.TH" }
          , { rename = "Torch.Indef.Dynamic.Tensor.Random.TH"           , to = "Torch.Indef.${namespace}.Dynamic.Tensor.Random.TH" }
          , { rename = "Torch.Indef.Dynamic.Tensor.Math.Random.TH"      , to = "Torch.Indef.${namespace}.Dynamic.Tensor.Math.Random.TH" }
          , { rename = "Torch.Undefined.Tensor.Random.THC"              , to = "Torch.Undefined.${namespace}.Tensor.Random.THC" }
          ]
        else
          [ { rename = "Torch.Undefined.Tensor.Random.TH"               , to = "Torch.Undefined.${namespace}.Tensor.Random.TH" }
          , { rename = "Torch.Undefined.Tensor.Math.Random.TH"          , to = "Torch.Undefined.${namespace}.Tensor.Math.Random.TH" }
          , { rename = "Torch.Indef.Static.Tensor.Random.THC"           , to = "Torch.Indef.${namespace}.Tensor.Random.THC" }
          , { rename = "Torch.Indef.Dynamic.Tensor.Random.THC"          , to = "Torch.Indef.${namespace}.Dynamic.Tensor.Random.THC" }
          ]
