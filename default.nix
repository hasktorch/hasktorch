let
  shared = import ./nix/shared.nix { };
in
  { inherit (shared)
      hasktorch-codegen
      libtorch-ffi_cpu
      hasktorch_cpu
      hasktorch-examples_cpu
      hasktorch-experimental_cpu
      hasktorch-docs
    ;

    ${shared.nullIfDarwin "libtorch-ffi_cudatoolkit_9_2"} = shared.libtorch-ffi_cudatoolkit_9_2;
    ${shared.nullIfDarwin "hasktorch_cudatoolkit_9_2"} = shared.hasktorch_cudatoolkit_9_2;
    ${shared.nullIfDarwin "hasktorch-examples_cudatoolkit_9_2"} = shared.hasktorch-examples_cudatoolkit_9_2;
    ${shared.nullIfDarwin "hasktorch-experimental_cudatoolkit_9_2"} = shared.hasktorch-experimental_cudatoolkit_9_2;

    ${shared.nullIfDarwin "libtorch-ffi_cudatoolkit_10_2"} = shared.libtorch-ffi_cudatoolkit_10_2;
    ${shared.nullIfDarwin "hasktorch_cudatoolkit_10_2"} = shared.hasktorch_cudatoolkit_10_2;
    ${shared.nullIfDarwin "hasktorch-examples_cudatoolkit_10_2"} = shared.hasktorch-examples_cudatoolkit_10_2;
    ${shared.nullIfDarwin "hasktorch-experimental_cudatoolkit_10_2"} = shared.hasktorch-experimental_cudatoolkit_10_2;
  }
