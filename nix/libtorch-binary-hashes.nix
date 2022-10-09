version : builtins.getAttr version {
  "1.11.0" = {
    x86_64-darwin-cpu = {
      name = "libtorch-macos-1.11.0.zip";
      url = "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.11.0.zip";
      hash = "sha256-oTbvPjrREXPQanSjxzHbgJOtY5Yzb9FFgQsUG78o6eQ=";
    };
    x86_64-linux-cpu = {
      name = "libtorch-cxx11-abi-shared-with-deps-1.11.0-cpu.zip";
      url = "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcpu.zip";
      hash = "sha256-zC3lyeQrZJAkwingIbbjTkLXngsEaVv6VCl3QbAjOMQ=";
    };
    x86_64-linux-cuda-10 = {
      name = "libtorch-cxx11-abi-shared-with-deps-1.11.0-cu113.zip";
      url = "https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu102.zip";
      hash = "1qznvi3yjlpf0vxf9hnr59ajin17xfz85w3axrsc9xnmpkyhi6p8";
    };
    x86_64-linux-cuda-11 = {
      name = "libtorch-cxx11-abi-shared-with-deps-1.11.0-cu113.zip";
      url = "https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu113.zip";
      hash = "11zjggaw2qi4fh7v36ygb0pqlbphihq46yfvbgrv93a7x6f857ld";
    };
  };
}
