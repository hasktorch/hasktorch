{
  description = "Easy example of using stacklock2nix to build a Haskell project";

  # This is a flake reference to the stacklock2nix repo.
  #
  # Note that if you copy the `./flake.lock` to your own repo, you'll likely
  # want to update the commit that this stacklock2nix reference points to:
  #
  # $ nix flake lock --update-input stacklock2nix
  #
  # You may also want to lock stacklock2nix to a specific release:
  #
  # inputs.stacklock2nix.url = "github:cdepillabout/stacklock2nix/v1.5.0";
  inputs.stacklock2nix.url = "github:cdepillabout/stacklock2nix/main";

  # This is a flake reference to Nixpkgs.
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    tokenizers = {
      url = "github:hasktorch/tokenizers/master";
    };
    naersk.follows = "tokenizers/naersk";
  };

  outputs = { self, nixpkgs, stacklock2nix, tokenizers, naersk }:
    let
      # System types to support.
      supportedSystems = [
        "x86_64-linux"
        "aarch64-darwin"
      ];

      # Helper function to generate an attrset '{ x86_64-linux = f "x86_64-linux"; ... }'.
      forAllSystems = f: nixpkgs.lib.genAttrs supportedSystems (system: f system);

      # Nixpkgs instantiated for supported system types.
      nixpkgsFor =
        forAllSystems (system: import nixpkgs { inherit system; overlays = [ stacklock2nix.overlay naersk.overlay tokenizers.overlays.default self.overlay ]; });

#      deviceConfig = {cudaSupport=true;device="cuda-11";};
      deviceConfig = {cudaSupport=false;device="cpu";};
    in
    {
      # A Nixpkgs overlay.
      overlay = final: prev: {
        # This is a top-level attribute that contains the result from calling
        # stacklock2nix.
        torch = prev.callPackage ./nix/libtorch.nix deviceConfig;
        c10 = prev.callPackage ./nix/libtorch.nix deviceConfig;
        torch_cpu = prev.callPackage ./nix/libtorch.nix deviceConfig;
        torch_cuda = if deviceConfig.cudaSupport then prev.callPackage ./nix/libtorch.nix deviceConfig else null;
        
        hasktorch-stacklock = final.stacklock2nix {
          stackYaml = ./stack.yaml;

          # The Haskell package set to use as a base.  You should change this
          # based on the compiler version from the resolver in your stack.yaml.
          baseHaskellPkgSet = final.haskell.packages.ghc928;

          # Any additional Haskell package overrides you may want to add.
          additionalHaskellPkgSetOverrides = hfinal: hprev: {
            # The servant-cassava.cabal file is malformed on GitHub:
            # https://github.com/haskell-servant/servant-cassava/pull/29
            libtorch-ffi =
              final.haskell.lib.compose.overrideCabal
               {
                 configureFlags =
                   [] ++ (if deviceConfig.cudaSupport then [" -f cuda"] else []) ++ (if prev.stdenv.hostPlatform.isDarwin then [" -f gcc"] else []);
               }
               hprev.libtorch-ffi;
            tar =
              final.haskell.lib.compose.overrideCabal
               { doCheck = false;
               }
               hprev.tar;
            half =
              final.haskell.lib.compose.overrideCabal
               { doCheck = false;
               }
               hprev.half;
            sysinfo =
              final.haskell.lib.compose.overrideCabal
               { doCheck = false;
               }
               hprev.sysinfo;
            inline-c-cpp =
              final.haskell.lib.compose.overrideCabal
                { preConfigure = ''
                    sed -i -e 's/ghc-options:/ghc-options: -fcompact-unwind /g' *.cabal
                    cat *.cabal
                 '';
                  doCheck = true;
                  configureFlags = [
                    "--extra-lib-dirs=${prev.pkgs.libcxx}/lib"
                  ];
                }
                hprev.inline-c-cpp;
            happy =
              final.haskell.lib.compose.overrideCabal
               { doCheck = false;
               }
               hprev.happy;
            streaming-cassava =
              final.haskell.lib.compose.overrideCabal
               { preConfigure = ''
                   sed -i 's/2.8/2.10.8/' *.cabal
                 '';
               }
               hprev.streaming-cassava;
            term-rewriting =
              final.haskell.lib.compose.overrideCabal
               { doCheck = false;
               }
               hprev.term-rewriting;
            type-errors-pretty =
              final.haskell.lib.compose.overrideCabal
               { preConfigure = ''
                   sed -i 's/doctest >= 0.16 && < 0.19/doctest >=0.16/' *.cabal
                 '';
                 doCheck = false;
               }
               hprev.type-errors-pretty;
            hashable =
              final.haskell.lib.compose.overrideCabal
               { preConfigure = ''
                   ls
                   cat *.cabal
                 '';
                 doCheck = false;
               }
               hprev.hashable;
            haskell-language-server =
              final.haskell.lib.compose.overrideCabal
                { doCheck = false;
                } hprev.haskell-language-server;
            lsp =
              final.haskell.lib.compose.overrideCabal
                { doCheck = false;
                } hprev.lsp;
            http2 =
              final.haskell.lib.compose.overrideCabal
                { doCheck = false;
                } hprev.http2;
            torch = null;
            tokenizers =
              final.haskell.lib.compose.overrideCabal
                { doCheck = false;
                } hprev.tokenizers;

          };

          cabal2nixArgsOverrides = args: args // {
            "splitmix" = ver: {};
            "tokenizers" = ver: { "tokenizers_haskell" = prev.pkgs.tokenizersPackages.tokenizers-haskell; };
#            "inline-c-cpp" = ver: { "c++" = prev.pkgs.libcxx; };
          };          

          # Additional packages that should be available for development.
          additionalDevShellNativeBuildInputs = stacklockHaskellPkgSet: [
            # Some Haskell tools (like cabal-install and ghcid) can be taken from the
            # top-level of Nixpkgs.
            final.cabal-install
            final.ghcid
            final.stack
            # Some Haskell tools need to have been compiled with the same compiler
            # you used to define your stacklock2nix Haskell package set.  Be
            # careful not to pull these packages from your stacklock2nix Haskell
            # package set, since transitive dependency versions may have been
            # carefully setup in Nixpkgs so that the tool will compile, and your
            # stacklock2nix Haskell package set will likely contain different
            # versions.
            final.haskell-language-server
            # Other Haskell tools may need to be taken from the stacklock2nix
            # Haskell package set, and compiled with the example same dependency
            # versions your project depends on.
            #stacklockHaskellPkgSet.some-haskell-lib
          ];

          # When creating your own Haskell package set from the stacklock2nix
          # output, you may need to specify a newer all-cabal-hashes.
          #
          # This is necessary when you are using a Stackage snapshot/resolver or
          # `extraDeps` in your `stack.yaml` file that is _newer_ than the
          # `all-cabal-hashes` derivation from the Nixpkgs you are using.
          #
          # If you are using the latest nixpkgs-unstable and an old Stackage
          # resolver, then it is usually not necessary to override
          # `all-cabal-hashes`.
          #
          # If you are using a very recent Stackage resolver and an old Nixpkgs,
          # it is almost always necessary to override `all-cabal-hashes`.
          all-cabal-hashes = final.fetchurl {
            name = "all-cabal-hashes";
            url = "https://github.com/commercialhaskell/all-cabal-hashes/archive/e4e35502b729d90aee1ba8d43c5d76977afd4622.tar.gz";
            sha256 = "sha256-nryIxlik0dyBWPAgCGKvsjF0IaSYy3zDzQ56849L5TU=";
          };
        };

        # One of our local packages.
        hasktorch-examples = final.hasktorch-stacklock.pkgSet.examples;

        # You can also easily create a development shell for hacking on your local
        # packages with `cabal`.
        hasktorch-dev-shell = final.hasktorch-stacklock.devShell;
      };

      lib = forAllSystems (system: {
        hasktorch = nixpkgsFor.${system}.hasktorch-stacklock;
      });

      packages = forAllSystems (system: {
        hasktorch = nixpkgsFor.${system}.hasktorch-stacklock.pkgSet.hasktorch;
        codegen = nixpkgsFor.${system}.hasktorch-stacklock.pkgSet.codegen;
        examples = nixpkgsFor.${system}.hasktorch-stacklock.pkgSet.examples;
        hasktorch-gradually-typed = nixpkgsFor.${system}.hasktorch-stacklock.pkgSet.hasktorch-gradually-typed;
      });

      defaultPackage = forAllSystems (system: self.packages.${system}.hasktorch);

      devShells = forAllSystems (system: {
        hasktorch-dev-shell = nixpkgsFor.${system}.hasktorch-dev-shell;
      });

      devShell = forAllSystems (system: self.devShells.${system}.hasktorch-dev-shell);
    };
}
