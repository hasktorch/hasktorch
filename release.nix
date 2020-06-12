let
  hsPkgs = import ./default.nix { };
in
  {
  	libtorch = hsPkgs.libtorch-ffi.components.all;

  	hasktorch = hsPkgs.hasktorch.components.all;

  	all-examples = hsPkgs.examples.components.all;

  	alexNet = hsPkgs.examples.components.exes.alexNet;
	autograd = hsPkgs.examples.components.exes.autograd;
	distill = hsPkgs.examples.components.exes.distill;
	gaussian-process = hsPkgs.examples.components.exes.gaussian-process;
	gd-field = hsPkgs.examples.components.exes.gd-field;
	image-processing = hsPkgs.examples.components.exes.image-processing;
	load-torchscript = hsPkgs.examples.components.exes.load-torchscript;
	matrix-factorization = hsPkgs.examples.components.exes.matrix-factorization;
	minimal-text-example = hsPkgs.examples.components.exes.minimal-text-example;
	mnist-mlp = hsPkgs.examples.components.exes.mnist-mlp;
	optimizers = hsPkgs.examples.components.exes.optimizers;
	regression = hsPkgs.examples.components.exes.regression;
	rnn = hsPkgs.examples.components.exes.rnn;
	serialization = hsPkgs.examples.components.exes.serialization;
	static-mnist = hsPkgs.examples.components.sublibs.static-mnist;
	static-mnist-cnn = hsPkgs.examples.components.exes.static-mnist-cnn;
	static-mnist-mlp = hsPkgs.examples.components.exes.static-mnist-mlp;
	static-xor-mlp = hsPkgs.examples.components.exes.static-xor-mlp;
	typed-transformer = hsPkgs.examples.components.exes.typed-transformer;
	vae = hsPkgs.examples.components.exes.vae;
	xor-mlp = hsPkgs.examples.components.exes.xor-mlp;

	all-experimental = hsPkgs.experimental.components.all;

  	dataloader-cifar10 = hsPkgs.experimental.components.exes.dataloader-cifar10;
  	untyped-nlp = hsPkgs.experimental.components.exes.untyped-nlp;
  }