{ runCommand, lib, ghc, pkgs}:
{ hsdocs
, prologue ? null # Optionally, a file to be used for the Haddock "--prologue" option.
}: 
runCommand "haddock-join" { buildInputs = [ hsdocs pkgs.jq ]; } ''
  # Merge all the docs from the packages. We don't use symlinkJoin because:
  # - We are going to want to redistribute this, so we don't want any symlinks.
  # - We want to be selective about what we copy (we don't need the hydra 
  #   tarballs from the other packages, for example.
  ${ghc}/bin/ghc --version
  mkdir -p "$out/share/doc"
  for pkg in ${lib.concatStringsSep " " hsdocs}; do
    cp -R $pkg/share/doc/* "$out/share/doc"
  done
  # We're going to sed all the files so they'd better be writable!
  chmod -R +w $out/share/doc

  # We're now going to rewrite all the pre-generated Haddock HTML output
  # so that links point to the appropriate place within our combined output,
  # rather than into the store.
  root=$out/share/doc
  for f in $(find $out -name "*.html"); do
    # Replace all links to the docs we're processing with relative links 
    # to the root of the doc directory we're creating - the rest of the link is
    # the same.
    # Also, it's not a a file:// link now because it's a relative URL instead
    # of an absolute one.
    relpath=$(realpath --relative-to=$(dirname $f) --no-symlinks $root)
    pkgsRegex="${"file://(" + (lib.concatStringsSep "|" hsdocs) + ")/share/doc"}"
    sed -i -r "s,$pkgsRegex,$relpath,g" "$f"
    # Now also replace the index/contents links so they point to (what will be) 
    # the combined ones instead.
    # Match the enclosing quotes to make sure the regex for index.html doesn't also match
    # the trailing part of doc-index.html
    sed -i -r "s,\"index\.html\",\"$relpath/share/doc/index.html\",g" "$f"
    sed -i -r "s,\"doc-index\.html\",\"$relpath/share/doc/doc-index.html\",g" "$f"
  done

  # Move to the docdir. We do this so that we can give relative docpaths to
  # Haddock so it will generate relative (relocatable) links in the index.
  cd $out/share/doc
  # Collect all the interface files and their docpaths (in this case
  # we can just use the enclosing directory).
  interfaceOpts=()
  for interfaceFile in $(find . -name "*.haddock"); do
    docdir=$(dirname $interfaceFile)
    interfaceOpts+=("--read-interface=$docdir,$interfaceFile")
  done

  # Generate the contents and index
  ${ghc}/bin/haddock \
    --gen-contents \
    --gen-index \
    ${lib.optionalString (prologue != null) "--prologue ${prologue}"} \
    "''${interfaceOpts[@]}"

  echo "[]" > "$root/doc-index.json"
  for file in $(ls $root/*/*/doc-index.json); do
    project=$(basename $(dirname $(dirname $file)))"/html"
    jq -s \
      ".[0] + [.[1][] | (. + {link: (\"$project/\" + .link)}) ]" \
      "$root/doc-index.json" \
      $file \
      > /tmp/doc-index.json
    mv /tmp/doc-index.json "$root/doc-index.json"
  done
''
