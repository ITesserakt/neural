{ pkgs ? import <nixpkgs> {} }: let
    toolchain = pkgs.rust.packages.stable;
in pkgs.mkShell rec {
  packages = with pkgs; [
    toolchain.cargo
    toolchain.rustc
    toolchain.rustfmt
    toolchain.clippy
    pkg-config

    blas
    openblas
    openssl

    (with python3Packages; [
      datasets
      numpy
      pillow
    ])
  ];

  RUSTFLAGS = "-lblas";

  RUST_SRC_PATH = toolchain.rustPlatform.rustLibSrc;
  RUST_TOOLCHAIN_PATH = pkgs.symlinkJoin {
      name = "neural-toolchain";
      paths = packages;
  };
}
