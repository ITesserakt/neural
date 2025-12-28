{
  pkgs ? import <nixpkgs> { },
}:
pkgs.mkShell {
  packages = with pkgs; [
    rustup

    pkg-config

    blas
    # openblas
    # openssl
    # gfortran

    (with python3Packages; [
      datasets
      numpy
      pillow
    ])
  ];

  RUSTFLAGS = "-lblas";
}
