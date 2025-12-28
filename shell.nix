{
  pkgs ? import <nixpkgs> { },
}:
pkgs.mkShell {
  packages = with pkgs; [
    rustup

    pkg-config
    llvmPackages.bintools

    # blas
    # openblas
    # openssl
    # gfortran

    (with python3Packages; [
      datasets
      numpy
      pillow
    ])
  ];

  # RUSTFLAGS = "-lblas";
}
