{
  description = "Python development environment with Jupyter and polars";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          jupyter
          polars
          matplotlib
          numpy
          seaborn
          ipykernel
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            pythonEnv
            nodejs  # for Jupyter extensions
          ];
          
          shellHook = ''
            # Add python symlink if it doesn't exist
            if ! command -v python >/dev/null 2>&1; then
              mkdir -p $PWD/.nix-bin
              ln -sf ${pythonEnv}/bin/python3 $PWD/.nix-bin/python
              export PATH="$PWD/.nix-bin:$PATH"
            fi
            
            # Setup messages and kernel registration
            echo "Python development environment loaded"
            
            # Auto-register Jupyter kernel if not already registered
            KERNEL_NAME="feels-floor-sim"
            if ! jupyter kernelspec list | grep -q "$KERNEL_NAME"; then
              echo "Registering Jupyter kernel: $KERNEL_NAME"
              python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "Feels Floor Sim (Nix)"
            else
              echo "Jupyter kernel '$KERNEL_NAME' already registered"
            fi
            
            # Update VSCode settings with current Python path
            if [ -f ".vscode/settings.json" ]; then
              # Create a temporary settings file with relative/dynamic paths
              cat > .vscode/settings.json.tmp << EOF
{
  "python.defaultInterpreterPath": "./.vscode/python",
  "python.terminal.activateEnvironment": false,
  "jupyter.kernels.filter": [
    {
      "path": "$HOME/Library/Jupyter/kernels/feels-floor-sim/kernel.json",
      "type": "jupyterKernel"
    }
  ],
  "jupyter.defaultKernel": "feels-floor-sim",
  "jupyter.interactiveWindow.creationMode": "perFile",
  "files.associations": {
    "*.ipynb": "jupyter-notebook"
  }
}
EOF
              mv .vscode/settings.json.tmp .vscode/settings.json
              
              # Create a symlink to the current Python interpreter
              ln -sf "$(which python3)" .vscode/python
              echo "Updated .vscode/settings.json with dynamic Python symlink"
            fi
          '';
        };
      });
}