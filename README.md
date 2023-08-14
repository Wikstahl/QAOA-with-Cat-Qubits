# QAOA with Cat Qubits

## Installation <a name="installation"></a>
  Clone the files
  ```
  git clone git@github.com:Wikstahl/QAOA-with-Cat-Qubits
  cd QAOA-with-Cat-Qubits
  ```
  Recommended: Create a new virtual environment using **conda** or **venv**
  ```
  conda create -n qaoa_with_cats python=3.11
  ```
  activate it using
  ```
  conda activate qaoa_with_cats
  ```
  install the required packages
  ```
  pip install -r requirements.txt
  ```
  install the module
  ```
  python setup.py
  ```

## Examples <a name="examples"></a>
Examples of obtaining the average gate fidelities applied to qaoa states and thermal states are given in the folder `examples`

## Usage <a name="usage"></a>
The `src` folder contains the scripts used for producing the results.
    - The folder `avg_fid` contain the source files for computing the average gate fidelities for RX, RZ, RY, and RZZ-gates for the cat qubit.
    - The folder `kraus` contain the source files for constructing the kraus operators 
