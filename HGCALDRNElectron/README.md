# Electron Energy Regression in HGCAL using DRN

First, clone the repository and create a new branch with your username. Then, create a conda environment using yaml file.
```
git clone git@github.com:bmjoshi/HGCALDRNElectron.git
cd HGCALDRNElectron/
git checkout -b <username>-working
conda env create -f conda_torch1.7.yml
```

Once you have created the environment, you can activate it every time you want to use the repository by running
```
conda activate ~/.conda/env/torch1.7
```

Exercise-01: Fitting Data with Gaussian Distribution
```
cd Exercise-01
```
