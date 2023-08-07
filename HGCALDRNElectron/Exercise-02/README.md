## Generate pickles from ntuples

All the ntuples are listed in ntuples.dat. 

nTuple2pkls.py can be used to extract pickles from ntuples (files with .root extension). You can extract pickled arrays from a single file using "file" as INPUT_TYPE or you could also extract them for multiple files using "folder" as INPUT_TYPE and providing path to folder containing root files as PATH_TO_INPUT. as Please refer to the usage below.

```
usage: nTuple2pkls.py [-h] [-i PATH_TO_INPUT] [-t INPUT_TYPE]
                      [-o OUTPUT_FOLDER] [-d DATA_TYPE] [--mc]

optional arguments:
  -h, --help            show this help message and exit
  -i PATH_TO_INPUT, --path_to_input PATH_TO_INPUT
  -t INPUT_TYPE, --input_type INPUT_TYPE
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
  -d DATA_TYPE, --data_type DATA_TYPE
  --mc
```

If the files are large and require additional resources, one can submit SLURM job. To submit jobs run
```
sbatch get_pickles_example.slurm
```
