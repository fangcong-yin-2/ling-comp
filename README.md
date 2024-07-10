# ling-comp
Linguistic Compression in Single-Sentence Human-Written Summaries

## Word-level Analysis
We provide the script to replicate the word-level analysis in section 3.3 with dynamic time warping barycenter averaging (DBA). 

To run the script, first install Cython by 'pip install cython' and build DBA with `bash ./DBA/cython/build.sh`. Then, go to `./DBA/cython/run_analysis.py` and define your data path and the name of the feature to be analyzed. Finally, run `python ./DBA/cython/run_analysis.py`.
