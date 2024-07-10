DBA
===

DBA stands for Dynamic Time Warping Barycenter Averaging. DBA is an averaging method that is consistent with Dynamic Time Warping. I give below an example of the difference between the traditional arithmetic mean of the set of time series and DBA. 

![arithmetic mean](https://raw.githubusercontent.com/fpetitjean/DBA/master/images/arithmetic.png)

![DBA](https://raw.githubusercontent.com/fpetitjean/DBA/master/images/DBA.png)


# Underlying research and scientific papers

This code is supporting 3 research papers:
* [Pattern Recognition 2011](http://francois-petitjean.com/Research/Petitjean2011-PR.pdf): A global averaging method for Dynamic Time Warping
* [ICDM 2014](http://francois-petitjean.com/Research/Petitjean2014-ICDM-DTW.pdf): Dynamic Time Warping Averaging of Time Series allows Faster and more Accurate Classification
* [ICDM 2017](http://francois-petitjean.com/Research/ForestierPetitjean2017-ICDM.pdf): Generating synthetic time series to augment sparse datasets

When using this repository, please cite:
```
@ARTICLE{Petitjean2011-DBA,
  title={A global averaging method for dynamic time warping, with applications to clustering},
  author={Petitjean, Fran{\c{c}}ois and Ketterlin, Alain and Gan{\c{c}}arski, Pierre},
  journal={Pattern Recognition},
  volume={44},
  number={3},
  pages={678--693},
  year={2011},
  publisher={Elsevier}
}

@INPROCEEDINGS{Petitjean2014-ICDM-2,
  title={Dynamic time warping averaging of time series allows faster and more accurate classification},
  author={Petitjean, Fran{\c{c}}ois and Forestier, Germain and Webb, Geoffrey I and Nicholson, Ann E and Chen, Yanping and Keogh, Eamonn},
  booktitle={Data Mining (ICDM), 2014 IEEE International Conference on},
  pages={470--479},
  year={2014},
  organization={IEEE}
}

@INPROCEEDINGS{Forestier2017-ICDM,
  title={Generating synthetic time series to augment sparse datasets},
  author={Forestier, Germain and Petitjean, Fran{\c{c}}ois and Dau, Hoang Anh and Webb, Geoffrey I and Keogh, Eamonn},
  booktitle={Data Mining (ICDM), 2017 IEEE International Conference on},
  pages={865--870},
  year={2017},
  organization={IEEE}
}
```

# Organisation of the repository

This repository gives you different versions of DBA for different programming language, whether you want to have a warping window or not, etc. Apologies for the inconsistencies between versions but I've basically created them as the need arose. 

Each file corresponds to one of these combinations; if one is missing for your usage, let me know. Currently the length is limited to 1,000 (let me know if you need more). 

* `DBA.java` standard DBA in Java with no warping window and memory allocated statically
* `DBAWarpingWindow.java` same as `DBA.java` but with a warping window as a parameter
* `DBA.m` Matlab implementation of DBA with no windows 
* `DBA.py` Fast Python implementation of DBA with no windows
* `DBA_multivariate.py` Fast Python implementation of DBA for multi-variate time series with no windows
* `cython/*` Cython implementation (thus usable in Python) of DBA with warping window (mono-variate)
