Detection of magnetopause and bow shock crossings on Mercury based on MESSENGER magnetometer data
==============================

The project is intended for detecting concept drift in MESSENGER magnetomoter data, sampling the data according to the detected drifts, and training a CRNN model to predict the crossings. Here, project's file structure and instructions for using it are provided.

Setup
------------

Before starting drift detection, orbits with initially known drifts need to be prepared. Here are several groups of adjacent orbits, each group can be assigned its own drift label:

- 232-247
- 380-399
- 553-570
- 607-627
- 750-764
- 1383-1396
- 1460-1474
- 1489-1505
- 1507-1526
- 1560-1577

All files corresponding to these orbits need to be placed in `data/drifts/`. Program was tested with 8 groups of orbits with known drifts, consisting of the following 100 orbits:

```
1383  1464  1497  381  558  626
1384  1465  1498  382  559  627
1385  1466  1499  384  560  751
1386  1467  233   387  561  752
1387  1468  234   388  562  753
1388  1469  235   389  564  754
1389  1470  236   390  565  755
1390  1471  237   391  566  756
1391  1474  238   394  567  757
1392  1489  239   396  569  758
1393  1490  240   397  570  759
1394  1491  241   398  607  760
1395  1492  242   399  612  761
1396  1493  244   553  613  762
1460  1494  245   554  619  763
1461  1495  247   556  621
1462  1496  380   557  625
```

All other orbits for drift detection and crossing prediction need to be put in `data/orbits/`. Orbits with known drifts from `data/drifts/` are included automatically, so their files do not need to be added to `data/orbits/` again. Program was tested on datasets of different sizes between 100 to 3000 orbits.

Drift detection
------------
Before crossing prediction, drift detection has to be performed first with the following command :

`python gan.py logs/gan 1 cuda`

There are 3 command line arguments:
1. Directory for logs and output - `logs/gan`
2. Dataset number - `1` by default
3. Device name - `cuda` or `cpu`

Dataset number is used later when performing drift detection on several different dataset samples in a row. By default its value should be `1`, as it corresponds to full dataset, which means that all orbits from `data/orbits/` are used.

Another parameter is the features that are selected to be trained on. These features are written down in `data/features_gan.txt` file, separated by newline, and they can be changed if needed. Full list of available features: `X_MSO, Y_MSO, Z_MSO, BX_MSO, BY_MSO, BZ_MSO, DBX_MSO, DBY_MSO, DBZ_MSO, RHO_DIPOLE, PHI_DIPOLE, THETA_DIPOLE, BABS_DIPOLE, BX_DIPOLE, BY_DIPOLE, BZ_DIPOLE, RHO, RXY, X, Y, Z, VX, VY, VZ, VABS, D, COSALPHA, EXTREMA`

Script `./run-gan.sh` is used for performing drift detection on multiple samples of the dataset. It also adds a timestamp to log directory. Script is executed with the following command:

`./run-gan.sh cuda`

The datasets themselves are defined on line 939 of `gan.py` and can be edited if, for example, the sampling needs to be changed or there are less orbits available in total.

The output of drift detection is stored in 2 text files: `log_set1.txt` and `drifts_set1.txt`. The number at the end corresponds to the dataset sample, so there will be more output files if drift detection is performed with `./run-gan.sh`. File `log_set1.txt` is updated as drift detection is happening, and the output contains drift labels that are assigned to sets of orbits with probabilities, an example of which looks like this:

```
113/2312 orbits 2 - 20 (13) -- drift 4, prob 0.9999923706054688
123/2312 orbits 21 - 33 (10) -- drift 7, prob 0.9974427223205566
127/2312 orbits 35 - 39 (4) -- drift 3, prob 0.9999980926513672
```

File `drifts_set1.txt` contains orbit numbers and drift labels assigned to them, in the format of Python dictionary. This is intended for passing the results of drift detection to crossing prediction. A sample of these results looks like this:

```
30 7
31 7
33 7
35 3
37 3
38 3
```

Crossing prediction
------------
After drift detection is finished, its results need to be manually moved to the `data` directory, so `drifts_set1.txt` (`set2` and so on) file needs to be moved from `logs/gan` to `data`. This is done to decouple drift detection from crossing prediction, so that these programs can be executed and tested separately.

Once the files with orbit numbers and drift labels are ready, crossing prediction can be performed with the following command:

`python cnn.py logs/cnn 1 23 5`

There are 4 command line arguments:
1. Directory for logs and output - `logs/cnn`
2. Dataset number - `1` by default
3. Plot numbers - `23`, which correspond to true and predicted labels of testing set
4. Value of `max_orbits` parameter - `5`

Dataset number serves the same purpose as in drift detection. Plot numbers are a string with digits that are used to indicate what types of plots need to be drawn. `0` corresponds to true labels of training orbits, `1` - predicted labels of training orbits, `2` - true labels of testing orbits, `3` - predicted labels of testing orbits. By default it's `23` because prediction results of testing orbits are usually of most interest. Last argument is `max_orbits`, which defines how many training orbits per drift label are going to be sampled. Value of `5` means that only 5 orbits with highest entropy from each drift are selected and used for training the classifier. A total amount of training orbits can be estimated as value of `max_orbits` multiplied by a number of detected drifts.

As in drift detection, features for training in crossing prediction are selected in `data/features_cnn.txt` file. Full list of available features remains the same.

Script `./run-cnn.sh` also has the same purpose for crossing prediction as `./run-gan.sh` for drift detection. Script is executed with the following command:

`./run-cnn.sh 23 5`

Values `23` and `5` correspond to plot numbers and `max_orbits`.

The output of crossing prediction consists of text log and plots. Log shows several things:
1. What orbits are selected for training and testing from each drift (`drift 1 training orbits`, `drift 1 testing orbits`, ...)
2. Metric summary for testing data in `= TESTING =` section
3. Metric values for each orbit in `= EVALUATION =` section

Plots are distributed in subdirectories like `test_true`, `test_pred`, `test_all` (merged plots) and are stored as PNG files like `fig240_drift1.png`.

After crossing prediction is finished, its performance values and plots can be used for further analysis.

Project Organization
------------

    ├── README.md                  <- Top-level README for using this project
    ├── data
    │   ├── drifts                 <- Orbit files with initially known drifts
    │   ├── orbits                 <- Orbit files for drift detection
    |   ├── drifts_set1.txt        <- Detected drifts for use during crossing prediction
    |   ├── features_cnn.txt       <- Selected features for crossing prediction
    |   └── features_gan.txt       <- Selected features for drift detection
    │
    ├── logs                       <- Generated logs
    |   ├── cnn                    <- Logs for crossing prediction
	|   |   ├── plots              <- Plotted orbits with true and predicted crossings
	|   |   └── log_cnn_set1.txt   <- Text log
	|   |
    |   └── gan                    <- Logs for drift detection
	|       ├── drifts_set1.txt    <- Detected drift for each orbit
	|       └── log.txt            <- Text log
    │
    ├── cnn.py                     <- Main script for crossing prediction
    ├── gan.py                     <- Main script for drift detection
    ├── util.py                    <- Helper functions loading data
    ├── run-cnn.sh                 <- Script for performing crossing prediction with different arguments
    └── run-gan.sh                 <- Script for performing drift detection with different arguments
    
