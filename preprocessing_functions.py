import numpy as np

### Remove NaN-heavy features (and datapoints if required)

def nan_feature_filter(X, cutoff_ratio=0.8):
    """
    Removes features (columns) from X that have a proportion of NaN values greater than or equal to cutoff_ratio.
    Args:
        X (np.ndarray): Input dataset of shape (n_samples, n_features).
        cutoff_ratio (float): Threshold ratio for removing features (default 0.8).
    Returns:
        np.ndarray: Dataset with features removed.
        np.ndarray: Boolean mask of kept features (True = kept, False = removed).
    """
    nan_proportions = np.mean(np.isnan(X), axis=0)
    keep_mask = nan_proportions < cutoff_ratio
    X_new = X[:, keep_mask]
    return X_new, keep_mask

def nan_datapoint_filter(X, cutoff_ratio=0.8):
    """
    Removes datapoints (rows) from X that have a proportion of NaN values greater than or equal to cutoff_ratio.
    Args:
        X (np.ndarray): Input dataset of shape (n_samples, n_features).
        cutoff_ratio (float): Threshold ratio for removing datapoints (default 0.8).
    Returns:
        np.ndarray: Dataset with datapoints removed.
        np.ndarray: Boolean mask of kept datapoints (True = kept, False = removed).
    """
    nan_proportions = np.mean(np.isnan(X), axis=1)
    keep_mask = nan_proportions < cutoff_ratio
    X_new = X[keep_mask, :]
    return X_new, keep_mask



### Separate data into continuous and categorical columns

def is_categorical(M, feature_mask):
    """
    Function
    ---------
    Separates the features into categorical or continuous/ordinal categorical
    For now, done by hand due to the nature of the provided data

    Inputs
    ---------
    M : number of features in the dataset
    feature_mask : (1 x P) mask of the P filtered features from the dataset
    
    Outputs
    ---------
    filtered_continuous_mask   :     (1 x P) 
    filtered_categorical_mask    :     (1 x P) 
    """

    # CSV has the following shape : [       A                B             C             D            ...   letter n° M+1 ]
    #                               [       Id            _State         FMONTH          IDATE        ...   Feature n° M  ]
    #                               [ Not a feature      Feature n°1   Feature n°2      Feature n°3   ...   Feature n° M  ]
    #                               [      idx[0]         idx[1]        idx[2]          idx[3]]       ...       idx[M]    ]


    continuous_idx = [27, 28, 29, 30, 34, 50, 53, 61, 63, 64, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 
                      90, 91, 93, 94, 95, 9, 99, 100, 111, 112, 113, 114, 115, 116, 121, 122, 128, 129, 130, 132, 
                      138, 139, 140, 141, 144, 146, 148, 149, 150, 151, 152, 153, 154, 155, 163, 172, 174, 176, 179, 
                      181, 184, 189, 193, 194, 196, 198, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 247, 253, 
                      257, 258, 259, 260, 267, 268, 269, 270, 271, 272, 277, 278, 288, 289, 292, 293, 294, 295, 296, 
                      297, 298, 300, 301, 302, 303, 304, 305, 306]   

    #   IDX     Column  Name        Signification                                     
    #   27      AB      GENHLTH     General health?                     Encode 7 and 9 as NANs,     !! 1 means excellent, 5 means poor !! 
    #   28      AC      PHYSHLTH    Physical health not good?           Encode 77 and 99 as NANs, encode 88 as zeros
    #   29      AD      MENTHLTH    Mental health                       Encode 77 and 99 as NANs, encode 88 as zeros
    #   30      AE      POORHLTH                                        Encode 77 and 99 as NANs, encode 88 as zeros
    #   34      AI      CHECKUP1    Frequence of doctors                Encode 7 and 9 as NANs, encode 8 as zeros,   !! 1 means excellent, 4 means poor !!
    #   38      AM      CHOLCHK     Frequence of cholesterol check
    #   50      AY      DIABAGE2    Age when learned had diabetes       Encoding is related to categorical feature n°49 
    #   53      BB      EDUCA       Level of education                  Ordinal categorical
    #   61      BJ      INCOME2     Annual income
    #   63      BL      WEIGHT2
    #   64      BM      HEIGHT3
    #   74      BW      SMOKEDAY2   Frequency of smoking - cigarettes
    #   76      BY      LASTSMK2    Time spent since the last cigarette
    #   77      BZ      USENOW      Frequency of smoking - general
    #   78      CA      ALCDAY5     Frequency of alcohol
    #   79      CB      AVEDRNK2    Alcohol comsumption
    #   80      CC      DRNK3GE5    Alcohol consumption
    #   81      CD      MAXDRNKS    Alcohol consumption
    #   82      CE      FRUITJU1    Food habits
    #   83      CF      FRUIT1      Food habits
    #   84-->87                     Food habits
    #   89      CM      EXEROFT1    Physical activity
    #   91, 93--> 95                Physical activity
    #   98, 99  CU, CV              Arthrisis / joint pain   
    #   100             SEATBELT    Why ask seatbelt in this study? except if means too poor physical condition --> cardivascular disease?
    #   111-->116  DH   BLDSUGAR... Diabetes
    #   121     DR      CRGVLNG1    Caregiving
    #   122     DS      CRGVHRS1    Caregiving
    #   128 --> 130,132             Visual impairment
    #   138-->141                   Cognitive decline
    #   144             LONGWTCH    Salt intake
    #   146             ASTHMAGE    Asthma history 
    #   148             ASERVIST    Asthma   
    #   149             ASDRVIST    Asthma
    #   150             ASRCHKUP    Asthma
    #   151             ASACTLIM    Asthma
    #   152             ASYMPTOM    Asthma
    #   153             ASNOSLEP    Asthma
    #   154             ASTHMED3    Asthma
    #   155             ASINHALR    Asthma
    #   163             ARTTODAY    Arthrisis
    #   172             HOWLONG     Breast and cervical cancer
    #   174             LASTAP2     Breast and cervical cancer
    #   176             HLPSTTST    Breast and cervical cancer
    #   179             LENGEXAM    Breast and cervical cancer
    #   181             LSTBLDS3    Colorectal cancer
    #   184             LASTSIG3    Colorectal cancer
    #   189             PSATIME     Prostate cancer
    #   193             SCNTMNY1    Social context
    #   194             SCNTMEL1    Social context
    #   196             SCNTWRK1    Social context
    #   198             SCNTLWK1    Social context
    #   205             EMTSUPRT    Emotional support and life satisfaction
    #   206             LSATISFY    Emotional support and life satisfaction
    #   207             ADPLEASR    Anxiety and depression
    #   208             ADDOWN      Anxiety and depression
    #   209             ADSLEEP     Anxiety and depression
    #   210             ADENERGY    Anxiety and depression
    #   211             ADEAT1      Anxiety and depression
    #   212             ADFAIL      Anxiety and depression
    #   213             ADTHINK     Anxiety and depression
    #   214             ADMOVE      Anxiety and depression
    #   247             _AGE5YR     Computed age categories
    #   252             HTM4        Reported height in meters
    #   253             WTKG3       Reported weight in kilograms 
    #   254             _BMI5       Body mass index
    #   255             _BMI5CAT    Four-categories of BMI
    #   ... Other computed variables corresponding to the previous questions/variables
    #   288             MAXVO2_     Estimated VO2max
    #   289             FC60_       Estimated functionnal capacity
    #   292-->306                   Calculated minutes of activity

    continuous_mask = np.zeros(M, dtype=bool)
    continuous_mask[continuous_idx] = True

    filtered_continuous_mask  = continuous_mask[feature_mask]
    
    categorical_mask = np.ones(M, dtype=bool)
    categorical_mask[continuous_mask] = False
    filtered_categorical_mask = categorical_mask[feature_mask]

    return filtered_continuous_mask, filtered_categorical_mask

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------

    #   IMPORTANT NOTES :

    #   ORDINAL CATEGORICAL features should be checked : 

    #                       Need to re-encode some values as NAN (ex: 77: did not want to answer)
    #                       Need to re-encode some values as zeros (ex: 88: "none")
    #                       Need to check the ordinality : do the low "answers" have low encodings?

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------

    #   Features that are CATEGORICAL : 
    #   IDX     Column  Name           Comments   
    #   1       B       State                   
    #   31      AF      HLTHPLN1       Encode 7 and 9 as NANs
    #   32      AE      PERSDOC2       Encode 7 and 9 ans NANs, Encode 3 as zero
    #   33                             Encode 7 and 9 as NANs
    #   35      AJ      BPHIGH4        Encode 7 and 9 as NANs
    #   36 
    #   37      AL      BLOODCHO
    #   ... 
    #      
    #   Too many features to note them all, but need to reencode the 7-77 and 9-99 and also often the zeros     
    #   Columns ~36 to ~71 : interesting features about heart and general health


    # --------------------------------------------------------------------------------------------------------------------------------------------------------------

    #   Features that should be SUPPRESSED : 

    #   IDX     Column  Name        Comments       
    #   2       C       FMONTH
    #   3       D       IDate
    #   4       E       IMONTH
    #   5       F       IDAY
    #   6       G       IYEAR
    #   7       H       DISPCODE    Interview completed or not
    #   8       I       SEQNO       Record identification
    #   9       J       PSU         Sampling Unit
    #   55      BD      NUMHHOL2    
    #   56      BE      NUMPHON2
    #   57      BF      CPDEMO1
    #   62      BK      INTERNET
    #   217     HJ      QSTVER      Questionnaire version, land or cell phone
    #   218     HK      QSTLANG     Language
    #   228     HU      _DUALUSE    Phone
    #   229     UV      _DUALCOR    Phone
    #   230     HW      _LLCPWT     Phone weighting variables
    #   251     IR      HTIN4       Reported height in inches

    #   Many features would not have a significant incidence on the result of the study, including : 
    #   Phone number, housing in a private residence, housing in college, state residence, are you an adult (legality), number of adults in the household,

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------

    #  Features that  are in the grey zone :
    #  IDX          Column      Name                    Comments 
    #  106          DC          HIVTSTD3                Date of last HIV test. ordered categorical, but tricky encoding and maybe not worth it to hot one reincode it
    #  220..> 223   HM ..> HP   _STSTR --> _WT2RAKE     Weighting variables. Seem unuseful but not sure. Same for the phone weighting variable _LLCPWT
    #  250          IQ          _AGE_G                  Multiple calculated variables for age, not understood if really useful

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Features not found (hidden): 

    # PAINACT2
    # QLMENTL2
    # QLSTRES2
    # QLHLTH2

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Features that may be useful : calculated variables : 

    #   IDX     Column      Name        Comments 
    #   231     HX          _RFHLTH     "General level of health"  
    #   235     IB          _RFCHOL     "High cholesterol"    



### Impute missing values in continuous and categorical data

def mean_imputation(x, means=None, fill_all_nan_with=0.0):
    """
    Impute NaNs in continuous data by column means.

    Args:
        x: array of shape (N, D), can contain np.nan
        means: optional array of shape (D,). If None, compute from x.
        fill_all_nan_with: value used when a column has all NaNs in the 'fit' data

    Returns:
        x_filled: imputed copy of x
        means: column means used (shape (D,))
    """
    x = x.astype(float, copy=True)

    if means is None:
        means = np.nanmean(x, axis=0)
        # Handle columns that are all-NaN (nanmean -> NaN)
        all_nan_cols = np.isnan(means)
        if np.any(all_nan_cols):
            means[all_nan_cols] = fill_all_nan_with

    # Replace NaNs with the corresponding column mean
    nan_rows, nan_cols = np.where(np.isnan(x))
    x[nan_rows, nan_cols] = means[nan_cols]
    return x, means

def mode_imputation(x, modes=None, fill_all_nan_with=0):
    """
    Impute NaNs in categorical data by column modes.

    Args:
        x: array of shape (N, D), can contain np.nan (=> dtype will be float)
        modes: optional array of shape (D,). If None, compute from x.
        fill_all_nan_with: value used when a column has all NaNs in the 'fit' data

    Returns:
        x_filled: imputed copy of x
        modes: column modes used (shape (D,))
    """
    x = x.copy()

    if modes is None:
        D = x.shape[1]
        modes = np.empty(D)

        for j in range(D):
            col = x[:, j]
            col_no_nan = col[~np.isnan(col)]
            if col_no_nan.size == 0:
                # all NaN -> fallback
                modes[j] = fill_all_nan_with
                continue

            # Compute mode via counts; tie-break by smallest value
            vals, counts = np.unique(col_no_nan, return_counts=True)
            maxc = counts.max()
            candidates = vals[counts == maxc]
            modes[j] = np.min(candidates)  # tie-break: smallest value

    # Fill NaNs using modes
    nan_rows, nan_cols = np.where(np.isnan(x))
    x[nan_rows, nan_cols] = modes[nan_cols]
    return x, modes

def impute_train_test(x_train_cont, x_test_cont, x_train_cat, x_test_cat,
                      cast_categorical_to_int=True):
    """
    Fit imputation stats on train; apply to both train and test.

    Returns:
        x_train_cont_imp, x_test_cont_imp, x_train_cat_imp, x_test_cat_imp, means, modes
    """
    # Continuous
    x_train_cont_imp, means = mean_imputation(x_train_cont, means=None)
    x_test_cont_imp, _ = mean_imputation(x_test_cont, means=means)

    # Categorical
    x_train_cat_imp, modes = mode_imputation(x_train_cat, modes=None)
    x_test_cat_imp, _ = mode_imputation(x_test_cat, modes=modes)

    if cast_categorical_to_int:
        # Safe to cast back to integer type after mode imputation (no NaNs remain)
        x_train_cat_imp = x_train_cat_imp.astype(int, copy=False)
        x_test_cat_imp = x_test_cat_imp.astype(int, copy=False)

    return (x_train_cont_imp, x_test_cont_imp,
            x_train_cat_imp, x_test_cat_imp,
            means, modes)

