import numpy as np
import pandas as pd
import sys

#from fancyimpute import  KNN

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import svd
from collections import defaultdict

from scipy.stats import mode, itemfreq
import matplotlib.pylab as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC as SVM


def drop( x, missing_data_cond):
        """ Drops all observations that have missing data
        Parameters
        ----------
        x : np.ndarray
            Matrix with categorical data, where rows are observations and
            columns are features
        missing_data_cond : function
            Method that takes one value and returns True if it represents
            missing data or false otherwise.
        """

        # drop observations with missing values
        return x[np.sum(missing_data_cond(x), axis=1) == 0]

def replace( x, missing_data_cond, in_place=False):
        """ Replace missing data with a random observation with data
        Parameters
        ----------
        x : np.ndarray
            Matrix with categorical data, where rows are observations and
            columns are features
        missing_data_cond : function
            Method that takes one value and returns True if it represents
            missing data or false otherwise.
        """
        if in_place:
            data = x
        else:
            data = np.copy(x)

        for col in xrange(x.shape[1]):
            nan_ids = missing_data_cond(x[:, col])
            val_ids = np.random.choice(np.where(~nan_ids)[0],  np.sum(nan_ids))
            data[nan_ids, col] = data[val_ids, col]
        return data

def summarize( x, summary_func, missing_data_cond, in_place=False):
        """ Substitutes missing values with a statistical summary of each
        feature vector
        Parameters
        ----------
        x : numpy.array
            Assumes that each feature column is of single type. Converts
            digit string features to float.
        summary_func : function
            Summarization function to be used for imputation
            (mean, median, mode, max, min...)
        missing_data_cond : function
            Method that takes one value and returns True if it represents
            missing data or false otherwise.
        """

        if in_place:
            data = x
        else:
            data = np.copy(x)

        # replace missing values with the summarization function
        for col in xrange(x.shape[1]):
            nan_ids = missing_data_cond(x[:, col])
            if True in nan_ids:
                val = summary_func(x[~nan_ids, col])
                data[nan_ids, col] = val

        return data

def one_hot( x, missing_data_cond, weighted=False, in_place=False):
        """Create a one-hot row for each observation
        Parameters
        ----------
        x : np.ndarray
            Matrix with categorical data, where rows are observations and
            columns are features
        missing_data_cond : function
            Method that takes one value and returns True if it represents
            missing data or false otherwise.
        weighted : bool
            Replaces one-hot by n_classes-hot.
        Returns
        -------
        data : np.ndarray
            Matrix with categorical data replaced with one-hot rows
        """

        if in_place:
            data = x
        else:
            data = np.copy(x)

        # find rows and columns with missing data
        _, miss_cols = np.where(missing_data_cond(data))
        miss_cols_uniq = np.unique(miss_cols)

        for miss_col in miss_cols_uniq:
            uniq_vals, indices = np.unique(data[:, miss_col],
                                           return_inverse=True)
            if weighted:
                data = np.column_stack((data, np.eye(uniq_vals.shape[0],
                                        dtype=int)[indices]*uniq_vals.shape[0]))
            else:
                data = np.column_stack((data, np.eye(uniq_vals.shape[0],
                                                     dtype=int)[indices]))

        # remove categorical columns with missing data
        data = np.delete(data, miss_cols, 1)
        return data

def knn( x, k, summary_func, missing_data_cond, cat_cols,
            weighted=False, in_place=False):
        """ Replace missing values with the summary function of K-Nearest
        Neighbors
        Parameters
        ----------
        x : np.ndarray
            Matrix with categorical data, where rows are observations and
            columns are features
        k : int
            Number of nearest neighbors to be used
        summary_func : function
            Summarization function to be used for imputation
            (mean, median, mode, max, min...)
        missing_data_cond : function
            Method that takes one value and returns True if it represents
            missing data or false otherwise.
        cat_cols : int tuple
            Index of columns that are categorical
        """
        if in_place:
            data = x
        else:
            data = np.copy(x)


        # first transform features with categorical missing data into one hot
        data_complete = one_hot(data, missing_data_cond, weighted=weighted)

        # binarize complete categorical variables and convert to int
        col = 0
        cat_ids_comp = []
        while col < max(cat_cols):
            if isinstance(data_complete[0, col], basestring) \
                    and not data_complete[0, col].isdigit():
                cat_ids_comp.append(col)
            col += 1

        data_complete = binarize_data(data_complete,
                                          cat_ids_comp).astype(float)

        # normalize features
        scaler = StandardScaler().fit(data_complete)
        data_complete = scaler.transform(data_complete)
        # create dict with missing rows and respective columns
        missing = defaultdict(list)
        map(lambda (x, y): missing[x].append(y),
            np.argwhere(missing_data_cond(data)))
        # create mask to build NearestNeighbors with complete observations only
        mask = np.ones(len(data_complete), bool)
        mask[missing.keys()] = False
        # fit nearest neighbors and get knn ids of missing observations
        print 'Computing k-nearest neighbors'
        nbrs = NearestNeighbors(n_neighbors=k, metric='cosine',algorithm = 'brute').fit(
            data_complete[mask])
        ids = nbrs.kneighbors(data_complete[missing.keys()],
                              return_distance=False)

        def substituteValues(i):
            row = missing.keys()[i]
            cols = missing[row]
            data[row, cols] = mode(data[mask][ids[i]][:, cols])[0].flatten()

        print 'Substituting missing values'
        map(substituteValues, xrange(len(missing)))
        return data

def predict( x, cat_cols, missing_data_cond, clf, inc_miss=True,
                in_place=False):
        """ Uses random forest for predicting missing values
        Parameters
        ----------
        cat_cols : int tuple
            Index of columns that are categorical
        missing_data_cond : function
            Method that takes one value and returns True if it represents
            missing data or false otherwise.
        clf : object
            Object with fit and predict methods, e.g. sklearn's Decision Tree
        inc_miss : bool
            Include missing data in fitting the model?
        """

        if in_place:
            data = x
        else:
            data = np.copy(x)

        # find rows and columns with missing data
        miss_rows, miss_cols = np.where(missing_data_cond(data))
        miss_cols_uniq = np.unique(miss_cols)

        if inc_miss:
            valid_cols = np.arange(data.shape[1])
        else:
            valid_cols = [n for n in xrange(data.shape[1])
                          if n not in miss_cols_uniq]

        # factorize valid cols
        data_factorized = np.copy(data)

        # factorize categorical variables and store transformation
        factor_labels = {}
        for cat_col in cat_cols:
            # factors, labels = pd.factorize(data[:, cat_col])
            labels, factors = np.unique(data[:, cat_col], return_inverse=True)
            factor_labels[cat_col] = labels
            data_factorized[:, cat_col] = factors

        # values are integers, convert accordingly
        data_factorized = data_factorized.astype(int)

        # update each column with missing features
        for miss_col in miss_cols_uniq:
            # extract valid observations given current column missing data
            valid_obs = [n for n in xrange(len(data))
                         if data[n, miss_col] != '?']

            # prepare independent and dependent variables, valid obs only
            data_train = data_factorized[:, valid_cols][valid_obs]
            y_train = data_factorized[valid_obs, miss_col]

            # train random forest classifier
            clf.fit(data_train, y_train)

            # given current feature, find obs with missing vals
            miss_obs_iddata = miss_rows[miss_cols == miss_col]

            # predict missing values
            y_hat = clf.predict(data_factorized[:, valid_cols][miss_obs_iddata])

            # replace missing data with prediction
            data_factorized[miss_obs_iddata, miss_col] = y_hat

        # replace values on original data
        for col in factor_labels.keys():
            data[:, col] = factor_labels[col][data_factorized[:, col]]

        return data

def factor_analysis( x, cat_cols, missing_data_cond, threshold=0.9,
                        technique='SVD', in_place=False):
        """ Performs low-rank matrix approximation via dimensioality reduction
        and replaces missing data with values obtained from the data projected
        onto N principal components or singular values or eigenvalues...
        cat_cols : int tuple
            Index of columns that are categorical
        missing_data_cond : function
            Method that takes one value and returns True if it represents
            missing data or false otherwise.
        threshold : float
            Variance threshold that must be explained by eigen values.
        technique : str
            Technique used for low-rank approximation. 'SVD' is supported
        """

        def _mode(d):
            return mode(d)[0].flatten()

        if in_place:
            data = x
        else:
            data = np.copy(x)

        data_summarized =summarize(x, _mode, missing_data_cond)

        # factorize categorical variables and store encoding
        factor_labels = {}
        for cat_col in cat_cols:
            labels, factors = np.unique(data_summarized[:, cat_col],
                                        return_inverse=True)
            factor_labels[cat_col] = labels
            data_summarized[:, cat_col] = factors

        data_summarized = data_summarized.astype(float)
        if technique == 'SVD':
            lsvec, sval, rsvec = svd(data_summarized)
            # find number of singular values that explain 90% of variance
            n_singv = 1
            while np.sum(sval[:n_singv]) / np.sum(sval) < threshold:
                n_singv += 1

            # compute low rank approximation
            data_summarized = np.dot(
                lsvec[:, :n_singv],
                np.dot(np.diag(sval[:n_singv]), rsvec[:n_singv, ]))
        else:
            raise Exception("Technique {} is not supported".format(technique))

        # get missing data indices
        nans = np.argwhere(missing_data_cond(x))

        # update data given projection
        for col in np.unique(nans[:, 1]):
            obs_ids = nans[nans[:, 1] == col, 0]
            # clip low rank approximation to be within factor labels
            proj_cats = np.clip(
                data_summarized[obs_ids, col], 0, len(factor_labels[col])-1)
            # round categorical variable factors to int
            proj_cats = proj_cats.round().astype(int)
            data[obs_ids, col] = factor_labels[col][proj_cats]

        return data

def factorize_data( x, cols, in_place=False):
        """Replace column in cols with factors of cols
        Parameters
        ----------
        x : np.ndarray
            Matrix with categorical data
        cols: tuple <int>
            Index of columns with categorical data
        Returns
        -------
        d : np.ndarray
            Matrix with categorical data replaced with factors
        """

        if in_place:
            data = x
        else:
            data = np.copy(x)

        factors_labels = {}
        for col in cols:
            # factors, labels = pd.factorize(data[:, col])
            labels, factors = np.unique(data[:, col], return_inverse=True)
            factors_labels[col] = labels
            data[:, col] = factors

        return data, factors_labels

def binarize_data( x, cols, miss_data_symbol=False,
                      one_minus_one=True, in_place=False):
        """Replace column in cols with one-hot representation of cols
        Parameters
        ----------
        x : np.ndarray
            Matrix with categorical data, where rows are observations and
            columns are features
        cols: tuple <int>
            Index of columns with categorical data
        Returns
        -------
        d : np.ndarray
            Matrix with categorical data replaced with one-hot rows
        """

        if in_place:
            data = x
        else:
            data = np.copy(x)
        for col in cols:
            uniq_vals, indices = np.unique(data[:, col], return_inverse=True)

            if one_minus_one:
                data = np.column_stack(
                    (data,
                     (np.eye(uniq_vals.shape[0], dtype=int)[indices] * 2) - 1))
            else:
                data = np.column_stack((data, np.eye(uniq_vals.shape[0],
                                                     dtype=int)[indices]))
            # add missing data column to feature
            if miss_data_symbol is not False and \
                    miss_data_symbol not in uniq_vals:
                data = np.column_stack(
                    (data, -one_minus_one * np.ones((len(data), 1), dtype=int)))

        # remove columns with categorical variables
        val_cols = [n for n in xrange(data.shape[1]) if n not in cols]
        data = data[:, val_cols]
        return data

rng = np.random.RandomState(0)

reload(sys)
sys.setdefaultencoding('utf8')

df_main_excel = pd.read_excel('A:/Capstone_Code/Modelling/Modelling_Final_File_Imputation.xlsx',sheetname='Sheet1')

df_impute_rev = df_main_excel.filter(['Company','Age','Target_Sector_Hierarchy','Target_Countries','Target_Dom_Country','Revenue','Company_Type','Employees'])

df_impute_rev.drop_duplicates()
#df_buyer_filter = df_main_excel.filter(['Buyer','Buyer_Age','Buyer_Sector_Hierarchy','Buyer_Countries','Buyer_Dom_Country','Buyer_Revenue','Buyer_Company_Type','Employees'])
#df_buyer_filter =df_buyer_filter.loc[df_buyer_filter['Buyer_Revenue'].notnull()]
#df_buyer_filter =df_buyer_filter.loc[df_buyer_filter['Buyer_Revenue']<>0]
#df_buyer_filter['RightN_Cand_Acq'] = 'N'
#df_buyer_filter = df_buyer_filter.drop_duplicates()
#df_impute = pd.concat([df_impute_rev,df_buyer_filter.rename(columns = {'Buyer':'Company','Buyer_Age':'Age','Buyer_Sector_Hierarchy':'Target_Sector_Hierarchy',
#                                                                        'Buyer_Countries':'Target_Countries','Buyer_Dom_Country':'Target_Dom_Country',
#                                                                           'Buyer_Revenue':'Revenue','Buyer_Company_Type':'Company_Type'})],ignore_index=True)
#df_impute_rev = df_impute_rev.append(df_buyer_filter)
df_impute = df_impute_rev
#df_impute = df_impute.drop_duplicates()
df_impute.reset_index()
df_impute.index = np.arange(1, len(df_impute) + 1)

df_impute['Number_Countries'] = map(lambda x: (len(x.split(',')) if str(x)<>'nan' else 0), df_impute['Target_Countries'])
df_impute['Age'] = map(lambda x: ( x*-1 if x<0 else x), df_impute['Age'])


df_impute['Sector_Dummy'] = pd.Categorical(df_impute['Target_Sector_Hierarchy']).codes
df_impute['Country_Dummy'] = pd.Categorical(df_impute['Target_Dom_Country']).codes
df_impute['Type_Dummy'] = pd.Categorical(df_impute['Company_Type']).codes
df_impute['Revenue'] =df_impute['Revenue'].fillna('?') 
df_impute['Employees'] =df_impute['Employees'].fillna('?') 

#df_impute['Revenue'] = df_impute['Revenue'].astype('float')
#df_impute= df_impute.loc[df_impute['Revenue']<> '?']
df_impute = df_impute.filter(['Company','Age','Sector_Dummy','Number_Countries','Country_Dummy','Revenue','Employees'])

df_impute.reset_index()
df_impute.index = np.arange(1, len(df_impute) + 1)


#df_full =df_impute.filter(['Sector_Dummy','Number_Countries','Country_Dummy','Type_Dummy','Revenue','RightN_Cand_Acq'])



df_impute_compt = df_impute.filter(['Sector_Dummy','Number_Countries','Country_Dummy','Revenue','Employees'])

#df_impute_compt['Revenue'] = df_impute_compt['Revenue'].convert_objects(convert_numeric=True)
df_impute_compt['Revenue'] =map(lambda x: str(x).replace('-','?'),df_impute_compt['Revenue'] )

df_impute_compt['Revenue'] =map(lambda x: (long(x) if type(x)=='long' else x),df_impute_compt['Revenue'] )
X_Complete = df_impute_compt.as_matrix()
df_incompt = df_impute_compt
df_incompt = df_incompt.filter(['Sector_Dummy','Number_Countries','Country_Dummy','Revenue','Employees'])

X_Incomplete =df_incompt.as_matrix()
#X_Incomplete = df_modelling_file.as_matrix()
#df_full.filter(['Sector_Dummy','Number_Countries','Country_Dummy','Type_Dummy','Revenue']).as_matrix(),df_full['RightN_Cand_Acq'].as_matrix()

missing_data_cond = lambda x: x == '?'

  
    
cat_cols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24)#,25) #,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,
             #63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,
             # 119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,
             #  164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,
              #  205,206,207,208,209,210,211,212,213,214,215,216,217,218,219)

n_neighbors = 4

print 'imputing with random replacement'
data_replace = replace(X_Incomplete, missing_data_cond)

# replace missing values with feature summary
print 'imputing with feature summarization (mode)'
summ_func = lambda x: mode(x)[0]
data_mode = summarize(X_Incomplete, summ_func, missing_data_cond)

print 'imputing with predicted values from random forest'
clf = RandomForestClassifier(n_estimators=100, criterion='gini')
data_rf = predict(X_Incomplete[:,1:], cat_cols, missing_data_cond, clf)

print 'imputing with predicted values usng SVM'
clf = clf = SVM(
    penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', 
    fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, 
    random_state=None, max_iter=1000)
data_svm = predict(X_Incomplete, cat_cols, missing_data_cond, clf)

# replace missing data with knn
print 'imputing with K-Nearest Neighbors'
data_knn = knn(X_Incomplete, n_neighbors, np.mean, missing_data_cond, cat_cols)




# replace missing data with predictions using logistic regression
print 'imputing with predicted values usng logistic regression'
clf = LogisticRegression(
            penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
            intercept_scaling=1)
data_logistic = predict(X_Incomplete, cat_cols, missing_data_cond, clf)




# replace missing data with values obtained after factor analysis
print 'imputing with factor analysis'
data_facanal = factor_analysis(X_Incomplete, cat_cols, missing_data_cond)



def compute_histogram(data, labels):
    histogram = itemfreq(sorted(data))
    for label in labels:
        if label not in histogram[:,0]:
            histogram = np.vstack((histogram,
                                   np.array([[label, 0]], dtype=object)))
    histogram = histogram[histogram[:,0].argsort()]
    return histogram

# compute histograms
labels = np.unique(X_Incomplete[:,1])
freq_data = {}
freq_data['Raw data'] = compute_histogram(X_Incomplete[:,3], labels)
# freq_data['Drop missing'] = compute_histogram(data_drop[:,1], labels)
freq_data['Random replace'] = compute_histogram(data_replace[:,3], labels)
freq_data['Summary'] = compute_histogram(data_mode[:,3], labels)
freq_data['Random forests'] = compute_histogram(data_rf[:,3], labels)
freq_data['SVM'] = compute_histogram(data_svm[:,3], labels)
freq_data['Logistic regression'] = compute_histogram(data_logistic[:,3], labels)
freq_data['PCA'] = compute_histogram(data_facanal[:,3], labels)
freq_data['KNN'] = compute_histogram(data_knn[:,3], labels)


#df_impute['SVM'] = data_svm[:,3]
df_impute['Revenue'] = data_knn[:,3]
#df_impute['RF'] = data_rf[:,3]
#df_impute['PCA'] = data_facanal[:,3]
#df_impute['LReg'] = data_logistic[:,3]
df_impute['Employees'] = data_knn[:,4]



writer = pd.ExcelWriter('A:/Capstone_Code/Modelling/Impt_FullData.xlsx', engine='xlsxwriter')
df_impute.to_excel(writer,'Sheet1')
writer.save()

# plot histograms given feature with missing data
n_methods = len(freq_data.keys())
bins = np.arange(len(labels))
width = .25
fig, ax = plt.subplots(figsize=(12,8))

for i in xrange(1,n_methods):
    key = sorted(freq_data.keys())[i]
    offset = i*2*width/float(n_methods)
    ax.bar(bins+offset, freq_data[key][:,1].astype(int), width, label=key,
           color=plt.cm.hot(i/float(n_methods)), align='center')

ax.set_xlabel('Categories', size=15)
ax.set_ylabel('Count', size=15)
ax.set_title('Company Data SetRevenue', size=15, fontweight='bold')
ax.set_xticks(bins + width)
ax.set_xticklabels(labels, rotation=45)
plt.legend(loc=2)
plt.tight_layout()
plt.show()







#df_sectors.reset_index()
#df_sectors.index = np.arange(1, len(df_sectors) + 1)

#df_main_excel['Target_Dom_Country'] = map(lambda x: x.strip().replace('United States of America','USA'),df_main_excel['Target_Dom_Country'])
#df_main_excel['Buyer_Dom_Country'] = map(lambda x: x.strip().replace('United States of America','USA').replace('United States','USA'),df_main_excel['Buyer_Dom_Country'])

#for i in xrange(1,len(df_main_excel)+1):
#
#    #target_sector = df_sectors.loc[df_sectors['Hierarchy_Lower'] == df_main_excel.loc[i,'Dominant_Sector'].lower(),'Target Sector Hierarchy'].values
#    #
#    #buyer_sector = (df_sectors.loc[df_sectors['Hierarchy_Lower'] == df_main_excel.loc[i,'Industry_BB_Buyer'].lower(),'Target Sector Hierarchy'].values
#    #                 if str(df_main_excel.loc[i,'Industry_BB_Buyer'])<> 'nan' else [])
#    #if str(target_sector)<>'nan' and len(target_sector)>0:
#    #    df_main_excel.at[i,'Target_Sector_Hierarchy'] =target_sector[0]
#    #if str(buyer_sector)<>'nan' and len(buyer_sector)>0:
#    #    df_main_excel.at[i,'Buyer_Sector_Hierarchy'] =buyer_sector[0]
#    target_countries = df_main_excel.loc[i,'Target_Countries']
#    buyer_countries = df_main_excel.loc[i,'Buyer_Countries']
#    if str(target_countries)<> 'nan':
#        target_countries = target_countries.strip().replace('United States of America','USA').replace('United States','USA')
#        ls_target =target_countries.split(';')
#        ls_target = unique(ls_target)
#        if len(ls_target)>1:
#           df_main_excel.at[i,'Target_Countries'] = ','.join(ls_target)
#        else:
#           df_main_excel.at[i,'Target_Countries'] = ''.join(ls_target)
#    if str(buyer_countries)<> 'nan':
#        buyer_countries = buyer_countries.strip().replace('United States of America','USA').replace('United States','USA')
#        ls_buyer =buyer_countries.split(';')
#        ls_buyer = unique(ls_buyer)
#        if len(ls_buyer)>1:
#           df_main_excel.at[i,'Buyer_Countries'] = ','.join(ls_buyer)
#        else:
#           df_main_excel.at[i,'Buyer_Countries'] = ''.join(ls_buyer)
#             
#           
#    

#writer = pd.ExcelWriter('A:/Capstone_Code/files/Modelling_Final_File_new.xlsx', engine='xlsxwriter')
#df_main_excel.to_excel(writer,'Sheet1')
#writer.save()