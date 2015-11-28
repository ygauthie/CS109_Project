
# coding: utf-8

# # Are We Rich Yet? - Ensemble version
# 
# ## This notebook attempts to split the data into training, validation, and test in order to facilitate ensemble regression later. Training is 2009-2013, Validation is 2014, and Test is 2015.
# 
# ### The predictions and ROIs are stored for each classifier run on training. A final ensemble regression is run over validation results and then applied to test results.
# 
# #### I have quite the headache.

# In[1]:

get_ipython().magic(u'matplotlib inline')

get_ipython().magic(u'run talibref.py')


# ### Get data

# In[2]:

#df=pd.read_csv("data/IYZ.csv")
ticker = 'IYZ'
startdate=datetime.date(2009, 1, 1)
enddate=datetime.date.today()

sdatev=datetime.date(2014, 1, 1)
edatev=datetime.date(2014, 12, 31)
sdatet=datetime.date(2015, 1, 1)
edatet=datetime.date.today()

df = generate_ticker_data(ticker, startdate, enddate)
df.head()


# In[3]:

df.tail()


# In[4]:

dftouse=df.copy()


# ### Feature Engineering

# In[5]:

IGNORE = ['date', 'result_1','close_1','perf_1','result_14','close_14','perf_14','results']


# In[6]:

INDICATORS=[]
for v in df.columns:
    l=df[v].unique()
    if len(l) <= 10 and v not in IGNORE:
        print v, l
        INDICATORS.append(v)


# In[7]:

STANDARDIZABLE = []
for v in df.columns:
    if v not in INDICATORS and v not in IGNORE:
        print v
        STANDARDIZABLE.append(v)


# In[8]:

# from sklearn.cross_validation import train_test_split
# itrain, itest = train_test_split(xrange(dftouse.shape[0]), train_size=0.7)
# mask=np.ones(dftouse.shape[0], dtype='int')
# mask[itrain]=1
# mask[itest]=0
# mask = (mask==1)
# mask.shape, mask.sum()


# In[9]:

#dftouse['date'] = pd.to_datetime(dftouse['date'])
#mask = (dftouse.date < '2015-01-01').values
#mask.shape, mask.sum()

mask = dftouse['date'] < sdatev
maskv = (dftouse['date'] >= sdatev) & (dftouse['date'] <= edatev)
maskt = (dftouse['date'] >= sdatet) & (dftouse['date'] <= edatet)


# ### Check ROI of signals, alone:

# In[10]:

#print "ROI baseline: 1.12%"
#print 'ROI "result" buy-only: 67.45%'
#print 'ROI "result" buy-sell: 172.49%'
    
def signalperf(signal, valtest):
    ypred = df[signal]
    if valtest == "v":
        df_pred = df[maskv].reset_index(drop=True)
        sdate = sdatev
        edate = edatev
    else:
        df_pred = df[maskt].reset_index(drop=True)
        sdate = sdatet
        edate = edatet
    df_pred['pred_result'] = ypred
    df_pred['result_baseline'] = np.ones(df_pred.shape[0])

    balance, profit, ROI2, balovertime, signals = evaluate_profit(df_pred, sdate, edate, 10000, 'pred_result', 'close', False, [0])
    #print 'ROI "pred" buy-only: {0:.2f}%'.format(ROI*100)

    balance, profit, ROI, balovertime, signals = evaluate_profit(df_pred, sdate, edate, 10000, 'pred_result', 'close', False, [1])
    print signal + ': ROI "pred" buy-only: {0:.2f}%'.format(ROI2*100), 'buy-sell: {0:.2f}%'.format(ROI*100)
    return


# In[11]:

print 'validation: ' + sdatev.strftime('%Y-%m-%d') + ' - ' + edatev.strftime('%Y-%m-%d')

for v in INDICATORS:
    signalperf(v, "v")


# In[12]:

print 'test: ' + sdatet.strftime('%Y-%m-%d') + '-' + edatet.strftime('%Y-%m-%d')

for v in INDICATORS:
    signalperf(v, "t")


# #### 1.2 Standardize the data

# Use the mask to compute the training and test parts of the dataframe. Use `StandardScaler` from `sklearn.preprocessing` to "fit" the columns in `STANDARDIZABLE` on the training set. Then use the resultant estimator to transform both the training and the test parts of each of the columns in the dataframe, replacing the old unstandardized values in the `STANDARDIZABLE` columns of `dftouse` by the new standardized ones.

# In[13]:

#your code here
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(dftouse[mask][STANDARDIZABLE])
dftouse[STANDARDIZABLE] = scaler.transform(dftouse[STANDARDIZABLE])
dftouse.head()


# We create a list `lcols` of the columns we will use in our classifier. This list should not contain the response `RESP`. How many features do we have?

# In[14]:

#lcols=list(dftouse.columns)
#lcols.remove(u'results')
lcols=[]
#lcols.append('cv_signal')
for c in list(dftouse.columns):
    #if c not in INDICATORS and c not in IGNORE:
    #if c in INDICATORS:
    if c not in IGNORE:  #Original
        lcols.append(c)
print len(lcols)


# ### EDA for the data

# We create a variable `ccols` which contains all variables not in our indicators list

# In[15]:

ccols=[]
for c in lcols:
    if c not in INDICATORS and c not in IGNORE:
        ccols.append(c)
print len(ccols), len(INDICATORS)
ccols


# #### 1.4 Train a SVM on this data.

# In[16]:

from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV


# In[17]:

"""
Function
--------
cv_optimize

Inputs
------
clf : an instance of a scikit-learn classifier
parameters: a parameter grid dictionary thats passed to GridSearchCV (see above)
X: a samples-features matrix in the scikit-learn style
y: the response vectors of 1s and 0s (+ives and -ives)
n_folds: the number of cross-validation folds (default 5)
score_func: a score function we might want to pass (default python None)
   
Returns
-------
The best estimator from the GridSearchCV, after the GridSearchCV has been used to
fit the model.
     
Notes
-----
see do_classify and the code below for an example of how this is used
"""
#your code here
def cv_optimize(clf, parameters, X, y, n_folds, score_func):
    fitmodel = GridSearchCV(clf, param_grid=parameters, cv=n_folds, scoring=score_func)
    fitmodel.fit(X, y)
    return fitmodel.best_estimator_


# In[18]:

from sklearn.metrics import confusion_matrix
def do_classify(clf, parameters, indf, featurenames, targetname, target1val, mask=mask, reuse_split=None, score_func=None, n_folds=5):
    subdf=indf[featurenames]
    X=subdf#.values
    y=(indf[targetname].values==target1val)*1
    y=indf[targetname]
    #if mask !=None:
    #    print "using mask"
    #    Xtrain, Xval, Xtest, ytrain, yval, ytest = X[mask], X[maskv], X[maskt], y[mask], y[maskv], y[maskt]
    Xtrain, Xval, Xtest, ytrain, yval, ytest = X[mask], X[maskv], X[maskt], y[mask], y[maskv], y[maskt]
    if reuse_split !=None:
        print "using reuse split"
        Xtrain, Xval, Xtest, ytrain, yval, ytest = reuse_split['Xtrain'], reuse_split['Xval'], reuse_split['Xtest'], reuse_split['ytrain'], reuse_split['yval'], reuse_split['ytest']
    if parameters:
        clf = cv_optimize(clf, parameters, Xtrain, ytrain, n_folds=n_folds, score_func=score_func)
    clf=clf.fit(Xtrain, ytrain)
    training_accuracy = clf.score(Xtrain, ytrain)
    val_accuracy = clf.score(Xval, yval)
    test_accuracy = clf.score(Xtest, ytest)
    print "############# based on standard predict ################"
    print "Accuracy on training data: %0.2f" % (training_accuracy)
    print "Accuracy on validation data: %0.2f" % (val_accuracy)
    print "Accuracy on test data:     %0.2f" % (test_accuracy)
    print confusion_matrix(ytest, clf.predict(Xtest))
    print "########################################################"
    return clf, Xtrain, ytrain, Xval, yval, Xtest, ytest


# In[19]:

get_ipython().run_cell_magic(u'time', u'', u'clfsvm, Xtrain, ytrain, Xval, yval, Xtest, ytest = do_classify(LinearSVC(loss="hinge"), {"C": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]}, dftouse, lcols, u\'results\', 1, mask=mask)')


# In[20]:

clfsvm


# In[21]:

predsv={}
predsv_ROI_long={}
predsv_ROI_longshort={}

predst={}
predst_ROI_long={}
predst_ROI_longshort={}


# In[22]:

df_pred = df[maskv].reset_index(drop=True)
df_pred['baseline'] = np.ones(df_pred.shape[0])

balance, profit, ROI, balovertime, signals = evaluate_profit(df_pred, sdatev, edatev, 10000, 'baseline', 'close', False, [0])
print "ROI baseline: {0:.2f}%".format(ROI*100)
predsv["baseline_long"] = df_pred['baseline']
predsv_ROI_long["baseline_long"] = ROI
predsv_ROI_longshort["baseline_long"] = ROI

balance, profit, ROI, balovertime, signals = evaluate_profit(df_pred, sdatev, edatev, 10000, 'results', 'close', False, [0])
print 'ROI "result" buy-only: {0:.2f}%'.format(ROI*100)
predsv["baseline_ema"] = df_pred['results']
predsv_ROI_long["baseline_ema"] = ROI
balance, profit, ROI, balovertime, signals = evaluate_profit(df_pred, sdatev, edatev, 10000, 'results', 'close', False, [1])
print 'ROI "result" buy-sell: {0:.2f}%'.format(ROI*100)
predsv_ROI_longshort["baseline_ema"] = ROI

balance, profit, ROI, balovertime, signals = evaluate_profit(df_pred, sdatev, edatev, 10000, 'result_1', 'close', False, [0])
print 'ROI "result" buy-only: {0:.2f}%'.format(ROI*100)
predsv["baseline_max"] = df_pred['result_1']
predsv_ROI_long["baseline_max"] = ROI
balance, profit, ROI, balovertime, signals = evaluate_profit(df_pred, sdatev, edatev, 10000, 'result_1', 'close', False, [1])
print 'ROI "result" buy-sell: {0:.2f}%'.format(ROI*100)
predsv_ROI_longshort["baseline_max"] = ROI


# In[23]:

df_pred = df[maskt].reset_index(drop=True)
df_pred['baseline'] = np.ones(df_pred.shape[0])

balance, profit, ROI, balovertime, signals = evaluate_profit(df_pred, sdatet, edatet, 10000, 'baseline', 'close', False, [0])
print "ROI baseline: {0:.2f}%".format(ROI*100)
predst["baseline_long"] = df_pred['baseline']
predst_ROI_long["baseline_long"] = ROI
predst_ROI_longshort["baseline_long"] = ROI

balance, profit, ROI, balovertime, signals = evaluate_profit(df_pred, sdatet, edatet, 10000, 'results', 'close', False, [0])
print 'ROI "result" buy-only: {0:.2f}%'.format(ROI*100)
predst["baseline_ema"] = df_pred['results']
predst_ROI_long["baseline_ema"] = ROI
balance, profit, ROI, balovertime, signals = evaluate_profit(df_pred, sdatet, edatet, 10000, 'results', 'close', False, [1])
print 'ROI "result" buy-sell: {0:.2f}%'.format(ROI*100)
predst_ROI_longshort["baseline_ema"] = ROI

balance, profit, ROI, balovertime, signals = evaluate_profit(df_pred, sdatet, edatet, 10000, 'result_1', 'close', False, [0])
print 'ROI "result" buy-only: {0:.2f}%'.format(ROI*100)
predst["baseline_max"] = df_pred['result_1']
predst_ROI_long["baseline_max"] = ROI
balance, profit, ROI, balovertime, signals = evaluate_profit(df_pred, sdatet, edatet, 10000, 'result_1', 'close', False, [1])
print 'ROI "result" buy-sell: {0:.2f}%'.format(ROI*100)
predst_ROI_longshort["baseline_max"] = ROI


# In[24]:

def evaluate(clf, desc):

    #Validation
    ypred = clf.predict(Xval)
    df_pred = df[maskv].reset_index(drop=True)
    df_pred['pred_result'] = ypred
    df_pred['result_baseline'] = np.ones(df_pred.shape[0])
    print "accuracy on validation set: {0:.3f}".format((df_pred.result_14 == df_pred.pred_result).sum()/float(len(df_pred)))
    
    balance, profit, ROI, balovertime, signals = evaluate_profit(df_pred, startdate, enddate, 10000, 'pred_result', 'close', False, [0])
    print 'ROI "pred" buy-only: {0:.2f}%'.format(ROI*100)
    predsv[desc] = ypred
    predsv_ROI_long[desc] = ROI

    balance, profit, ROI, balovertime, signals = evaluate_profit(df_pred, startdate, enddate, 10000, 'pred_result', 'close', False, [1])
    print 'ROI "pred" buy-sell: {0:.2f}%'.format(ROI*100)
    predsv_ROI_longshort[desc] = ROI
    
    #Test
    ypred = clf.predict(Xtest)
    df_pred = df[maskt].reset_index(drop=True)
    df_pred['pred_result'] = ypred
    df_pred['result_baseline'] = np.ones(df_pred.shape[0])
    print "accuracy on test set: {0:.3f}".format((df_pred.result_14 == df_pred.pred_result).sum()/float(len(df_pred)))
    
    balance, profit, ROI, balovertime, signals = evaluate_profit(df_pred, startdate, enddate, 10000, 'pred_result', 'close', False, [0])
    print 'ROI "pred" buy-only: {0:.2f}%'.format(ROI*100)
    predst[desc] = ypred
    predst_ROI_long[desc] = ROI

    balance, profit, ROI, balovertime, signals = evaluate_profit(df_pred, startdate, enddate, 10000, 'pred_result', 'close', False, [1])
    print 'ROI "pred" buy-sell: {0:.2f}%'.format(ROI*100)
    predst_ROI_longshort[desc] = ROI


# In[25]:

evaluate(clfsvm, "svm")


# The results obtained ought to be very similar to the efforts you put in earlier. If not its likely you wrote `cv_optimize` wrong. (Remember that we are using the same mask).
# 
# We'll reuse the training and test sets you computed above later in the homework. We do this by putting them into a dictionary `reuse_split`

# In[26]:

reuse_split=dict(Xtrain=Xtrain, Xval=Xval, Xtest=Xtest, ytrain=ytrain, yval=yval, ytest=ytest)


# ## 2. Estimate costs and benefits from assumptions and data

# ### Our data is highly asymmetric

# First notice that our data set is very highly asymmetric, with positive `RESP`onses only making up 16-17% of the samples.

# In[27]:

print "whole data set", dftouse['results'].mean()#Highly asymmetric
print "training set", dftouse['results'][mask].mean(), "val set", dftouse['results'][maskv].mean(), "test set", dftouse['results'][maskt].mean()


# This means that a classifier which predicts that EVERY customer is a negative has an accuracy rate of 83-84%. By this we mean that **a classifier that predicts that no customer will respond to our mailing** has an accuracy of 83-84%!

# In[28]:

#your code here
from sklearn.linear_model import LogisticRegression
clflog = LogisticRegression(penalty="l1")
parameters = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
clflog, Xtrain, ytrain, Xval, yval, Xtest, ytest = do_classify(clflog, parameters, dftouse, lcols, u'results',1, mask=mask)#, reuse_split=reuse_split)


# In[29]:

clflog


# In[30]:

evaluate(clflog, "logistic_regression")


# In[31]:

from sklearn.ensemble import RandomForestClassifier
clfraf = RandomForestClassifier(n_jobs=2)
parameters = {}#{"C": [0.001, 0.01, 0.1, 1, 10, 100]}
clfraf, Xtrain, ytrain, Xval, yval, Xtest, ytest = do_classify(clfraf, parameters, dftouse, lcols, u'results',1, mask=mask, reuse_split=reuse_split)


# In[32]:

evaluate(clfraf, "random_forest")


# ## 4. Trying to improve the SVM: Feature Selection and Data Balancing

# If you did the previous section right, you will find that the logistic regression model provides a better profit over some section of the profit curve than the baseline "send to everyone" classifier, while the SVM classifier is generally poor. At this might we might want to try all kinds of classifiers: from perceptrons to random forests. In the interest of time, and to study the SVM in some more detail, we'll restrict ourselves to trying to improve the SVM performance here. In real life you would try other classifiers as well.
# 
#  We wont be exhaustive in this improvement process either(which is something you should do on your project) in the interests of time, but we'll explore if feature-selection on the  SVM, and data balancing on the SVM (SVM's are known to perform better on balanced data) help.
#  
# ( An aside: many classifiers such as SVM and decision trees struggle in their techniques on imbalanced data. You can read more at: see Weiss, Gary M., and Foster Provost. "The effect of class distribution on classifier learning: an empirical study." Rutgers Univ (2001). Also see http://pages.stern.nyu.edu/~fprovost/Papers/skew.PDF and http://www.cs.ox.ac.uk/people/vasile.palade/papers/Class-Imbalance-SVM.pdf for multiple ways to deal with the imbalance problem: balancing is not always the best option. `Sklearn` also provides a class weighting strategy: http://scikit-learn.org/stable/modules/svm.html#unbalanced-problems ). 

# ### Feature Selection

# The Lasso, for example, implements internally, a form of feature selection by setting many coefficients to zero. Let us find coefficients that are non-zero.

# #### Non zero lasso features

# We write a function `nonzero_lasso` which takes the fit classifier `clfloglasso` as an argument, and spits out a dataframe of coefficients, sorted by the absolute magnitude of the coefficients. This way we can see which features dominated the logistic regression.

# In[33]:

def nonzero_lasso(clf):
    featuremask=(clf.coef_ !=0.0)[0]
    return pd.DataFrame(dict(feature=lcols, coef=clf.coef_[0], abscoef=np.abs(clf.coef_[0])))[featuremask].sort('abscoef', ascending=False)


# In[34]:

lasso_importances=nonzero_lasso(clflog)
lasso_importances.set_index("feature", inplace=True)
lasso_importances.head(10)


# #### 4.1 Feature importance using correlations

# We can also get a notion of which features are important in the classification process by seeing how they correlate with the response. Implement some code to obtain the Pearson correlation coefficient between each of our features and the response. Do this on the training set only! Create a dataframe indexed by the features, which has columns `abscorr` the absolute value of the correlation and `corr` the value of the correlation. Sort the dataframe by `abscorr`, highest first, and show the top 25 features with the highest absolute correlation. Is there much overlap with the feature selection performed by the LASSO?

# In[35]:

from scipy.stats.stats import pearsonr
correlations=[]
dftousetrain=dftouse[mask]
for col in lcols:
    r=pearsonr(dftousetrain[col], dftousetrain['results'])[0]
    correlations.append(dict(feature=col,corr=r, abscorr=np.abs(r)))

bpdf=pd.DataFrame(correlations).sort('abscorr', ascending=False)
bpdf.set_index(['feature'], inplace=True)
bpdf.head(25)


# #### Why Feature Select?

# One of the reasons feature selection is done, automatically or otherwise, is that there might be strong correlations between features. Also recall polynomial regression: a large number of features can lead to overfitting. Feature selection helps curb the problem of the curse of dimensionality, where centrality measures often used in statistics go wonky at higher dimensions. Between feature-engineering which we did some of, earlier, and feature selection, is where a lot of smarts and domain knowledge comes in. You will gain this with experience.

# ### Create a pipeline to feature-select, standardize and train!

# We shall use sklearn pipelines to do correlation-with-response based feature selection for our SVM model. Maybe such feature-selection will improve the abysmal performance. 
# 
# This does not reduce the collinearity amongst the features, for which one either needs PCA, ICA, or some feature selection using the forward-backward algorithm. We do not have the time to approach it here. 
# 
# Its very important to do response based feature selection in the right way. If you remember, we separately standardized the training and test sets. This was to prevent **any** information about the overall mean and standard deviation leaking into the test set. 
# 
# But we played a bit loose with the rules there. We standardized on the entire training set. Instead we should have been standardizing separately in each cross-validation fold. There the original training set would be broken up into a sub-training and validation set, the standardization needed to be done on those separately. This can be implemented with `sklearn` pipelines.
# 
# Such kind of "data snooping" is relatively benign though, as it used no information about the response variable. But if you do any feature selection which uses the response variable, such as choosing the "k" most correlated variables from above, its not benign any more. This is because you have leaked the response from the validation into your sub-training set, and cannot thus be confident about your predictions: you might overfit. In such a situation, you must do the feature selection inside the cross-validation fold. See http://nbviewer.ipython.org/github/cs109/content/blob/master/lec_10_cross_val.ipynb from the 2013 course for a particularly dastardly case of this, where you see that the problem is particularly exacerbated when you have many more features than samples.
# 
# Lets do this here using sklearn pipelines.

# In[36]:

from sklearn import feature_selection
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest


# Lets define a scorer which returns the absolute values of the pearson correlation between the feature and the response for each sample. The specific form of the scorer is dictated to us in the API docs for `SelectKBest`, see [here](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html): the first argument must be an array of scores, and the second an array of p-values.

# In[37]:

def pearson_scorer(X,y):
    rs=np.zeros(X.shape[1])
    pvals=np.zeros(X.shape[1])
    i=0
    for v in X.T:
        rs[i], pvals[i]=pearsonr(v, y)
        i=i+1
    return np.abs(rs), pvals    


# Lets apply the feature selection to a model which did not have any automatic feature selection and performed rather poorly before: the linear SVM. 
# 
# The `Pipeline` feature of sklearn chains various parts of a machine learning algorithm together. In this case we want to chain feature-selection and training in such a way that both happen freshly for each cross-validation fold (we wont bother to standardize in each cross-validation fold separately here for brevity, although you might want to do this).
# We use the `SelectKBest` meta estimator to select the 25 most correlated/anti-correlated features. We create an instance of this meta-estimator, `selectorlinearsvm`. We then combine it with the linear SVC estimators into the pipeline `pipelinearsvm`: the `Pipeline` function simply takes a list of `scikit-learn` estimators and wraps them together into a new estimator object, which can then be passed to `GridSearchCV` via our `do_classify` function. Notice how this new estimator object can be used exactly the same way as a single classifier can be used in `scikit-learn`..this uniformity of interface is one of the nice features of `sklearn`!

# In[38]:

selectorlinearsvm = SelectKBest(k=25, score_func=pearson_scorer)
pipelinearsvm = Pipeline([('select', selectorlinearsvm), ('svm', LinearSVC(loss="hinge"))])


# #### Let us run the pipelined classifier 

# We'll run the classifier and compare the results using the ROC curve to the previous SVM result.

# In[39]:

pipelinearsvm, _,_,_,_,_,_  = do_classify(pipelinearsvm, {"svm__C": [0.00001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}, dftouse,lcols, u'results',1, reuse_split=reuse_split)


# What features did the pipelined classifier use? We can access them so:

# In[40]:

np.array(lcols)[pipelinearsvm.get_params()['select'].get_support()]


# We plot the ROC curves, using the label `svm-feature-selected` for the pipelined classifier `pipelinearsvm`. We plot it alongside the older logistic-with lasso and all-features SVM for comparison

# In[41]:

evaluate(pipelinearsvm, "pipelinearsvm")


# #### 4.3 Implement a RBF based pipelined (feature-selected) classifier on the balanced set.

# In[42]:

from sklearn.svm import SVC
reuse_split_new = reuse_split


# In[43]:

get_ipython().run_cell_magic(u'time', u'', u'selectorsvm2 = SelectKBest(k=25, score_func=pearson_scorer)\npipesvm2 = Pipeline([(\'select2\', selectorsvm2), (\'svm2\', SVC())])\npipesvm2, _,_,_,_,_,_  = do_classify(pipesvm2, {"svm2__C": [1e8, 1e9, 1e10], "svm2__gamma": [ 1e-9, 1e-10, 1e-11]}, dftouse,lcols, u\'results\',1, reuse_split=reuse_split)')


# In[44]:

pipesvm2.get_params()


# In[45]:

evaluate(pipesvm2, "pipesvm2")


# ## Yvan ML

# In[46]:

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score

# selector of  features
featSelector = SelectKBest(k=18, score_func=pearson_scorer)

print "#############====================== Log Regression =====================#############"
pipeLR = Pipeline([('select', featSelector), ('LR', LogisticRegression(penalty="l1"))])
pipeLR, _,_,_,_,_,_  = do_classify(pipeLR, {"LR__C": [0.005, 0.01, 0.02, 0.05, 10.0]}, dftouse,lcols, u'results',1, reuse_split=reuse_split_new)
evaluate(pipeLR, "pipe_Log_Regr")
print "#############====================== Linear SVM ========================#############"
clfsvm_b = Pipeline([('select', featSelector), ('svm', LinearSVC(loss="hinge"))])
clfsvm_b, _,_,_,_,_,_  = do_classify(clfsvm_b, {"svm__C": [0.00001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}, dftouse,lcols, u'results',1, reuse_split=reuse_split_new)
evaluate(clfsvm_b, "linear_svm")
print "#############====================== RBF SVC ===========================#############"
pipesvm2 = Pipeline([('select', featSelector), ('svm2', SVC())])
pipesvm2, _,_,_,_,_,_  = do_classify(pipesvm2, {"svm2__C": [1e8, 1e9, 1e10], "svm2__gamma": [ 1e-9, 1e-10, 1e-11]}, dftouse,lcols, u'results',1, reuse_split=reuse_split_new)
evaluate(pipesvm2, "rbf_svc")
print "#############====================== Random Forest =====================#############"
pipeRF = Pipeline([('select', featSelector), ('RF', RandomForestClassifier())])
pipeRF, _,_,_,_,_,_  = do_classify(pipeRF, {"RF__max_depth": [3,5,7,10,15,25,50], "RF__n_estimators": [5,10,20,40],"RF__max_features": [1,2,3]}, dftouse,lcols, u'results',1, reuse_split=reuse_split_new)
evaluate(pipeRF, "rand_forest")
print "#############====================== Extra Trees= =====================#############"
pipeET = Pipeline([('select', featSelector), ('ET', RandomForestClassifier())])
pipeET, _,_,_,_,_,_  = do_classify(pipeET, {"ET__max_depth": [3,5,7,10,15,25,50], "ET__n_estimators": [5,10,20,40],"ET__max_features": [1,2,3]}, dftouse,lcols, u'results',1, reuse_split=reuse_split_new)
evaluate(pipeET, "extra_trees")
print "#############====================== AdaBoost ==========================#############"
pipeAda = Pipeline([('select', featSelector), ('Ada', AdaBoostClassifier())])
pipeAda, _,_,_,_,_,_  = do_classify(pipeAda, {"Ada__n_estimators": [5,10,20,40],"Ada__learning_rate": [0.1,0.5,1.0]}, dftouse,lcols, u'results',1, reuse_split=reuse_split_new)
evaluate(pipeAda, "ada_boost")
print "#############====================== Gaussian NB ==========================#############"
pipeNB = Pipeline([('select', featSelector), ('NB', GaussianNB())])
pipeNB, _,_,_,_,_,_  = do_classify(pipeNB, {}, dftouse,lcols, u'results',1, reuse_split=reuse_split_new)
evaluate(pipeNB, "gaussian_nb")
print "#############====================== Gradient Boosting ====================#############"
pipeGB = Pipeline([('select', featSelector), ('GB', GradientBoostingClassifier())])
pipeGB, _,_,_,_,_,_  = do_classify(pipeGB, {"GB__n_estimators": [5,10,20,40],"GB__learning_rate": [0.1,0.5,1.0]}, dftouse,lcols, u'results',1, reuse_split=reuse_split_new)
evaluate(pipeGB, "gradient_boosting")


# ## Ensemble

# In[47]:

predsv.keys()


# In[48]:

for k in predsv_ROI_long:
    print k + ' ROI buy-only: {0:.2f}%'.format(predsv_ROI_long[k]*100)


# In[49]:

for k in predsv_ROI_longshort:
    print k + ' ROI buy-only: {0:.2f}%'.format(predsv_ROI_longshort[k]*100)


# In[50]:

predst.keys()


# In[51]:

for k in predst_ROI_long:
    print k + ' ROI buy-only: {0:.2f}%'.format(predst_ROI_long[k]*100)


# In[52]:

for k in predst_ROI_longshort:
    print k + ' ROI buy-only: {0:.2f}%'.format(predst_ROI_longshort[k]*100)


# In[53]:

dfensemble=pd.DataFrame.from_dict({'svm':predsv['svm'],
                                     'pipelinearsvm':predsv['pipelinearsvm'],
                                     'rbf_svc':predsv['rbf_svc'],
                                     'linear_svm':predsv['linear_svm'],
                                     'ada_boost':predsv['ada_boost'],
                                     'rand_forest':predsv['rand_forest'],
                                     'pipesvm2':predsv['pipesvm2'],
                                     'extra_trees':predsv['extra_trees'],
                                     'logistic_regression':predsv['logistic_regression'],
                                     'pipe_Log_Regr':predsv['pipe_Log_Regr'],
                                     'random_forest':predsv['random_forest'],
                                     'gaussian_nb':predsv['gaussian_nb'],
                                     'gradient_boosting':predsv['gradient_boosting'],
                                   'y':predsv['baseline_ema']})


# In[54]:

from sklearn.linear_model import LinearRegression
X = dfensemble.drop('y', axis = 1)
lm = LinearRegression()
valreg = lm.fit(X, dfensemble.y)


# In[55]:

dfensembletest=pd.DataFrame.from_dict({'svm':predst['svm'],
                                     'pipelinearsvm':predst['pipelinearsvm'],
                                     'rbf_svc':predst['rbf_svc'],
                                     'linear_svm':predst['linear_svm'],
                                     'ada_boost':predst['ada_boost'],
                                     'rand_forest':predst['rand_forest'],
                                     'pipesvm2':predst['pipesvm2'],
                                     'extra_trees':predst['extra_trees'],
                                     'logistic_regression':predst['logistic_regression'],
                                     'pipe_Log_Regr':predst['pipe_Log_Regr'],
                                     'random_forest':predst['random_forest'],
                                     'gaussian_nb':predst['gaussian_nb'],
                                     'gradient_boosting':predst['gradient_boosting'],
                                   'y':predst['baseline_ema']})


# In[56]:

Xt = dfensembletest.drop('y', axis = 1)
epreds = valreg.predict(Xt)


# ### Ensemble Results

# In[57]:

#Test
df_pred = df[maskt].reset_index(drop=True)
df_pred['pred_result'] = 1*(epreds>0.5)
df_pred.head()

balance, profit, ROI, balovertime, signals = evaluate_profit(df_pred, sdatet, edatet, 10000, 'pred_result', 'close', False, [0])
print 'ROI "pred" buy-only: {0:.2f}%'.format(ROI*100)
#predst[desc] = ypred
#predst_ROI_long[desc] = ROI

balance, profit, ROI, balovertime, signals = evaluate_profit(df_pred, sdatet, edatet, 10000, 'pred_result', 'close', False, [1])
print 'ROI "pred" buy-sell: {0:.2f}%'.format(ROI*100)
#predst_ROI_longshort[desc] = ROI

