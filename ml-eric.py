
# coding: utf-8

# # Are We Rich Yet?

# In[1]:

get_ipython().magic(u'matplotlib inline')

get_ipython().magic(u'run talibref.py')


# ### Get data

# In[2]:

#df=pd.read_csv("data/IYZ.csv")
ticker = 'IYZ'
startdate=datetime.date(2009, 1, 1)
enddate=datetime.date.today()
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

dftouse['date'] = pd.to_datetime(dftouse['date'])
mask = (dftouse.date < '2015-01-01').values
mask.shape, mask.sum()

#mask = dftouse['date'] < datetime.date(2014, 1, 1)
#maskv = (dftouse['date'] > datetime.date(2013, 12, 31)) & (dftouse['date'] < datetime.date(2015, 1, 1))
#maskt = dftouse['date'] > datetime.date(2014, 12, 31)


# ### Check ROI of signals, alone:

# In[10]:

#print "ROI baseline: 1.12%"
#print 'ROI "result" buy-only: 67.45%'
#print 'ROI "result" buy-sell: 172.49%'
    
def signalperf(signal):
    ypred = df[signal]
    df_pred = df[~mask].reset_index(drop=True)
    df_pred['pred_result'] = ypred
    df_pred['result_baseline'] = np.ones(df_pred.shape[0])

    balance, profit, ROI2, balovertime = evaluate_profit(df_pred, startdate, enddate, 10000, 'pred_result', 'close', False)
    #print 'ROI "pred" buy-only: {0:.2f}%'.format(ROI*100)

    balance, profit, ROI, balovertime = evaluate_profit2(df_pred, startdate, enddate, 10000, 'pred_result', 'close', False)
    print signal + ': ROI "pred" buy-only: {0:.2f}%'.format(ROI2*100), 'buy-sell: {0:.2f}%'.format(ROI*100)
    return


# In[11]:

for v in INDICATORS:
    signalperf(v)


# #### 1.2 Standardize the data

# Use the mask to compute the training and test parts of the dataframe. Use `StandardScaler` from `sklearn.preprocessing` to "fit" the columns in `STANDARDIZABLE` on the training set. Then use the resultant estimator to transform both the training and the test parts of each of the columns in the dataframe, replacing the old unstandardized values in the `STANDARDIZABLE` columns of `dftouse` by the new standardized ones.

# In[12]:

#your code here
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(dftouse[mask][STANDARDIZABLE])
dftouse[STANDARDIZABLE] = scaler.transform(dftouse[STANDARDIZABLE])
dftouse.head()


# We create a list `lcols` of the columns we will use in our classifier. This list should not contain the response `RESP`. How many features do we have?

# In[13]:

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

# In[14]:

ccols=[]
for c in lcols:
    if c not in INDICATORS and c not in IGNORE:
        ccols.append(c)
print len(ccols), len(INDICATORS)
ccols


# #### 1.4 Train a SVM on this data.

# In[15]:

from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV


# In[16]:

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


# In[17]:

from sklearn.metrics import confusion_matrix
def do_classify(clf, parameters, indf, featurenames, targetname, target1val, mask=None, reuse_split=None, score_func=None, n_folds=5):
    subdf=indf[featurenames]
    X=subdf.values
    y=(indf[targetname].values==target1val)*1
    if mask !=None:
        print "using mask"
        Xtrain, Xtest, ytrain, ytest = X[mask], X[~mask], y[mask], y[~mask]
    if reuse_split !=None:
        print "using reuse split"
        Xtrain, Xtest, ytrain, ytest = reuse_split['Xtrain'], reuse_split['Xtest'], reuse_split['ytrain'], reuse_split['ytest']
    if parameters:
        clf = cv_optimize(clf, parameters, Xtrain, ytrain, n_folds=n_folds, score_func=score_func)
    clf=clf.fit(Xtrain, ytrain)
    training_accuracy = clf.score(Xtrain, ytrain)
    test_accuracy = clf.score(Xtest, ytest)
    print "############# based on standard predict ################"
    print "Accuracy on training data: %0.2f" % (training_accuracy)
    print "Accuracy on test data:     %0.2f" % (test_accuracy)
    print confusion_matrix(ytest, clf.predict(Xtest))
    print "########################################################"
    return clf, Xtrain, ytrain, Xtest, ytest


# In[18]:

get_ipython().run_cell_magic(u'time', u'', u'clfsvm, Xtrain, ytrain, Xtest, ytest = do_classify(LinearSVC(loss="hinge"), {"C": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]}, dftouse, lcols, u\'result_14\',1, mask=mask)')


# In[19]:

clfsvm


# In[20]:

preds={}
preds_ROI_long={}
preds_ROI_longshort={}


# In[21]:

df_pred = df[~mask].reset_index(drop=True)
df_pred['baseline'] = np.ones(df_pred.shape[0])

balance, profit, ROI, balovertime = evaluate_profit(df_pred, startdate, enddate, 10000, 'baseline', 'close', False)
print "ROI baseline: {0:.2f}%".format(ROI*100)
preds["baseline_long"] = df_pred['baseline']
preds_ROI_long["baseline_long"] = ROI
preds_ROI_longshort["baseline_long"] = ROI

balance, profit, ROI, balovertime = evaluate_profit(df_pred, startdate, enddate, 10000, 'results', 'close', False)
print 'ROI "result" buy-only: {0:.2f}%'.format(ROI*100)
preds["baseline_ema"] = df_pred['results']
preds_ROI_long["baseline_ema"] = ROI
balance, profit, ROI, balovertime = evaluate_profit2(df_pred, startdate, enddate, 10000, 'results', 'close', False)
print 'ROI "result" buy-sell: {0:.2f}%'.format(ROI*100)
preds_ROI_longshort["baseline_ema"] = ROI

balance, profit, ROI, balovertime = evaluate_profit(df_pred, startdate, enddate, 10000, 'result_1', 'close', False)
print 'ROI "result" buy-only: {0:.2f}%'.format(ROI*100)
preds["baseline_max"] = df_pred['result_1']
preds_ROI_long["baseline_max"] = ROI
balance, profit, ROI, balovertime = evaluate_profit2(df_pred, startdate, enddate, 10000, 'result_1', 'close', False)
print 'ROI "result" buy-sell: {0:.2f}%'.format(ROI*100)
preds_ROI_longshort["baseline_max"] = ROI


# In[22]:

def evaluate(clf, desc):

    ypred = clf.predict(Xtest)
    df_pred = df[~mask].reset_index(drop=True)
    df_pred['pred_result'] = ypred
    df_pred['result_baseline'] = np.ones(df_pred.shape[0])
    print "accuracy on test set: {0:.3f}".format((df_pred.result_14 == df_pred.pred_result).sum()/float(len(df_pred)))
    
    balance, profit, ROI, balovertime = evaluate_profit(df_pred, startdate, enddate, 10000, 'pred_result', 'close', False)
    print 'ROI "pred" buy-only: {0:.2f}%'.format(ROI*100)
    preds[desc] = ypred
    preds_ROI_long[desc] = ROI

    balance, profit, ROI, balovertime = evaluate_profit2(df_pred, startdate, enddate, 10000, 'pred_result', 'close', False)
    print 'ROI "pred" buy-sell: {0:.2f}%'.format(ROI*100)
    preds_ROI_longshort[desc] = ROI


# In[23]:

evaluate(clfsvm, "svm")


# The results obtained ought to be very similar to the efforts you put in earlier. If not its likely you wrote `cv_optimize` wrong. (Remember that we are using the same mask).
# 
# We'll reuse the training and test sets you computed above later in the homework. We do this by putting them into a dictionary `reuse_split`

# In[24]:

reuse_split=dict(Xtrain=Xtrain, Xtest=Xtest, ytrain=ytrain, ytest=ytest)


# ## 2. Estimate costs and benefits from assumptions and data

# ### Our data is highly asymmetric

# First notice that our data set is very highly asymmetric, with positive `RESP`onses only making up 16-17% of the samples.

# In[25]:

print "whole data set", dftouse['result_14'].mean()#Highly asymmetric
print "training set", dftouse['result_14'][mask].mean(), "test set", dftouse['result_14'][~mask].mean()


# This means that a classifier which predicts that EVERY customer is a negative has an accuracy rate of 83-84%. By this we mean that **a classifier that predicts that no customer will respond to our mailing** has an accuracy of 83-84%!

# In[26]:

#your code here
from sklearn.linear_model import LogisticRegression
clflog = LogisticRegression(penalty="l1")
parameters = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
clflog, Xtrain, ytrain, Xtest, ytest = do_classify(clflog, parameters, dftouse, lcols, u'result_14',1, mask=mask, reuse_split=reuse_split)


# In[27]:

clflog


# In[28]:

evaluate(clflog, "logistic_regression")


# In[29]:

from sklearn.ensemble import RandomForestClassifier
clfraf = RandomForestClassifier(n_jobs=2)
parameters = {}#{"C": [0.001, 0.01, 0.1, 1, 10, 100]}
clfraf, Xtrain, ytrain, Xtest, ytest = do_classify(clfraf, parameters, dftouse, lcols, u'result_14',1, mask=mask, reuse_split=reuse_split)


# In[30]:

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

# In[31]:

def nonzero_lasso(clf):
    featuremask=(clf.coef_ !=0.0)[0]
    return pd.DataFrame(dict(feature=lcols, coef=clf.coef_[0], abscoef=np.abs(clf.coef_[0])))[featuremask].sort('abscoef', ascending=False)


# In[32]:

lasso_importances=nonzero_lasso(clflog)
lasso_importances.set_index("feature", inplace=True)
lasso_importances.head(10)


# #### 4.1 Feature importance using correlations

# We can also get a notion of which features are important in the classification process by seeing how they correlate with the response. Implement some code to obtain the Pearson correlation coefficient between each of our features and the response. Do this on the training set only! Create a dataframe indexed by the features, which has columns `abscorr` the absolute value of the correlation and `corr` the value of the correlation. Sort the dataframe by `abscorr`, highest first, and show the top 25 features with the highest absolute correlation. Is there much overlap with the feature selection performed by the LASSO?

# In[33]:

from scipy.stats.stats import pearsonr
correlations=[]
dftousetrain=dftouse[mask]
for col in lcols:
    r=pearsonr(dftousetrain[col], dftousetrain['result_14'])[0]
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

# In[34]:

from sklearn import feature_selection
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest


# Lets define a scorer which returns the absolute values of the pearson correlation between the feature and the response for each sample. The specific form of the scorer is dictated to us in the API docs for `SelectKBest`, see [here](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html): the first argument must be an array of scores, and the second an array of p-values.

# In[35]:

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

# In[36]:

selectorlinearsvm = SelectKBest(k=25, score_func=pearson_scorer)
pipelinearsvm = Pipeline([('select', selectorlinearsvm), ('svm', LinearSVC(loss="hinge"))])


# #### Let us run the pipelined classifier 

# We'll run the classifier and compare the results using the ROC curve to the previous SVM result.

# In[37]:

pipelinearsvm, _,_,_,_  = do_classify(pipelinearsvm, {"svm__C": [0.00001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}, dftouse,lcols, u'result_14',1, reuse_split=reuse_split)


# What features did the pipelined classifier use? We can access them so:

# In[38]:

np.array(lcols)[pipelinearsvm.get_params()['select'].get_support()]


# We plot the ROC curves, using the label `svm-feature-selected` for the pipelined classifier `pipelinearsvm`. We plot it alongside the older logistic-with lasso and all-features SVM for comparison

# In[39]:

evaluate(pipelinearsvm, "pipelinearsvm")


# ### Balancing train set to test set for training.

# In[40]:

jtrain=np.arange(0, ytrain.shape[0])
n_pos=len(jtrain[ytrain==1])
n_neg=len(jtrain[ytrain==0])
print n_pos, n_neg


# There are many more negative samples in the training set. We want to balance the negative samples to the positive samples. So lets sample $n_{+}$ samples from the negative samples in the training set (without replacement).

# In[41]:

ineg = np.random.choice(jtrain[ytrain==0], 500, replace=False)#n_pos, replace=False)


# We concatenate all the indexes and use them to select a new training set from the old one.

# In[42]:

alli=np.concatenate((jtrain[ytrain==1], ineg))
alli.shape


# In[43]:

Xtrain_new = Xtrain[alli]
ytrain_new = ytrain[alli]
Xtrain_new.shape, ytrain_new.shape


# We store these into a new split variable `reuse_split_new`.

# In[44]:

reuse_split_new=dict(Xtrain=Xtrain_new, Xtest=Xtest, ytrain=ytrain_new, ytest=ytest)


# Note that the test sets are identical as before. This is as, even though we are training the SVM classifier in the "naturally" unfound situation of balanced classes, we **must test it in the real-world scenario of imbalance**.

# #### 4.2 Train a linear SVM on this balanced set

# Train a non-feature-selected linear SVM on this new balanced set as a comparison to both our old SVM on the imbalanced data set `clfsvm` and the feature-selected linear SVM `pipelinearsvm`. Store this new classifier in the variable `clfsvm_b`.
# 
# Compare the performances of all three of these classifiers using the roc curve plot, with the new `clfsvm_b` labeled as `svm-all-features-balanced`. 

# In[45]:

#your code here
clfsvm_b = Pipeline([('select', selectorlinearsvm), ('svm', LinearSVC(loss="hinge"))])
clfsvm_b, _,_,_,_  = do_classify(clfsvm_b, {"svm__C": [0.00001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}, dftouse,lcols, u'result_14',1, reuse_split=reuse_split_new)


# In[46]:

clfsvm_b


# In[47]:

evaluate(clfsvm_b, "svm_balanced")


# #### 4.3 Implement a RBF based pipelined (feature-selected) classifier on the balanced set.

# In[48]:

from sklearn.svm import SVC


# In[49]:

get_ipython().run_cell_magic(u'time', u'', u'selectorsvm2 = SelectKBest(k=25, score_func=pearson_scorer)\npipesvm2 = Pipeline([(\'select2\', selectorsvm2), (\'svm2\', SVC())])\npipesvm2, _,_,_,_  = do_classify(pipesvm2, {"svm2__C": [1e8, 1e9, 1e10], "svm2__gamma": [ 1e-9, 1e-10, 1e-11]}, dftouse,lcols, u\'result_14\',1, reuse_split=reuse_split_new)')


# In[50]:

pipesvm2.get_params()


# In[51]:

evaluate(pipesvm2, "pipesvm2")


# ## Yvan ML

# In[52]:

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
pipeLR, _,_,_,_  = do_classify(pipeLR, {"LR__C": [0.005, 0.01, 0.02, 0.05, 10.0]}, dftouse,lcols, u'results',1, reuse_split=reuse_split_new)
evaluate(pipeLR, "pipe_Log_Regr")
print "#############====================== Linear SVM ========================#############"
clfsvm_b = Pipeline([('select', featSelector), ('svm', LinearSVC(loss="hinge"))])
clfsvm_b, _,_,_,_  = do_classify(clfsvm_b, {"svm__C": [0.00001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}, dftouse,lcols, u'results',1, reuse_split=reuse_split_new)
evaluate(clfsvm_b, "linear_svm")
print "#############====================== RBF SVC ===========================#############"
pipesvm2 = Pipeline([('select', featSelector), ('svm2', SVC())])
pipesvm2, _,_,_,_  = do_classify(pipesvm2, {"svm2__C": [1e8, 1e9, 1e10], "svm2__gamma": [ 1e-9, 1e-10, 1e-11]}, dftouse,lcols, u'results',1, reuse_split=reuse_split_new)
evaluate(pipesvm2, "rbf_svc")
print "#############====================== Random Forest =====================#############"
pipeRF = Pipeline([('select', featSelector), ('RF', RandomForestClassifier())])
pipeRF, _,_,_,_  = do_classify(pipeRF, {"RF__max_depth": [3,5,7,10,15,25,50], "RF__n_estimators": [5,10,20,40],"RF__max_features": [1,2,3]}, dftouse,lcols, u'results',1, reuse_split=reuse_split_new)
evaluate(pipeRF, "rand_forest")
print "#############====================== Extra Trees= =====================#############"
pipeET = Pipeline([('select', featSelector), ('ET', RandomForestClassifier())])
pipeET, _,_,_,_  = do_classify(pipeET, {"ET__max_depth": [3,5,7,10,15,25,50], "ET__n_estimators": [5,10,20,40],"ET__max_features": [1,2,3]}, dftouse,lcols, u'results',1, reuse_split=reuse_split_new)
evaluate(pipeET, "extra_trees")
print "#############====================== AdaBoost ==========================#############"
pipeAda = Pipeline([('select', featSelector), ('Ada', AdaBoostClassifier())])
pipeAda, _,_,_,_  = do_classify(pipeAda, {"Ada__n_estimators": [5,10,20,40],"Ada__learning_rate": [0.1,0.5,1.0]}, dftouse,lcols, u'results',1, reuse_split=reuse_split_new)
evaluate(pipeAda, "ada_boost")
print "#############====================== Gaussian NB ==========================#############"
pipeNB = Pipeline([('select', featSelector), ('NB', GaussianNB())])
pipeNB, _,_,_,_  = do_classify(pipeNB, {}, dftouse,lcols, u'results',1, reuse_split=reuse_split_new)
evaluate(pipeNB, "gaussian_nb")
print "#############====================== Gradient Boosting ====================#############"
pipeGB = Pipeline([('select', featSelector), ('GB', GradientBoostingClassifier())])
pipeGB, _,_,_,_  = do_classify(pipeGB, {"GB__n_estimators": [5,10,20,40],"GB__learning_rate": [0.1,0.5,1.0]}, dftouse,lcols, u'results',1, reuse_split=reuse_split_new)
evaluate(pipeGB, "gradient_boosting")


# ## Ensemble

# In[53]:

preds.keys()


# In[54]:

dfensemble=pd.DataFrame.from_dict({'svm':preds['svm'],
                                     'pipelinearsvm':preds['pipelinearsvm'],
                                     'rbf_svc':preds['rbf_svc'],
                                     'linear_svm':preds['linear_svm'],
                                     'ada_boost':preds['ada_boost'],
                                     'rand_forest':preds['rand_forest'],
                                     'pipesvm2':preds['pipesvm2'],
                                     'extra_trees':preds['extra_trees'],
                                     'logistic_regression':preds['logistic_regression'],
                                     'pipe_Log_Regr':preds['pipe_Log_Regr'],
                                     'random_forest':preds['random_forest'],
                                     'gaussian_nb':preds['gaussian_nb'],
                                     'svm_balanced':preds['svm_balanced'],
                                     'gradient_boosting':preds['gradient_boosting'],
                                   'y':preds['baseline_ema']})


# In[55]:

from sklearn.linear_model import LinearRegression
X = dfensemble.drop('y', axis = 1)
lm = LinearRegression()
valreg = lm.fit(X, dfensemble.y)


# In[56]:

#Xt = dfensembletest.drop('y', axis = 1)
#epreds = valreg.predict(Xt)


# In[ ]:



