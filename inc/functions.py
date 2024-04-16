# Functions File
import pandas as pd
import PyPDF2, os, re
## Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor 
from sklearn.neighbors import KNeighborsRegressor

# Modelling Helpers
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def return_r2(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)**0.5
    r2 = r2_score(y_test, y_pred)
    return r2, mse, mae, rmse

def regression_model_test(X_train, y_train, X_test, y_test):
    """
    """
    # Collect all R2 Scores.
    R2_Scores = []
    models = ['Linear Regression', 'Lasso Regression', 'AdaBoost Regression', 
            'Ridge Regression', 'GradientBoosting Regression',
            'RandomForest Regression', 'KNeighbours Regression']
    
    """ Linear Regression """
    clf_lr = LinearRegression()
    clf_lr.fit(X_train , y_train)
    accuracies = cross_val_score(estimator = clf_lr, 
                                 X = X_train, y = y_train, 
                                 cv = 5,verbose = 0)
    y_pred = clf_lr.predict(X_test)

    # Use function to return r2
    r2, mse, mae, rmse = return_r2(y_test, y_pred)

    # Append to R2_Scores
    print("Completed Linear Regression model...")
    R2_Scores.append(r2)

    """ Lasso Regression """
    clf_la = Lasso()
    clf_la.fit(X_train , y_train)
    accuracies = cross_val_score(estimator = clf_la, 
                                 X = X_train, y = y_train, 
                                 cv = 5,verbose = 0)
    y_pred = clf_la.predict(X_test)

    # Use function to return r2
    r2, mse, mae, rmse = return_r2(y_test, y_pred)

    # Append to R2_Scores
    print("Completed Lasso Regression model...")
    R2_Scores.append(r2)

    """ AdaBoostRegressor """
    clf_ar = AdaBoostRegressor(n_estimators=1000)
    clf_ar.fit(X_train , y_train)
    accuracies = cross_val_score(estimator = clf_ar, 
                                 X = X_train, y = y_train, 
                                 cv = 5, verbose = 0)
    y_pred = clf_ar.predict(X_test)

    # Use function to return r2
    r2, mse, mae, rmse = return_r2(y_test, y_pred)

    # Append to R2_Scores
    print("Completed AdaBoost Regression model...")
    R2_Scores.append(r2)

    """ Ridge Regression """
    clf_rr = Ridge()
    clf_rr.fit(X_train , y_train)
    accuracies = cross_val_score(estimator = clf_rr, 
                                 X = X_train, y = y_train, 
                                 cv = 5,verbose = 0)
    y_pred = clf_rr.predict(X_test)

    # Use function to return r2
    r2, mse, mae, rmse = return_r2(y_test, y_pred)

    # Append to R2_Scores
    print("Completed Ridge Regression model...")
    R2_Scores.append(r2)

    """ Gradient Boosting Regression """
    clf_gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                        max_depth=1, random_state=0, 
                                        loss='squared_error', verbose=0)
    clf_gbr.fit(X_train , y_train)
    accuracies = cross_val_score(estimator = clf_gbr, 
                                 X = X_train, y = y_train, 
                                 cv = 5,verbose = 0)
    y_pred = clf_gbr.predict(X_test)

    # Use function to return r2
    r2, mse, mae, rmse = return_r2(y_test, y_pred)

    # Append to R2_Scores
    print("Completed Gradient Boosting Regression model...")
    R2_Scores.append(r2)

    """ Random Forest """
    clf_rf = RandomForestRegressor()
    clf_rf.fit(X_train , y_train)
    accuracies = cross_val_score(estimator = clf_rf, 
                                 X = X_train, y = y_train, 
                                 cv = 5,verbose = 0)
    y_pred = clf_rf.predict(X_test)

    # Fine Tune Random Forest
    no_of_test=[100]
    params_dict={'n_estimators':no_of_test,'n_jobs':[-1],
                 'max_features':["auto",'sqrt','log2']}
    clf_rf=GridSearchCV(estimator=RandomForestRegressor(), 
                        param_grid=params_dict,scoring='r2')
    clf_rf.fit(X_train,y_train)

    pred=clf_rf.predict(X_test)

    # Use function to return r2
    r2, mse, mae, rmse = return_r2(y_test, pred)

    # Append to R2_Scores
    print("Completed Random Forest Regression model...")
    R2_Scores.append(r2)

    """ KNeighbors Regression """
    clf_knn = KNeighborsRegressor()
    clf_knn.fit(X_train , y_train)
    accuracies = cross_val_score(estimator = clf_knn, 
                                 X = X_train, y = y_train, 
                                 cv = 5,verbose = 0)
    y_pred = clf_knn.predict(X_test)

    # Fine Tune KNeighbors
    n_neighbors=[]
    for i in range (0,50,5):
        if(i!=0):
            n_neighbors.append(i)
    params_dict={'n_neighbors':n_neighbors,'n_jobs':[-1]}
    clf_knn=GridSearchCV(estimator=KNeighborsRegressor(), 
                         param_grid=params_dict,scoring='r2')
    clf_knn.fit(X_train,y_train)

    pred=clf_knn.predict(X_test)

    # Use function to return r2
    r2, mse, mae, rmse = return_r2(y_test, pred)

    # Append to R2_Scores
    print("Completed KNeighbors Regression model...")
    R2_Scores.append(r2)

    # Return Results
    print("---------------------")
    print("Finalizing results...")
    compare = pd.DataFrame({'Algorithms' : models , 'R2-Scores' : R2_Scores})
    
    return compare.sort_values(by='R2-Scores' ,ascending=False)

def get_diamond_info():
    """
    Returns cleaned_text as letters and numbers from a list of pdfs in 
    resources/pdfs folder
    """
    pdf_list = os.listdir('resources/pdfs')
    text = ''

    for pdf in pdf_list:
    # Open the PDF file in binary mode
        with open(f'resources/pdfs/{pdf}', 'rb') as file:
            # Create a PDF file reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Loop through each page in the PDF
            for page_num in range(len(pdf_reader.pages)):
                # Get the page object
                page = pdf_reader.pages[page_num]
                
                # Extract text from the page
                text += page.extract_text()

        text += "\n"

    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    return cleaned_text