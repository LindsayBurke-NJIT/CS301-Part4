import io
import pandas as pd
import base64
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def preprocess(df):
    for col in df.select_dtypes(include='object').columns:
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)
    
    for col in df.select_dtypes(include='number').columns:
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)
    return df

def parseDf(filename, contents):
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    if filename.endswith('.csv'):
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    elif filename.endswith('.xls') or filename.endswith('.xlsx'):
        df = pd.read_excel(io.BytesIO(decoded))
    else:
        return None

    return df
    
def getPipeline(X, y):
    numeric_features = X.select_dtypes(include='number').columns
    categorical_features = X.select_dtypes(include='object').columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    gradBoost =  GradientBoostingRegressor(random_state=42)
    bagging_model = BaggingRegressor(estimator=gradBoost, n_estimators=20, random_state=42)

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', bagging_model)
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)
    return pipe, X_test, y_test

def gradBoostRegr(df, targetVar, checklistVals):
    if(checklistVals==[]):
        return None
    
    y = df[targetVar]
    X = df[checklistVals]

    pipe, X_test, y_test = getPipeline(X, y)
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    return r2