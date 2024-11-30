import io
import pandas as pd
import base64
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def preprocess(df):
    #Do label encoding for the low medium and high variables
    lowMedHighVars = ["Parental_Involvement", "Access_to_Resources", "Motivation_Level", "Family_Income", "Teacher_Quality"]
    labels={'Low': 1, 'Medium': 2, 'High': 3}
    for cat in range(len(lowMedHighVars)):
        df[lowMedHighVars[cat]]=df[lowMedHighVars[cat]].map(labels)

    #Do One-Hot Encoding (drop the first) for Yes/No variables
    df = pd.get_dummies(df,columns=['Extracurricular_Activities'],drop_first=True)
    df = pd.get_dummies(df,columns=['Internet_Access'],drop_first=True)
    df = pd.get_dummies(df,columns=['Learning_Disabilities'],drop_first=True)

    #Do label encoding for positive, negative, neutral Peer_Influence
    labels={'Positive': 1, 'Negative': -1, 'Neutral': 0}
    df["Peer_Influence"] = df["Peer_Influence"].map(labels)

    #Do One-Hot encoding for School_Type_Public (public -> 1, private -> 0)
    df = pd.get_dummies(df,columns=['School_Type'],drop_first=True)

    #Do One-Hot encoding for Gender_Male (Male -> 1, Female -> 0)
    df = pd.get_dummies(df,columns=['Gender'],drop_first=True)

    #Do label encoding for distance_from_home
    labels={'Near': 1, 'Moderate': 2, 'Far': 3}
    df["Distance_from_Home"] = df["Distance_from_Home"].map(labels)

    #Do label encoding for Parent_Education_Level
    labels={'High School': 1, 'College': 2, 'Postgraduate': 3}
    df["Parental_Education_Level"] = df["Parental_Education_Level"].map(labels)

    df["Parental_Education_Level"] = df["Parental_Education_Level"].fillna(df["Parental_Education_Level"].mode()[0])
    df["Teacher_Quality"] = df["Teacher_Quality"].fillna(df["Teacher_Quality"].mode()[0])
    df["Distance_from_Home"] = df["Distance_from_Home"].fillna(df["Distance_from_Home"].mode()[0])
    
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
    
def gradBoostRegr(df, targetVar):
    y = df[targetVar]
    X = df.drop(columns=targetVar, inplace=False)

    numeric_features = df.select_dtypes(include='number')
    categorical_features = df.select_dtypes(include='object')
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
        ('feature_selection', SelectKBest(f_regression, k=21)),
        ('model', bagging_model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)