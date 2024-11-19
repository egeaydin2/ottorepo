import numpy as np
import os
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from optbinning import BinningProcess
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_predict
from datetime import datetime
import json
import joblib
import cx_Oracle
from dash import dash_table
from optuna.integration import OptunaSearchCV
#os.chdir("C:/Oracle/instantclient_21_3")

# Oracle SQL connection details
user = "A68699"
pw = "Ta9263J8"
dsn = cx_Oracle.makedsn(host='ddm_edw.finansbank.com.tr', port=9522, service_name='enduser')

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout of the app with 10 tabs on the left side
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Nav([
                dbc.NavLink("Main", href="/tab-1", id="tab-1-link", className="text-center"),
                dbc.NavLink("Data Import & Binning", href="/tab-2", id="tab-2-link", className="text-center"),
                dbc.NavLink("Feature Elimination", href="/tab-3", id="tab-3-link", className="text-center"),
                dbc.NavLink("Algorithm Lab", href="/tab-4", id="tab-4-link", className="text-center"),
                dbc.NavLink("Algorithm Lab II", href="/tab-5", id="tab-5-link", className="text-center")
            ] + [dbc.NavLink(f"Tab {i}", href=f"/tab-{i}", id=f"tab-{i}-link", className="text-center") for i in range(6, 11)], vertical=True, pills=True)
        ], width=2, style={'backgroundColor': '#f8f9fa'}),
        dbc.Col([
            dcc.Location(id='url', refresh=False),
            html.Div(id='page-content')
        ], width=10)
    ])
], fluid=True)


# Callback to update page content based on selected tab
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    search_methods = [
        {'label': 'Randomized Search', 'value': 'randomized'},
        {'label': 'Grid Search', 'value': 'grid'},
        {'label': 'Optuna Search', 'value': 'optuna'},
        {'label': 'Bayes Search', 'value': 'bayes'}
    ]
    if pathname == '/tab-1':
        return html.Div([
            html.H1("AI Factory", className="text-center", style={'fontSize': '5em', 'color': 'black'}),
            html.P("Welcome to the AI Factory. Navigate through the tabs to explore various features.", className="text-center", style={'fontSize': '2em', 'color': 'black'})
        ], style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'alignItems': 'center', 'height': '100vh', 'backgroundColor': 'lightblue'})
    elif pathname == '/tab-2':
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("AI Factory Modeling Pipeline", className="text-center text-primary mb-4"),
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H2("Train and Test Datasets"),
                    dbc.Input(id="train-query", placeholder="Enter SQL query for train dataset", type="text", className="mb-2"),
                    dbc.Input(id="test-query", placeholder="Enter SQL query for test dataset", type="text", className="mb-2"),
                    dbc.Input(id="target-column", placeholder="Enter target column number (0-indexed)", type="number", className="mb-2"),
                    dbc.Button("Load Datasets", id="load-datasets-button", color="primary", className="mt-2", style={"backgroundColor": "#4B0082"}),
                    html.Div(id="dataset-status", className="mt-2")
                ], width=6, className="offset-md-3")
            ]),
            dbc.Row([
                dbc.Col([
                    html.H2("Feature Elimination"),
                    dbc.Input(id="missing-rate-threshold", placeholder="Enter missing rate threshold (0-100)", type="number", className="mb-2"),
                    dbc.Button("Eliminate Features", id="eliminate-features-button", color="primary", className="mt-2", style={"backgroundColor": "#4B0082"}),
                    html.Div(id="elimination-status", className="mt-2")
                ], width=6, className="offset-md-3")
            ]),
            dbc.Row([
                dbc.Col([
                    html.H2("Binning Process"),
                    dbc.Input(id="max-n-prebins", placeholder="Enter max number of prebins", type="number", className="mb-2"),
                    dbc.Input(id="min-bin-size", placeholder="Enter min bin size", type="number", className="mb-2"),
                    dbc.Input(id="max-n-bins", placeholder="Enter max number of bins", type="number", className="mb-2"),
                    dbc.Button("Start Binning", id="start-binning-button", color="primary", className="mt-2", style={"backgroundColor": "#4B0082"}),
                    html.Div(id="binning-status", className="mt-2")
                ], width=6, className="offset-md-3")
            ])
        ], fluid=True, style={"backgroundColor": "#ADD8E6", "height": "100vh"}) # Light blue background color
    elif pathname == '/tab-3':
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H2("Feature Elimination Based on Correlation"),
                    dbc.Input(id="correlation-threshold", placeholder="Enter correlation threshold (0-1)", type="number", className="mb-2"),
                    dbc.Button("Eliminate Features", id="eliminate-features-button-tab3", color="primary", className="mt-2", style={"backgroundColor": "#4B0082"}),
                    html.Div(id="elimination-status-tab3", className="mt-2")
                ], width=6, className="offset-md-3")
            ])
        ], fluid=True, style={"backgroundColor": "#ADD8E6", "height": "100vh"})
    elif pathname == '/tab-4':
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H2("Random Forest"),
                    dbc.Input(id="rf-n-estimators", placeholder="Enter number of estimators", type="number", className="mb-2", style={"width": "50%"}),
                    dbc.Input(id="rf-random-state", placeholder="Enter random state", type="number", className="mb-2", style={"width": "50%"}),
                    dcc.Dropdown(id="rf-search-method", options=search_methods, placeholder="Select search method", className="mb-2", style={"width": "50%"}),
                    dbc.Button("Train Random Forest", id="train-rf-button", color="primary", className="mt-2", style={"backgroundColor": "#4B0082"}),
                    html.Div(id="rf-gini", className="mt-2")
                ], width=6, className="offset-md-3")
            ]),
            dbc.Row([
                dbc.Col([
                    html.H2("Gradient Boosting"),
                    dbc.Input(id="gb-n-estimators", placeholder="Enter number of estimators", type="number", className="mb-2", style={"width": "50%"}),
                    dbc.Input(id="gb-random-state", placeholder="Enter random state", type="number", className="mb-2", style={"width": "50%"}),
                    dcc.Dropdown(id="gb-search-method", options=search_methods, placeholder="Select search method", className="mb-2", style={"width": "50%"}),
                    dbc.Button("Train Gradient Boosting", id="train-gb-button", color="primary", className="mt-2", style={"backgroundColor": "#4B0082"}),
                    html.Div(id="gb-gini", className="mt-2")
                ], width=6, className="offset-md-3")
            ]),
            dbc.Row([
                dbc.Col([
                    html.H2("Logistic Regression"),
                    dbc.Input(id="lr-random-state", placeholder="Enter random state", type="number", className="mb-2", style={"width": "50%"}),
                    dbc.Input(id="lr-max-iter", placeholder="Enter max iterations", type="number", className="mb-2", style={"width": "50%"}),
                    dcc.Dropdown(id="lr-search-method", options=search_methods, placeholder="Select search method", className="mb-2", style={"width": "50%"}),
                    dbc.Button("Train Logistic Regression", id="train-lr-button", color="primary", className="mt-2", style={"backgroundColor": "#4B0082"}),
                    html.Div(id="lr-gini", className="mt-2")
                ], width=6, className="offset-md-3")
            ]),
            dbc.Row([
                dbc.Col([
                    html.H2("Decision Tree"),
                    dbc.Input(id="dt-max-depth", placeholder="Enter max depth", type="number", className="mb-2", style={"width": "50%"}),
                    dbc.Input(id="dt-random-state", placeholder="Enter random state", type="number", className="mb-2", style={"width": "50%"}),
                    dcc.Dropdown(id="dt-search-method", options=search_methods, placeholder="Select search method", className="mb-2", style={"width": "50%"}),
                    dbc.Button("Train Decision Tree", id="train-dt-button", color="primary", className="mt-2", style={"backgroundColor": "#4B0082"}),
                    html.Div(id="dt-gini", className="mt-2")
                ], width=6, className="offset-md-3")
            ]),
            dbc.Row([
                dbc.Col([
                    html.H2("LightGBM"),
                    dbc.Input(id="lgbm-n-estimators", placeholder="Enter number of estimators", type="number", className="mb-2", style={"width": "50%"}),
                    dbc.Input(id="lgbm-random-state", placeholder="Enter random state", type="number", className="mb-2", style={"width": "50%"}),
                    dcc.Dropdown(id="lgbm-search-method", options=search_methods, placeholder="Select search method", className="mb-2", style={"width": "50%"}),
                    dbc.Button("Train LightGBM", id="train-lgbm-button", color="primary", className="mt-2", style={"backgroundColor": "#4B0082"}),
                    html.Div(id="lgbm-gini", className="mt-2")
                ], width=6, className="offset-md-3")
            ]),
            dbc.Row([
                dbc.Col([
                    html.H2("MLP"),
                    dbc.Input(id="mlp-hidden-layers", placeholder="Enter hidden layer sizes", type="text", className="mb-2", style={"width": "50%"}),
                    dbc.Input(id="mlp-random-state", placeholder="Enter random state", type="number", className="mb-2", style={"width": "50%"}),
                    dcc.Dropdown(id="mlp-search-method", options=search_methods, placeholder="Select search method", className="mb-2", style={"width": "50%"}),
                    dbc.Button("Train MLP", id="train-mlp-button", color="primary", className="mt-2", style={"backgroundColor": "#4B0082"}),
                    html.Div(id="mlp-gini", className="mt-2")
                ], width=6, className="offset-md-3")
            ]),
            dbc.Row([
                dbc.Col([
                    html.H2("AdaBoost"),
                    dbc.Input(id="adaboost-n-estimators", placeholder="Enter number of estimators", type="number", className="mb-2", style={"width": "50%"}),
                    dbc.Input(id="adaboost-learning-rate", placeholder="Enter learning rate", type="number", className="mb-2", style={"width": "50%"}),
                    dbc.Input(id="adaboost-random-state", placeholder="Enter random state", type="number", className="mb-2", style={"width": "50%"}),
                    dcc.Dropdown(id="adaboost-search-method", options=search_methods, placeholder="Select search method", className="mb-2", style={"width": "50%"}),
                    dbc.Button("Train AdaBoost", id="train-adaboost-button", color="primary", className="mt-2", style={"backgroundColor": "#4B0082"}),
                    html.Div(id="adaboost-gini", className="mt-2")
                ], width=6, className="offset-md-3")
            ]),
            dbc.Row([
                dbc.Col([
                    html.H2("Ensemble Method"),
                    dbc.Input(id="rf-n-estimators", placeholder="Enter RF number of estimators", type="number", className="mb-2", style={"width": "50%"}),
                    dbc.Input(id="rf-max-depth", placeholder="Enter RF max depth", type="number", className="mb-2", style={"width": "50%"}),
                    dbc.Input(id="gb-n-estimators", placeholder="Enter GB number of estimators", type="number", className="mb-2", style={"width": "50%"}),
                    dbc.Input(id="gb-learning-rate", placeholder="Enter GB learning rate", type="number", className="mb-2", style={"width": "50%"}),
                    dbc.Input(id="ada-n-estimators", placeholder="Enter AdaBoost number of estimators", type="number", className="mb-2", style={"width": "50%"}),
                    dbc.Input(id="ada-learning-rate", placeholder="Enter AdaBoost learning rate", type="number", className="mb-2", style={"width": "50%"}),
                    dbc.Input(id="ensemble-voting", placeholder="Enter voting method (hard/soft)", type="text", className="mb-2", style={"width": "50%"}),
                    dbc.Input(id="ensemble-weights", placeholder="Enter weights (comma-separated)", type="text", className="mb-2", style={"width": "50%"}),
                    dcc.Dropdown(id="ensemble-search-method", options=search_methods, placeholder="Select search method", className="mb-2", style={"width": "50%"}),
                    dbc.Button("Train Ensemble", id="train-ensemble-button", color="primary", className="mt-2", style={"backgroundColor": "#4B0082"}),
                    html.Div(id="ensemble-gini", className="mt-2")
                ], width=6, className="offset-md-3")
            ]),
            dbc.Row([
                dbc.Col([
                    html.H2("Auto ML"),
                    dbc.Input(id="time-left-for-task", placeholder="Enter time left for task (seconds)", type="number", className="mb-2", style={"width": "75%"}),
                    dbc.Input(id="per-run-time-limit", placeholder="Enter per run time limit (seconds)", type="number", className="mb-2", style={"width": "75%"}),
                    dbc.Button("Run Auto ML", id="run-auto-ml-button", color="primary", className="mt-2", style={"backgroundColor": "#4B0082"}),
                    html.Div(id="auto-ml-status", className="mt-2")
                ], width=6, className="offset-md-3")
            ])
        ], fluid=True, style={"backgroundColor": "#ADD8E6", "height": "250vh"})
    elif pathname == '/tab-5':
        return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Button("Refresh Results", id="refresh-results-button", color="primary", className="mt-2", style={"backgroundColor": "#4B0082"}),
                html.Div(id="model-results", className="mt-2")
            ], width=12)
        ])
        ], fluid=True, style={"backgroundColor": "#ADD8E6", "height": "100vh"})
    else:
        return html.Div([
            html.H3(f"You are on {pathname}", className="text-center")
        ], style={'paddingTop': '50px'})

@app.callback(
    Output("model-results", "children"),
    Input("refresh-results-button", "n_clicks")
)
def update_results(n_clicks):
    if n_clicks:
        try:
            # List of model JSON files
            model_files = [
                "RandomForest_models.json",
                "GradientBoosting_models.json",
                "LogisticRegression_models.json",
                "DecisionTree_models.json",
                "LightGBM_models.json",
                "MLP_models.json",
                "AdaBoost_models.json",
                "Ensemble_models.json",
                "AutoML_models.json"
            ]
            
            results = []
            
            for model_file in model_files:
                try:
                    with open(model_file, "r") as file:
                        lines = file.readlines()
                    
                    # Parse the last line to get the latest model details
                    latest_model_details = json.loads(lines[-1])
                    
                    # Extract details
                    model_name = latest_model_details["model_name"]
                    version = latest_model_details["version"]
                    run_time = latest_model_details["run_time"]
                    gini_scores = latest_model_details["gini_scores"]
                    train_gini = gini_scores["train_gini"]
                    cross_val_gini = gini_scores["cross_val_gini"]
                    test_gini = gini_scores["test_gini"]
                    
                    # Append the details to the results list
                    results.append({
                        "Model Name": model_name,
                        "Version": version,
                        "Run Time": run_time,
                        "Train Gini": train_gini,
                        "Cross-Validation Gini": cross_val_gini,
                        "Test Gini": test_gini
                    })
                except Exception as e:
                    results.append({
                        "Model Name": model_file.replace("_models.json", ""),
                        "Version": "N/A",
                        "Run Time": "N/A",
                        "Train Gini": "N/A",
                        "Cross-Validation Gini": "N/A",
                        "Test Gini": f"Failed to read: {e}"
                    })
            
            # Create the DataTable
            table = dash_table.DataTable(
                data=results,
                columns=[
                    {"name": "Model Name", "id": "Model Name"},
                    {"name": "Version", "id": "Version"},
                    {"name": "Run Time", "id": "Run Time"},
                    {"name": "Train Gini", "id": "Train Gini"},
                    {"name": "Cross-Validation Gini", "id": "Cross-Validation Gini"},
                    {"name": "Test Gini", "id": "Test Gini"}
                ],
                style_table={'overflowX': 'auto'},
                style_header={
                    'backgroundColor': 'rgb(30, 30, 30)',
                    'color': 'white',
                    'fontWeight': 'bold'
                },
                style_cell={
                    'backgroundColor': 'rgb(50, 50, 50)',
                    'color': 'white',
                    'textAlign': 'center'
                }
            )
            
            return table
        except Exception as e:
            return f"Failed to update results: {e}"
    return ""

# Function to calculate Gini score
def calculate_gini(y_true, y_pred_proba):
    auc = roc_auc_score(y_true, y_pred_proba)
    return round(2 * auc - 1, 2)

# Function to store model details
def store_model_details(model_name, version, gini_scores, best_params):
    model_details = {
        "model_name": model_name,
        "version": version,
        "gini_scores": gini_scores,
        "best_params": best_params,
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(f"{model_name}_models.json", "a") as file:
        file.write(json.dumps(model_details) + "\n")

# Callback to load datasets
@app.callback(
    Output("dataset-status", "children"),
    Input("load-datasets-button", "n_clicks"),
    State("train-query", "value"),
    State("test-query", "value"),
    State("target-column", "value")
)
def load_datasets(n_clicks, train_query, test_query, target_column):
    global df_train_x, df_train_y, df_test_x, df_test_y,df_train,df_test
    
    def downcast_dtypes(df):
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            if df[col].dtype == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif df[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
        return df    

    if n_clicks:
        try:
            connection = cx_Oracle.connect(user, pw, dsn)
            
            # Load train dataset
            cursor = connection.cursor()
            cursor.execute(train_query)
            
            Columns_Names_List = [desc[0] for desc in cursor.description]
            
            # Fetch data in chunks
            batch_size = 1000
            all_chunks = []

            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                # Convert rows to DataFrame
                df_chunk = pd.DataFrame(rows, columns=Columns_Names_List)
                # Downcast numeric data
                df_chunk = downcast_dtypes(df_chunk)
                all_chunks.append(df_chunk)

            # Concatenate all chunks
            df_train = pd.concat(all_chunks, ignore_index=True)

            # Split train dataset
            df_train_y = df_train.iloc[:, target_column]
            df_train_x = df_train.iloc[:,target_column:]
            df_train_x = df_train_x.drop(df_train_x.columns[0], axis=1)
            
            
            # Load test dataset
            cursor.execute(test_query)
            
            all_chunks_test = []

            while True:
                rows_test = cursor.fetchmany(batch_size)
                if not rows_test:
                    break
                # Convert rows to DataFrame
                df_chunk_test = pd.DataFrame(rows_test, columns=Columns_Names_List)
                # Downcast numeric data
                df_chunk_test = downcast_dtypes(df_chunk_test)
                all_chunks_test.append(df_chunk_test)
            
            df_test = pd.concat(all_chunks_test,ignore_index=True)
    
            # Split test dataset
            df_test_y = df_test.iloc[:, target_column]
            df_test_x = df_test.iloc[:,target_column:]
            df_test_x = df_test_x.drop(df_test_x.columns[0], axis=1)
            
            connection.close()
            return (f"Datasets loaded successfully! "
                    f"Train shape: {df_train.shape}, Train X shape: {df_train_x.shape}, Train Y shape: {df_train_y.shape}. "
                    f"Test shape: {df_test.shape}, Test X shape: {df_test_x.shape}, Test Y shape: {df_test_y.shape}.")
        except Exception as e:
            return f"Failed to load datasets: {e}"
    return ""

# Callback to eliminate features based on missing rate
@app.callback(
    Output("elimination-status", "children"),
    Input("eliminate-features-button", "n_clicks"),
    State("missing-rate-threshold", "value")
)
def eliminate_features(n_clicks, threshold):
    global df_train_x, df_test_x
    if n_clicks:
        try:
            threshold = threshold / 100.0  # Convert percentage to proportion
            missing_rate_train = df_train_x.isnull().mean()
            missing_rate_test = df_test_x.isnull().mean()
            
            # Identify columns to drop
            columns_to_drop = missing_rate_train[missing_rate_train > threshold].index.union(
                missing_rate_test[missing_rate_test > threshold].index)
            
            # Drop columns from train and test datasets
            df_train_x = df_train_x.drop(columns=columns_to_drop)
            df_test_x = df_test_x.drop(columns=columns_to_drop)
            
            return (f"Features eliminated successfully! "
                    f"Remaining Train X shape: {df_train_x.shape}, Remaining Test X shape: {df_test_x.shape}.")
        except Exception as e:
            return f"Failed to eliminate features: {e}"
    return ""

# Callback to start binning process for train and test datasets
@app.callback(
    Output("binning-status", "children"),
    Input("start-binning-button", "n_clicks"),
    State("max-n-prebins", "value"),
    State("min-bin-size", "value"),
    State("max-n-bins", "value")
)
def start_binning(n_clicks, max_n_prebins, min_bin_size, max_n_bins):
    global df_train_x, df_train_y, df_test_x, df_test_y, list_features, list_categorical, df_train_binned, df_test_binned
    if n_clicks:
        try:
            # Define list of features and categorical ones
            list_features = df_train_x.columns.values
            list_categorical = df_train_x.select_dtypes(include=['object', 'category']).columns.values
            
            # Initialize and fit binning process for train and test datasets
            binning_process = BinningProcess(
                categorical_variables=list_categorical,
                variable_names=list_features,
                max_n_prebins=max_n_prebins,
                min_bin_size=min_bin_size,
                max_n_bins=max_n_bins
            )
            df_train_binned = binning_process.fit_transform(df_train_x, df_train_y, metric_missing='empirical')
            df_test_binned = binning_process.transform(df_test_x)
            
            return (f"Binning process completed! "
                    f"Binned Train shape: {df_train_binned.shape}, Binned Test shape: {df_test_binned.shape}.")
        except Exception as e:
            return f"Failed to start binning process: {e}"
    return ""

@app.callback(
    Output("elimination-status-tab3", "children"),
    Input("eliminate-features-button-tab3", "n_clicks"),
    State("correlation-threshold", "value")
)
def eliminate_features_tab3(n_clicks, threshold):
    global df_train_binned, df_test_binned, x_train_final, x_test_final
    if n_clicks:
        try:
            # Calculate correlation matrix
            correla_matrix = df_train_binned.corr().abs()
            upper = np.triu(np.ones(correla_matrix.shape), k=1).astype(bool)
            
            # Identify columns to drop based on correlation threshold
            to_drop = [correla_matrix.columns[j] for i in range(correla_matrix.shape) 
                       for j in range(i+1, correla_matrix.shape) if correla_matrix.iloc[i, j] > threshold]
            
            # Drop highly correlated features
            x_train_final = df_train_binned.drop(columns=to_drop)
            x_test_final = df_test_binned.drop(columns=to_drop)
            
            return (f"Features eliminated successfully! "
                    f"Remaining Train shape: {x_train_final.shape}, Remaining Test shape: {x_test_final.shape}.")
        except Exception as e:
            return f"Failed to eliminate features: {e}"
    return ""

@app.callback(
    Output("rf-gini", "children"),
    Input("train-rf-button", "n_clicks"),
    State("rf-n-estimators", "value"),
    State("rf-random-state", "value"),
    State("rf-search-method", "value")
)
def train_random_forest(n_clicks, n_estimators, random_state, search_method):
    if n_clicks:
        try:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=random_state)
            
            if search_method == 'randomized':
                search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=3, random_state=random_state)
            elif search_method == 'grid':
                search = GridSearchCV(model, param_grid=param_grid, cv=3)
            elif search_method == 'optuna':
                search = OptunaSearchCV(model, param_distributions=param_grid, n_trials=10, cv=3, random_state=random_state)
            elif search_method == 'bayes':
                search = BayesSearchCV(model, search_spaces=param_grid, n_iter=10, cv=3, random_state=random_state)
            else:
                return "Invalid search method"
            
            search.fit(df_train_binned, df_train_y)
            best_params = search.best_params_
            
            model = RandomForestClassifier(**best_params, random_state=random_state)
            model.fit(df_train_binned, df_train_y)
            
            # Calculate Gini scores
            train_gini = calculate_gini(df_train_y, model.predict_proba(df_train_binned)[:, 1])
            cross_val_gini = calculate_gini(df_train_y, cross_val_predict(model, df_train_binned,df_train_y, cv=3, method='predict_proba')[:, 1])
            test_gini = calculate_gini(df_test_y, model.predict_proba(df_test_binned)[:, 1])
            
            gini_scores = {
                "train_gini": train_gini,
                "cross_val_gini": cross_val_gini,
                "test_gini": test_gini
            }
            
            # Generate unique version name
            version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Store model details
            store_model_details("RandomForest", version, gini_scores, best_params)
            
            # Serialize and save the model with a unique filename
            model_filename = f"RandomForest_{version}.joblib"
            joblib.dump(model, model_filename)
            
            return f"Train Gini: {train_gini}, Cross-Validation Gini: {cross_val_gini}, Test Gini: {test_gini}, Best Params: {best_params}, Version: {version}, Model saved as: {model_filename}"
        except Exception as e:
            return f"Failed to train Random Forest: {e}"
    return ""

# Callback to train Gradient Boosting model and display Gini score
@app.callback(
    Output("gb-gini", "children"),
    Input("train-gb-button", "n_clicks"),
    State("gb-n-estimators", "value"),
    State("gb-random-state", "value"),
    State("gb-search-method", "value")
)
def train_gradient_boosting(n_clicks, n_estimators, random_state, search_method):
    if n_clicks:
        try:
            global df_train_y, df_train_binned, df_test_y, df_test_binned
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            model = GradientBoostingClassifier(random_state=random_state)
            
            if search_method == 'randomized':
                search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=3, random_state=random_state)
            elif search_method == 'grid':
                search = GridSearchCV(model, param_grid=param_grid, cv=3)
            elif search_method == 'optuna':
                search = OptunaSearchCV(model, n_trials=10, cv=3, random_state=random_state)
            elif search_method == 'bayes':
                search = BayesSearchCV(model, search_spaces=param_grid, n_iter=10, cv=3, random_state=random_state)
            else:
                return "Invalid search method"
            
            search.fit(df_train_binned, df_train_y)
            best_params = search.best_params_
            
            model = GradientBoostingClassifier(**best_params, random_state=random_state)
            model.fit(df_train_binned, df_train_y)
            
            # Calculate Gini scores
            train_gini = calculate_gini(df_train_y, model.predict_proba(df_train_binned)[:, 1])
            cross_val_gini = calculate_gini(df_train_y, cross_val_predict(model, df_train_binned,df_train_y, cv=3, method='predict_proba')[:, 1])
            test_gini = calculate_gini(df_test_y, model.predict_proba(df_test_binned)[:, 1])
            
            gini_scores = {
                "train_gini": train_gini,
                "cross_val_gini": cross_val_gini,
                "test_gini": test_gini
            }
            
            # Generate unique version name
            version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Store model details
            store_model_details("GradientBoosting", version, gini_scores, best_params)
            
            # Serialize and save the model with a unique filename
            model_filename = f"GradientBoosting_{version}.joblib"
            joblib.dump(model, model_filename)
            
            return f"Train Gini: {train_gini}, Cross-Validation Gini: {cross_val_gini}, Test Gini: {test_gini}, Best Params: {best_params}, Version: {version}, Model saved as: {model_filename}"
        except Exception as e:
            return f"Failed to train Gradient Boosting: {e}"
    return ""

# Callback to train Logistic Regression model and display Gini score
@app.callback(
    Output("lr-gini", "children"),
    Input("train-lr-button", "n_clicks"),
    State("lr-random-state", "value"),
    State("lr-max-iter", "value"),
    State("lr-search-method", "value")
)
def train_logistic_regression(n_clicks, random_state, max_iter, search_method):
    if n_clicks:
        try:
            param_grid = {
                'C': [0.01, 0.1, 1, 10],
                'solver': ['liblinear', 'saga']
            }
            model = LogisticRegression(random_state=random_state, max_iter=max_iter)
            
            if search_method == 'randomized':
                search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=3, random_state=random_state)
            elif search_method == 'grid':
                search = GridSearchCV(model, param_grid=param_grid, cv=3)
            elif search_method == 'optuna':
                search = OptunaSearchCV(model, param_distributions=param_grid, n_trials=10, cv=3, random_state=random_state)
            elif search_method == 'bayes':
                search = BayesSearchCV(model, search_spaces=param_grid, n_iter=10, cv=3, random_state=random_state)
            else:
                return "Invalid search method"
            
            search.fit(df_train_binned, df_train_y)
            best_params = search.best_params_
            
            model = LogisticRegression(**best_params, random_state=random_state, max_iter=max_iter)
            model.fit(df_train_binned, df_train_y)
            
            # Calculate Gini scores
            train_gini = calculate_gini(df_train_y, model.predict_proba(df_train_binned)[:, 1])
            cross_val_gini = calculate_gini(df_train_y, cross_val_predict(model, df_train_binned,df_train_y, cv=3, method='predict_proba')[:, 1])
            test_gini = calculate_gini(df_test_y, model.predict_proba(df_test_binned)[:, 1])
            
            gini_scores = {
                "train_gini": train_gini,
                "cross_val_gini": cross_val_gini,
                "test_gini": test_gini
            }
            
            # Generate unique version name
            version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Store model details
            store_model_details("LogisticRegression", version, gini_scores, best_params)
            
            # Serialize and save the model with a unique filename
            model_filename = f"LogisticRegression_{version}.joblib"
            joblib.dump(model, model_filename)
            
            return f"Train Gini: {train_gini}, Cross-Validation Gini: {cross_val_gini}, Test Gini: {test_gini}, Best Params: {best_params}, Version: {version}, Model saved as: {model_filename}"
        except Exception as e:
            return f"Failed to train Logistic Regression: {e}"
    return ""

# Callback to train Decision Tree model and display Gini score
@app.callback(
    Output("dt-gini", "children"),
    Input("train-dt-button", "n_clicks"),
    State("dt-max-depth", "value"),
    State("dt-random-state", "value"),
    State("dt-search-method", "value")
)
def train_decision_tree(n_clicks, max_depth, random_state, search_method):
    if n_clicks:
        try:
            param_grid = {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = DecisionTreeClassifier(random_state=random_state)
            
            if search_method == 'randomized':
                search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=3, random_state=random_state)
            elif search_method == 'grid':
                search = GridSearchCV(model, param_grid=param_grid, cv=3)
            elif search_method == 'optuna':
                search = OptunaSearchCV(model, param_distributions=param_grid, n_trials=10, cv=3, random_state=random_state)
            elif search_method == 'bayes':
                search = BayesSearchCV(model, search_spaces=param_grid, n_iter=10, cv=3, random_state=random_state)
            else:
                return "Invalid search method"
            
            search.fit(df_train_binned, df_train_y)
            best_params = search.best_params_
            
            model = DecisionTreeClassifier(**best_params, random_state=random_state)
            model.fit(df_train_binned, df_train_y)
            
            # Calculate Gini scores
            train_gini = calculate_gini(df_train_y, model.predict_proba(df_train_binned)[:, 1])
            cross_val_gini = calculate_gini(df_train_y, cross_val_predict(model, df_train_binned,df_train_y, cv=3, method='predict_proba')[:, 1])
            test_gini = calculate_gini(df_test_y, model.predict_proba(df_test_binned)[:, 1])
            
            gini_scores = {
                "train_gini": train_gini,
                "cross_val_gini": cross_val_gini,
                "test_gini": test_gini
            }
            
            # Generate unique version name
            version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Store model details
            store_model_details("DecisionTree", version, gini_scores, best_params)
            
            # Serialize and save the model with a unique filename
            model_filename = f"DecisionTree_{version}.joblib"
            joblib.dump(model, model_filename)
            
            return f"Train Gini: {train_gini}, Cross-Validation Gini: {cross_val_gini}, Test Gini: {test_gini}, Best Params: {best_params}, Version: {version}, Model saved as: {model_filename}"
        except Exception as e:
            return f"Failed to train Decision Tree: {e}"
    return ""

# Callback to train LightGBM model and display Gini score
@app.callback(
    Output("lgbm-gini", "children"),
    Input("train-lgbm-button", "n_clicks"),
    State("lgbm-n-estimators", "value"),
    State("lgbm-random-state", "value"),
    State("lgbm-search-method", "value")
)
def train_lightgbm(n_clicks, n_estimators, random_state, search_method):
    if n_clicks:
        try:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100]
            }
            model = LGBMClassifier(random_state=random_state)
            
            if search_method == 'randomized':
                search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=3, random_state=random_state)
            elif search_method == 'grid':
                search = GridSearchCV(model, param_grid=param_grid, cv=3)
            elif search_method == 'optuna':
                search = OptunaSearchCV(model, param_distributions=param_grid, n_trials=10, cv=3, random_state=random_state)
            elif search_method == 'bayes':
                search = BayesSearchCV(model, search_spaces=param_grid, n_iter=10, cv=3, random_state=random_state)
            else:
                return "Invalid search method"
            
            search.fit(df_train_binned, df_train_y)
            best_params = search.best_params_
            
            model = LGBMClassifier(**best_params, random_state=random_state)
            model.fit(df_train_binned, df_train_y)
            
            # Calculate Gini scores
            train_gini = calculate_gini(df_train_y, model.predict_proba(df_train_binned)[:, 1])
            cross_val_gini = calculate_gini(df_train_y, cross_val_predict(model, df_train_binned,df_train_y, cv=3, method='predict_proba')[:, 1])
            test_gini = calculate_gini(df_test_y, model.predict_proba(df_test_binned)[:, 1])
            
            gini_scores = {
                "train_gini": train_gini,
                "cross_val_gini": cross_val_gini,
                "test_gini": test_gini
            }
            
            # Generate unique version name
            version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Store model details
            store_model_details("LightGBM", version, gini_scores, best_params)
            
            # Serialize and save the model with a unique filename
            model_filename = f"LightGBM_{version}.joblib"
            joblib.dump(model, model_filename)
            
            return f"Train Gini: {train_gini}, Cross-Validation Gini: {cross_val_gini}, Test Gini: {test_gini}, Best Params: {best_params}, Version: {version}, Model saved as: {model_filename}"
        except Exception as e:
            return f"Failed to train LightGBM: {e}"
    return ""

# Callback to train MLP model and display Gini score
@app.callback(
    Output("mlp-gini", "children"),
    Input("train-mlp-button", "n_clicks"),
    State("mlp-hidden-layers", "value"),
    State("mlp-random-state", "value"),
    State("mlp-search-method", "value")
)
def train_mlp(n_clicks, hidden_layers, random_state, search_method):
    if n_clicks:
        try:
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam', 'sgd']
            }
            model = MLPClassifier(random_state=random_state)
            
            if search_method == 'randomized':
                search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=3, random_state=random_state)
            elif search_method == 'grid':
                search = GridSearchCV(model, param_grid=param_grid, cv=3)
            elif search_method == 'optuna':
                search = OptunaSearchCV(model, param_distributions=param_grid, n_trials=10, cv=3, random_state=random_state)
            elif search_method == 'bayes':
                search = BayesSearchCV(model, search_spaces=param_grid, n_iter=10, cv=3, random_state=random_state)
            else:
                return "Invalid search method"
            
            search.fit(df_train_binned, df_train_y)
            best_params = search.best_params_
            
            model = MLPClassifier(**best_params, random_state=random_state)
            model.fit(df_train_binned, df_train_y)
            
            # Calculate Gini scores
            train_gini = calculate_gini(df_train_y, model.predict_proba(df_train_binned)[:, 1])
            cross_val_gini = calculate_gini(df_train_y, cross_val_predict(model, df_train_binned,df_train_y, cv=3, method='predict_proba')[:, 1])
            test_gini = calculate_gini(df_test_y, model.predict_proba(df_test_binned)[:, 1])
            
            gini_scores = {
                "train_gini": train_gini,
                "cross_val_gini": cross_val_gini,
                "test_gini": test_gini
            }
            
            # Generate unique version name
            version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Store model details
            store_model_details("MLP", version, gini_scores, best_params)
            
            # Serialize and save the model with a unique filename
            model_filename = f"MLP_{version}.joblib"
            joblib.dump(model, model_filename)
            
            return f"Train Gini: {train_gini}, Cross-Validation Gini: {cross_val_gini}, Test Gini: {test_gini}, Best Params: {best_params}, Version: {version}, Model saved as: {model_filename}"
        except Exception as e:
            return f"Failed to train MLP: {e}"
    return ""

# Callback to train AdaBoost model and display Gini score
@app.callback(
    Output("adaboost-gini", "children"),
    Input("train-adaboost-button", "n_clicks"),
    State("adaboost-n-estimators", "value"),
    State("adaboost-random-state", "value"),
    State("adaboost-search-method", "value")
)
def train_adaboost(n_clicks, n_estimators, random_state, search_method):
    if n_clicks:
        try:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1]
            }
            model = AdaBoostClassifier(random_state=random_state)
            
            if search_method == 'randomized':
                search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=3, random_state=random_state)
            elif search_method == 'grid':
                search = GridSearchCV(model, param_grid=param_grid, cv=3)
            elif search_method == 'optuna':
                search = OptunaSearchCV(model, param_distributions=param_grid, n_trials=10, cv=3, random_state=random_state)
            elif search_method == 'bayes':
                search = BayesSearchCV(model, search_spaces=param_grid, n_iter=10, cv=3, random_state=random_state)
            else:
                return "Invalid search method"
            
            search.fit(df_train_binned, df_train_y)
            best_params = search.best_params_
            
            model = AdaBoostClassifier(**best_params, random_state=random_state)
            model.fit(df_train_binned, df_train_y)
            
            # Calculate Gini scores
            train_gini = calculate_gini(df_train_y, model.predict_proba(df_train_binned)[:, 1])
            cross_val_gini = calculate_gini(df_train_y, cross_val_predict(model, df_train_binned,df_train_y, cv=3, method='predict_proba')[:, 1])
            test_gini = calculate_gini(df_test_y, model.predict_proba(df_test_binned)[:, 1])
            
            gini_scores = {
                "train_gini": train_gini,
                "cross_val_gini": cross_val_gini,
                "test_gini": test_gini
            }
            
            # Generate unique version name
            version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Store model details
            store_model_details("AdaBoost", version, gini_scores, best_params)
            
            # Serialize and save the model with a unique filename
            model_filename = f"AdaBoost_{version}.joblib"
            joblib.dump(model, model_filename)
            
            return f"Train Gini: {train_gini}, Cross-Validation Gini: {cross_val_gini}, Test Gini: {test_gini}, Best Params: {best_params}, Version: {version}, Model saved as: {model_filename}"
        except Exception as e:
            return f"Failed to train AdaBoost: {e}"
    return ""

@app.callback(
    Output("ensemble-gini", "children"),
    Input("train-ensemble-button", "n_clicks"),
    [State("rf-n-estimators", "value"),
     State("rf-max-depth", "value"),
     State("gb-n-estimators", "value"),
     State("gb-learning-rate", "value"),
     State("ada-n-estimators", "value"),
     State("ada-learning-rate", "value"),
     State("ensemble-voting", "value"),
     State("ensemble-weights", "value"),
     State("ensemble-search-method", "value")]
)
def train_ensemble(n_clicks, rf_n_estimators, rf_max_depth, gb_n_estimators, gb_learning_rate, ada_n_estimators, ada_learning_rate, voting, weights, search_method):
    if n_clicks:
        try:
            param_grid_rf = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20]
            }
            param_grid_gb = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1]
            }
            param_grid_ada = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1]
            }
            model1 = RandomForestClassifier(random_state=42)
            model2 = GradientBoostingClassifier(random_state=42)
            model3 = AdaBoostClassifier(random_state=42)

            if search_method == 'randomized':
                search_rf = RandomizedSearchCV(model1, param_distributions=param_grid_rf, n_iter=10, cv=3, random_state=42)
                search_gb = RandomizedSearchCV(model2, param_distributions=param_grid_gb, n_iter=10, cv=3, random_state=42)
                search_ada = RandomizedSearchCV(model3, param_distributions=param_grid_ada, n_iter=10, cv=3, random_state=42)
            elif search_method == 'grid':
                search_rf = GridSearchCV(model1, param_grid=param_grid_rf, cv=3)
                search_gb = GridSearchCV(model2, param_grid=param_grid_gb, cv=3)
                search_ada = GridSearchCV(model3, param_grid=param_grid_ada, cv=3)
            elif search_method == 'optuna':
                search_rf = OptunaSearchCV(model1, param_distributions=param_grid_rf, n_trials=10, cv=3, random_state=42)
                search_gb = OptunaSearchCV(model2, param_distributions=param_grid_gb, n_trials=10, cv=3, random_state=42)
                search_ada = OptunaSearchCV(model3, param_distributions=param_grid_ada, n_trials=10, cv=3, random_state=42)
            elif search_method == 'bayes':
                search_rf = BayesSearchCV(model1, search_spaces=param_grid_rf, n_iter=10, cv=3, random_state=42)
                search_gb = BayesSearchCV(model2, search_spaces=param_grid_gb, n_iter=10, cv=3, random_state=42)
                search_ada = BayesSearchCV(model3, search_spaces=param_grid_ada, n_iter=10, cv=3, random_state=42)
            else:
                return "Invalid search method"

            search_rf.fit(df_train_binned, df_train_y)
            search_gb.fit(df_train_binned, df_train_y)
            search_ada.fit(df_train_binned, df_train_y)
            best_params_rf = search_rf.best_params_
            best_params_gb = search_gb.best_params_
            best_params_ada = search_ada.best_params_

            model1 = RandomForestClassifier(**best_params_rf, random_state=42)
            model2 = GradientBoostingClassifier(**best_params_gb, random_state=42)
            model3 = AdaBoostClassifier(**best_params_ada, random_state=42)
            models = [('rf', model1), ('gb', model2), ('ada_boost', model3)]

            for name, model in models:
                model.fit(df_train_binned, df_train_y)

            ensemble = VotingClassifier(estimators=models, voting=voting, weights=weights)
            ensemble.fit(df_train_binned, df_train_y)

            # Calculate Gini scores
            train_gini = calculate_gini(df_train_y, ensemble.predict_proba(df_train_binned)[:, 1])
            cross_val_gini = calculate_gini(df_train_y, cross_val_predict(ensemble, df_train_binned,df_train_y, cv=3, method='predict_proba')[:, 1])
            test_gini = calculate_gini(df_test_y, ensemble.predict_proba(df_test_binned)[:, 1])

            gini_scores = {
                "train_gini": train_gini,
                "cross_val_gini": cross_val_gini,
                "test_gini": test_gini
            }

            # Generate unique version name
            version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"

            # Store model details
            store_model_details("Ensemble", version, gini_scores, {"rf": best_params_rf, "gb": best_params_gb, "ada": best_params_ada})

            # Serialize and save the model with a unique filename
            model_filename = f"Ensemble_{version}.joblib"
            joblib.dump(ensemble, model_filename)

            return f"Train Gini: {train_gini}, Cross-Validation Gini: {cross_val_gini}, Test Gini: {test_gini}, Best Params: RF: {best_params_rf}, GB: {best_params_gb}, Ada: {best_params_ada}, Version: {version}, Model saved as: {model_filename}"
        except Exception as e:
            return f"Failed to train ensemble model: {e}"
    return ""

@app.callback(
    Output("automl-gini", "children"),
    [Input("train-automl-button", "n_clicks"),
     State("time-left-for-task", "value"),
     State("per-run-time-limit", "value")]
)
def train_automl(n_clicks, time_left_for_task, per_run_time_limit):
    if n_clicks:
        try:
            automl = AutoSklearnClassifier(time_left_for_this_task=time_left_for_task, per_run_time_limit=per_run_time_limit)
            automl.fit(df_train_binned, df_train_y)

            # Calculate Gini scores
            train_gini = calculate_gini(df_train_y, automl.predict_proba(df_train_binned)[:, 1])
            cross_val_gini = calculate_gini(df_train_y, cross_val_predict(automl, df_train_binned,df_train_y, cv=3, method='predict_proba')[:, 1])
            test_gini = calculate_gini(df_test_y, automl.predict_proba(df_test_binned)[:, 1])

            gini_scores = {
                "train_gini": train_gini,
                "cross_val_gini": cross_val_gini,
                "test_gini": test_gini
            }

            # Generate unique version name
            version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"

            # Store model details
            store_model_details("AutoML", version, gini_scores, {})

            # Serialize and save the model with a unique filename
            model_filename = f"AutoML_{version}.joblib"
            joblib.dump(automl, model_filename)

            return f"Train Gini: {train_gini}, Cross-Validation Gini: {cross_val_gini}, Test Gini: {test_gini}, Version: {version}, Model saved as: {model_filename}"
        except Exception as e:
            return f"Failed to train AutoML: {e}"
    return ""

# Run the app on a specified port
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8080)


