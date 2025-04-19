import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import joblib

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# 샘플 데이터 로드 함수
def load_sample_data(case):
    file_map = {
        "normal": "samples/sample_normal.csv",
        "hyper": "samples/sample_hyper.csv",
        "hypo": "samples/sample_hypo.csv"
    }
    try:
        return pd.read_csv(file_map[case])
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

# 모델 및 임퓨터 로드 함수
model_cache = {}
# imputer_cache = {}

def load_model_and_imputer(case):
    if case not in model_cache:
        model_cache[case] = joblib.load(f"model/rf_{case}_model.pkl")
    return model_cache[case]


# 입력 영역
input_card = dbc.Card([
    html.H3("Type 1 Diabetes Dashboard", className="text-center mb-2 mt-3"),
    dbc.Alert([
        html.H5("User Guidelines:"),
        html.P("1. This dashboard is designed for youth(2~20) with Type 1 Diabetes.", className="mb-1"),
        html.P("2. Previous 24-hour CGM (Continuous Glucose Monitoring) data is required.", className="mb-0"),
        html.P("3. To help you explore the features, this dashboard includes sample cases of normal, hyper-, and hypo-glycemia.", className="mb-0")
    ], color="danger", className="mb-4"),

    # html.H5("Patient Information"),
    html.Div([
    html.H4("Patient Information", className="d-inline-block"),
    dbc.Button([
        html.I(className="bi bi-smartwatch", style={"marginRight": "5px"}),
        "Connect Device (Garmin)"
    ], color="primary", size="sm", outline=True, className="ms-2 d-inline-block", id="connect_watch_button")
    ]),
    dbc.Label("Gender:"),
    dcc.Dropdown(
        options=[{"label": "1: Male", "value": 1}, {"label": "2: Female", "value": 2}],
        id="gender", className="mb-3"
    ),
    dbc.Label("Age:"),
    dbc.Input(type="number", id="age", placeholder="Enter age (ages 2–20)", className="mb-3"),
    dbc.Label("Height (cm):"),
    dbc.Input(type="number", id="height", placeholder="Enter height", className="mb-3"),
    dbc.Label("Weight (kg):"),
    dbc.Input(type="number", id="weight", placeholder="Enter weight", className="mb-3"),
    dbc.Label("Upload CGM Data (24hrs before exercise):", className="mt-3"),
    dcc.Upload(
        id="upload_cgm",
        children=html.Div([
            "Drag and Drop or ",
            html.A("Select CGM File")
            ]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "marginBottom": "20px"
                },
                multiple=False
                ),

    html.H4("Activity Information", className="mt-4"),
    dbc.Label("Exercise Intensity Level:"),
    dcc.Dropdown(
        options=[{"label": f"{i}: {desc}", "value": i} for i, desc in enumerate([
            "Low - Pretty easy", "Mild - Working a bit", "Moderate - Working to keep up",
            "Heavy - Hard to keep going but did it", "Exhaustive - Too tough / Had to stop"], start=1)],
        id="exercise_intensity", className="mb-3"
    ),
    dbc.Label("Exercise Duration (minutes):"),
    dbc.Input(type="number", id="duration", placeholder="Enter duration", className="mb-3"),

    # html.H5("Insulin Information", className="mt-4"),
    # dbc.Label("Insulin Delivery Method:"),
    # dcc.Dropdown(
    #     options=[{"label": i, "value": i} for i in ["Pump", "Injection"]],
    #     id="insulin_method", className="mb-3"
    # ),

    html.H4("Samples", className="mt-4"),
    dbc.RadioItems(
        options=[
            {"label": "Normal Case", "value": "normal"},
            {"label": "Hyperglycemia Case", "value": "hyper"},
            {"label": "Hypoglycemia Case", "value": "hypo"}
        ],
        id="sample_case",
        inline=True,
        className="mb-3"
    ),
    dbc.Button("Run Prediction Model", id="run_button", color="primary", className="mt-2")
], style={"height": "100%"})

# 결과 영역
result_card = dbc.Card(
    dbc.CardBody([
        html.H4("Results", className="mt-3"),

        html.H5("BMI Distribution:", className="mt-4"),
        html.Div(id="bmi_plot_output"),  # BMI plot placeholder

        html.H4("Predicted Glucose Outcomes within 2 Hours After Exercise:"),
        html.Div(id="prediction_result"),

        # html.H5("Loaded Sample Data Preview:"),
        # html.Div(id="sample_data_output", style={"overflowX": "auto"})
    ]),
    style={"height": "100%"}
)

# 전체 레이아웃
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(input_card, width=6),
        dbc.Col(result_card, width=6, style={"display": "flex", "flexDirection": "column"})
    ])
], fluid=True)

# 샘플 데이터 출력 콜백
# @app.callback(
#     Output("sample_data_output", "children"),
#     Input("sample_case", "value")
# )
# def display_sample(case):
#     if case:
#         df = load_sample_data(case)
#         df = df.drop(columns=["first_condition_label"], errors="ignore")
#         return dbc.Table.from_dataframe(df.head(), striped=True, bordered=True, hover=True)
#     return "Select a sample case to preview data."

# 예측 실행 콜백
@app.callback(
    Output("prediction_result", "children"),
    Input("run_button", "n_clicks"),
    State("sample_case", "value")
)
def run_model(n_clicks, case):
    if n_clicks and case:
        try:
            import plotly.graph_objects as go
            df = load_sample_data(case)
            df = df.drop(columns=["first_condition_label"], errors="ignore")
            X_input = df.select_dtypes(include='number').copy()
            X_input = X_input.drop(columns=["original_label", "subject_id"], errors="ignore")

            result = {}
            for model_name in ["hyper", "hypo", "normal"]:
                model = load_model_and_imputer(model_name)
                
                Z_input = pd.DataFrame(np.ones((X_input.shape[0], 1)), columns=["intercept"])
                
                cluster_input = df["subject_id"] if "subject_id" in df.columns else pd.Series([0]*X_input.shape[0])
                
                prob = model.predict(X_input.head(1), Z_input.head(1), cluster_input.head(1))[0]
                result[model_name.capitalize()] = round(prob * 100, 1)

            # color_map = {
            #      "Normal": "green",
            #      "Hypo": "blue",
            #      "Hyper": "red"
            #        }
            gauges = [
    dcc.Graph(
        figure=go.Figure(go.Indicator(
            mode="gauge+number",
            value=v,  
            # number={"suffix": "%"}, 
            title={"text": f"P({k})"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {
                    "color": "red" if v > 50 else "green",
                    "thickness": 0.8
                }
            }
        )),
        style={"width": "105%", "height": "250px"}
    ) for k, v in result.items()
]


            top_label = max(result, key=result.get)
            top_value = round(result[top_label],2)

            disclaimer = html.P(
                "Note: The predicted result is for reference only and should not be used as a basis for medical decisions.",
                style={"fontSize": "0.85rem", "color": "gray", "fontStyle": "italic", "marginTop": "10px"}
                )

            summary = html.P(
                f"There is a {top_value}% chance of {top_label} within 2 hours after exercise.",
                style={"color": "red", "fontWeight": "bold", "fontSize": "23px"}
            )
            
            sample_info = html.P(
                f"Sample used: {case.capitalize()} case",
                style={"fontSize": "0.9rem", "color": "gray"}
            )

            return html.Div([
                sample_info,
                dbc.Row([dbc.Col(g, width=4) for g in gauges]),
                disclaimer,
                summary
                
            ])
        except Exception as e:
            return f"Prediction failed: {e}"
    return html.Div("[Placeholder]")

# BMI 시각화 콜백
@app.callback(
    Output("bmi_plot_output", "children"),
    Input("run_button", "n_clicks"),
    State("sample_case", "value")
)
def plot_bmi(n_clicks, case):
    if not n_clicks or not case:
        return ""

    import plotly.graph_objects as go
    try:
        bmi_df = pd.read_csv("bmi/bmi.csv")
        df_sample = load_sample_data(case)

        if "subject_id" not in df_sample.columns:
            return html.P("subject_id not found in sample data.")

        subject_id = df_sample["subject_id"].iloc[0]

        fig = go.Figure()

        # 전체 회색 점
        fig.add_trace(go.Scatter(
            x=bmi_df["weight_kg"],
            y=bmi_df["height_cm"],
            mode="markers",
            marker=dict(color="gray", opacity=0.3, size=6),
            name="All",
            hoverinfo="skip"
        ))

        # 빨간 점 + BMI Category 텍스트
        selected = bmi_df[bmi_df["subject_id"] == subject_id]
        if not selected.empty:
            fig.add_trace(go.Scatter(
                x=selected["weight_kg"],
                y=selected["height_cm"],
                mode="markers",
                marker=dict(color="red", size=10),
                name= "You"
            ))
            bmi_cat = selected["bmi_category"].iloc[0]
            bmi_value = selected["bmi"].iloc[0]
            bmi_text = html.P(f"BMI : {bmi_value} ({bmi_cat})", style={"color": "red", "fontWeight": "bold", "marginTop": "10px", "fontSize": "23px"})
        else:
            bmi_text = html.P("BMI : Not found", style={"color": "red", "fontWeight": "bold", "marginTop": "10px"})

        fig.update_layout(
            title="BMI Scatterplot",
            xaxis_title="Weight (kg)",
            yaxis_title="Height (cm)",
            height=400,
            showlegend=False
        )

        return html.Div([
            dcc.Graph(figure=fig),
            bmi_text
        ])

    except Exception as e:
        return html.P(f"Failed to generate plot: {e}")

if __name__ == "__main__":
    app.run(debug=True)
