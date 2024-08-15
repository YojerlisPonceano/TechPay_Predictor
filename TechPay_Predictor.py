#data processing libraries
import pandas as pd
import numpy as np
import plotly.express as px

#dash libraries
from dash import Dash, dcc, html, Input, Output  
from dash import callback_context
import dash_bootstrap_components as dbc

#to load the model
import joblib

#to get the data
from get_data import get_and_clean_data

#getting the data
cleaned_data, raw_data = get_and_clean_data()

#importing the model
xgb_model = joblib.load("https://github.com/YojerlisPonceano/TechPay_Predictor/raw/main/assets/xgb_model.joblib")
                         
#getting the model score
score = ""
with open('https://raw.githubusercontent.com/YojerlisPonceano/TechPay_Predictor/main/assets/model_score.txt', 'r') as file:
    score = file.read().strip()


def get_positions_order() -> dict:
    """
    Creates a dictionary where each position from the dataset is a key, 
    and the corresponding integer value represents that position.

    Returns:
        dict: Positions with its value as an interger.
    """
    positions = cleaned_data.columns[:10]
    positions_dict_order = {}
    i = 0 
    for position in positions:
        positions_dict_order[position[9:]] = i
        i +=1
    return positions_dict_order


def get_values_for_prediction(position, gender, years_exp) -> list:
    """
    Function that creates a list of numbers based on the users selections 
    that is later used to make the prediction.

    Args:
        position (str): The user role.
        gender (str): Gender.
        years_exp (str): Years of work experience.

    Returns:
        list: list of number
    """
    positions_dict_order = get_positions_order()
    values_for_prediction = [0,0,0,0,0,0,0,0,0,0]
    gender_number = 0
    if gender == "Male":
        gender_number = 1
        
    if position in positions_dict_order.keys():
        values_for_prediction[positions_dict_order[position]] = 1
    values_for_prediction.append(gender_number)
    values_for_prediction.append(years_exp)
    
    return values_for_prediction

#getting the labels for the positions(Roles)
positions_labels = raw_data["Position"].unique()


external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets, 
            suppress_callback_exceptions = True)
server = app.server

# ------------------------------------------------------------------------------------------------------
# App layout
app.layout = dbc.Container([

    html.H1("TechPay Predictor",
            className = "app-title"),
    
    dbc.Row([html.Br()]),

    dbc.Row([ html.H5("TechPay Predictor offers personalized salary forecasts for IT professionals based on role, experience, and gender.", 
              className = "app-description")]),

    dbc.Row([html.Br()]),

    dbc.Row([
        dbc.Col([
             html.H5("Role", className = "role-title"),
             dcc.Dropdown(id="slct_position",
                 className = "role-dropdown",
                 options= positions_labels,
                 multi=False,
                 value="Database Administrator (DBA)"
                 )  
        ]),

        dbc.Col([
            html.H5("Gender", className = "gender-title"),
            dcc.Dropdown(id="slct_gender",
                 className = "gender-dropdown",
                 options=["Male", "Female"],
                 multi=False,
                 value='Male'
                 )
        ]),

        dbc.Col([
            html.H5("Experience (Years)", className = "exp-title"),
            dcc.Dropdown(id="years_exp",
                      className = "exp-input",
                      options=[str(i) for i in range(51)],
                      multi=False,
                      value='1'
                    )
        ])
    ]),
   
    dbc.Row([html.Br()]),

    dbc.Row([
        dbc.Col([
            html.Button('Submit', 
                    id='submit_val',
                    className = "submit-button",
                    n_clicks=0
                    )

            
        ])
    ]),

    dbc.Row([html.Br()]),

    html.Div(id='output_container', className="output", 
                      children=[]),
    dbc.Row([html.Br()]),
    dbc.Row([
        dbc.Col([dcc.Graph(id='roles_barchart', className = "barchart",figure={})])
        
    ]),
    dbc.Row([html.Br()]),

    dbc.Row([ html.P("""This graph compares the predicted salary for the selected role, gender, 
                        and years of experience with the salaries of other roles, maintaining 
                        the same gender and years of experience. This shows the potential salary 
                        you could earn if you had chosen different roles with the same years of experience.
                        The model being used has a R2 score of {}.""".format(score),
              className = "barchar-description")])
    
], fluid=True)


#-------------------------------------------------------------------------------------------------------------------
#App callback

@app.callback(
    [Output(component_id='output_container', component_property='children'),
    Output(component_id='roles_barchart', component_property='figure')],
    [Input(component_id='slct_position', component_property='value'),
    Input(component_id='slct_gender', component_property='value'),
    Input(component_id='years_exp', component_property='value'),
    Input(component_id='submit_val', component_property='n_clicks')]
)
def update_graph(position_slctd, gender_slctd, years_exp_slctd, n_clicks):
    if years_exp_slctd == None:
        years_exp_slctd = 0

    #getting the posible salary with the model
    postion_salary = []
    for position in positions_labels:
        possible_salary = np.round(float(xgb_model.predict([get_values_for_prediction(position, gender_slctd, int(years_exp_slctd))])[0]),2)
        postion_salary.append({"Position":position, "possible_salary": possible_salary})


    #getting the possible salary of the selected position
    possible_salary = np.round(float(xgb_model.predict([get_values_for_prediction(position_slctd, gender_slctd, int(years_exp_slctd))])[0]),2)

    #creating an empty figure to return if the user leaves any required fields unfilled
    empty_df = pd.DataFrame(columns=['x', 'y'])
    fig = px.bar(empty_df, x='x', y='y', title='Submit your selection to display the graph here')

    #initial output text
    container = 'Click "Submit" to view the estimated salary based on your selected role, gender, and years of experience.'

    #list to check if submit was click, to use as a green light to show the predicted outputs
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]

    #to validate if submit was clicked
    if 'submit_val' in changed_id:
        if (position_slctd is not None) and (gender_slctd is not None):

            #creating the output text
            container = f"As a {position_slctd} with {years_exp_slctd} years of experience and identifying as {gender_slctd}, your estimated annual salary is approximately ${possible_salary} USD."

            #creating formatting  the dataFame to make the figure
            postion_salary_df = pd.DataFrame(postion_salary)
            postion_salary_df['Highlight'] = postion_salary_df['Position'].apply(lambda x: 'Selected Role' if x == position_slctd else 'Other Roles')
            postion_salary_df = postion_salary_df.sort_values(by="possible_salary")

            # Create the bar chart
            fig = px.bar(postion_salary_df, 
                        x='Position', 
                        y='possible_salary', 
                        color='Highlight',
                        color_discrete_map={'Selected Role': '#54bebe', 'Other Roles': '#d0d0d0'},
                        title="Salary Prediction per Role",
                        text='possible_salary')

            # Format fig
            fig.update_layout(
                height = 600,
                autosize =True,
                title_font=dict(
                    family="Helvetica Neue, Arial, sans-serif", 
                    size=24, 
                    color="black"
                ),
                title_x=0.5,  
                xaxis_title='',
                yaxis_title='', 
                yaxis=dict(
                    tickvals=[], 
                    showticklabels=False,
                ),
                xaxis=dict(
                    tickangle=-45,  
                    title_font=dict(
                        family="Helvetica Neue, Arial, sans-serif",  
                        size=14,
                        color="black"
                    )
                ),
                plot_bgcolor='white',
                margin=dict(l=50, r=30, t=50, b=70),  
                legend=dict(
                    orientation='h',
                    title="",
                    yanchor='top',  
                    y=-0.4,  
                    xanchor='center',
                    x=0.5
                ),
                font=dict(
                    family="Helvetica Neue, Arial, sans-serif",  
                    size=12,
                    color="black"
                )
            )

            # Update data labels with better formatting
            fig.update_traces(
                texttemplate='%{text:.0f}', 
                textposition='outside', 
                marker=dict(
                    line=dict(
                        color='black', 
                        width=1  
                    )
                )
            )

        else:
            container = "Make sure that the Role and the Gender are selected."
    return container, fig


if __name__ == '__main__':
    app.run_server(debug=True) 
