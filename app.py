from dash import Dash, html, dcc
import plotly.express as px

df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species", template="plotly_white")

app = Dash(__name__)
server = app.server   # ←これが必須

app.layout = html.Div([
    html.H2("Dash + Plotly 動作確認"),
    dcc.Graph(figure=fig)
])

if __name__ == "__main__":
    app.run(debug=True)