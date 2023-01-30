import dash
import numpy as np
import plotly.graph_objects as go
from dash import dcc, callback_context
from dash import html
from dash.dependencies import Input, Output
from dash_extensions import EventListener
from plotly.subplots import make_subplots
import plotly.express as px


class presentation_viewer():
    def __init__(self):
        app = dash.Dash()
        dives_window=html.Div([Keyboard(id="keyboard"),html.Div(id='presentation_window', style={"white-space": "pre"})])
        app.layout = html.Div(dives_window)
        self.current_slide=0

        @app.callback(Output("output", "children"), [Input("keyboard", "keydown")])
        def keydown(event):
            return f"hey {event}"

        app.run_server(debug=True, use_reloader=False)
