#!/usr/bin/env python3

import yaml
import pandas as pd

import plotly.graph_objects as go

import os
import glob


def load_yaml_data(path: str) -> dict:
    """Load all of the yaml data"""
    data = {}
    files = glob.glob(os.path.join(path, "*.yaml"))
    for file in files:
        name = os.path.basename(file).split('.')[0]
        try:
            with open(file) as stream:
                data[name] = yaml.safe_load(stream=stream)
        except yaml.YAMLError as e:
            print(e)
    return data

def get_dataframe(data: dict) -> pd.DataFrame:
    """Load the dictionary into a DataFrame"""
    big_df = pd.DataFrame()
    for trial, trial_data in data.items():
        df = pd.json_normalize(data=trial_data["apples"])
        df.insert(loc=1, column="trial", value=trial, allow_duplicates=True)
        big_df = pd.concat((big_df, df), ignore_index=True)
    return big_df

def plot_radius_estimates(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # Add control trace
    df_control = df.loc[df["trial"] == "control_apples"]
    fig.add_trace(
        go.Scatter(
            x=df_control["id"],
            y=df_control["radius"],
            name=str(df_control["trial"].iloc[0]),
            mode="markers",
            marker=dict(
                color="LightSkyBlue",
                size=20
            )
        )
    )

    # Add the trial data
    df_trials = df.loc[df["trial"] != "control_apples"]
    for trial in df_trials["trial"].unique():
        df_trial = df_trials.loc[df_trials["trial"] == trial]
        trial_name = df_trial["trial"].iloc[0]
        if trial_name.startswith("random"):
            color = "darkorange"
        else:
            color = "mediumseagreen"
        fig.add_trace(
            go.Scatter(
                x=df_trial["id"],
                y=df_trial["radius"],
                name=str(trial_name),
                mode="markers",
                marker=dict(
                    size=10,
                    color=color
                ),
                
            )
        )

    fig.update_layout(
        title="Radius estimates by apple ID",
        xaxis=dict(
            title="Apple ID"
        ),
        yaxis=dict(
            title="Radius (m)"
        )
    )

    return fig


def main():
    __here__ = os.path.dirname(__file__)
    data = load_yaml_data(path=__here__)
    df = get_dataframe(data=data)
    fig = plot_radius_estimates(df=df)
    fig.show()

    return


if __name__ == "__main__":
    main()