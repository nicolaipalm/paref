from pathlib import Path
from typing import List

import numpy as np
import plotly.graph_objects as go
import yaml
from pymoo.indicators.hv import Hypervolume

from src.weimoo.moos.helper_functions.return_pareto_front_2d import return_pareto_front_2d


class EvaluationMOOSingleRun:
    def __init__(self, path: Path):
        self.path = path

        with open(path) as f:
            content = yaml.load(f, Loader=yaml.FullLoader)

            self.meta_data = content.get("meta_data")
            self.input_dimension = content.get("input_dimension")
            self.data = content.get("data")

            self.x = [np.array(element.get("x")) for element in self.data]
            self.y = np.array([np.array(element.get("y")) for element in self.data])
            self.pareto_front_evaluations = return_pareto_front_2d(self.y)

    def plot_dots_2d(self, pareto_front: np.ndarray = None):

        if pareto_front is None:
            data = [
                go.Scatter(x=self.y.T[0], y=self.y.T[1], mode="markers", name="Dominated valuations"),
                go.Scatter(x=self.pareto_front_evaluations.T[0], y=self.pareto_front_evaluations.T[1], mode="markers",
                           name="Pareto front of evaluations"),

            ]
        else:
            data = [
                go.Scatter(x=pareto_front.T[0], y=pareto_front.T[1], mode="markers", name="Real pareto front"),
                go.Scatter(x=self.y.T[0], y=self.y.T[1], mode="markers", name="Dominated evaluations"),
                go.Scatter(x=self.pareto_front_evaluations.T[0], y=self.pareto_front_evaluations.T[1], mode="markers",
                           name="Pareto front of evaluations"),

            ]

        fig1 = go.Figure(data=data)

        fig1.update_layout(
            width=800,
            height=600,
            plot_bgcolor="rgba(0,0,0,0)",
            # title=f"({input_dimensions}-dim) GPR weight based MOO: relative Hypervolume: {hypervolume_weight / hypervolume_max * 100}%"
        )

        fig1.show()

    def calculate_hyper_volume_2d(self, reference_point: np.ndarray):
        metric = Hypervolume(ref_point=reference_point, normalize=False)
        return metric.do(self.pareto_front_evaluations)

    def calculate_relative_hyper_volume_2d(self, reference_point: np.ndarray, pareto_front: np.ndarray):
        metric = Hypervolume(ref_point=reference_point, normalize=False)
        hypervolume_max = metric.do(pareto_front)
        hypervolume_evaluations = metric.do(self.pareto_front_evaluations)

        return hypervolume_evaluations / hypervolume_max


class EvaluationMOO:
    def __init__(self, paths: List[Path]):
        self.evaluations_moos = [EvaluationMOOSingleRun(path=path) for path in paths]

    def plot_hyper_volumes_against_dimension(self, reference_point: np.ndarray):
        data = [
            go.Scatter(x=[evaluation_moo.input_dimension for evaluation_moo in self.evaluations_moos],
                       y=[evaluation_moo.calculate_hyper_volume_2d(reference_point=reference_point) for evaluation_moo
                          in self.evaluations_moos],
                       mode="markers", name="Hypervolumes"),
        ]

        fig1 = go.Figure(data=data)

        fig1.update_layout(
            width=800,
            height=600,
            plot_bgcolor="rgba(0,0,0,0)",
            # title=f"({input_dimensions}-dim) GPR weight based MOO: relative Hypervolume: {hypervolume_weight / hypervolume_max * 100}%"
        )

        fig1.show()

    def plot_relative_hyper_volumes_against_dimension(self, reference_point: np.ndarray, pareto_front: np.ndarray):
        data = [
            go.Scatter(x=[evaluation_moo.input_dimension for evaluation_moo in self.evaluations_moos],
                       y=[evaluation_moo.calculate_relative_hyper_volume_2d(reference_point=reference_point,
                                                                            pareto_front=pareto_front) for
                          evaluation_moo
                          in self.evaluations_moos],
                       mode="markers", name="Hypervolumes"),
        ]

        fig1 = go.Figure(data=data)

        fig1.update_layout(
            width=800,
            height=600,
            plot_bgcolor="rgba(0,0,0,0)",
            # title=f"({input_dimensions}-dim) GPR weight based MOO: relative Hypervolume: {hypervolume_weight / hypervolume_max * 100}%"
        )

        fig1.show()
