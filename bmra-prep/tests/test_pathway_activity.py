import bmra_prep.pathway_activity
import numpy as np
import pandas as pd


def test_simple():
    rng = np.random.default_rng(42)

    x = rng.random((1000, 4))
    y = rng.random((4, 4))

    pert_matrix = np.eye(4)

    bmra_prep.pathway_activity.predict_coeffs(x, y, pert_matrix, 10, 1.0, 1.0, 1.0, 1.0)


def test_process_inhibitor_data():
    inhib_conc_df = pd.DataFrame(
        {
            "pert_name": ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p7", "p8", "p8"],
            "drug": ["d1", "d1", "d2", "d2", "d3", "d3", "d4", "d2", "d4", "d2"],
            "dose": [7, 14, 8, 16, 2, 4, 8, 10, 8, 10],
        }
    )
    ic50_df = pd.DataFrame(
        {
            "drug": ["d1", "d2", "d2", "d3", "d4"],
            "module": ["m1", "m2", "m3", "m3", "m3"],
            "ic50": [10, 10, 30, 5, 10],
        }
    )

    y_true, pert_df = bmra_prep.pathway_activity.process_inhibitor_data(
        inhib_conc_df, ic50_df
    )

    correct_y_true = pd.DataFrame(
        np.array(
            [
                [0.58823529, 0.41666667, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 0.55555556, 0.38461538, 1.0, 1.0, 0.5, 0.5],
                [
                    1.0,
                    1.0,
                    0.78947368,
                    0.65217391,
                    0.71428571,
                    0.55555556,
                    0.55555556,
                    0.55555556,
                ],
            ]
        ),
        index=["m1", "m2", "m3"],
        columns=["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"],
    )
    correct_pert_df = pd.DataFrame(
        np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1],
            ]
        ),
        index=["m1", "m2", "m3"],
        columns=["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"],
    )

    pd.testing.assert_frame_equal(y_true, correct_y_true)
    pd.testing.assert_frame_equal(pert_df, correct_pert_df)
