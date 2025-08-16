#!/usr/bin/env python3
"""
Web-based interactive viewer for a subject's neuron point cloud with toggleable
brain regions using Dash + Plotly (WebGL 3D scatter).

Usage:
  python -m GenerativeBrainModel.scripts.web_view_subject_masks \
    --subject subject_1 \
    --data-dir /home/user/gbm3/GBM3/processed_spike_voxels_2018 \
    --masks-dir /home/user/gbm3/GBM3/processed_spike_voxels_2018_masks \
    --points 120000 --top-k 12 --host 0.0.0.0 --port 8050

Then open http://<server_ip>:8050 in your browser.
"""

import os
import argparse
import numpy as np
import h5py

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go


def _subject_files(subject: str, data_dir: str, masks_dir: str):
    data_h5 = os.path.join(data_dir, f"{subject}.h5")
    mask_h5 = os.path.join(masks_dir, f"{subject}_mask.h5")
    if not os.path.exists(data_h5):
        raise FileNotFoundError(data_h5)
    if not os.path.exists(mask_h5):
        raise FileNotFoundError(mask_h5)
    return data_h5, mask_h5


def _load_positions(h5_path: str) -> np.ndarray:
    with h5py.File(h5_path, 'r') as f:
        pos = f['cell_positions'][:].astype(np.float32)
        pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        pos = np.clip(pos, 0.0, 1.0)
        return pos


def _load_labels(mask_h5: str):
    with h5py.File(mask_h5, 'r') as f:
        label_xyz = f['label_volume'][:]
        region_names = np.array(f['region_names']).astype(str)
    return label_xyz, region_names


def _labels_for_points(pos01: np.ndarray, label_xyz: np.ndarray) -> np.ndarray:
    X, Y, Z = label_xyz.shape
    xi = np.clip(np.round(pos01[:, 0] * (X - 1)).astype(np.int32), 0, X - 1)
    yi = np.clip(np.round(pos01[:, 1] * (Y - 1)).astype(np.int32), 0, Y - 1)
    zi = np.clip(np.round(pos01[:, 2] * (Z - 1)).astype(np.int32), 0, Z - 1)
    labels = label_xyz[xi, yi, zi]
    return labels.astype(np.int32)


def _top_k_regions(labels: np.ndarray, k: int, max_label: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=max_label + 1)
    counts[0] = 0  # background
    order = np.argsort(counts)[::-1]
    return order[:k]


def build_app(subject: str, pos01: np.ndarray, labels: np.ndarray, region_names: np.ndarray, points: int, top_k: int):
    # Sample for performance
    n = pos01.shape[0]
    if n > points:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=points, replace=False)
        pos01 = pos01[idx]
        labels = labels[idx]

    max_label = int(labels.max())
    top_regions = [int(r) for r in _top_k_regions(labels, top_k, max_label) if r > 0][: top_k]

    # Build a list of (id, name)
    options = []
    for rid in range(1, max_label + 1):
        nm = region_names[rid - 1] if (rid - 1) < len(region_names) else f"Region {rid}"
        label = nm
        options.append({'label': label, 'value': rid})

    # Initial active regions: top_k
    initial_active = top_regions

    app = dash.Dash(__name__)
    app.title = f"Subject viewer: {subject}"

    app.layout = html.Div([
        html.Div([
            html.H3(f"Subject: {subject}"),
            html.Div([
                html.Label("Active regions"),
                dcc.Dropdown(id='region-dropdown', options=options, value=initial_active, multi=True,
                             placeholder='Select regions to display'),
            ]),
            html.Div([
                dcc.Checklist(id='show-background', options=[{'label': 'Show background (unselected) points', 'value': 'bg'}],
                               value=['bg'])
            ], style={'marginTop': '8px'}),
            html.Div([
                html.Label("Marker size"),
                dcc.Slider(id='marker-size', min=1, max=5, step=1, value=2, marks=None),
            ], style={'marginTop': '8px'}),
        ], style={'width': '24%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '8px', 'boxSizing': 'border-box'}),

        html.Div([
            dcc.Graph(id='scatter3d', style={'height': '92vh'}),
        ], style={'width': '76%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        # Hidden stores for data
        dcc.Store(id='pos-x', data=pos01[:, 0].astype(np.float32).tolist()),
        dcc.Store(id='pos-y', data=pos01[:, 1].astype(np.float32).tolist()),
        dcc.Store(id='pos-z', data=pos01[:, 2].astype(np.float32).tolist()),
        dcc.Store(id='labels', data=labels.astype(np.int32).tolist()),
        dcc.Store(id='region-names', data=list(map(str, region_names))),
    ])

    @app.callback(
        Output('scatter3d', 'figure'),
        Input('region-dropdown', 'value'),
        Input('show-background', 'value'),
        Input('marker-size', 'value'),
        State('pos-x', 'data'),
        State('pos-y', 'data'),
        State('pos-z', 'data'),
        State('labels', 'data'),
        State('region-names', 'data'),
        prevent_initial_call=False,
    )
    def update_figure(active_regions, show_bg, marker_size, x, y, z, lbls, names):
        import plotly.colors as pc
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        z = np.asarray(z, dtype=np.float32)
        lbls = np.asarray(lbls, dtype=np.int32)
        names = list(names)

        active_set = set(active_regions or [])
        traces = []

        # Background trace (points with label 0 or not in active)
        if 'bg' in (show_bg or []):
            bg_mask = (lbls == 0) | (~np.isin(lbls, list(active_set)))
            if np.any(bg_mask):
                traces.append(go.Scatter3d(
                    x=x[bg_mask], y=y[bg_mask], z=z[bg_mask],
                    mode='markers',
                    marker=dict(size=marker_size, color='rgba(150,150,150,0.15)')
                ))

        # One trace per active region for legend toggling
        # Use Plotly qualitative colors cyclically
        palettes = pc.qualitative.Plotly + pc.qualitative.Dark24 + pc.qualitative.Set3
        for i, rid in enumerate(sorted(active_set)):
            mask = (lbls == rid)
            if not np.any(mask):
                continue
            color = palettes[i % len(palettes)]
            name = names[rid - 1] if (rid - 1) < len(names) else f"Region {rid}"
            traces.append(go.Scatter3d(
                x=x[mask], y=y[mask], z=z[mask],
                mode='markers',
                name=name,
                marker=dict(size=marker_size, color=color, opacity=0.85)
            ))

        layout = go.Layout(
            scene=dict(
                xaxis=dict(title='X', range=[0, 1]),
                yaxis=dict(title='Y', range=[0, 1]),
                zaxis=dict(title='Z', range=[0, 1]),
                aspectmode='cube',
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            legend=dict(itemsizing='constant')
        )
        fig = go.Figure(data=traces, layout=layout)
        return fig

    return app


def main():
    ap = argparse.ArgumentParser(description='Web viewer for subject neuron masks')
    ap.add_argument('--subject', type=str, required=True)
    ap.add_argument('--data-dir', type=str, default='/home/user/gbm3/GBM3/processed_spike_voxels_2018')
    ap.add_argument('--masks-dir', type=str, default='/home/user/gbm3/GBM3/processed_spike_voxels_2018_masks')
    ap.add_argument('--points', type=int, default=120000)
    ap.add_argument('--top-k', type=int, default=12)
    ap.add_argument('--host', type=str, default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8050)
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    data_h5, mask_h5 = _subject_files(args.subject, args.data_dir, args.masks_dir)
    pos01 = _load_positions(data_h5)
    label_xyz, region_names = _load_labels(mask_h5)
    labels = _labels_for_points(pos01, label_xyz)

    app = build_app(args.subject, pos01, labels, region_names, points=args.points, top_k=args.top_k)
    app.run_server(debug=args.debug, host=args.host, port=args.port)


if __name__ == '__main__':
    main()





