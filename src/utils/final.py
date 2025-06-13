import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

metric_paths = {
    "knn_uu": r"C:\Users\usuario\TFG-Justin\results\filtrado_uu\metrics\metricas.csv",
    "knn_ii": r"C:\Users\usuario\TFG-Justin\results\filtrado_ii\metrics\metricas.csv",
    "svd": r"C:\Users\usuario\TFG-Justin\results\svd_pytorch\modelo_final\metricas\metricas_final.csv",
    "mlp": r"C:\Users\usuario\TFG-Justin\results\mlp_embeddings\final\metrics_mlp.csv",
    "content_based_mlp": r"C:\Users\usuario\TFG-Justin\results\content_based\mlp_noleak\metrics1.csv",
    "openp5": r"C:\Users\usuario\TFG-Justin\results\openp5\metrics1.csv"
}

output_base = r"C:\Users\usuario\TFG-Justin\results\graficas"
os.makedirs(output_base, exist_ok=True)

DEFAULT_EXCLUDE = {"Training_time_s", "nDCGf@10"}
EXTRA_EXCLUDE = {"Diversity", "Novelty"}

special_models = ["openp5", "content_based_mlp"]
max_value = 0.0

for model in special_models:
    path = metric_paths[model]
    df = pd.read_csv(path)
    values = df.iloc[0].to_dict()
    values = {k: v for k, v in values.items() if k not in DEFAULT_EXCLUDE and k not in EXTRA_EXCLUDE}
    max_value = max(max_value, max(values.values()))

scale_common = 1.0 if max_value == 0 else 1.0 / max_value


def plot_radar(metrics, model_name, output_dir, scale=1.0):
    labels = list(metrics.keys())
    values = [v * scale for v in metrics.values()]

    norm_values = []
    for label, val in zip(labels, values):
        if label in {"RMSE", "MAE"}:
            norm_values.append(1 - (val / max(val, 1)))
        else:
            norm_values.append(val)

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles_cycle = np.concatenate([angles, [angles[0]]])
    values_cycle = np.concatenate([norm_values, [norm_values[0]]])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles_cycle, values_cycle, marker="o", label=model_name)
    ax.fill(angles_cycle, values_cycle, alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_title(f"Radar Chart - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_radar_chart.png"))
    plt.close()


def plot_bars_and_lines(metrics, model_name, output_dir, scale=1.0):
    labels = list(metrics.keys())
    values = [v * scale for v in metrics.values()]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color="skyblue")
    plt.title(f"Bar Chart - {model_name}")
    plt.ylabel("Valor")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_bar_chart.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(labels, values, marker="o", linestyle="-", color="tomato")
    plt.title(f"Line Plot - {model_name}")
    plt.ylabel("Valor")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_line_plot.png"))
    plt.close()


for model_name, path in metric_paths.items():
    if not os.path.exists(path):
        print(f"Ruta no encontrada para {model_name}")
        continue

    df = pd.read_csv(path)
    metrics = df.iloc[0].to_dict()
    scale = scale_common if model_name in special_models else 1.0
    filtered = {k: v for k, v in metrics.items() if k not in DEFAULT_EXCLUDE}

    model_output = os.path.join(output_base, model_name)
    os.makedirs(model_output, exist_ok=True)

    plot_bars_and_lines(filtered, model_name, model_output, scale)
    plot_radar(filtered, model_name, model_output, scale)

    if model_name in special_models:
        base_dir = os.path.dirname(path)

        path1 = os.path.join(base_dir, "metrics1.csv")
        if os.path.exists(path1):
            df1 = pd.read_csv(path1)
            m1 = df1.iloc[0].to_dict()
            m1 = {k: v for k, v in m1.items() if k not in DEFAULT_EXCLUDE and k not in EXTRA_EXCLUDE}
            subdir1 = os.path.join(model_output, "sin_div_nov")
            os.makedirs(subdir1, exist_ok=True)
            plot_bars_and_lines(m1, model_name, subdir1, scale_common)
            plot_radar(m1, model_name, subdir1, scale_common)

        path2 = os.path.join(base_dir, "metrics2.csv")
        if os.path.exists(path2):
            df2 = pd.read_csv(path2)
            m2 = df2.iloc[0].to_dict()
            subdir2 = os.path.join(model_output, "con_div_nov")
            os.makedirs(subdir2, exist_ok=True)
            plot_bars_and_lines(m2, model_name, subdir2, scale_common)
            plot_radar(m2, model_name, subdir2, scale_common)
