import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

input_base = r"C:\Users\usuario\TFG-Justin\results\graficas"

rating_models = ["knn_uu", "knn_ii", "svd", "mlp"]

ranking_models_variants = {
    "sin_div_nov": [
        os.path.join("content_based_mlp", "sin_div_nov"),
        os.path.join("openp5", "sin_div_nov")
    ],
    "con_div_nov": [
        os.path.join("content_based_mlp", "con_div_nov"),
        os.path.join("openp5", "con_div_nov")
    ]
}

output_rating = os.path.join(input_base, "global_rating")
output_ranking_base = os.path.join(input_base, "global_ranking")

os.makedirs(output_rating, exist_ok=True)
os.makedirs(output_ranking_base, exist_ok=True)


def create_combined_chart(model_paths, output_path, chart_type):
    images = []
    labels = []

    for model_path in model_paths:
        model_name = os.path.basename(os.path.dirname(model_path))
        file_path = os.path.join(input_base, model_path, f"{model_name}_{chart_type}.png")
        if os.path.exists(file_path):
            images.append(mpimg.imread(file_path))
            labels.append(model_name)

    if not images:
        return

    fig, axes = plt.subplots(1, len(images), figsize=(6 * len(images), 5))
    if len(images) == 1:
        axes = [axes]

    for ax, img, label in zip(axes, images, labels):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(label)

    plt.tight_layout()
    output_file = os.path.join(output_path, f"composite_{chart_type}.png")
    plt.savefig(output_file)
    plt.close()


for chart_type in ["bar_chart", "radar_chart", "line_plot"]:
    create_combined_chart(rating_models, output_rating, chart_type)

for variant, model_paths in ranking_models_variants.items():
    output_variant = os.path.join(output_ranking_base, variant)
    os.makedirs(output_variant, exist_ok=True)
    for chart_type in ["bar_chart", "radar_chart", "line_plot"]:
        create_combined_chart(model_paths, output_variant, chart_type)
