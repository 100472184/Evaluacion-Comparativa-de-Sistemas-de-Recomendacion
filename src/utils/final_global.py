import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

input_base = r"C:\Users\usuario\TFG-Justin\results\graficas"
rating_models = ["knn_uu", "knn_ii", "svd", "mlp"]
output_rating = os.path.join(input_base, "global_rating")
os.makedirs(output_rating, exist_ok=True)


def create_combined_chart(model_list, output_path, chart_type):
    images = []
    for model in model_list:
        path = os.path.join(input_base, model, f"{model}_{chart_type}.png")
        if os.path.exists(path):
            images.append(mpimg.imread(path))

    if not images:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, img, model in zip(axes, images, model_list):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(model)

    for ax in axes[len(images):]:
        ax.axis("off")

    plt.tight_layout()
    output_file = os.path.join(output_path, f"composite_{chart_type}.png")
    plt.savefig(output_file)
    plt.close()


for chart_type in ["bar_chart", "radar_chart", "line_plot"]:
    create_combined_chart(rating_models, output_rating, chart_type)
