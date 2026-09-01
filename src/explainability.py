import os
import shap
import matplotlib.pyplot as plt

def generate_shap_plots(model, X_sample, output_dir: str = "reports/figures"):
    """Generate global SHAP feature importance plot."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    
    output_path = os.path.join(output_dir, "shap_summary.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved SHAP summary plot to '{output_path}'")
