import argparse
import onnx
from onnxsim import simplify

def simplify_onnx(input_model_path, output_model_path="export/simplified_model.onnx"):
    """
    Simplifies an ONNX model and saves the result.

    Args:
        input_model_path (str): Path to the input ONNX model.
        output_model_path (str): Path to save the simplified ONNX model.
    """
    # Load the original ONNX model
    model = onnx.load(input_model_path)

    # Simplify the model
    print("Simplifying the ONNX model...")
    simplified_model, check = simplify(model)

    # Check if the simplified model is valid
    if not check:
        raise ValueError("Simplified ONNX model could not be validated")

    # Save the simplified model
    onnx.save(simplified_model, output_model_path)
    print(f"Simplified model saved to {output_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplify an ONNX model using ONNX Simplifier.")
    parser.add_argument(
        "--input_model_path", 
        type=str, 
        required=True, 
        help="Path to the input ONNX model"
    )
    parser.add_argument(
        "--output_model_path", 
        type=str, 
        default="export/simplified_model.onnx", 
        help="Path to save the simplified ONNX model (default: export/simplified_model.onnx)"
    )

    args = parser.parse_args()
    simplify_onnx(args.input_model_path, args.output_model_path)

'''
python simplify_onnx.py --input_model_path /path/to/model.onnx --output_model_path /path/to/simplified_model.onnx
'''