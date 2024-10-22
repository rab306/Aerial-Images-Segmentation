import sys  
from Src.models.train import train_model  
from Src.models.evaluation import evaluate_model
from Src.models.Unet_model import compile_model

def main():
    """
    Main entry point of the project to handle training and evaluation.
    """

    # Step 1: Compile the model
    print("Compiling the model...")
    model = compile_model()

    # Step 2: Train the model
    print("Training the model...")
    trained_model, history = train_model()
    print("Training completed.")

    # Step 3: Evaluate the model
    print("Evaluating the model...")
    evaluation_results = evaluate_model(trained_model)
    print(f"Evaluation Results: {evaluation_results}")

    print("All steps completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}", file=sys.stderr)
        sys.exit(1)  # Exit with an error code
