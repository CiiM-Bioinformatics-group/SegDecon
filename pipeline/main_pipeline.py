# main_pipeline.py

from denoising_and_segmentation import run_pipeline as preprocess_run_pipeline
from deconvolution import run_pipeline as deconvolution_run_pipeline
from postprocess import run_pipeline as postprocess_run_pipeline

def main():
    """Main function to run the entire pipeline in sequence"""
    print("Starting pipeline...")

    # Step 1: Run preprocessing and segmentation
    preprocess_run_pipeline()
    
    # Step 2: Run deconvolution
    deconvolution_run_pipeline()
    
    # Step 3: Run postprocessing
    postprocess_run_pipeline()

    print("Pipeline finished.")

if __name__ == "__main__":
    main()
