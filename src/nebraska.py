"""
Description:
    Nebraska model
"""
from prep.prep_nbi import *

def main():
    # Prepare dataset
    trainX, trainY = data_preprocessing()

    #TODO: Include IRIS dataset
    # Training and evaluating
    scores, histories = evaluate_model(trainX, trainY)

    # Learning curves
    summarize_diagnostics(histories)

    # Summarize estimated performance
    summarize_performance(scores)

if __name__ =='__main__':
    main()
