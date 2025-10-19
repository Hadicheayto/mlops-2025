
TRAIN_RAW="data/titanic/train.csv"
TEST_RAW="data/titanic/test.csv"

TRAIN_PROCESSED="data/titanic/processed/train_processed.csv"
TEST_PROCESSED="data/titanic/processed/test_processed.csv"

TRAIN_FEATURES="data/featurized/train_features.csv"
TEST_FEATURES="data/featurized/test_features.csv"

TRANSFORMER_DIR="data/transformers"
EVAL_DIR="data/eval"
TRAIN_DIR="data/train"

MODEL_PATH="data/models/pipeline_model.pkl"
METRICS_JSON="data/eval/metrics.json"
PREDICTIONS_CSV="data/prediction/predictions.csv"


mkdir -p data/titanic/processed data/featurized data/models data/eval data/prediction


echo "Running preprocessing..."
python scripts/preprocess.py \
    --train_path $TRAIN_RAW \
    --test_path $TEST_RAW \
    --output_train $TRAIN_PROCESSED \
    --output_test $TEST_PROCESSED


echo "Running feature engineering..."
python scripts/featurize.py \
    --train_path $TRAIN_PROCESSED \
    --test_path $TEST_PROCESSED \
    --output_train $TRAIN_FEATURES \
    --output_test $TEST_FEATURES \
    --transformer_dir $TRANSFORMER_DIR \
    --eval_dir $EVAL_DIR \
    --train_dir $TRAIN_DIR \
    --test_size 0.2


echo "Training model..."
python scripts/train.py \
    --train_dir $TRAIN_DIR \
    --transformer_dir $TRANSFORMER_DIR \
    --output_model $MODEL_PATH


echo "Evaluating model..."
python scripts/evaluate.py \
    --model_path $MODEL_PATH \
    --X_path $EVAL_DIR/X_eval.csv \
    --y_path $EVAL_DIR/y_eval.csv \
    --output_json $METRICS_JSON


echo "Running inference on test set..."
python scripts/predict.py \
    --model_path $MODEL_PATH \
    --features_path $TEST_FEATURES \
    --output_path $PREDICTIONS_CSV

echo "Pipeline completed successfully!"
