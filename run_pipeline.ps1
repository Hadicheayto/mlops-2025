# run_pipeline.ps1
# PowerShell script to run the full ML pipeline

Write-Host "Step 1: Preprocessing..."
python scripts/preprocess.py `
    --train_path data/titanic/train.csv `
    --test_path data/titanic/test.csv `
    --output_train data/titanic/processed/train_processed.csv `
    --output_test data/titanic/processed/test_processed.csv

Write-Host "Step 2: Feature engineering..."
python scripts/featurize.py `
    --train_path data/titanic/processed/train_processed.csv `
    --test_path data/titanic/processed/test_processed.csv `
    --output_train data/featurized/train_features.csv `
    --output_test data/featurized/test_features.csv `
    --transformer_dir data/transformers `
    --eval_dir data/eval `
    --train_dir data/train `
    --test_size 0.2

Write-Host "Step 3: Training the model..."
python scripts/train.py `
    --train_dir data/featurized/train `
    --transformer_dir data/transformers `
    --output_model data/models/pipeline_model.pkl

Write-Host "Step 4: Evaluating the model..."
python scripts/evaluate.py `
    --model_path data/models/pipeline_model.pkl `
    --X_path data/featurized/eval/X_eval.csv `
    --y_path data/featurized/eval/y_eval.csv `
    --output_json data/eval/metrics.json

Write-Host "Step 5: Running inference on test data..."
python scripts/predict.py `
    --model_path data/models/pipeline_model.pkl `
    --features_path data/featurized/test_features.csv `
    --output_path data/prediction/predictions.csv

Write-Host "Pipeline finished!"
