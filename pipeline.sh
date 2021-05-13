python scripts/generate_time_series.py

python time_series_forecasting/data_utils.py --csv_path "data/data.csv" \
                                             --out_path "data/processed_data.csv" \
                                             --config_path "data/config.json"

# Train
# Trains a model and saves the model in models/ts_models/

python time_series_forecasting/training.py --data_csv_path "data/processed_data.csv" \
                                           --feature_target_names_path "data/config.json" \
                                           --output_json_path "models/trained_config.json" \
                                           --log_dir "models/ts_views_logs" \
                                           --model_dir "models/ts_views_models"

python time_series_forecasting/evaluation.py --data_csv_path "data/processed_data.csv" \
                                             --feature_target_names_path "data/config.json" \
                                             --trained_json_path "data/trained_config.json" \
                                             --eval_json_path "data/eval.json" \
                                             --data_for_visualization_path "data/visualization.json"

python time_series_forecasting/plot_images.py