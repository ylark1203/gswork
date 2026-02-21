# python train_offline.py --subject subject4 --work_name bbw_2000_covarience_dX512 --config config/offline_GBS_dataset.yaml --preload
python calculate_metrics.py --subject subject4 --work_name bbw_2000_covarience_dX512  --output_dir output/HR
# python render.py --subject subject4 --work_name bbw_2000_covarience_dX512 --output_dir output/HR --white_bg