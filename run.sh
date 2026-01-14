python train_offline.py --subject bala --work_name offline_spiral_mean_std2 --config config/offline_spiral.yaml --preload
python calculate_metrics.py --subject bala --work_name offline_spiral_mean_std2  --output_dir output/INSTA


# python train_offline.py --subject subject1 --work_name gcn_mesh --config config/offline_GBS_spiral.yaml --preload
# python calculate_metrics.py --subject subject1 --work_name gcn_mesh  --output_dir output/HR

# python train_offline.py --subject subject2 --work_name gcn_mesh --config config/offline_GBS_spiral.yaml --preload
# python calculate_metrics.py --subject subject2 --work_name gcn_mesh  --output_dir output/HR

# python train_offline.py --subject subject3 --work_name gcn_mesh --config config/offline_GBS_spiral.yaml --preload
# python calculate_metrics.py --subject subject3 --work_name gcn_mesh  --output_dir output/HR

# python train_offline.py --subject subject4 --work_name gcn_mesh --config config/offline_GBS_spiral.yaml --preload
# python calculate_metrics.py --subject subject4 --work_name gcn_mesh  --output_dir output/HR