# python train_offline.py --subject biden --work_name reproduction --config config/offline.yaml --preload
# python calculate_metrics.py --subject subject1 --work_name reproduction --output_dir output/HR

# python train_offline.py --subject justin --work_name reproduction --config config/offline.yaml --preload
python calculate_metrics.py --subject subject2 --work_name reproduction --output_dir output/HR

# python train_offline.py --subject malte_1 --work_name reproduction --config config/offline.yaml --preload
python calculate_metrics.py --subject subject3 --work_name reproduction --output_dir output/HR

python train_offline.py --subject subject4 --work_name reproduction --config config/offline_HR.yaml --preload
python calculate_metrics.py --subject subject4 --work_name reproduction --output_dir output/HR