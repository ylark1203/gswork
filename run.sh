python train_offline.py --subject bala --work_name learnableshear_weight_to_xyz --config config/offline.yaml --preload
python calculate_metrics.py --subject bala --work_name learnableshear_weight_to_xyz  --output_dir output/INSTA
python render.py --subject nf_03 --work_name reproduction --output_dir output/INSTA

python train_offline.py --subject biden --work_name learnableshear_weight_to_xyz --config config/offline.yaml --preload
python calculate_metrics.py --subject biden --work_name learnableshear_weight_to_xyz  --output_dir output/INSTA
python render.py --subject nf_03 --work_name reproduction --output_dir output/INSTA

python train_offline.py --subject justin --work_name learnableshear_weight_to_xyz --config config/offline.yaml --preload
python calculate_metrics.py --subject justin --work_name learnableshear_weight_to_xyz  --output_dir output/INSTA
python render.py --subject nf_03 --work_name reproduction --output_dir output/INSTA

python train_offline.py --subject malte_1 --work_name learnableshear_weight_to_xyz --config config/offline.yaml --preload
python calculate_metrics.py --subject malte_1 --work_name learnableshear_weight_to_xyz  --output_dir output/INSTA
python render.py --subject nf_03 --work_name reproduction --output_dir output/INSTA

python train_offline.py --subject marcel --work_name learnableshear_weight_to_xyz --config config/offline.yaml --preload
python calculate_metrics.py --subject marcel --work_name learnableshear_weight_to_xyz  --output_dir output/INSTA
python render.py --subject nf_03 --work_name reproduction --output_dir output/INSTA

python train_offline.py --subject nf_01 --work_name learnableshear_weight_to_xyz --config config/offline.yaml --preload
python calculate_metrics.py --subject nf_01 --work_name learnableshear_weight_to_xyz  --output_dir output/INSTA
python render.py --subject nf_03 --work_name reproduction --output_dir output/INSTA

python train_offline.py --subject nf_03 --work_name learnableshear_weight_to_xyz --config config/offline.yaml --preload
python calculate_metrics.py --subject nf_03 --work_name learnableshear_weight_to_xyz  --output_dir output/INSTA
python render.py --subject nf_03 --work_name reproduction --output_dir output/INSTA

python train_offline.py --subject wojtek_1 --work_name learnableshear_weight_to_xyz --config config/offline.yaml --preload
python calculate_metrics.py --subject wojtek_1 --work_name learnableshear_weight_to_xyz  --output_dir output/INSTA
python render.py --subject nf_03 --work_name reproduction --output_dir output/INSTA

python train_offline.py --subject subject1 --work_name learnableshear_weight_to_xyz --config config/offline_GBS_dataset.yaml --preload
python calculate_metrics.py --subject subject1 --work_name learnableshear_weight_to_xyz  --output_dir output/HR
python render.py --subject nf_03 --work_name reproduction --output_dir output/INSTA

python train_offline.py --subject subject2 --work_name learnableshear_weight_to_xyz --config config/offline_GBS_dataset.yaml --preload
python calculate_metrics.py --subject subject2 --work_name learnableshear_weight_to_xyz  --output_dir output/HR
python render.py --subject nf_03 --work_name reproduction --output_dir output/INSTA

python train_offline.py --subject subject3 --work_name learnableshear_weight_to_xyz --config config/offline_GBS_dataset.yaml --preload
python calculate_metrics.py --subject subject3 --work_name learnableshear_weight_to_xyz  --output_dir output/HR
python render.py --subject nf_03 --work_name reproduction --output_dir output/INSTA

python train_offline.py --subject subject4 --work_name learnableshear_weight_to_xyz --config config/offline_GBS_dataset.yaml --preload
python calculate_metrics.py --subject subject4 --work_name learnableshear_weight_to_xyz  --output_dir output/HR
python render.py --subject nf_03 --work_name reproduction --output_dir output/INSTA