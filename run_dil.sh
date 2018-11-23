python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --batch_size 100 --seed 0 --num_filters 32 --dim_reduction_type dilated_convolution --experiment_name new_filter_32_dil_1 --use_gpu True --stride 1 --dilation 1 --num_epochs 40

python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --batch_size 100 --seed 0 --num_filters 32 --dim_reduction_type dilated_convolution --experiment_name new_filter_32_dil_2 --use_gpu True --stride 1 --dilation 2 --num_epochs 40

python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --batch_size 100 --seed 0 --num_filters 32 --dim_reduction_type dilated_convolution --experiment_name new_filter_32_dil_4 --use_gpu True --stride 1 --dilation 4 --num_epochs 40

python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --batch_size 100 --seed 0 --num_filters 32 --dim_reduction_type dilated_convolution --experiment_name new_filter_32_dil_8 --use_gpu True --stride 1 --dilation 8 --num_epochs 40
