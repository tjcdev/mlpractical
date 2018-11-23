python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --batch_size 100 --seed 0 --num_filters 32 --dim_reduction_type strided_convolution --experiment_name new_filter_32_stride_2 --use_gpu True --stride 2 --dilation 1 --num_epochs 40

python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --batch_size 100 --seed 0 --num_filters 32 --dim_reduction_type strided_convolution --experiment_name new_filter_32_stride_4 --use_gpu True --stride 4 --dilation 1 --num_epochs 40

python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --batch_size 100 --seed 0 --num_filters 32 --dim_reduction_type strided_convolution --experiment_name new_filter_32_stride_8 --use_gpu True --stride 8 --dilation 1 --num_epochs 40
