[max-pooling]
[DONE]
python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --batch_size 100 --seed 0 --num_filters 48 --dim_reduction_type max_pooling --experiment_name max_pooling_1 --use_gpu True --num_epochs 100

[avg_pooling]
python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --batch_size 100 --seed 0 --num_filters 48 --dim_reduction_type avg_pooling --experiment_name avg_pooling_1 --use_gpu True --num_epochs 100

[our-model]
