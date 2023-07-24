python run_glue_no_trainer_v2.py \
--model_name_or_path  path_to_bert \
--per_device_train_batch_size 32 \
--learning_rate 2e-5 \
--num_train_epochs 3 \
--output_dir path_to_output \
--train_file ../data/novel_train_generate_v2.csv \
--test_file ../data/novel_valid_generate_v2.csv