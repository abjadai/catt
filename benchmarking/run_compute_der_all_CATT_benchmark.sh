#!/bin/bash

set -o xtrace
for with_case_ending in yes no
do
    echo "CATT Benchmark"
    echo "##############################################################################"
    python compute_der.py benchmarking/all_models_CATT_data/CATT_data_gt.txt benchmarking/all_models_CATT_data/CATT_data_ed_mlm_ns_epoch_178.txt $with_case_ending
    python compute_der.py benchmarking/all_models_CATT_data/CATT_data_gt.txt benchmarking/all_models_CATT_data/CATT_data_eo_mlm_ns_epoch_193.txt $with_case_ending
    python compute_der.py benchmarking/all_models_CATT_data/CATT_data_gt.txt benchmarking/all_models_CATT_data/CATT_data_CBHG_output_200K.txt $with_case_ending
    python compute_der.py benchmarking/all_models_CATT_data/CATT_data_gt.txt benchmarking/all_models_CATT_data/CATT_data_Command_R_Plus_fixed.txt $with_case_ending
    python compute_der.py benchmarking/all_models_CATT_data/CATT_data_gt.txt benchmarking/all_models_CATT_data/CATT_data_GPT4_output_fixed.txt $with_case_ending
    python compute_der.py benchmarking/all_models_CATT_data/CATT_data_gt.txt benchmarking/all_models_CATT_data/CATT_data_Sakhr_output.txt $with_case_ending
    python compute_der.py benchmarking/all_models_CATT_data/CATT_data_gt.txt benchmarking/all_models_CATT_data/CATT_data_Farasa_output.txt $with_case_ending
    python compute_der.py benchmarking/all_models_CATT_data/CATT_data_gt.txt benchmarking/all_models_CATT_data/CATT_data_deep_diacritization_d2_output.txt $with_case_ending
    python compute_der.py benchmarking/all_models_CATT_data/CATT_data_gt.txt benchmarking/all_models_CATT_data/CATT_data_deep_diacritization_d3_output.txt $with_case_ending
    python compute_der.py benchmarking/all_models_CATT_data/CATT_data_gt.txt benchmarking/all_models_CATT_data/CATT_data_TashkeelAlkhalil_output.txt $with_case_ending
    python compute_der.py benchmarking/all_models_CATT_data/CATT_data_gt.txt benchmarking/all_models_CATT_data/CATT_data_Mishkal_output.txt $with_case_ending
    python compute_der.py benchmarking/all_models_CATT_data/CATT_data_gt.txt benchmarking/all_models_CATT_data/CATT_data_Multilevel_diacritizer.txt $with_case_ending
    python compute_der.py benchmarking/all_models_CATT_data/CATT_data_gt.txt benchmarking/all_models_CATT_data/CATT_data_Shakkala_output.txt $with_case_ending
done
