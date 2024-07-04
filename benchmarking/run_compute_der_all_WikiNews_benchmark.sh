#!/bin/bash

set -o xtrace

for with_case_ending in yes no
do
    echo "WikiNews Benchmark"
    echo "##############################################################################"
    python compute_der.py benchmarking/all_models_WikiNews_data/WikiNews_data_gt.txt benchmarking/all_models_WikiNews_data/WikiNews_data_ed_mlm_ns_epoch_178.txt $with_case_ending
    python compute_der.py benchmarking/all_models_WikiNews_data/WikiNews_data_gt.txt benchmarking/all_models_WikiNews_data/WikiNews_data_eo_mlm_ns_epoch_193.txt $with_case_ending
    python compute_der.py benchmarking/all_models_WikiNews_data/WikiNews_data_gt.txt benchmarking/all_models_WikiNews_data/WikiNews_data_CBHG_200K_output.txt $with_case_ending
    python compute_der.py benchmarking/all_models_WikiNews_data/WikiNews_data_gt.txt benchmarking/all_models_WikiNews_data/WikiNews_data_Command_R_Plus_v2.txt $with_case_ending
    python compute_der.py benchmarking/all_models_WikiNews_data/WikiNews_data_gt.txt benchmarking/all_models_WikiNews_data/WikiNews_data_GPT4_output.txt $with_case_ending
    python compute_der.py benchmarking/all_models_WikiNews_data/WikiNews_data_gt.txt benchmarking/all_models_WikiNews_data/WikiNews_data_Sakhr_output.txt $with_case_ending
    python compute_der.py benchmarking/all_models_WikiNews_data/WikiNews_data_gt.txt benchmarking/all_models_WikiNews_data/WikiNews_data_Farasa_output.txt $with_case_ending
    python compute_der.py benchmarking/all_models_WikiNews_data/WikiNews_data_gt.txt benchmarking/all_models_WikiNews_data/WikiNews_data_no_tashkeel_deep_diacritization_d2_output.txt $with_case_ending
    python compute_der.py benchmarking/all_models_WikiNews_data/WikiNews_data_gt.txt benchmarking/all_models_WikiNews_data/WikiNews_data_no_tashkeel_deep_diacritization_d3_output.txt $with_case_ending
    python compute_der.py benchmarking/all_models_WikiNews_data/WikiNews_data_gt.txt benchmarking/all_models_WikiNews_data/WikiNews_data_TashkeelAlkhalil_output.txt $with_case_ending
    python compute_der.py benchmarking/all_models_WikiNews_data/WikiNews_data_gt.txt benchmarking/all_models_WikiNews_data/WikiNews_data_Mishkal_output.txt $with_case_ending
    python compute_der.py benchmarking/all_models_WikiNews_data/WikiNews_data_gt.txt benchmarking/all_models_WikiNews_data/WikiNews_data_Multilevel_diacritizer.txt $with_case_ending
    python compute_der.py benchmarking/all_models_WikiNews_data/WikiNews_data_gt.txt benchmarking/all_models_WikiNews_data/WikiNews_data_Shakkala_output.txt $with_case_ending
done
