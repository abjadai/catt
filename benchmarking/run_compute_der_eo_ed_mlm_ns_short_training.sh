#!/bin/bash

set -o xtrace
for with_case_ending in yes no
do
    echo "CATT Benchmark"
    echo "##############################################################################"
    python compute_der.py benchmarking/eo_ed_mlm_ns/catt_data/CATT_data_gt.txt benchmarking/eo_ed_mlm_ns/catt_data/CATT_data_eo_epoch_5.txt $with_case_ending
    python compute_der.py benchmarking/eo_ed_mlm_ns/catt_data/CATT_data_gt.txt benchmarking/eo_ed_mlm_ns/catt_data/CATT_data_eo_mlm_epoch_5.txt $with_case_ending
    python compute_der.py benchmarking/eo_ed_mlm_ns/catt_data/CATT_data_gt.txt benchmarking/eo_ed_mlm_ns/catt_data/CATT_data_ed_epoch_5.txt $with_case_ending
    python compute_der.py benchmarking/eo_ed_mlm_ns/catt_data/CATT_data_gt.txt benchmarking/eo_ed_mlm_ns/catt_data/CATT_data_ed_mlm_epoch_5.txt $with_case_ending

done

for with_case_ending in yes no
do
    echo "WikiNews Benchmark"
    echo "##############################################################################"
    python compute_der.py benchmarking/eo_ed_mlm_ns/wikinews_data/WikiNews_data_gt.txt benchmarking/eo_ed_mlm_ns/wikinews_data/WikiNews_data_eo_epoch_5.txt $with_case_ending
    python compute_der.py benchmarking/eo_ed_mlm_ns/wikinews_data/WikiNews_data_gt.txt benchmarking/eo_ed_mlm_ns/wikinews_data/WikiNews_data_eo_mlm_epoch_5.txt $with_case_ending
    python compute_der.py benchmarking/eo_ed_mlm_ns/wikinews_data/WikiNews_data_gt.txt benchmarking/eo_ed_mlm_ns/wikinews_data/WikiNews_data_ed_epoch_5.txt $with_case_ending
    python compute_der.py benchmarking/eo_ed_mlm_ns/wikinews_data/WikiNews_data_gt.txt benchmarking/eo_ed_mlm_ns/wikinews_data/WikiNews_data_ed_mlm_epoch_5.txt $with_case_ending
done
