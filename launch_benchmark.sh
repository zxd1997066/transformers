#!/bin/bash
set -xe

function main {
    # set common info
    source oob-common/common.sh
    init_params $@
    fetch_device_info
    set_environment

    # requirements
    if [ "${EXAMPLE_ARGS}" == "" ];then
        set +x
        echo "[ERROR] Please set EXAMPLE_ARGS before launch"
        echo "  export EXAMPLE_ARGS=/example/args/use"
        echo "      Example: export EXAMPLE_ARGS='text-classification/run_glue.py --task_name MRPC --model_name_or_path xlnet-base-cased' "
        exit 1
        set -x
    fi
    model_args="  "
    pip uninstall -y transformers tokenizers || true
    rm -rf ~/.cache/huggingface && rm -rf /tmp/output* || true
    pip install -e .
    # huggingface models deps
    pip install nltk rouge_score seqeval sacrebleu sentencepiece jiwer zstandard
    # speechbrain and its deps
    pip install --no-deps speechbrain
    pip install hyperpyyaml joblib ruamel.yaml ruamel.yaml.clib scipy
    pip install librosa soundfile
    if [ "${device}" == "cuda" ];then
        pip install typing_extensions==4.7.1 opencv-python==4.8.0.74 opencv-python-headless==4.8.0.74 
    fi

    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        # cache
        python examples/${framework}/$(echo ${EXAMPLE_ARGS}) \
            --do_eval --output_dir /tmp/output0 --overwrite_output_dir \
            --per_device_eval_batch_size 1 \
            --num_iter 3 --num_warmup 1 \
            --precision $precision \
            --channels_last $channels_last \
            --device_oob $device \
            ${addtion_options} || true
        #
        for batch_size in ${batch_size_list[@]}
        do
            if [ $batch_size -le 0 ];then
                batch_size=8
            fi
            # clean workspace
            logs_path_clean
            # generate launch script for multiple instance
            # export TORCH_LOGS="recompiles"
            if [ "${OOB_USE_LAUNCHER}" == "1" ] && [ "${device}" != "cuda" ];then
                generate_core_launcher
            else
                generate_core
            fi
            # launch
            echo -e "\n\n\n\n Running..."
            cat ${excute_cmd_file} |column -t > ${excute_cmd_file}.tmp
            mv ${excute_cmd_file}.tmp ${excute_cmd_file}
            source ${excute_cmd_file}
            echo -e "Finished.\n\n\n\n"
            # collect launch result
            collect_perf_logs
        done
    done
}

# run
function generate_core {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${device_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        # instances
        if [ "${device}" != "cuda" ];then
            OOB_EXEC_HEADER=" numactl -m $(echo ${device_array[i]} |awk -F ';' '{print $2}') "
            OOB_EXEC_HEADER+=" -C $(echo ${device_array[i]} |awk -F ';' '{print $1}') "
        else
            OOB_EXEC_HEADER=" CUDA_VISIBLE_DEVICES=${device_array[i]} "
        fi
        printf " ${OOB_EXEC_HEADER} \
            python examples/${framework}/$(echo ${EXAMPLE_ARGS}) \
                --do_eval --output_dir /tmp/output1 --overwrite_output_dir \
                --per_device_eval_batch_size $batch_size \
                --num_iter $num_iter --num_warmup $num_warmup \
                --precision $precision \
                --channels_last $channels_last \
                --device_oob $device \
                ${addtion_options} \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
        if [ "${numa_nodes_use}" == "0" ];then
            break
        fi
    done
    echo -e "\n wait" >> ${excute_cmd_file}
}

function generate_core_launcher {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${device_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        printf "python -m oob-common.launch --enable_jemalloc \
                    --core_list $(echo ${device_array[@]} |sed 's/;.//g') \
                    --log_file_prefix rcpi${real_cores_per_instance} \
                    --log_path ${log_dir} \
                    --ninstances ${#device_array[@]} \
                    --ncore_per_instance ${real_cores_per_instance} \
            examples/${framework}/$(echo ${EXAMPLE_ARGS}) \
                --do_eval --output_dir /tmp/output1 --overwrite_output_dir \
                --per_device_eval_batch_size $batch_size \
                --num_iter $num_iter --num_warmup $num_warmup \
                --precision $precision \
                --channels_last $channels_last \
                ${addtion_options} \
        > /dev/null 2>&1 &  \n" |tee -a ${excute_cmd_file}
        break
    done
    echo -e "\n wait" >> ${excute_cmd_file}
}

# download common files
rm -rf oob-common && git clone https://github.com/intel-sandbox/oob-common.git

# Start
main "$@"
