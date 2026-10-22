source /media/data4/zhangzw/course_cifar_sampler/.venv_d031/bin/activate
cd /media/data4/zhangzw/cifar10-fewstep-sampling
export OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=1 python scripts/run_experiment.py --phase all > outputs/logs/dpmpp_cn4.log 2>&1 &
sleep 30
CUDA_VISIBLE_DEVICES=6 python scripts/run_unipc.py --phase all > outputs/logs/unipc_cn4.log 2>&1 &
wait
echo ALL_DONE > outputs/logs/done_cn4.flag
