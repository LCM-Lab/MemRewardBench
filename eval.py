import os

from loomeval.scripts.utils.RetrieveBenchmark import RetrieveDefalutCfgPath
from loomeval.benchmarks.utils.run_benchmark import BenchmarkRunner
import traceback
def main():

    run_inference = 1
    run_eval = 1     
    seed = 42   
    output_dir = "/data/lcm_lab/sora/results"                    # None = use default path       
    model_list = [
        "/data/hf_models/Qwen3-8B",
        "/data/hf_models/Qwen2.5-14B-Instruct",
        "/data/hf_models/Qwen3-4B",
        "/data/hf_models/Llama-3.1-8B-Instruct",
        "/data/hf_models/Llama-3.1-70B-Instruct"
        ]  # model_list
    benchmarks = ["MemRewardBench"]  # Run loomeval.benchmarks to see which benchmarks we support
    cfg_path_list = [
        RetrieveDefalutCfgPath(benchmark_name) for benchmark_name in benchmarks
    ] # These benchmark lists have corresponding configuration files in a fixed order. If you wish to use different config files, modify them sequentially according to this order. If no changes are made, the default cfg_path will be used.

    servers = ["vllm","transformers"]              # transformers / vLLM / RWKV / SGLang
    acceleration = ""                    # SparseAttn, SnapKV, etc.
    max_model_len = 131072             # Maximum context length
    max_length = 131072             # Maximum input sequence length
    gpu_memory_utilization = 0.95       # GPU memory utilization (0.0-1.0)

    # GPU Settings
    gp_list = [0,1,2,3,4,5,6,7]                        # Available GPU device IDs
    gp_num = "1"                         # GPUs per task instance


    assert len(cfg_path_list)==len(benchmarks)
    for server in servers:
        for model_path in model_list:
            for i,benchmark_name in enumerate(benchmarks):
                runner = BenchmarkRunner(benchmark_name)
                save_tag = f"{os.path.basename(model_path.rstrip()) }_{benchmark_name}_{server}" # save_tag; defaults to Modle_benchmark
                try:
                    run_succeeded = True
                    if run_inference:
                        print("🚀 Starting inference...\n")
                        runner.run_inference(
                            model_path=model_path,
                            cfg_path=cfg_path_list[i],
                            save_tag=save_tag,
                            server=server,
                            acceleration=acceleration,
                            gp_list=gp_list,
                            gp_num=gp_num,
                            dynamic_gpu_allocation=dynamic_gpu_allocation,
                            gpu_scaling_map=gpu_scaling_map,
                            max_model_len=max_model_len,
                            gpu_memory_utilization=gpu_memory_utilization,
                            max_length=max_length,
                            skip_existing=skip_existing,
                            limit=limit,
                            limit_model = limit_model,
                            infer_kwargs = infer_kwargs,
                            test_kawrgs=test_kwargs,
                            output_dir=output_dir,
                            seed=seed,
                        )
                    else:
                        print("⏭️  Skipping inference (run_inference=False)\n")
                except Exception as e:
                    error_detail = traceback.format_exc()
                    run_succeeded = False
                    print(error_detail)

                # Step 2: Run Evaluation
                if run_eval and run_succeeded:
                    print("\n📈 Starting evaluation...\n")
                    runner.run_evaluation(
                        save_tag=save_tag,
                        config_paths=cfg_path_list,
                        output_dir=output_dir
                    )
                else:
                    print("\n⏭️  Skipping evaluation (run_eval=False)\n")

if __name__ == "__main__":
    main()