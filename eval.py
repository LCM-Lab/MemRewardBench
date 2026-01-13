
import os, argparse
from loomeval.benchmarks.utils.run_benchmark import BenchmarkRunner
from loomeval.scripts.utils.RetrieveBenchmark import RetrieveDefalutCfgPath
import traceback
def main(args):    
    runner = BenchmarkRunner("MemRewardBench")
    try:
        run_succeeded = True
        print("🚀 Starting inference...\n")
        runner.run_inference(
            model_path=args.model_path,
            cfg_path=RetrieveDefalutCfgPath("MemRewardBench"),
            save_tag=f"{os.path.basename(args.model_path.rstrip()) }_MemRewardBench", # save_tag; defaults to Modle_benchmark",
            server="vllm",  # transformers / vLLM / RWKV / SGLang
            acceleration = "", # SparseAttn, SnapKV, etc.
            gp_list=args.gpus,
            gp_num=args.tp_size,
            dynamic_gpu_allocation=False,
            gpu_scaling_map= "17384:1,70536:2,148000:8", # Length:GPU mapping
            max_model_len = 131072,              # Maximum context length
            max_length = 131072,             # Maximum input sequence length
            gpu_memory_utilization = 0.95,       # GPU memory utilization (0.0-1.0)
            batch_size = 1,
            skip_existing=True,
            limit=1,
            limit_model = "global",
            infer_kwargs = None,
            output_dir=args.save_path,
            seed=42,
        )
    except Exception as e:
        error_detail = traceback.format_exc()
        run_succeeded = False
        print(error_detail)
    
    # Step 2: Run Evaluation
    if  run_succeeded:
        print("\n📈 Starting evaluation...\n")
        runner.run_evaluation(
            save_tag=f"{os.path.basename(args.model_path.rstrip()) }_MemRewardBench",
            config_paths=[RetrieveDefalutCfgPath("MemRewardBench")],
            output_dir=args.save_path
        )
    else:
        print("\n⏭️  Skipping evaluation (run_eval=False)\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type = str, required = True
    )

    parser.add_argument(
        "--save-path", type = str, default = "./Results",
        help = "The Evaluating Results Saving path"
    )
    
    parser.add_argument(
        "--gpus",type = int, nargs = "+", default = [0, 1, 2, 3, 4, 5, 6, 7]
    )

    parser.add_argument(
        "--tp-size",type = str, default = "1"
    )

    args = parser.parse_args()
    main(args)
        