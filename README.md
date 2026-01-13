# 📜 MemRewardBench

> *The first benchmark to systematically evaluate Reward Models' ability to assess long-term memory management in LLMs across contexts up to 128K tokens.*

---

## Introduction

**MemRewardBench** is the first dedicated benchmark for evaluating **Reward Models (RMs)** in their ability to judge long-term memory management processes in Large Language Models. Unlike existing benchmarks that evaluate LLMs directly, MemRewardBench focuses on assessing how well RMs can evaluate the quality of intermediate memory states and memory management trajectories.

The benchmark includes **2,400 high-quality samples** across **three core tasks**: **Long-context Reasoning**, **Multi-turn Dialogue Understanding**, and **Long-form Generation**, with context lengths ranging from **8K to 128K tokens**. Each sample provides:
- A question with long context
- Two memory management trajectories (chosen and rejected)
- Ground-truth judgments based on both outcome correctness and process quality

---

## How to Use

### Data Format

All data in MemRewardBench are standardized to the following format:
```json
{
    "task": "Task category (Long-context_Reasoning, Multi-turn_Dialogue, Long-form_Generation)",
    "chosen": "Higher-quality memory management trajectory with better intermediate states",
    "rejected": "Lower-quality memory management trajectory with suboptimal memory updates",
    "subtask": "Specific setting (e.g., 'Sequential-Noise', 'Parallel', 'Mem0-Out')",
    "ctx_length": "Context length in tokens (8k, 16k, 32k, 64k, or 128k)",
    "question": "The evaluation question along with the full context"
}
```

**Field Descriptions**:
- **task**: Broad task category covering the three main evaluation scenarios
- **chosen**: Memory trajectory that demonstrates superior memory management (more concise, accurate, and logically coherent)
- **rejected**: Memory trajectory with issues like redundant information, dropped critical details, or delayed updates
- **subtask**: Specific memory management pattern and error type (e.g., Sequential-Noise, Mixed-Drop, A-Mem-Mem)
- **ctx_length**: Token-based context length, testing RM capability across different sequence lengths
- **question**: Complete input including the question and long context for evaluation

---

## Evaluation

### 1. Clone and Install
```bash
git clone https://github.com/LCM-Lab/LOOM-Eval
cd LOOM-Eval/loomeval
pip install -e .
```

---

### 2. Evaluate

Once the environment is set up, you can proceed with model evaluation. Here's how you can do it.

### **Run the Evaluation Script**:

```bash
python ./eval.py --model-path meta-llama/Meta-Llama-3.1-8B-Instruct  --gpus 0 1 2 3 4 5 6 7 --tp-size 1
```

This command will start the evaluation for the model. Ensure that the `--model-path` points to the directory of the model you wish to evaluate

#### Parameters:

- `--model-path`: Path to the model directory (required).
- `--save-path`: (Optional) Path to save evaluation results. If not specified, results are saved to the `./results` directory.

---

## Benchmark Statistics

| Task Type | Settings | Context Length Distribution | Total |
|-----------|----------|----------------------------|-------|
| | | 8k / 16k / 32k / 64k / 128k | |
| **Long-context Reasoning** | Sequential-Noise | 101 / 44 / 43 / 36 / 31 | 255 |
| | Sequential-Drop | 35 / 22 / 22 / 40 / 15 | 134 |
| | Mixed-Noise | 22 / 33 / 49 / 46 / 34 | 184 |
| | Mixed-Drop | 19 / 65 / 72 / 43 / 28 | 227 |
| **Multi-turn Dialogue** | Mem0-Out | 27 / 27 / 42 / 48 / 23 | 167 |
| | Mem0-Mem | 25 / 25 / 41 / 47 / 21 | 159 |
| | A-Mem-Out | 42 / 42 / 48 / 50 / 47 | 229 |
| | A-Mem-Mem | 48 / 45 / 49 / 53 / 50 | 245 |
| **Long-form Generation** | Sequential | 49 / 152 / 147 / 67 / 42 | 457 |
| | Parallel | 51 / 48 / 53 / 133 / 58 | 343 |
| **Total** | 10 settings | 419 / 503 / 566 / 563 / 349 | **2,400** |

---

## 📬 Contact

Questions? Suggestions? Reach out at: `iiiigray19@gmail.com` and `zctang2000@gmail.com`

---

## License
This benchmark is released under the Apache-2.0 License.