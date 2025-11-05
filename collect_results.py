import os
import re
import pandas as pd
import argparse

def parse_log(filepath):
    """Parse a single log file and return dict {task: value}"""
    results = {}
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    in_table = False
    for line in lines:
        line = line.strip()
        if line.startswith("|") and "Tasks" in line and "Metric" in line:
            in_table = True
            continue
        if re.match(r"^\|\s*-+\s*\|", line):
            continue
        if in_table:
            if not line.startswith("|"):# or re.match(r"^\|\s*-+\s*\|", line):
                # 表格结束
                in_table = False
                continue

            parts = [p.strip() for p in line.split("|")[1:-1]]  # 去掉两边的 |
            if len(parts) < 6:
                continue

            task, version, flt, nshot, metric, value = parts[:6]

            # 过滤：只保留 acc
            if metric != "acc":
                continue

            # 删除 mmlu 的下属 (以 `- `开头的)
            if task.startswith("- "):
                continue

            # 记录结果
            try:
                results[task] = float(value)
            except ValueError:
                pass
    return results


def main(log_dir, output_file="results.md"):
    all_results = {}
    tasks_set = set()

    for fname in os.listdir(log_dir):
        if not fname.startswith("eval_") or not fname.endswith(".log"):
            continue
        method = fname[len("eval_"):-len(".log")]
        fpath = os.path.join(log_dir, fname)
        res = parse_log(fpath)
        if res:
            all_results[method] = res
            tasks_set.update(res.keys())

    tasks = sorted(tasks_set)
    exclude_tasks = {"truthfulqa_mc2"}  # 👈 自己改
    tasks = sorted(tasks_set - exclude_tasks)
    print(tasks)
    # tasks_set.remove("arc_easy")
    df = pd.DataFrame(all_results).T  # method 作为 index
    df = df[tasks]  # 按任务列排序

    # 计算每个 method 的平均值
    df["average"] = df.mean(axis=1)

    # 输出为 Markdown
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("### Results\n\n") # <-- 新增标题
        f.write(df.to_markdown(floatfmt=".4f"))

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", help="Directory containing eval_*.log files")
    parser.add_argument("--output", default="results.md", help="Output markdown file")
    args = parser.parse_args()

    main(args.log_dir, args.output)