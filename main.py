from pathlib import Path
import subprocess
import sys
import json

def main():
    project_dir = Path(__file__).resolve().parent
    # 读取config.json
    config_file = project_dir / "config.json"
    if not config_file.exists():
        print(f"错误：未找到配置文件：{config_file}")
        sys.exit(1)

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # 获取json中配置的python路径
    target_python = config["python_path"]
    target_python_path = Path(target_python)
    if not target_python_path.exists():
        print(f"错误：配置的Python不存在 -> {target_python}")
        sys.exit(1)
    print(f"使用配置指定Python: {target_python}")

    requirements_file = project_dir / "requirements.txt"
    if not requirements_file.exists():
        print(f"错误：未找到依赖文件：{requirements_file}")
        sys.exit(1)

    print("开始安装 Python 依赖...")

    # ✅ 使用config里指定的python执行pip安装
    command = [
        str(target_python_path),
        "-m",
        "pip",
        "install",
        "-r",
        str(requirements_file),
    ]

    try:
        subprocess.run(command, check=True)
        print("\n依赖安装完成,可以运行 chunker.py。")
        # ✅ 同样用配置里的python运行chunker.py
        chunker_file = project_dir / "chunker.py"
        subprocess.run([str(target_python_path), str(chunker_file)], check=True)
    except subprocess.CalledProcessError as exc:
        print(f"\n执行失败,退出码:{exc.returncode}")
        sys.exit(exc.returncode)

if __name__ == "__main__":
    main()