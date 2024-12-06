import os
import argparse
import platform
import subprocess
from pathlib import Path

def run_command(command):
    """运行系统命令并返回输出"""
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              encoding='utf-8')
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e.stderr}")
        return None

def deploy_to_pi(pi_host, pi_user, project_dir, target_dir):
    """部署项目到树莓派"""
    # 确保项目目录存在
    project_path = Path(project_dir)
    if not project_path.exists():
        print(f"Error: Project directory {project_dir} does not exist!")
        return False
    
    # 构建scp命令
    if platform.system() == 'Windows':
        # Windows下使用pscp
        scp_cmd = f'pscp -r "{project_dir}" {pi_user}@{pi_host}:{target_dir}'
        print("\nWindows系统检测到，请确保已安装PuTTY并将pscp添加到系统PATH中")
        print("如果没有安装PuTTY，请从以下地址下载：")
        print("https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html")
    else:
        # Linux/Mac下使用scp
        scp_cmd = f'scp -r "{project_dir}" {pi_user}@{pi_host}:{target_dir}'
    
    print(f"\n开始传输项目到树莓派...")
    print(f"命令: {scp_cmd}")
    
    # 执行传输
    result = run_command(scp_cmd)
    
    if result is not None:
        print("\n项目成功传输到树莓派！")
        
        # 打印后续步骤说明
        print("\n后续步骤：")
        print("1. SSH登录到树莓派:")
        print(f"   ssh {pi_user}@{pi_host}")
        print("\n2. 进入项目目录:")
        print(f"   cd {target_dir}/yolov5-lite")
        print("\n3. 安装依赖:")
        print("   pip3 install -r requirements.txt")
        print("\n4. 运行测试:")
        print("   python3 detect.py --source 0  # 使用摄像头")
        print("   python3 detect.py --source data/images  # 检测图片")
        return True
    else:
        print("\n项目传输失败！请检查:")
        print("1. 树莓派是否开机并连接到网络")
        print("2. IP地址是否正确")
        print("3. 用户名是否正确")
        print("4. SSH是否已在树莓派上启用")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='部署YOLOv5-lite项目到树莓派')
    parser.add_argument('--host', type=str, required=True, help='树莓派的IP地址')
    parser.add_argument('--user', type=str, default='pi', help='树莓派的用户名(默认: pi)')
    parser.add_argument('--project-dir', type=str, default='.',
                        help='项目目录路径(默认: 当前目录)')
    parser.add_argument('--target-dir', type=str, default='~/projects',
                        help='树莓派上的目标目录(默认: ~/projects)')
    
    args = parser.parse_args()
    deploy_to_pi(args.host, args.user, args.project_dir, args.target_dir)
