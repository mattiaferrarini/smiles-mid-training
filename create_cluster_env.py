import os
import argparse
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create cluster environment script")
    parser.add_argument("-e", "--env_file", type=str, required=True, help="Name of the environment file to adapt for cluster use")
    args = parser.parse_args()

    env_file = args.env_file
    if os.path.exists(env_file):
        directory = os.path.dirname(env_file)
        filename = os.path.basename(env_file)
        cluster_env_file = os.path.join(directory, f"cluster_{filename}")
        with open(env_file, 'r') as f:
            lines = f.readlines()

        with open(cluster_env_file, 'w') as f:
            for line in lines:
                if line.startswith('prefix'):
                    # Skip the prefix line for portability
                    continue
                if '=' in line and not line.strip().startswith('-'):
                    # Remove hash from pip packages in requirements list: package==version=hash -> package==version
                    cleaned_line = re.sub(r'(.*==[\d\.]+)=[a-f0-9_]+', r'\1', line)
                elif '=' in line:
                    # Remove build info from conda packages in environment file: package=version=build -> package=version
                    cleaned_line = re.sub(r'(.*=[\d\.]+)=[\w]+_[\d]+', r'\1', line)
                else:
                    cleaned_line = line
                f.write(cleaned_line)
        print(f"Cluster environment file '{cluster_env_file}' created successfully.")
    else:
        print(f"Environment file '{env_file}' does not exist.")